#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficient Vision Transformer with Hybrid Optical-Electronic Architecture

ARCHITECTURE DESIGN (Option A):
    - PatchEmbedding: Standard Conv2d (large kernel, one-time operation)
    - Attention: Standard PyTorch MultiheadAttention
    - ConvFFN Expand (dim→hidden): Optical Conv using CNN_layer (spatial parallelism advantage)
    - ConvFFN Compress (hidden→dim): Standard 1x1 Conv (no spatial locality, GPU-optimized)

RATIONALE:
    1. Optical computation excels at spatial parallel operations (3x3 conv)
    2. 1x1 convolution (pointwise) is already highly optimized on GPUs
    3. Hybrid approach maximizes performance while maintaining optical computing benefits
    4. Future work: Explore pure optical architecture (Option B) with architectural innovations

Using optimized CNN_layer from ONN-MZI codebase for optical convolution layers.
"""

import sys
import os
import math
import time
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to import from ONN-MZI
parent_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(parent_dir / "ONN-MZI"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# Import efficient CNN modules from ONN-MZI
from module.CNN import CNN_layer
from module.MZI_array.mzi import MZI

print("[Import] Successfully imported CNN_layer from ONN-MZI")


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class CFG:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 5e-2
    label_smoothing: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    depth: int = 2  # Number of transformer blocks
    embed_dim: int = 20  # Must be compatible with optical core (divisible by in_channels)
    num_heads: int = 4
    mlp_ratio: float = 4.0


# ----------------------------
# Efficient Optical Conv using CNN_layer
# ----------------------------

class EfficientOpticalConv(nn.Module):
    """
    Optical convolution using the efficient CNN_layer from ONN-MZI

    This replaces the slow OpticalConv2d with nested loops.
    CNN_layer uses pre-computed transfer matrices and batched operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 mzi_repeat_num=5, detection_mode='power'):
        super().__init__()

        # Validate channel compatibility
        if out_channels % in_channels != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by in_channels ({in_channels})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Use CNN_layer from ONN-MZI (10x10 MZI mesh, efficient implementation)
        self.cnn_layer = CNN_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            mzi_repeat_num=mzi_repeat_num,
            mzi_row_num=5,  # 10x10 mesh: 5 MZIs per row
            mzi_column_num=4,  # 4 MZIs per column
            detection_mode=detection_mode
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, out_channels, H', W')
        """
        return self.cnn_layer(x)


# ----------------------------
# Vision Transformer Components
# ----------------------------

class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings using standard convolution

    Rationale: Use standard conv for patch embedding because:
    1. Large kernel (patch_size=4) may have compatibility issues with optical implementation
    2. This operation is done once at the beginning, not the bottleneck
    3. Optical computation advantage is maximized in spatial convolutions (ConvFFN)
    """
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=20):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Use standard convolution for reliable patch projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.proj(x)  # (B, embed_dim, 7, 7)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 49, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed

        return x


class ConvFFN(nn.Module):
    """
    Feed-Forward Network using hybrid architecture

    Uses optical conv for expansion and standard 1x1 conv for compression,
    because CNN_layer requires out_channels % in_channels == 0,
    which doesn't hold for compression (hidden_dim -> dim when hidden_dim > dim).
    """
    def __init__(self, dim, hidden_dim, detection_mode='power'):
        super().__init__()

        # HYBRID ARCHITECTURE:
        # 1. Expand with optical conv (dim -> hidden_dim, where hidden_dim % dim == 0)
        # 2. Compress with standard 1x1 conv (hidden_dim -> dim, any ratio)

        # First layer: expand using optical convolution
        self.conv1 = EfficientOpticalConv(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=3,
            detection_mode=detection_mode
        )

        # Second layer: compress using standard 1x1 convolution
        # This avoids the CNN_layer constraint of out_channels % in_channels == 0
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=1,  # 1x1 conv, no spatial reduction
            bias=True
        )

        # Activation
        self.act = nn.GELU()

        # Padding only for optical conv (kernel=3)
        # Standard 1x1 conv doesn't change spatial dimensions
        self.pad = nn.ConstantPad2d(1, 0)  # Pad by 1 for kernel=3

    def forward(self, x):
        """
        Args:
            x: (B, N, C) - sequence format from transformer
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        # Reshape to spatial format
        x = x.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # Expand with optical conv: dim -> hidden_dim
        # Input: (B, C, H, W)
        # Pad: (B, C, H+2, W+2)
        # Optical Conv (kernel=3): (B, hidden_dim, H, W)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.act(x)

        # Compress with 1x1 conv: hidden_dim -> dim
        # No padding needed for 1x1 conv
        # (B, hidden_dim, H, W) -> (B, C, H, W)
        x = self.conv2(x)

        # Reshape back to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        return x


class Block(nn.Module):
    """Transformer block with attention + optical conv FFN"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0,
                 drop=0.0, drop_path=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(dim, hidden_dim)

        self.drop_path = nn.Identity()  # Simplified for now

    def forward(self, x):
        # Attention block
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])

        # MLP block (using optical conv)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with Optical Convolution layers"""
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10,
                 embed_dim=20, depth=2, num_heads=4, mlp_ratio=4.0):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling + classification
        x = x.mean(dim=1)  # (B, embed_dim)
        x = self.head(x)  # (B, num_classes)

        return x


# ----------------------------
# Distributed Training Setup
# ----------------------------

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        raise RuntimeError("Cannot detect distributed environment")

    # Set device BEFORE initializing process group to avoid NCCL conflicts
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Rank {rank}] local_rank={local_rank}, visible_gpus={num_gpus}")
        
        if local_rank >= num_gpus:
             raise RuntimeError(
                f"Process with local_rank={local_rank} tried to access GPU {local_rank}, "
                f"but only {num_gpus} GPUs are visible to this process.\n"
                f"Check CUDA_VISIBLE_DEVICES and SLURM configuration."
            )
        torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}] Detected torchrun environment: world_size={world_size}, local_rank={local_rank}")

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    print(f"[Rank {rank}] Initializing distributed training...")
    dist.barrier()
    print(f"[Rank {rank}] Distributed training initialized successfully with nccl backend")

    return rank, local_rank, world_size


def is_main_process(rank):
    return rank == 0


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_mzi_parameters(model, json_path, rank):
    """
    加载校准的 MZI 参数并应用到模型中的所有 MZI 实例。
    """
    if not is_main_process(rank):
        return

    if not os.path.exists(json_path):
        print(f"[Warning] MZI 参数文件未找到于 {json_path}。将使用随机/默认初始化参数。")
        return

    print(f"[Info] 正在从 {json_path} 加载 MZI 参数...")
    with open(json_path, 'r') as f:
        params_dict = json.load(f)

    count = 0
    # 遍历模型中的所有模块
    for name, module in model.named_modules():
        if isinstance(module, MZI):
            if module.index is not None:
                # JSON 中的键是字符串，所以将索引转换为字符串进行查找
                idx_str = str(module.index)
                if idx_str in params_dict:
                    p = params_dict[idx_str]
                    module.load_physical_parameters(
                        a=p.get('a'),
                        b=p.get('b'),
                        delta_r=p.get('delta_r'),
                        phi0=p.get('phi0')
                    )
                    module.freeze_fabrication_parameters()
                    count += 1
    
    if count > 0:
        print(f"[Info] 已成功为 {count} 个 MZI 实例加载了校准参数。")
    else:
        print(f"[Warning] 未找到 MZI 实例或参数文件中没有匹配的索引，未加载任何 MZI 参数。")


# ----------------------------
# Training Functions
# ----------------------------

def get_dataloaders(cfg, distributed=False):
    """Create MNIST dataloaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)

    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_ds,
            shuffle=False
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            sampler=test_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

    return train_loader, test_loader


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, cfg, args, rank):
    """Save training checkpoint"""
    if not is_main_process(rank):
        return

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Unwrap DDP model
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
        'config': vars(cfg),
        'args': vars(args)
    }

    # Save latest
    latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    # Save epoch checkpoint
    if epoch % args.save_every == 0:
        epoch_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
        torch.save(checkpoint, epoch_path)

    print(f"[Checkpoint] Saved to {latest_path}")


def load_checkpoint(model, optimizer, scheduler, scaler, args, rank):
    """Load training checkpoint"""
    if args.no_resume:
        if is_main_process(rank):
            print("No checkpoint found, starting training from scratch")
        return 0, 0.0

    checkpoint_path = args.resume if args.resume else os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')

    if not os.path.exists(checkpoint_path):
        if is_main_process(rank):
            print("No checkpoint found, starting training from scratch")
        return 0, 0.0

    if is_main_process(rank):
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer, scheduler, scaler
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('best_acc', 0.0)

    if is_main_process(rank):
        print(f"Resumed from epoch {checkpoint['epoch']}, best acc: {best_acc:.2f}%")

    return start_epoch, best_acc


# ----------------------------
# Main Training Loop
# ----------------------------

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--use-ideal-mzi", action="store_true")
    parser.add_argument("--finetune-epoch", type=int, default=None)

    # Checkpoint arguments
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)

    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    # Print startup info
    print("\n" + "=" * 60)
    print("Starting Efficient ViT Training with CNN_layer")
    print("=" * 60)
    print(f"Arguments: {vars(args)}")
    print("=" * 60 + "\n")

    # Setup distributed training
    if args.distributed or 'RANK' in os.environ or 'SLURM_PROCID' in os.environ:
        rank, local_rank, world_size = setup_distributed()
        distributed = True
    else:
        rank, local_rank, world_size = 0, 0, 1
        distributed = False

    # Set device
    if torch.cuda.is_available():
        if distributed:
            # Check GPU availability
            num_gpus = torch.cuda.device_count()
            if local_rank >= num_gpus:
                raise RuntimeError(
                    f"local_rank={local_rank} but only {num_gpus} GPUs available. "
                    f"torchrun requested {world_size} processes but SLURM only allocated {num_gpus} GPUs. "
                    f"Fix: Use --gpus-per-node={world_size} or --gres=gpu:v100:{world_size} in SLURM script."
                )
            # torch.cuda.set_device(local_rank) is already called in setup_distributed
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Only print on main process
    is_main = is_main_process(rank)

    cfg = CFG(epochs=args.epochs, batch_size=args.bs, lr=args.lr, amp=not args.no_amp, depth=args.depth)
    set_seed(cfg.seed + rank)

    # Load data
    if is_main:
        print("Loading MNIST dataset...")
    train_loader, test_loader = get_dataloaders(cfg, distributed)
    if is_main:
        print(f"[OK] Dataset loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")

    # Create model
    if is_main:
        print("Creating ViT model with efficient optical convolution...")

    model = VisionTransformer(
        img_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    ).to(device)

    if is_main:
        print("[OK] Model created successfully")

    # Load calibrated MZI parameters if not using ideal MZIs
    if not args.use_ideal_mzi:
        json_path = os.path.join(Path(__file__).parent, "results", "mzi_parameters.json")
        load_mzi_parameters(model, json_path, rank)
        
    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"Using DistributedDataParallel on {world_size} GPUs")

    # Print model info
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"=== EFFICIENT OPTICAL ViT Model ===")
        print(f"Device: {device}")
        print(f"Distributed: {distributed} (World Size: {world_size})")
        print(f"Total parameters: {total_params:,}")
        print(f"Batch size per GPU: {cfg.batch_size}")
        print(f"Effective batch size: {cfg.batch_size * world_size}")
        print()

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Load checkpoint
    start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, scaler, args, rank)

    if is_main:
        print("Starting training with EFFICIENT OPTICAL CONVOLUTION layers...")
        print()

    # Training loop
    step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.epochs):
        # Set epoch for distributed sampler
        if distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        batch_times = []

        for batch_idx, (x, y) in enumerate(train_loader):
            batch_start = time.time()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if batch_idx % 10 == 0 and is_main:
                # Memory stats
                mem_str = ""
                if torch.cuda.is_available():
                    try:
                        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                        mem_str = f"Mem: {mem_allocated:.1f}/{mem_reserved:.1f}GB"
                    except Exception as e:
                        mem_str = f"Mem: Error ({e})"

                avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d}, "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                      f"Time: {avg_batch_time:.2f}s/batch, "
                      f"{mem_str}", flush=True)

        # Evaluation
        model.eval()
        correct, n = 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                n += y.size(0)

        # Gather metrics from all processes
        if distributed:
            correct_tensor = torch.tensor([correct], device=device)
            n_tensor = torch.tensor([n], device=device)
            dist.all_reduce(correct_tensor)
            dist.all_reduce(n_tensor)
            correct = correct_tensor.item()
            n = n_tensor.item()

        acc = 100.0 * correct / n
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start

        if is_main:
            print(f"\n[Epoch {epoch+1}/{cfg.epochs}] "
                  f"Loss: {avg_loss:.4f}, "
                  f"Acc: {acc:.2f}%, "
                  f"Time: {epoch_time:.1f}s\n", flush=True)

        # Save checkpoint
        if acc > best_acc:
            best_acc = acc
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, cfg, args, rank)

    if is_main:
        print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
