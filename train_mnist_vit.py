#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, time, argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import FIXED MZI implementations
from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row

# ----------------------------
# 1) Physical Core & Virtual Layers
# ----------------------------

class OpticalCore10x10(nn.Module):
    """
    Represents the single physical 10x10 MZI mesh hardware available.
    It contains 50 specific MZIs (Indices 0-49) with calibrated defects.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mzi_counts = []
        
        # We supply voltages externally from BlockMZILinear, so disable internal trainable voltages
        mzi_kwargs = {"trainable_voltage": False}
        
        # Build the Clements-style 10x10 mesh structure (50 MZIs total)
        # Structure: 5x [Row(5) -> Col(4)] -> Row(5)
        mzi_idx = 0
        for _ in range(5):
            self.layers.append(MZIlayer_row(num=5, start_index=mzi_idx, mzi_kwargs=mzi_kwargs))
            self.mzi_counts.append(5)
            mzi_idx += 5
            
            self.layers.append(MZIlayer_column(num=4, start_index=mzi_idx, mzi_kwargs=mzi_kwargs))
            self.mzi_counts.append(4)
            mzi_idx += 4
            
        self.layers.append(MZIlayer_row(num=5, start_index=mzi_idx, mzi_kwargs=mzi_kwargs))
        self.mzi_counts.append(5)
        
        self.total_mzis = mzi_idx + 5
        assert self.total_mzis == 50, f"Expected 50 MZIs, got {self.total_mzis}"

    def forward(self, x, voltages):
        """
        x: (Batch, 10) - complex or real input
        voltages: (Batch, 50) or (50,) - control voltages for this pass
        """
        # Slice the big voltage vector into chunks for each layer
        current_v_idx = 0
        
        for i, layer in enumerate(self.layers):
            count = self.mzi_counts[i]
            
            if voltages.dim() == 1:
                v_chunk = voltages[current_v_idx : current_v_idx + count]
            else:
                v_chunk = voltages[:, current_v_idx : current_v_idx + count]
            
            x = layer(x, voltages=v_chunk)
            current_v_idx += count
            
        return x

class BlockMZILinear(nn.Module):
    """
    A logical linear layer that breaks down large matrix multiplication (e.g., 20x20)
    into 10x10 blocks to be executed on the OpticalCore10x10.
    """
    def __init__(self, in_features, out_features, optical_core):
        super().__init__()
        assert in_features % 10 == 0
        assert out_features % 10 == 0
        
        self.in_features = in_features
        self.out_features = out_features
        self.optical_core = optical_core
        
        self.n_row_blocks = in_features // 10
        self.n_col_blocks = out_features // 10
        
        # Parameters: One set of 50 voltages for EACH 10x10 block
        # Shape: (Out_Blocks, In_Blocks, 50)
        # Initialized with small random values
        self.voltages = nn.Parameter(
            torch.randn(self.n_col_blocks, self.n_row_blocks, 50) * 0.1
        )

    def forward(self, x):
        # x shape: (Batch, Seq, In_Features)
        B, S, D = x.shape
        assert D == self.in_features
        
        # Reshape to separate blocks: (Batch, Seq, Row_Blocks, 10)
        x_blocked = x.view(B, S, self.n_row_blocks, 10)
        
        # We need to accumulate results for each column block
        # Output shape: (Batch, Seq, Col_Blocks, 10)
        # But we'll sum in complex domain first if output of core is complex?
        # The core output is complex. We usually take magnitude at the very end of the logic layer,
        # but here we are doing linear combination.
        # Coherent addition (complex addition) is physically valid if using coherent detection,
        # but hard to implement if we measure magnitude after each pass.
        # ASSUMPTION: We are simulating a coherent system where we can sum complex fields,
        # OR we assume this is done digitally (electrical domain accumulation).
        # We will assume digital accumulation of complex values (Linear operation).
        
        output_blocks = []
        
        for col in range(self.n_col_blocks):
            col_acc = None
            for row in range(self.n_row_blocks):
                # Get input chunk: (Batch, Seq, 10)
                x_chunk = x_blocked[:, :, row, :]
                
                # Get voltage params for this block: (50,)
                v_params = self.voltages[col, row]
                
                # Flatten batch/seq for core processing
                # Core expects (Total_Batch, 10)
                x_flat = x_chunk.reshape(-1, 10)
                
                # Pass through Optical Core
                # Note: v_params is (50,), it will be broadcasted by core
                out_flat = self.optical_core(x_flat, v_params)
                
                # Reshape back
                out_chunk = out_flat.view(B, S, 10)
                
                if col_acc is None:
                    col_acc = out_chunk
                else:
                    col_acc = col_acc + out_chunk
            
            output_blocks.append(col_acc)
            
        # Concatenate results: (Batch, Seq, Out_Features)
        out = torch.cat(output_blocks, dim=2)
        return out

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=20):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, dim, optical_core, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        # Ensure hidden dims are multiples of 10
        hidden_dim = math.ceil(hidden_dim / 10) * 10
        
        self.fc1 = BlockMZILinear(dim, hidden_dim, optical_core)
        self.fc2 = BlockMZILinear(hidden_dim, dim, optical_core)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # FC1
        x = self.fc1(x)
        x = torch.abs(x) # Detection / Activation
        x = F.gelu(x)
        x = self.drop(x)
        
        # FC2
        x = self.fc2(x)
        x = torch.abs(x) # Detection
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, optical_core, mlp_ratio=4.0, attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, optical_core, mlp_ratio, drop)

    def forward(self, x):
        # MHSA (Standard Digital Attention for now)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        # MLP with Block-based Optical Processing
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob).div(keep_prob)
        return x * mask

class ViT(nn.Module):
    def __init__(
        self,
        img_size=14, patch_size=2, in_chans=1, num_classes=10,
        embed_dim=20, depth=8, num_heads=4,
        mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1
    ):
        super().__init__()
        
        # Instantiate the single physical hardware core
        self.optical_core = OpticalCore10x10()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.optical_core, mlp_ratio, attn_drop_rate, drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ----------------------------
# 2) Training Configuration
# ----------------------------
@dataclass
class CFG:
    img_size: int = 14
    patch: int = 2
    embed: int = 20
    depth: int = 8
    heads: int = 4
    epochs: int = 35
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 5e-2
    label_smoothing: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = True

def load_mzi_parameters(model, json_path="results/mzi_parameters.json"):
    import json
    import os
    
    if not os.path.exists(json_path):
        print(f"Warning: Calibration file {json_path} not found. Using ideal parameters.")
        return

    print(f"Loading calibrated MZI parameters from {json_path}...")
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    count = 0
    # Iterate over all modules to find MZI instances
    for name, module in model.named_modules():
        # Check for 'load_physical_parameters' method and 'index' attribute
        if hasattr(module, 'load_physical_parameters') and hasattr(module, 'index') and module.index is not None:
            idx_str = str(module.index)
            if idx_str in params:
                p = params[idx_str]
                module.load_physical_parameters(
                    a=p.get('a'),
                    b=p.get('b'),
                    delta_r=p.get('delta_r'),
                    phi0=p.get('phi0')
                )
                module.freeze_fabrication_parameters()
                count += 1
            else:
                 # Indices not in JSON are left as ideal (or whatever they were initialized as)
                 pass
    print(f"Applied calibrated parameters to {count} MZI instances.")

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_loaders(cfg: CFG, root="./data"):
    mean, std = 0.1307, 0.3081
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    test_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    train_ds = datasets.MNIST(root, train=True, download=True, transform=train_tf)
    test_ds  = datasets.MNIST(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader

def get_mzi_cache_stats(model):
    """Collect cache statistics from all MZI layers"""
    total_hits = 0
    total_misses = 0

    def collect_stats(module):
        nonlocal total_hits, total_misses
        if hasattr(module, 'get_cache_stats'):
            stats = module.get_cache_stats()
            # Assuming stats might return None if not implemented or initialized
            if stats: 
                total_hits += stats.get('cache_hits', 0)
                total_misses += stats.get('cache_misses', 0)
        
        # Helper for MZIlayer_row/col which don't expose get_cache_stats directly but have internal cache logic
        # Actually, looking at the code, they don't seem to expose a counter, just logic.
        # We might need to skip this or implement it if it was there.
        # For now, let's just recurse.
        
        for child in module.children():
            collect_stats(child)

    collect_stats(model)
    
    # Since we can't easily get stats from current MZI implementation without modifying it, 
    # we'll return placeholders or 0 to avoid errors.
    # The previous implementation of get_mzi_cache_stats relied on methods that might not exist 
    # in the standard nn.Module or the provided MZI classes.
    # If the MZI classes don't have get_cache_stats, this will just return 0s.

    total = total_hits + total_misses
    hit_rate = total_hits / total if total > 0 else 0

    return {
        'total_hits': total_hits,
        'total_misses': total_misses,
        'hit_rate': hit_rate
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--bs", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    cfg = CFG(epochs=args.epochs, batch_size=args.bs, lr=args.lr, amp=not args.no_amp)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(cfg)

    model = ViT(
        img_size=cfg.img_size, patch_size=cfg.patch, in_chans=1, num_classes=10,
        embed_dim=cfg.embed, depth=cfg.depth, num_heads=cfg.heads,
        mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1
    ).to(device)

    # Load calibrated hardware parameters
    # Note: load_mzi_parameters iterates over modules. 
    # Our OpticalCore10x10 contains the MZI layers with indices 0-49.
    # So this should work perfectly.
    load_mzi_parameters(model, json_path="results/mzi_parameters.json")

    total_params = sum(p.numel() for p in model.parameters())
    print("=== OPTIMIZED MZI ViT Model ===")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Batch size: {cfg.batch_size}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * math.ceil(len(train_loader.dataset)/cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_acc = 0.0
    step = 0

    print("Starting training with OPTIMIZED MZI implementation...")
    print()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
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

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d}, "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Evaluation
        model.eval()
        correct, n = 0, 0
        eval_start = time.time()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.amp):
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                n += y.size(0)

        acc = correct / n
        epoch_time = time.time() - epoch_start
        eval_time = time.time() - eval_start
        avg_loss = epoch_loss / num_batches

        # Get MZI cache statistics
        cache_stats = get_mzi_cache_stats(model)

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc*100:.2f}% | "
              f"Time: {epoch_time:.1f}s | "
              f"Eval: {eval_time:.1f}s | "
              f"Cache: {cache_stats['hit_rate']:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict()}, "mnist_vit_optimized_best.pt")
            print(f"  -> New best accuracy: {best_acc*100:.2f}%")

        print()

    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")

    # Final statistics
    final_cache_stats = get_mzi_cache_stats(model)
    print(f"Final cache statistics:")
    print(f"  Total cache hits: {final_cache_stats['total_hits']:,}")
    print(f"  Total cache misses: {final_cache_stats['total_misses']:,}")
    print(f"  Final cache hit rate: {final_cache_stats['hit_rate']:.3f}")
    
    # ---------------------------------------------------------
    # Collect Hardware Verification Data
    # ---------------------------------------------------------
    print("\nStarting hardware data collection...")
    collect_hardware_data(model, test_loader, device)

def collect_hardware_data(model, dataloader, device, save_dir="hardware_data"):
    import os
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Storage for collected data
    # Structure: List of dicts, where each dict is one 'pass' through the optical core
    collected_records = []
    
    # Define the hook function
    def core_hook(module, input_args, output):
        # input_args is (x, voltages)
        # x: (Batch, 10)
        # voltages: (Batch, 50) or (50,)
        # output: (Batch, 10)
        
        x_in = input_args[0].detach().cpu().numpy()
        voltages = input_args[1].detach().cpu().numpy()
        output_out = output.detach().cpu().numpy()
        
        # If voltages is 1D (shared across batch), repeat it for consistency if needed
        # Or just store it as is. Let's store as is to save space if it's constant.
        
        record = {
            "input_optical_state": x_in,     # (Batch, 10)
            "mzi_voltages": voltages,        # (Batch, 50) or (50,)
            "output_optical_state": output_out # (Batch, 10)
        }
        collected_records.append(record)

    # Register hook on the optical core
    # Access the core directly from the model
    if hasattr(model, 'optical_core'):
        handle = model.optical_core.register_forward_hook(core_hook)
    else:
        print("Error: Model does not have 'optical_core' attribute. Cannot collect data.")
        return

    print("Hook registered. Running inference on a single batch...")
    
    # Run inference on just one batch to avoid massive data files
    try:
        data_iter = iter(dataloader)
        x_batch, _ = next(data_iter)
        x_batch = x_batch.to(device)
        
        with torch.no_grad():
            _ = model(x_batch)
            
    except StopIteration:
        print("Error: Dataloader is empty.")
    except Exception as e:
        print(f"Error during collection inference: {e}")
    finally:
        handle.remove()
        print("Hook removed.")

    # Save to file
    if collected_records:
        save_path = os.path.join(save_dir, 'vit_hardware_verification.npy')
        
        # We also want to know which record corresponds to which logical layer.
        # Since the execution order is deterministic (FC1 block 0,0 -> 0,1... -> FC2...), 
        # we can reconstruct the mapping if we know the architecture.
        # For now, we save the raw sequence of operations.
        
        data_to_save = {
            "records": collected_records,
            "description": "Sequential records of every call to OpticalCore10x10.forward(x, v)."
        }
        
        np.save(save_path, data_to_save)
        print(f"Collected {len(collected_records)} hardware passes.")
        print(f"Data saved to: {save_path}")
    else:
        print("Warning: No data was collected.")

if __name__ == "__main__":
    main()