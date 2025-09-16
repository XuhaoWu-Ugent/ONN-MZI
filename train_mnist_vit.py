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

# Import FIXED MZI implementations (solves gradient issues)
from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row

# ----------------------------
# 1) Minimal ViT for MNIST with FIXED MZI
# ----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        # Use FIXED MZI implementations - fast and gradient-safe
        self.fc1 = nn.ModuleList()
        for _ in range(5):
            self.fc1.append(MZIlayer_row(num=5))
            self.fc1.append(MZIlayer_column(num=4))
        self.fc1.append(MZIlayer_row(num=5))

        self.fc2 = nn.ModuleList()
        for _ in range(5):
            self.fc2.append(MZIlayer_row(num=5))
            self.fc2.append(MZIlayer_column(num=4))
        self.fc2.append(MZIlayer_row(num=5))

        # Add fc3 layer, identical to fc1 and fc2
        self.fc3 = nn.ModuleList()
        for _ in range(5):
            self.fc3.append(MZIlayer_row(num=5))
            self.fc3.append(MZIlayer_column(num=4))
        self.fc3.append(MZIlayer_row(num=5))

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Process through first MZI chain (hidden layer 1)
        for layer in self.fc1:
            x = layer(x)
        x = torch.abs(x)  # Take magnitude for real activation
        x = F.gelu(x)
        x = self.drop(x)

        # Process through second MZI chain (hidden layer 2)
        for layer in self.fc2:
            x = layer(x)
        x = torch.abs(x)
        x = F.gelu(x)  # Add GELU for consistency as it's a hidden layer now
        x = self.drop(x)

        # Process through third MZI chain (output layer)
        for layer in self.fc3:
            x = layer(x)
        x = torch.abs(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        # MHSA
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        # MLP with FIXED MZI
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
        embed_dim=10, depth=8, num_heads=3,
        mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, attn_drop_rate, drop_rate, dpr[i])
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
    embed: int = 10
    depth: int = 8
    heads: int = 2
    epochs: int = 25
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 5e-2
    label_smoothing: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = True

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
            total_hits += stats['cache_hits']
            total_misses += stats['cache_misses']

        for child in module.children():
            collect_stats(child)

    collect_stats(model)

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

if __name__ == "__main__":
    main()