#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter sweep to find optimal MZI architecture
Search space:
- Number of fc chains (fc1, fc2, fc3, ...)
- Number of 11-layer MZI blocks per fc chain
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from dataclasses import dataclass

from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row


# ============================================================================
# Configurable MLP with variable fc chains and MZI blocks
# ============================================================================
class ConfigurableMLP(nn.Module):
    """
    MLP with configurable number of fc chains and MZI blocks per chain

    Args:
        dim: embedding dimension (fixed at 10)
        num_fc_chains: number of fc chains (1, 2, 3, 4, ...)
        num_mzi_blocks: number of 11-layer MZI blocks per fc chain (1, 2, 3, ...)
    """
    def __init__(self, dim, num_fc_chains=3, num_mzi_blocks=1, drop=0.0):
        super().__init__()
        self.num_fc_chains = num_fc_chains
        self.num_mzi_blocks = num_mzi_blocks

        # Create fc chains dynamically
        self.fc_chains = nn.ModuleList()
        for _ in range(num_fc_chains):
            fc_chain = self._create_mzi_block_chain(num_mzi_blocks)
            self.fc_chains.append(fc_chain)

        self.drop = nn.Dropout(drop)

    def _create_mzi_block_chain(self, num_blocks):
        """
        Create a chain with specified number of 11-layer MZI blocks
        Each block = 5x(row+column) + 1xrow = 11 layers
        """
        chain = nn.ModuleList()

        for _ in range(num_blocks):
            # One 11-layer MZI block
            for _ in range(5):
                chain.append(MZIlayer_row(num=5))
                chain.append(MZIlayer_column(num=4))
            chain.append(MZIlayer_row(num=5))

        return chain

    def forward(self, x):
        """
        Process through all fc chains with nonlinearity between them
        """
        for i, fc_chain in enumerate(self.fc_chains):
            # Process through MZI chain
            for layer in fc_chain:
                x = layer(x)

            # Apply magnitude and activation
            x = torch.abs(x)

            # Apply GELU for all chains except the last one
            if i < self.num_fc_chains - 1:
                x = F.gelu(x)

            x = self.drop(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=14, patch_size=2, in_chans=1, embed_dim=10):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_fc_chains, num_mzi_blocks,
                 attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                         dropout=attn_drop, batch_first=True)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ConfigurableMLP(dim, num_fc_chains, num_mzi_blocks, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x),
                                        self.norm1(x), need_weights=False)[0])
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


class ConfigurableViT(nn.Module):
    def __init__(self, img_size=14, patch_size=2, in_chans=1, num_classes=10,
                 embed_dim=10, depth=8, num_heads=2,
                 num_fc_chains=3, num_mzi_blocks=1,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, num_fc_chains, num_mzi_blocks,
                  attn_drop_rate, drop_rate, dpr[i])
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


# ============================================================================
# Training Configuration
# ============================================================================
@dataclass
class CFG:
    img_size: int = 14
    patch: int = 2
    embed: int = 10
    depth: int = 8
    heads: int = 2
    epochs: int = 35  # Keep same as original
    batch_size: int = 256
    lr: float = 5e-4  # Keep same as original
    weight_decay: float = 5e-2
    label_smoothing: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = True


def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    test_ds = datasets.MNIST(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader


def train_single_config(num_fc_chains, num_mzi_blocks, cfg, device,
                        train_loader, test_loader, save_dir):
    """
    Train one configuration and return best accuracy
    """
    print(f"\n{'='*80}")
    print(f"Training: fc_chains={num_fc_chains}, mzi_blocks={num_mzi_blocks}")
    print(f"{'='*80}")

    # Create model
    model = ConfigurableViT(
        img_size=cfg.img_size,
        patch_size=cfg.patch,
        in_chans=1,
        num_classes=10,
        embed_dim=cfg.embed,
        depth=cfg.depth,
        num_heads=cfg.heads,
        num_fc_chains=num_fc_chains,
        num_mzi_blocks=num_mzi_blocks,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * math.ceil(len(train_loader.dataset) / cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_acc = 0.0

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
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

        # Evaluation
        model.eval()
        correct, n = 0, 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.amp):
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                n += y.size(0)

        acc = correct / n
        avg_loss = epoch_loss / num_batches

        if acc > best_acc:
            best_acc = acc
            # Save best model
            model_name = f"vit_fc{num_fc_chains}_blocks{num_mzi_blocks}_best.pt"
            save_path = os.path.join(save_dir, model_name)
            torch.save({"model": model.state_dict(),
                       "accuracy": best_acc,
                       "num_fc_chains": num_fc_chains,
                       "num_mzi_blocks": num_mzi_blocks,
                       "total_params": total_params}, save_path)

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {acc*100:.2f}% | "
                  f"Best: {best_acc*100:.2f}%")

    print(f"Final best accuracy: {best_acc*100:.2f}%")
    return best_acc, total_params


def run_hyperparameter_sweep():
    """
    Main sweep function
    """
    print("="*80)
    print("MZI ARCHITECTURE HYPERPARAMETER SWEEP")
    print("="*80)
    print("\nSearch space:")
    print("  - Number of fc chains: 1, 2, 3, 4, 5")
    print("  - Number of MZI blocks per chain: 1, 2, 3, 4")
    print("  - Total configurations: 5 x 4 = 20")
    print("\nFixed parameters:")
    print("  - Epochs: 35")
    print("  - Learning rate: 5e-4")
    print("  - Batch size: 256")
    print("="*80)

    # Configuration
    cfg = CFG()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create save directory
    save_dir = "mzi_architecture_sweep_results"
    os.makedirs(save_dir, exist_ok=True)

    # Load data once
    print("\nLoading MNIST data...")
    train_loader, test_loader = get_loaders(cfg)

    # Define search space
    fc_chains_range = [1, 2, 3, 4, 5]
    mzi_blocks_range = [1, 2, 3, 4]

    # Results storage
    results = {}
    accuracy_matrix = np.zeros((len(fc_chains_range), len(mzi_blocks_range)))
    params_matrix = np.zeros((len(fc_chains_range), len(mzi_blocks_range)))

    # Run sweep
    total_configs = len(fc_chains_range) * len(mzi_blocks_range)
    config_count = 0

    for i, num_fc in enumerate(fc_chains_range):
        for j, num_blocks in enumerate(mzi_blocks_range):
            config_count += 1
            print(f"\n[{config_count}/{total_configs}] Configuration: "
                  f"fc_chains={num_fc}, mzi_blocks={num_blocks}")

            # Train
            best_acc, total_params = train_single_config(
                num_fc, num_blocks, cfg, device,
                train_loader, test_loader, save_dir
            )

            # Store results
            key = f"fc{num_fc}_blocks{num_blocks}"
            results[key] = {
                'num_fc_chains': num_fc,
                'num_mzi_blocks': num_blocks,
                'best_accuracy': float(best_acc),
                'total_params': int(total_params)
            }

            accuracy_matrix[i, j] = best_acc * 100
            params_matrix[i, j] = total_params

    # Save results
    results_file = os.path.join(save_dir, "sweep_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Results saved to {results_file}")

    # Create visualizations
    create_heatmaps(accuracy_matrix, params_matrix,
                   fc_chains_range, mzi_blocks_range, save_dir)

    # Print summary
    print_summary(results, accuracy_matrix, params_matrix,
                 fc_chains_range, mzi_blocks_range)

    return results


def create_heatmaps(accuracy_matrix, params_matrix, fc_range, blocks_range, save_dir):
    """
    Create heatmap visualizations
    """
    print("\n[INFO] Creating heatmap visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Heatmap 1: Accuracy
    ax = axes[0]
    im1 = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks and labels
    ax.set_xticks(range(len(blocks_range)))
    ax.set_yticks(range(len(fc_range)))
    ax.set_xticklabels(blocks_range, fontsize=18)
    ax.set_yticklabels(fc_range, fontsize=18)

    ax.set_xlabel('Number of MZI Blocks per FC Chain', fontsize=22, weight='bold')
    ax.set_ylabel('Number of FC Chains', fontsize=22, weight='bold')
    ax.set_title('Best Test Accuracy (%)', fontsize=24, weight='bold', pad=20)

    # Add text annotations
    for i in range(len(fc_range)):
        for j in range(len(blocks_range)):
            text = ax.text(j, i, f'{accuracy_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black",
                          fontsize=16, weight='bold')

    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('Accuracy (%)', fontsize=20, weight='bold')
    cbar1.ax.tick_params(labelsize=18)

    # Heatmap 2: Parameter count
    ax = axes[1]
    im2 = ax.imshow(params_matrix / 1000, cmap='Blues', aspect='auto')

    ax.set_xticks(range(len(blocks_range)))
    ax.set_yticks(range(len(fc_range)))
    ax.set_xticklabels(blocks_range, fontsize=18)
    ax.set_yticklabels(fc_range, fontsize=18)

    ax.set_xlabel('Number of MZI Blocks per FC Chain', fontsize=22, weight='bold')
    ax.set_ylabel('Number of FC Chains', fontsize=22, weight='bold')
    ax.set_title('Model Size (K parameters)', fontsize=24, weight='bold', pad=20)

    # Add text annotations
    for i in range(len(fc_range)):
        for j in range(len(blocks_range)):
            text = ax.text(j, i, f'{params_matrix[i, j]/1000:.1f}K',
                          ha="center", va="center", color="white" if params_matrix[i, j] > params_matrix.max()/2 else "black",
                          fontsize=16, weight='bold')

    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('Parameters (K)', fontsize=20, weight='bold')
    cbar2.ax.tick_params(labelsize=18)

    plt.suptitle('MZI Architecture Hyperparameter Sweep Results',
                fontsize=28, weight='bold', y=0.98)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'sweep_heatmaps.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"[SAVE] Heatmaps saved to {save_path}")
    plt.close()


def print_summary(results, accuracy_matrix, params_matrix, fc_range, blocks_range):
    """
    Print summary of results
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("="*80)

    # Find best configuration
    best_acc = 0
    best_config = None
    for key, data in results.items():
        if data['best_accuracy'] > best_acc:
            best_acc = data['best_accuracy']
            best_config = data

    print(f"\n[BEST CONFIGURATION]")
    print(f"  FC chains: {best_config['num_fc_chains']}")
    print(f"  MZI blocks per chain: {best_config['num_mzi_blocks']}")
    print(f"  Accuracy: {best_config['best_accuracy']*100:.2f}%")
    print(f"  Total parameters: {best_config['total_params']:,}")

    # Print top 5
    print(f"\n[TOP 5 CONFIGURATIONS]")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
    for rank, (key, data) in enumerate(sorted_results[:5], 1):
        print(f"  {rank}. fc={data['num_fc_chains']}, blocks={data['num_mzi_blocks']}: "
              f"{data['best_accuracy']*100:.2f}% ({data['total_params']:,} params)")

    # Analysis
    print(f"\n[ANALYSIS]")
    print(f"  Accuracy range: [{accuracy_matrix.min():.1f}%, {accuracy_matrix.max():.1f}%]")
    print(f"  Parameter range: [{int(params_matrix.min()):,}, {int(params_matrix.max()):,}]")
    print(f"  Best improvement over baseline: {accuracy_matrix.max() - accuracy_matrix.min():.1f}%")

    print("="*80)


if __name__ == "__main__":
    results = run_hyperparameter_sweep()
