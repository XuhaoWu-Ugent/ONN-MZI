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

        # We supply voltages externally, so disable internal trainable voltages
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

        # Hook for capturing optical core computations
        self.forward_power_hook = None
        self.hook_enabled = False

    def _collect_mzi_parameters(self):
        """
        Collect physical parameters from all 50 MZIs in the mesh.
        Returns a dict mapping MZI index to its physical parameters.
        """
        mzi_params = {}
        for layer in self.layers:
            if hasattr(layer, 'MZI'):
                for mzi in layer.MZI:
                    if hasattr(mzi, 'index') and mzi.index is not None:
                        idx = mzi.index
                        # Get physical parameters
                        if hasattr(mzi, 'physical_parameters'):
                            phys = mzi.physical_parameters()
                            mzi_params[idx] = {
                                'a': phys['a'].detach().cpu().item(),
                                'b': phys['b'].detach().cpu().item(),
                                'delta_r': phys['delta_r'].detach().cpu().item(),
                                'phi0': phys['phi0'].detach().cpu().item()
                            }
                        # Get voltage
                        if hasattr(mzi, '_voltage'):
                            if idx in mzi_params:
                                mzi_params[idx]['voltage'] = mzi._voltage.detach().cpu().item()
                            else:
                                mzi_params[idx] = {'voltage': mzi._voltage.detach().cpu().item()}
        return mzi_params

    def register_forward_power_hook(self, hook_fn):
        """
        Register a hook function to be called after each forward_power computation.

        Args:
            hook_fn: Callable that receives a dict with keys:
                - 'input_power': numpy array (Batch, 10) - input optical power
                - 'output_power': numpy array (Batch, 10) - output optical power
                - 'voltages': numpy array (50,) - voltages applied to all 50 MZIs
                - 'mzi_physical_parameters': dict - physical parameters of all MZIs
                - 'batch_size': int - batch size

        Returns:
            A handle object that can be used to remove the hook via handle.remove()
        """
        self.forward_power_hook = hook_fn
        self.hook_enabled = True

        class HookHandle:
            def __init__(self, parent):
                self.parent = parent

            def remove(self):
                self.parent.forward_power_hook = None
                self.parent.hook_enabled = False

        return HookHandle(self)

    def forward(self, x, voltages):
        """
        Coherent Forward Pass (Complex Field) - Standard Mode
        x: (Batch, 10) - complex or real input
        voltages: (Batch, 50) or (50,) - control voltages for this pass
        """
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

    def forward_power(self, x, voltages):
        """
        Incoherent Power Forward Pass (Spectral Diversity Mode).
        Simulates the summation of powers from different wavelengths (comb lines).

        Physics:
        - Input x is Optical Power (Intensity) >= 0.
        - The MZI mesh acts as a power transmission matrix T = |S|^2.
        - Output = T * x.
        - No coherent interference between inputs.

        x: (Batch, 10) - Real-valued power input
        voltages: (Batch, 50) or (50,)
        """
        current_v_idx = 0

        # Ensure input is float (power)
        if x.is_complex():
            x = x.abs().pow(2)

        device = x.device

        # Normalize input to (Batch_Total, 10)
        original_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, original_shape[-1])
        else:
            x_flat = x

        batch_size = x_flat.shape[0]
        current_state = x_flat # (Batch, 10)

        # Prepare voltages for hook (expand to full 50-element vector if needed)
        if voltages.dim() == 1:
            voltages_for_hook = voltages.detach().cpu().numpy()
        else:
            # For batched voltages, take the first sample as representative
            voltages_for_hook = voltages[0].detach().cpu().numpy() if batch_size > 0 else voltages.detach().cpu().numpy()

        for i, layer in enumerate(self.layers):
            count = self.mzi_counts[i]
            
            # Get voltages
            if voltages.dim() == 1:
                v_chunk = voltages[current_v_idx : current_v_idx + count] 
            else:
                v_chunk = voltages[:, current_v_idx : current_v_idx + count]
                # Handle broadcasting if needed
                if v_chunk.shape[0] != batch_size:
                    if v_chunk.shape[0] == 1:
                        v_chunk = v_chunk.expand(batch_size, -1)
                    elif batch_size % v_chunk.shape[0] == 0:
                        ratio = batch_size // v_chunk.shape[0]
                        v_chunk = v_chunk.repeat_interleave(ratio, dim=0)
            
            # 1. Get Coherent S-Matrix (Complex)
            # layer._build_transfer_matrix returns (2N, 2N) or (Batch, 2N, 2N)
            s_matrix = layer._build_transfer_matrix(device, voltage_overrides=v_chunk)
            
            # 2. Convert to Power Transmission Matrix T = |S|^2
            t_matrix = s_matrix.abs().pow(2) # Real, positive
            
            # 3. Apply T to current_state
            # Map input x to the correct input ports of the matrix
            # MZIlayer input ports are usually 0..N-1
            full_state = torch.zeros(batch_size, layer.matrix_size, device=device, dtype=t_matrix.dtype)
            
            # Use layer.num_ports / 2 for input count? 
            # MZIlayer_row(num=5) -> 10 ports total. Input is 10? 
            # In Clement mesh, we usually propagate 10 modes.
            input_dim = current_state.shape[1]
            full_state[:, :input_dim] = current_state
            
            # Multiply: new_state = full_state @ T.T
            if t_matrix.dim() == 3:
                new_state = torch.bmm(full_state.unsqueeze(1), t_matrix.transpose(-2, -1)).squeeze(1)
            else:
                new_state = torch.matmul(full_state, t_matrix.T)
                
            # Extract Output (Shifted by input_dim usually)
            # In MZIlayer implementation: output_flat = new_states[:, self.num_ports : self.num_ports + num_ports]
            # Here layer.num_ports seems to refer to 'num inputs' in the code's context? 
            # Let's check `MZIlayer_row`: num_ports = num*2 (e.g. 10). matrix_size = 20.
            # Inputs at 0..9. Outputs at 10..19.
            out_start = layer.num_ports
            out_end = layer.num_ports + layer.num_ports # Wait, MZIlayer_row.num_ports is 10.
            # In `forward`: `output_flat = new_states[:, self.num_ports : self.num_ports + num_ports]`
            # So yes, it extracts indices 10 to 19.
            
            # Wait, `num_ports` in MZIlayer_column is 2*(num+1) = 10 for num=4.
            # So `layer.num_ports` is reliable for the input/output dimension.
            
            current_state = new_state[:, layer.num_ports : layer.num_ports + input_dim]
            current_v_idx += count

        # Call hook if enabled
        if self.hook_enabled and self.forward_power_hook is not None:
            # Collect physical parameters from all 50 MZIs
            mzi_physical_params = self._collect_mzi_parameters()

            hook_data = {
                'input_power': x_flat.detach().cpu().numpy(),  # (Batch, 10)
                'output_power': current_state.detach().cpu().numpy(),  # (Batch, 10)
                'voltages': voltages_for_hook,  # (50,)
                'mzi_physical_parameters': mzi_physical_params,  # Dict with all MZI params
                'batch_size': batch_size
            }
            self.forward_power_hook(hook_data)

        if x.dim() == 3:
            return current_state.view(original_shape)
        return current_state


class OpticalConv2d(nn.Module):
    """
    Optical Convolution Layer using Incoherent Power Superposition (Spectral Diversity).
    
    Physics:
    - Inputs are encoded on different wavelengths (comb lines) corresponding to spatial kernel positions.
    - These pass through the MZI mesh simultaneously without interference.
    - The MZI mesh applies weights (attenuation/splitting).
    - The detector sums the power of all wavelengths: y = Sum(T_i * P_i).
    - To achieve negative weights, we use differential signaling:
      y = Core(x, V_pos) - Core(x, V_neg).
    """
    def __init__(self, in_channels, out_channels, kernel_size, optical_core, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.optical_core = optical_core
        
        # Calculate Unfolded Input Dimension
        # Standard Conv: Input (B, C, H, W) -> Unfold -> (B, C*K*K, L)
        self.input_dim = in_channels * kernel_size * kernel_size
        
        # We need to map this large input vector to 10x10 blocks for the core.
        assert self.input_dim % 10 == 0, f"Input dim {self.input_dim} must be divisible by 10 for 10x10 core."
        assert out_channels % 10 == 0, f"Out channels {out_channels} must be divisible by 10."
        
        self.n_row_blocks = self.input_dim // 10
        self.n_col_blocks = out_channels // 10
        
        # Parameters: Two sets of voltages (Pos, Neg) for differential weighting
        # Shape: (Out_Blocks, In_Blocks, 50)
        self.voltages_pos = nn.Parameter(torch.randn(self.n_col_blocks, self.n_row_blocks, 50) * 0.1)
        self.voltages_neg = nn.Parameter(torch.randn(self.n_col_blocks, self.n_row_blocks, 50) * 0.1)
        
        # Bias (Standard digital bias added after detection)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # x: (Batch, C, H, W)
        B, C, H, W = x.shape
        
        # 1. Im2Col (Unfold)
        # Output: (B, C*K*K, L) where L = H_out * W_out
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        L = x_unfolded.shape[2]
        
        # Transpose for block processing: (B, L, Input_Dim)
        x_inp = x_unfolded.transpose(1, 2) # (B, L, D)
        
        # Ensure input is Power (Positive)
        # Assuming previous layers output power or we rectifying here.
        x_inp = torch.abs(x_inp) 
        
        # Reshape for blocking: (B, L, Row_Blocks, 10)
        x_blocked = x_inp.view(B, L, self.n_row_blocks, 10)
        
        output_blocks = []
        
        # 2. Block Matrix Multiplication (Differential)
        for col in range(self.n_col_blocks):
            col_acc = None
            for row in range(self.n_row_blocks):
                x_chunk = x_blocked[:, :, row, :] # (B, L, 10)
                
                # Flatten: (B*L, 10)
                x_flat = x_chunk.reshape(-1, 10)
                
                # Get Voltages
                v_pos = self.voltages_pos[col, row]
                v_neg = self.voltages_neg[col, row]
                
                # Optical Pass (Incoherent Power Mode)
                # y = T_pos * x - T_neg * x
                out_pos = self.optical_core.forward_power(x_flat, v_pos)
                out_neg = self.optical_core.forward_power(x_flat, v_neg)
                
                diff_out = out_pos - out_neg
                
                # Reshape back
                out_chunk = diff_out.view(B, L, 10)
                
                if col_acc is None:
                    col_acc = out_chunk
                else:
                    col_acc = col_acc + out_chunk
            
            output_blocks.append(col_acc)
            
        # 3. Concatenate and Fold
        # (B, L, Out_Channels)
        out_cat = torch.cat(output_blocks, dim=2)
        
        # Add bias
        out_cat = out_cat + self.bias
        
        # Transpose back: (B, Out_Channels, L)
        out_trans = out_cat.transpose(1, 2)
        
        # Calculate Output Height/Width
        H_out = int((H + 2*self.padding - 1*(self.kernel_size-1) - 1)/self.stride + 1)
        W_out = int((W + 2*self.padding - 1*(self.kernel_size-1) - 1)/self.stride + 1)
        
        out = out_trans.view(B, self.out_channels, H_out, W_out)
        
        return out


class ConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network replacing the standard MLP.
    Based on 'Less-Attention' principles and Optical Convolution.
    Structure: Conv2d (3x3) -> GELU -> Conv2d (1x1)
    """
    def __init__(self, dim, hidden_dim, optical_core, drop=0.0):
        super().__init__()
        # Ensure dims are compatible with 10x10 core
        hidden_dim = math.ceil(hidden_dim / 10) * 10
        dim = math.ceil(dim / 10) * 10 
        
        # Layer 1: 3x3 Conv (Spatial + Channel Mixing)
        self.conv1 = OpticalConv2d(dim, hidden_dim, kernel_size=3, padding=1, optical_core=optical_core)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        
        # Layer 2: 1x1 Conv (Projection)
        self.conv2 = OpticalConv2d(hidden_dim, dim, kernel_size=1, padding=0, optical_core=optical_core)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        # x input to Block is usually (B, Seq, Dim).
        # We need (B, Dim, H, W) for Conv2d.
        B, S, D = x.shape
        
        # Handle CLS Token (Seq Length = Patches + 1)
        # We assume the first token is CLS.
        num_patches = S - 1
        H = W = int(math.sqrt(num_patches))
        
        if H * W != num_patches:
             # Fallback if no CLS token or unexpected shape
             # Try assuming no CLS
             H = W = int(math.sqrt(S))
             if H * W == S:
                 # No CLS token case
                 patches = x
                 cls_token = None
             else:
                 raise ValueError(f"Sequence length {S} incompatible with square image + optional CLS.")
        else:
             # Standard ViT case
             cls_token = x[:, 0:1, :]
             patches = x[:, 1:, :]
        
        # Reshape Patches: (B, N, D) -> (B, D, H, W)
        patches = patches.transpose(1, 2).view(B, D, H, W)
        
        # Apply Optical Convolutions
        patches = self.conv1(patches)
        patches = self.act(patches)
        patches = self.drop1(patches)
        
        patches = self.conv2(patches)
        patches = self.drop2(patches)
        
        # Reshape back: (B, D, H, W) -> (B, N, D)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Recombine
        if cls_token is not None:
            # We treat CLS as Identity in FFN (bypass)
            x = torch.cat([cls_token, patches], dim=1)
        else:
            x = patches
            
        return x


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


class Block(nn.Module):
    def __init__(self, dim, num_heads, optical_core, mlp_ratio=4.0, attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        # Use ConvFFN instead of MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(dim, hidden_dim, optical_core, drop)

    def forward(self, x):
        # MHSA (Standard Digital Attention)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        
        # ConvFFN (Optical Convolution)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--bs", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--depth", type=int, default=CFG.depth)
    args = parser.parse_args()

    cfg = CFG(epochs=args.epochs, batch_size=args.bs, lr=args.lr, amp=not args.no_amp, depth=args.depth)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(cfg)

    # Initialize model with ConvFFN (Optical Convolution)
    model = ViT(
        img_size=cfg.img_size, patch_size=cfg.patch, in_chans=1, num_classes=10,
        embed_dim=cfg.embed, depth=cfg.depth, num_heads=cfg.heads,
        mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1
    ).to(device)

    load_mzi_parameters(model, json_path="results/mzi_parameters.json")

    total_params = sum(p.numel() for p in model.parameters())
    print("=== OPTIMIZED OPTICAL CONVOLUTION ViT Model ===")
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

    print("Starting training with OPTICAL CONVOLUTION layers...")
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

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc*100:.2f}% | "
              f"Time: {epoch_time:.1f}s | "
              f"Eval: {eval_time:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict()}, "mnist_vit_optical_conv_best.pt")
            print(f"  -> New best accuracy: {best_acc*100:.2f}%")

        print()

    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")
    
    # Optional: Add hardware data collection if needed (omitted for brevity unless requested)

if __name__ == "__main__":
    main()
