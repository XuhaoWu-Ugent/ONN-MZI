"""
Constrained linear FC for ablating the mathematical constraints of the
power-mode optical FC, parametrized directly without an MZI mesh.

Motivation:
    The 86.95% baseline (K=15 power optical FC) leaves a 6.88pt gap to
    the elec nn.Linear baseline (95.34%) under the same CNN. We don't
    know if this gap is due to:
        (a) fundamental math constraints (non-neg weights, column-sum
            energy conservation, K-way sharing, block structure)
        (b) mesh-specific factors (voltage->T parametrization difficulty,
            insertion loss alpha^11, thermal crosstalk)

    This module implements (a) directly via raw nn.Parameter + softmax/
    sigmoid bounding, skipping the mesh entirely. Comparing
    ConstrainedLinearFC's accuracy ceiling to optical FC's separates
    the two contribution sources.

Mathematical structure (matches OpticalSharedLinear at the W_eff level):
    y_j = output_scale_j * (Σ_i (T_pos_{block(i),j,(i mod r)}
                                  - T_neg_{block(i),j,(i mod r)}) · x_i) / √N
          + output_shift_j

    where T_pos, T_neg are (K, r, r) tensors with constraints:
      - non-neg: T entries in [0, 1]                       (lossless gate)
      - col-sum: Σ_j T_{k,j,i} = 1   (lossless energy conservation)
      - blocks:  N=ceil(in/r) slices of the input mapped via K cyclic blocks

    Each constraint is independently toggleable via env vars.

Activate via env var: USE_CONSTRAINED_FC=1
Constraint toggles (defaults match optical FC): CONSTRAINED_FC_NONNEG=1,
    CONSTRAINED_FC_COLSUM=1, CONSTRAINED_FC_BLOCKS=1
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedLinearFC(nn.Module):
    def __init__(self, in_features, out_features, r=10,
                 num_shared_weights=None, **_unused_kwargs):
        super().__init__()

        if r < 1:
            raise ValueError(f"r must be a positive integer, got {r}")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.n_slices = math.ceil(in_features / r)
        self.padded_in = self.n_slices * r
        self.num_shared_weights = num_shared_weights
        # K = number of distinct blocks. None => no sharing (K = N).
        K = self.n_slices if num_shared_weights is None else min(num_shared_weights, self.n_slices)
        self.K = K

        # Constraint flags (env-controlled, default: match optical FC)
        self.use_nonneg = os.environ.get('CONSTRAINED_FC_NONNEG', '1') == '1'
        self.use_colsum = os.environ.get('CONSTRAINED_FC_COLSUM', '1') == '1'
        self.use_blocks = os.environ.get('CONSTRAINED_FC_BLOCKS', '1') == '1'

        if self.use_blocks:
            # K distinct (out_features × r) blocks; cyclically shared across
            # N slices. Each block maps r inputs → out_features outputs
            # directly, mimicking a partial readout from an r-port mesh
            # (only out_features of the r mesh-output ports are detected).
            # No electronic output_proj needed; the block itself is the
            # readout.
            self.raw_pos = nn.Parameter(torch.randn(K, out_features, r) * 0.1)
            self.raw_neg = nn.Parameter(torch.randn(K, out_features, r) * 0.1)
        else:
            # No slicing → single (out, padded_in) matrix; non-neg + col-sum
            # still applied if respective flags are on.
            self.raw_pos = nn.Parameter(torch.randn(out_features, self.padded_in) * 0.1)
            self.raw_neg = nn.Parameter(torch.randn(out_features, self.padded_in) * 0.1)

        # Diagonal affine (matches optical FC clean mode)
        self.output_scale = nn.Parameter(torch.ones(out_features))
        self.output_shift = nn.Parameter(torch.zeros(out_features))

        # API compat with OpticalSharedLinear
        self._fc_input_hook = None
        self._enable_fc_input_hook = False

        self._print_architecture()

    def _bound_T(self, raw):
        """Apply (configurable) non-negativity + column-sum constraints."""
        if self.use_colsum:
            # softmax along the output-port axis enforces column sum = 1
            # (and non-negativity automatically). For 3D blocks (K,r,r) the
            # output axis is dim=-2; for 2D full (out, in) it's dim=0.
            return F.softmax(raw, dim=-2)
        elif self.use_nonneg:
            return torch.sigmoid(raw)  # [0,1] without column-sum constraint
        else:
            return raw  # no constraints (≈ nn.Linear with the W = pos − neg parametrization)

    def _print_architecture(self):
        share_str = ("no sharing (K=N)" if self.num_shared_weights is None
                     else f"K={self.K} (cyclic across N={self.n_slices})")
        flags = (f"nonneg={self.use_nonneg}, "
                 f"colsum={self.use_colsum}, "
                 f"blocks={self.use_blocks}")
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*70}")
        print(f"ConstrainedLinearFC Initialized (no MZI mesh)")
        print(f"{'='*70}")
        print(f"Input -> Output: {self.in_features} -> {self.out_features}")
        print(f"Per-slice mesh size r: {self.r}")
        print(f"Slices (N): {self.n_slices}, {share_str}")
        print(f"Constraints: {flags}")
        print(f"Trainable params: {n_params}")
        print(f"{'='*70}\n")

    def forward(self, x):
        if self._enable_fc_input_hook:
            self._fc_input_hook = x.detach().cpu().clone()

        # Pad to padded_in if necessary
        if self.padded_in > self.in_features:
            x = F.pad(x, (0, self.padded_in - self.in_features))

        T_pos = self._bound_T(self.raw_pos)
        T_neg = self._bound_T(self.raw_neg)
        W_eff = T_pos - T_neg  # (K, r, r) or (out, padded_in)

        if self.use_blocks:
            x_sliced = x.view(x.size(0), self.n_slices, self.r)  # (B, N, r)
            block_indices = torch.arange(self.n_slices, device=x.device) % self.K
            W_per_slice = W_eff[block_indices]  # (N, r, r)
            # Per-slice: y[b,s,j] = Σ_i W[s,j,i] * x[b,s,i]
            y_per_slice = torch.einsum('bsi, sji -> bsj', x_sliced, W_per_slice)
            y = y_per_slice.sum(dim=1) / math.sqrt(self.n_slices)  # (B, r)
        else:
            # No slicing: full-matrix path
            y = F.linear(x, W_eff)  # (B, out_features)

        return self.output_scale * y + self.output_shift

    # ---- API compat with OpticalSharedLinear (no-ops mostly) ----
    def get_fc_input(self):
        return self._fc_input_hook

    def enable_hooks(self):
        self._enable_fc_input_hook = True

    def disable_hooks(self):
        self._enable_fc_input_hook = False
