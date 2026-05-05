#!/usr/bin/env python3
"""
extract_hardware_data.py - 完整的硬件数据提取脚本

改进:
1. CNN层: 提取所有 144 个 patches 的 in_power/out_power (已完成)
2. FC层: 提取所有 173 个 slices 的 in_power/out_power (新增)

数据结构:
- CNN: 每个filter拆分为pos/neg，各有144个patches的数据
- FC:
  - encoders: 15个processor，每个记录其处理的所有slices
  - decoders: 完整的hidden输入和输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import math
from module.ONN_shared import OpticalNetwork
from module.CNN import CNN_layer
from module.calibration_loader import load_mzi_calibration
from args import get_args
from torchvision import datasets, transforms


def load_mzi_parameters_from_json(model, json_path):
    """Thin wrapper kept for backward compatibility."""
    load_mzi_calibration(model, json_path)


def compute_mzi_output_power(filter_module, input_amplitude, device):
    """
    通过MZI阵列计算输出功率 (Power Mode)

    Args:
        filter_module: SingleChannelFilter 或 OpticalSliceProcessor
        input_amplitude: 复数振幅输入, shape可以是:
            - (batch, ports) for single slice
            - (batch, patches/slices, ports) for multiple
        device: torch device

    Returns:
        output_power: 输出功率，与输入shape相同
    """
    input_amplitude = input_amplitude.to(device)
    combined_matrix = filter_module._get_combined_matrix(device)

    original_shape = input_amplitude.shape

    if input_amplitude.dim() == 2:
        batch_size, num_ports = input_amplitude.shape
        num_items = 1
        input_amplitude = input_amplitude.unsqueeze(1)
    else:
        batch_size, num_items, num_ports = input_amplitude.shape

    matrix_size = filter_module.matrix_size

    output_powers = torch.zeros(batch_size, num_items, num_ports,
                               dtype=torch.float32, device=device)
    batch_total = batch_size * num_items

    for port_idx in range(num_ports):
        port_input = input_amplitude[:, :, port_idx].reshape(batch_total)
        state_vectors = torch.zeros(batch_total, matrix_size,
                                   dtype=torch.complex64, device=device)
        state_vectors[:, port_idx] = port_input
        new_states = torch.matmul(state_vectors, combined_matrix.T)
        port_outputs = new_states[:, num_ports:num_ports + num_ports]
        port_outputs = port_outputs.view(batch_size, num_items, num_ports)
        port_power = torch.abs(port_outputs) ** 2
        output_powers += port_power

    if len(original_shape) == 2:
        return output_powers.squeeze(1)
    return output_powers


def get_voltages(module):
    """提取模块中所有MZI的电压"""
    voltages = []
    for layer in module.layers:
        if hasattr(layer, 'MZI'):
            for mzi in layer.MZI:
                voltages.append(mzi.get_voltage().item())
    return voltages


def collect_hardware_data():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = OpticalNetwork(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        output_size=args.output_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        input_size=args.input_size,
        mzi_repeat_num=args.mzi_repeat_num,
        mzi_row_num=args.mzi_row_num,
        mzi_column_num=args.mzi_column_num,
        detection_mode=args.detection_mode,
        use_optical_fc=args.use_optical_fc,
        fc_activation_mode=args.fc_activation_mode,
        num_shared_weights=args.num_shared_weights,
        fc_pos_only=getattr(args, 'fc_pos_only', False)
    ).to(device)

    # 2. Load checkpoint
    checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_full_distill_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found.")
        return

    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    load_mzi_calibration(model, 'results/mzi_parameters_multi.json')
    model.eval()

    # 3. Prepare test input
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True
    )
    data, target = next(iter(test_loader))
    data = data.to(device)
    data_onn = F.interpolate(data, size=(args.input_size, args.input_size), mode='bilinear')

    print(f"Test input shape: {data_onn.shape}, target: {target.item()}")

    # 4. Results container
    results = {
        'metadata': {
            'K': args.num_shared_weights,
            'detection_mode': args.detection_mode,
            'input_size': args.input_size,
            'target_label': target.item(),
            'cnn_patches': 144,  # 12x12
            'fc_slices': 173
        },
        'layers': [],
        'fc': {}
    }

    # ========== 5. Extract CNN Layers (所有144个patches) ==========
    print("\n" + "="*60)
    print("Extracting CNN Layers (all 144 patches)")
    print("="*60)

    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            if isinstance(layer, CNN_layer):
                layer_info = {'layer_idx': layer_idx, 'filters': []}

                for filt_idx, filt in enumerate(layer.filters):
                    if layer_idx == 0:
                        x_input = data_onn.squeeze(1)
                    else:
                        x_input = data_onn.squeeze(1)

                    # 提取patches: (1, 144, 10)
                    patches = F.unfold(x_input.unsqueeze(1),
                                      kernel_size=filt.kernel_size,
                                      stride=1)
                    patches = patches.permute(0, 2, 1)
                    patches = F.pad(patches, (0, 1))

                    raw_input = patches.real if patches.is_complex() else patches
                    voltages = get_voltages(filt)

                    # ===== 正路 =====
                    pos_input_power = torch.abs(F.relu(raw_input)) ** 2
                    pos_input_amplitude = torch.sqrt(pos_input_power + 1e-10).to(torch.complex64)
                    pos_output_power = compute_mzi_output_power(filt, pos_input_amplitude, device)

                    layer_info['filters'].append({
                        'tag': f"filt{filt_idx}_pos",
                        'path_type': 'positive',
                        'component_type': 'cnn_filter',
                        'num_patches': 144,
                        'voltages': voltages,
                        'in_power': pos_input_power.cpu().numpy().tolist(),
                        'out_power': pos_output_power.cpu().numpy().tolist(),
                        'bias': filt.bias.detach().cpu().numpy().tolist(),
                        'diagonal': filt.diagonal_matrix.detach().cpu().numpy().tolist()
                    })

                    # ===== 负路 =====
                    neg_input_power = torch.abs(F.relu(-raw_input)) ** 2
                    neg_input_amplitude = torch.sqrt(neg_input_power + 1e-10).to(torch.complex64)
                    neg_output_power = compute_mzi_output_power(filt, neg_input_amplitude, device)

                    layer_info['filters'].append({
                        'tag': f"filt{filt_idx}_neg",
                        'path_type': 'negative',
                        'component_type': 'cnn_filter',
                        'num_patches': 144,
                        'voltages': voltages,
                        'in_power': neg_input_power.cpu().numpy().tolist(),
                        'out_power': neg_output_power.cpu().numpy().tolist(),
                        'bias': filt.bias.detach().cpu().numpy().tolist(),
                        'diagonal': filt.diagonal_matrix.detach().cpu().numpy().tolist()
                    })

                results['layers'].append(layer_info)
                print(f"  Layer {layer_idx}: {len(layer_info['filters'])} filter entries")
                print(f"    Each filter: 144 patches × 10 ports")

    # ========== 6. Extract FC Layer (所有173个slices) ==========
    print("\n" + "="*60)
    print("Extracting FC Layer (all 173 slices)")
    print("="*60)

    if hasattr(model, 'fc'):
        fc = model.fc
        K = len(fc.pos_encoder_slices)
        N = fc.n_slices  # 173
        r = fc.r  # 10

        # Clean FC: decoder is None, bias is None, uses output_scale/output_shift
        # Legacy FC: decoder + bias + running_logit_mean
        is_clean_fc = getattr(fc, 'use_clean_fc', False) or (fc.pos_decoder_slice is None)
        print(f"  FC mode: {'Clean (no decoder)' if is_clean_fc else 'Legacy (with decoder)'}")

        # 获取FC层输入
        with torch.no_grad():
            x = data_onn
            for layer in model.layers:
                x = layer(x)

            if model.detection_mode == 'coherent':
                x = torch.abs(x)
            elif model.detection_mode == 'power':
                x = torch.sqrt(x + 1e-8)
            x = model.activation(x)
            x = x.view(x.size(0), -1)
            fc_input = x ** 2

            # Padding and reshape
            padded_in = fc.padded_in
            if padded_in > fc.in_features:
                fc_input_padded = F.pad(fc_input, (0, padded_in - fc.in_features))
            else:
                fc_input_padded = fc_input

            fc_sliced = fc_input_padded.view(1, N, r)  # (1, 173, 10)

        print(f"  FC input: {fc_input.shape} -> sliced: {fc_sliced.shape}")
        print(f"  K={K} processors, N={N} slices")

        fc_res = {
            'fc_mode': 'clean' if is_clean_fc else 'legacy',
            'K': K,
            'N': N,
            'r': r,
            'encoders': [],  # 每个encoder处理的所有slices
            'decoders': []   # Empty list in clean mode
        }
        if is_clean_fc:
            # Clean mode: diagonal affine after (pos − neg)
            fc_res['output_scale'] = fc.output_scale.detach().cpu().numpy().tolist()
            fc_res['output_shift'] = fc.output_shift.detach().cpu().numpy().tolist()
            fc_res['global_bias'] = None  # Kept for JSON schema compat; use output_shift instead
        else:
            # Legacy mode: fc.bias + running_logit_mean
            fc_res['global_bias'] = fc.bias.detach().cpu().numpy().tolist()
            if hasattr(fc, 'running_logit_mean'):
                fc_res['running_logit_mean'] = fc.running_logit_mean.detach().cpu().numpy().tolist()

        # ===== Encoders: 记录每个processor处理的所有slices =====
        print(f"\n  Extracting encoders...")

        with torch.no_grad():
            for proc_idx in range(K):
                # 找出这个processor处理的所有slice indices
                slice_indices = list(range(proc_idx, N, K))
                num_slices_for_proc = len(slice_indices)

                # 获取这些slices的输入
                proc_input = fc_sliced[:, slice_indices, :]  # (1, num_slices, 10)
                proc_input_amplitude = torch.sqrt(proc_input + 1e-10).to(torch.complex64)

                # Positive encoder
                pos_processor = fc.pos_encoder_slices[proc_idx]
                pos_output = compute_mzi_output_power(pos_processor, proc_input_amplitude, device)

                fc_res['encoders'].append({
                    'tag': f"enc{proc_idx}_pos",
                    'path_type': 'positive',
                    'processor_idx': proc_idx,
                    'slice_indices': slice_indices,
                    'num_slices': num_slices_for_proc,
                    'voltages': get_voltages(pos_processor),
                    'in_power': proc_input.cpu().numpy().tolist(),   # (1, num_slices, 10)
                    'out_power': pos_output.cpu().numpy().tolist()   # (1, num_slices, 10)
                })

                # Negative encoder
                neg_processor = fc.neg_encoder_slices[proc_idx]
                neg_output = compute_mzi_output_power(neg_processor, proc_input_amplitude, device)

                fc_res['encoders'].append({
                    'tag': f"enc{proc_idx}_neg",
                    'path_type': 'negative',
                    'processor_idx': proc_idx,
                    'slice_indices': slice_indices,
                    'num_slices': num_slices_for_proc,
                    'voltages': get_voltages(neg_processor),
                    'in_power': proc_input.cpu().numpy().tolist(),
                    'out_power': neg_output.cpu().numpy().tolist()
                })

                if proc_idx == 0:
                    print(f"    Processor 0: handles {num_slices_for_proc} slices")
                    print(f"      slice_indices: {slice_indices[:5]}... (showing first 5)")

            print(f"  Extracted {len(fc_res['encoders'])} encoder entries")

            # ===== 计算完整的 hidden =====
            print(f"\n  Computing full hidden states...")

            pos_hidden = torch.zeros(1, r, device=device)
            neg_hidden = torch.zeros(1, r, device=device)

            # 累加所有173个slices的输出
            for slice_idx in range(N):
                proc_idx = slice_idx % K
                slice_input = fc_sliced[:, slice_idx:slice_idx+1, :]  # (1, 1, 10)
                slice_amp = torch.sqrt(slice_input + 1e-10).to(torch.complex64)

                pos_out = compute_mzi_output_power(fc.pos_encoder_slices[proc_idx], slice_amp, device)
                neg_out = compute_mzi_output_power(fc.neg_encoder_slices[proc_idx], slice_amp, device)

                pos_hidden += pos_out.squeeze(1)
                neg_hidden += neg_out.squeeze(1)

            # 归一化
            pos_hidden = pos_hidden / math.sqrt(N)
            neg_hidden = neg_hidden / math.sqrt(N)

            print(f"    pos_hidden: {pos_hidden.cpu().numpy()}")
            print(f"    neg_hidden: {neg_hidden.cpu().numpy()}")

            # ===== Decoders (Legacy only) =====
            if is_clean_fc:
                print(f"\n  Clean FC: skipping decoder extraction")
            else:
                print(f"\n  Extracting decoders...")

                # Positive decoder
                dec_pos = fc.pos_decoder_slice
                dec_in_amp_pos = torch.sqrt(pos_hidden + 1e-10).to(torch.complex64)
                dec_out_pos = compute_mzi_output_power(dec_pos, dec_in_amp_pos, device)

                fc_res['decoders'].append({
                    'tag': 'dec_pos',
                    'path_type': 'positive',
                    'voltages': get_voltages(dec_pos),
                    'in_power': pos_hidden.cpu().numpy().tolist(),
                    'out_power': dec_out_pos.cpu().numpy().tolist()
                })

                # Negative decoder
                dec_neg = fc.neg_decoder_slice
                dec_in_amp_neg = torch.sqrt(neg_hidden + 1e-10).to(torch.complex64)
                dec_out_neg = compute_mzi_output_power(dec_neg, dec_in_amp_neg, device)

                fc_res['decoders'].append({
                    'tag': 'dec_neg',
                    'path_type': 'negative',
                    'voltages': get_voltages(dec_neg),
                    'in_power': neg_hidden.cpu().numpy().tolist(),
                    'out_power': dec_out_neg.cpu().numpy().tolist()
                })

                print(f"  Extracted {len(fc_res['decoders'])} decoder entries")

            # Final output: use the model's actual forward instead of
            # manually replicating the post-mesh aggregation. This stays
            # correct under any combination of optical_linear_shared.py
            # forward variants (slice_weights, per-slice ReLU, output_proj
            # for r != out_features, etc.) without needing to keep this
            # script in sync.
            final_output = model(data_onn)

            # ===== 验证最终输出 =====
            pred = final_output.argmax(dim=-1).item()
            print(f"\n  Final output: {final_output.cpu().numpy().flatten()}")
            print(f"  Prediction: {pred}, Target: {target.item()}, Correct: {pred == target.item()}")

            fc_res['final_output'] = final_output.cpu().numpy().tolist()
            fc_res['prediction'] = pred

        results['fc'] = fc_res

    # ========== 7. Save ==========
    base_name = f"mzi_hardware_data_K{args.num_shared_weights}"
    npy_path = f"results/{base_name}.npy"
    json_path = f"results/{base_name}.json"

    np.save(npy_path, results)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ========== 8. Summary ==========
    print("\n" + "="*60)
    print("Extraction Complete - Full Data")
    print("="*60)
    print(f"  NPY:  {npy_path}")
    print(f"  JSON: {json_path}")
    print(f"\n  CNN Layer:")
    print(f"    Filters: 12 × 2 paths = 24 entries")
    print(f"    Each entry: 144 patches × 10 ports")
    print(f"    Total CNN data points: 24 × 144 × 10 = 34,560")
    fc_mode = results['fc'].get('fc_mode', 'legacy')
    print(f"\n  FC Layer ({fc_mode} mode):")
    print(f"    Encoders: {K} × 2 paths = {K*2} entries")
    total_slices = sum(len(e['slice_indices']) for e in results['fc']['encoders'])
    print(f"    Total slices covered: {total_slices // 2} × 2 = {total_slices}")
    dec_count = len(results['fc']['decoders'])
    print(f"    Decoders: {dec_count} entries" + (" (Clean FC: no decoder)" if dec_count == 0 else ""))
    fc_points = total_slices * 10 + dec_count * 10
    print(f"    Total FC data points: {total_slices} × 10 + {dec_count} × 10 = {fc_points}")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')
    collect_hardware_data()
