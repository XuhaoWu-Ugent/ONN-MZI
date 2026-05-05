import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
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
    Compute output power through MZI array given input amplitude.

    Args:
        filter_module: SingleChannelFilter with MZI layers
        input_amplitude: Complex amplitude input, shape (batch, num_patches, num_ports)
        device: torch device

    Returns:
        output_power: Power at output ports, shape (batch, num_patches, num_ports)
    """
    input_amplitude = input_amplitude.to(device)
    combined_matrix = filter_module._get_combined_matrix(device)

    batch_size, num_patches, num_ports = input_amplitude.shape
    matrix_size = filter_module.matrix_size

    if filter_module.detection_mode == 'power':
        # Power mode: process each input port independently, sum output powers
        output_powers = torch.zeros(batch_size, num_patches, num_ports,
                                   dtype=torch.float32, device=device)
        batch_total = batch_size * num_patches

        for port_idx in range(num_ports):
            port_input = input_amplitude[:, :, port_idx].reshape(batch_total)
            state_vectors = torch.zeros(batch_total, matrix_size,
                                       dtype=torch.complex64, device=device)
            state_vectors[:, port_idx] = port_input
            new_states = torch.matmul(state_vectors, combined_matrix.T)
            port_outputs = new_states[:, num_ports:num_ports + num_ports]
            port_outputs = port_outputs.view(batch_size, num_patches, num_ports)
            port_power = torch.abs(port_outputs) ** 2
            output_powers += port_power

        return output_powers
    else:
        # Coherent mode: process all ports together
        batch_total = batch_size * num_patches
        patches_flat = input_amplitude.view(batch_total, num_ports)
        state_vectors = torch.zeros(batch_total, matrix_size,
                                   dtype=torch.complex64, device=device)
        state_vectors[:, :num_ports] = patches_flat
        new_states = torch.matmul(state_vectors, combined_matrix.T)
        output_flat = new_states[:, num_ports:num_ports + num_ports]
        output_patches = output_flat.view(batch_size, num_patches, num_ports)
        # Return power (intensity)
        return (torch.abs(output_patches) ** 2)

def collect_hardware_data():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model
    model = OpticalNetwork(
        input_channels=args.input_channels, hidden_channels=args.hidden_channels,
        output_size=args.output_size, num_layers=args.num_layers, kernel_size=args.kernel_size,
        input_size=args.input_size, mzi_repeat_num=args.mzi_repeat_num,
        mzi_row_num=args.mzi_row_num, mzi_column_num=args.mzi_column_num,
        detection_mode=args.detection_mode, use_optical_fc=args.use_optical_fc,
        fc_activation_mode=args.fc_activation_mode, num_shared_weights=args.num_shared_weights,
        fc_pos_only=args.fc_pos_only
    ).to(device)

    # 2. Load the best weights
    checkpoint_override = os.environ.get('CHECKPOINT_PATH', '').strip()
    if checkpoint_override:
        if not os.path.exists(checkpoint_override):
            print(f"Error: CHECKPOINT_PATH={checkpoint_override} does not exist.")
            return
        checkpoint_path = checkpoint_override
        print(f"Using checkpoint from CHECKPOINT_PATH env: {checkpoint_path}")
    else:
        checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_ch{args.hidden_channels}_full_distill_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_full_distill_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_alpha0.5_sigma0.15_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"optimized_shared_K{args.num_shared_weights}_ch{args.hidden_channels}_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"optimized_shared_K{args.num_shared_weights}_best.pt"

        if not os.path.exists(checkpoint_path):
            print(f"Error: No checkpoint found.")
            return
    
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    load_mzi_calibration(model, 'results/mzi_parameters_multi.json')
    model.eval()

    # 3. Enable Hooks
    print("Enabling hardware hooks...")
    for layer in model.layers:
        if isinstance(layer, CNN_layer):
            for filt in layer.filters:
                filt.enable_hooks()
    if hasattr(model, 'fc'):
        model.fc.enable_hooks()

    # 4. Prepare Input (保留 Normalize 以维持模型精度)
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

    # 5. Run Forward Pass
    print("Running inference...")
    with torch.no_grad():
        _ = model(data_onn)

    # 6. Extract Data
    results = {
        'metadata': {
            'K': args.num_shared_weights,
            'detection_mode': args.detection_mode,
        },
        'layers': [],
        'fc': {}
    }

    def get_info(module, bias_val=None, path_type=None, component_type=None,
                 override_input_power=None, override_output_power=None, tag=None):
        """
        Extract hardware data from a module.

        Args:
            module: Filter module with hook_data
            bias_val: Bias parameter tensor
            path_type: 'positive' or 'negative'
            component_type: 'cnn_filter', 'encoder', or 'decoder'
            override_input_power: If provided, use this instead of hook_data
            override_output_power: If provided, use this instead of hook_data
            tag: Identifier string
        """
        info = {
            'tag': tag,
            'path_type': path_type,
            'component_type': component_type,
            'voltages': module.hook_data['mzi_voltages'].numpy() if module.hook_data['mzi_voltages'] is not None else None,
            'bias': bias_val.detach().cpu().numpy() if bias_val is not None else None
        }

        # Handle in_power
        if override_input_power is not None:
            info['in_power'] = override_input_power.detach().cpu().numpy() if torch.is_tensor(override_input_power) else override_input_power
        else:
            info['in_power'] = module.hook_data['optical_input'].numpy() if module.hook_data['optical_input'] is not None else None

        # Handle out_power
        if override_output_power is not None:
            info['out_power'] = override_output_power.detach().cpu().numpy() if torch.is_tensor(override_output_power) else override_output_power
        else:
            info['out_power'] = module.hook_data['optical_output'].numpy() if module.hook_data['optical_output'] is not None else None

        return info

    # Extract CNN Layers (拆分为正负路对，分别计算out_power)
    print("Extracting CNN layers with pos/neg path separation...")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, CNN_layer):
            layer_info = {'layer_idx': i, 'filters': []}
            for filt_idx, filt in enumerate(layer.filters):
                raw_input = filt.hook_data['optical_input']
                if raw_input is not None:
                    # 原始输入是复数振幅，取实部作为基准
                    if raw_input.is_complex():
                        raw_input = raw_input.real

                    # ========== 正路 (Positive Path) ==========
                    # 输入功率: P_pos = ReLU(x)^2
                    pos_input_power = torch.abs(F.relu(raw_input)) ** 2
                    # 输入振幅: A_pos = sqrt(P_pos) = |ReLU(x)|
                    pos_input_amplitude = torch.sqrt(pos_input_power + 1e-10).to(torch.complex64)
                    # 重新计算输出功率
                    pos_output_power = compute_mzi_output_power(filt, pos_input_amplitude, device)

                    layer_info['filters'].append(get_info(
                        filt, bias_val=filt.bias, tag=f"filt{filt_idx}_pos",
                        path_type='positive', component_type='cnn_filter',
                        override_input_power=pos_input_power,
                        override_output_power=pos_output_power
                    ))

                    # ========== 负路 (Negative Path) ==========
                    # 输入功率: P_neg = ReLU(-x)^2
                    neg_input_power = torch.abs(F.relu(-raw_input)) ** 2
                    # 输入振幅: A_neg = sqrt(P_neg) = |ReLU(-x)|
                    neg_input_amplitude = torch.sqrt(neg_input_power + 1e-10).to(torch.complex64)
                    # 重新计算输出功率
                    neg_output_power = compute_mzi_output_power(filt, neg_input_amplitude, device)

                    layer_info['filters'].append(get_info(
                        filt, bias_val=filt.bias, tag=f"filt{filt_idx}_neg",
                        path_type='negative', component_type='cnn_filter',
                        override_input_power=neg_input_power,
                        override_output_power=neg_output_power
                    ))

            results['layers'].append(layer_info)
            print(f"  Layer {i}: Extracted {len(layer_info['filters'])} filter entries")

    # Extract FC Layer (按照正/负路成对提取，每个 processor 包含其处理的所有 slices)
    if hasattr(model, 'fc'):
        fc = model.fc
        fc_input = fc.get_fc_input()  # 获取记录的 FC 输入

        if fc_input is None:
            print("Warning: FC input not recorded. Make sure hooks are enabled.")
        else:
            print(f"FC input shape: {fc_input.shape}")  # (batch, in_features)

        # Clean FC: decoder is None, bias is None, uses output_scale/output_shift
        # Legacy FC: decoder + bias + running_logit_mean
        is_clean_fc = getattr(fc, 'use_clean_fc', False) or (fc.pos_decoder_slice is None)
        print(f"FC mode: {'Clean (no decoder)' if is_clean_fc else 'Legacy (with decoder)'}")

        fc_res = {
            'fc_mode': 'clean' if is_clean_fc else 'legacy',
            'n_slices': fc.n_slices,
            'num_processors': fc.num_processors,
            'processors': []  # 保留原结构
        }
        if is_clean_fc:
            fc_res['output_scale'] = fc.output_scale.detach().cpu().numpy()
            fc_res['output_shift'] = fc.output_shift.detach().cpu().numpy()
            fc_res['global_bias'] = None
        else:
            fc_res['global_bias'] = fc.bias.detach().cpu().numpy()
            if hasattr(fc, 'running_logit_mean'):
                fc_res['running_logit_mean'] = fc.running_logit_mean.detach().cpu().numpy()

        # 计算所有 slice 的输入输出
        if fc_input is not None:
            fc_input = fc_input.to(device)
            batch_size = fc_input.size(0)

            # Pad and reshape
            if fc.padded_in > fc.in_features:
                fc_input_padded = F.pad(fc_input, (0, fc.padded_in - fc.in_features))
            else:
                fc_input_padded = fc_input
            x_sliced = fc_input_padded.view(batch_size, fc.n_slices, fc.r)  # (batch, n_slices, 10)

            # 提取每个 processor 的 voltages
            def get_processor_voltages(processor):
                voltages = []
                for layer in processor.layers:
                    if hasattr(layer, 'MZI'):
                        for mzi in layer.MZI:
                            voltages.append(mzi.get_voltage().item())
                return voltages

            # 提取每个 processor 的 combined matrix
            pos_enc_matrices = [p._get_combined_matrix(device) for p in fc.pos_encoder_slices]
            neg_enc_matrices = [p._get_combined_matrix(device) for p in fc.neg_encoder_slices]

            num_ports = fc.pos_encoder_slices[0].num_ports
            matrix_size = fc.pos_encoder_slices[0].matrix_size

            # 计算单个 slice 通过 MZI 阵列的输出功率
            def compute_slice_output(slice_input, combined_matrix):
                x_amp = torch.sqrt(torch.abs(slice_input) + 1e-8).to(torch.complex64)
                output_power = torch.zeros(batch_size, num_ports, dtype=torch.float32, device=device)
                for port_idx in range(num_ports):
                    state = torch.zeros(batch_size, matrix_size, dtype=torch.complex64, device=device)
                    state[:, port_idx] = x_amp[:, port_idx]
                    new_state = torch.matmul(state, combined_matrix.T)
                    output_power += torch.abs(new_state[:, num_ports:2*num_ports]) ** 2
                return output_power

            # 找出每个 processor 处理的 slice 索引
            def get_slice_indices_for_processor(proc_idx):
                return [s for s in range(fc.n_slices) if s % fc.num_processors == proc_idx]

            print(f"Extracting {fc.n_slices} slices (K={fc.num_processors} processors)...")

            # 用于累加 decoder 输入
            pos_hidden_accum = torch.zeros(batch_size, num_ports, dtype=torch.float32, device=device)
            neg_hidden_accum = torch.zeros(batch_size, num_ports, dtype=torch.float32, device=device)

            # 1. 提取 Encoders (成对: Pos0, Neg0, Pos1, Neg1...)
            num_enc = len(fc.pos_encoder_slices)
            for i in range(num_enc):
                slice_indices = get_slice_indices_for_processor(i)

                # 收集该 processor 处理的所有 slices 的数据
                in_powers = []
                pos_out_powers = []
                neg_out_powers = []

                for slice_idx in slice_indices:
                    slice_input = x_sliced[:, slice_idx, :]  # (batch, 10)
                    # 注意: FC 输入已经是功率域 (在 ONN_shared.py 中 x = x**2)
                    # 所以这里不需要再平方
                    slice_input_power = torch.abs(slice_input)  # 直接取绝对值（已经是功率）

                    pos_out = compute_slice_output(slice_input, pos_enc_matrices[i])
                    neg_out = compute_slice_output(slice_input, neg_enc_matrices[i])

                    in_powers.append(slice_input_power.detach().cpu().numpy())
                    pos_out_powers.append(pos_out.detach().cpu().numpy())
                    neg_out_powers.append(neg_out.detach().cpu().numpy())

                    # 累加到 hidden
                    pos_hidden_accum += pos_out
                    neg_hidden_accum += neg_out

                # 正路 encoder
                fc_res['processors'].append({
                    'tag': f"enc{i}_pos",
                    'path_type': 'positive',
                    'component_type': 'encoder',
                    'processor_idx': i,
                    'slice_indices': slice_indices,
                    'voltages': get_processor_voltages(fc.pos_encoder_slices[i]),
                    'in_power': np.stack(in_powers, axis=1) if in_powers else None,  # (batch, num_slices, 10)
                    'out_power': np.stack(pos_out_powers, axis=1) if pos_out_powers else None,
                })

                # 负路 encoder
                fc_res['processors'].append({
                    'tag': f"enc{i}_neg",
                    'path_type': 'negative',
                    'component_type': 'encoder',
                    'processor_idx': i,
                    'slice_indices': slice_indices,
                    'voltages': get_processor_voltages(fc.neg_encoder_slices[i]),
                    'in_power': np.stack(in_powers, axis=1) if in_powers else None,  # 输入相同
                    'out_power': np.stack(neg_out_powers, axis=1) if neg_out_powers else None,
                })

            # Normalize hidden
            pos_hidden = pos_hidden_accum / (fc.n_slices ** 0.5)
            neg_hidden = neg_hidden_accum / (fc.n_slices ** 0.5)

            # 2. 提取 Decoders (成对: Pos, Neg) — Legacy only
            if is_clean_fc:
                print("  Clean FC: skipping decoder extraction")
            else:
                def compute_decoder_output(decoder, hidden_input):
                    hidden_input = torch.abs(hidden_input)
                    x_amp = torch.sqrt(hidden_input + 1e-8).to(torch.complex64)
                    combined_matrix = decoder._get_combined_matrix(device)
                    output_power = torch.zeros(batch_size, num_ports, dtype=torch.float32, device=device)
                    for port_idx in range(num_ports):
                        state = torch.zeros(batch_size, matrix_size, dtype=torch.complex64, device=device)
                        state[:, port_idx] = x_amp[:, port_idx]
                        new_state = torch.matmul(state, combined_matrix.T)
                        output_power += torch.abs(new_state[:, num_ports:2*num_ports]) ** 2
                    return output_power

                pos_dec_output = compute_decoder_output(fc.pos_decoder_slice, pos_hidden)
                neg_dec_output = compute_decoder_output(fc.neg_decoder_slice, neg_hidden)

                fc_res['processors'].append({
                    'tag': 'dec_pos',
                    'path_type': 'positive',
                    'component_type': 'decoder',
                    'voltages': get_processor_voltages(fc.pos_decoder_slice),
                    'in_power': pos_hidden.detach().cpu().numpy(),
                    'out_power': pos_dec_output.detach().cpu().numpy(),
                })
                fc_res['processors'].append({
                    'tag': 'dec_neg',
                    'path_type': 'negative',
                    'component_type': 'decoder',
                    'voltages': get_processor_voltages(fc.neg_decoder_slice),
                    'in_power': neg_hidden.detach().cpu().numpy(),
                    'out_power': neg_dec_output.detach().cpu().numpy(),
                })

        results['fc'] = fc_res

    # 7. Save (NPY + JSON)
    hw_data_suffix = os.environ.get('HW_DATA_SUFFIX', '').strip()
    base_name = f"mzi_hardware_data_K{args.num_shared_weights}_ch{args.hidden_channels}_full_distill{hw_data_suffix}"
    npy_path = f"results/{base_name}.npy"
    json_path = f"results/{base_name}.json"

    # Save NPY (for Python loading)
    np.save(npy_path, results)

    # Convert numpy arrays to lists for JSON serialization
    def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_list(item) for item in obj]
        else:
            return obj

    results_json = numpy_to_list(results)

    # Save JSON (for easier inspection and cross-platform use)
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4)

    # Print summary and validation
    print(f"\n{'='*60}")
    print(f"Hardware snapshot saved to:")
    print(f"  NPY:  {npy_path}")
    print(f"  JSON: {json_path}")
    print(f"\nData Summary:")
    print(f"  CNN layers: {len(results['layers'])}")
    if len(results['layers']) > 0:
        num_filters = len(results['layers'][0]['filters']) // 2
        print(f"  CNN Layer 0: {len(results['layers'][0]['filters'])} entries ({num_filters} filters × 2 paths)")

        # Validation: check in_power/out_power consistency
        filt0_pos = results['layers'][0]['filters'][0]
        in_shape = np.array(filt0_pos['in_power']).shape if filt0_pos['in_power'] is not None else None
        out_shape = np.array(filt0_pos['out_power']).shape if filt0_pos['out_power'] is not None else None
        print(f"\n  Validation (filt0_pos):")
        print(f"    in_power shape:  {in_shape}")
        print(f"    out_power shape: {out_shape}")

        # Check that out_power is zero when in_power is zero
        if in_shape is not None and out_shape is not None:
            in_power = np.array(filt0_pos['in_power'])
            out_power = np.array(filt0_pos['out_power'])
            zero_in_mask = np.all(in_power < 1e-8, axis=-1)  # patches where all in_power ≈ 0
            zero_in_count = np.sum(zero_in_mask)
            if zero_in_count > 0:
                out_at_zero_in = out_power[zero_in_mask]
                max_out_at_zero = np.max(out_at_zero_in)
                print(f"    Patches with zero input: {zero_in_count}")
                print(f"    Max out_power when in=0: {max_out_at_zero:.2e} (should be ~0)")

    if hasattr(model, 'fc') and 'processors' in results['fc']:
        fc_data = results['fc']
        num_enc = fc_data['num_processors']
        mode = fc_data.get('fc_mode', 'legacy')
        print(f"\n  FC Layer ({mode} mode):")
        print(f"    Total slices (N): {fc_data['n_slices']}")
        print(f"    Physical processors (K): {fc_data['num_processors']}")
        dec_count = 0 if mode == 'clean' else 2
        print(f"    Processor entries: {len(fc_data['processors'])} ({num_enc}×2 encoders + {dec_count} decoders)")
        # 验证 encoder 数据
        enc0_pos = fc_data['processors'][0]
        if enc0_pos['in_power'] is not None:
            in_shape = np.array(enc0_pos['in_power']).shape
            out_shape = np.array(enc0_pos['out_power']).shape
            print(f"    enc0_pos slice_indices: {enc0_pos.get('slice_indices', 'N/A')}")
            print(f"    enc0_pos in_power shape: {in_shape}")
            print(f"    enc0_pos out_power shape: {out_shape}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    collect_hardware_data()