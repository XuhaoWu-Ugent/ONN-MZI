import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from module.ONN_shared import OpticalNetwork
from module.CNN import CNN_layer
from args import get_args
from torchvision import datasets, transforms

def load_mzi_parameters_from_json(model, json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f: mzi_params = json.load(f)
    def load_mzi_layer_params(mzi_layer):
        if hasattr(mzi_layer, 'MZI'):
            for mzi in mzi_layer.MZI:
                if hasattr(mzi, 'index') and mzi.index is not None:
                    physical_index = mzi.index % 50
                    param_key = str(physical_index)
                    if param_key in mzi_params:
                        params = mzi_params[param_key]
                        mzi.load_physical_parameters(a=params['a'], b=params['b'], delta_r=params['delta_r'], phi0=params['phi0'])
                        mzi.freeze_fabrication_parameters()
    for layer in model.layers:
        if isinstance(layer, CNN_layer):
            for f in layer.filters:
                for ml in f.layers: load_mzi_layer_params(ml)
    if hasattr(model, 'fc'):
        for path in ['pos_encoder_slices', 'neg_encoder_slices', 'pos_decoder_slice', 'neg_decoder_slice']:
            if hasattr(model.fc, path):
                module = getattr(model.fc, path)
                if isinstance(module, nn.ModuleList):
                    for p in module: 
                        for ml in p.layers: load_mzi_layer_params(ml)
                elif hasattr(module, 'layers'):
                    for ml in module.layers: load_mzi_layer_params(ml)

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
    checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_alpha0.5_sigma0.15_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"optimized_shared_K{args.num_shared_weights}_best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found.")
        return
    
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    load_mzi_parameters_from_json(model, 'results/mzi_parameters.json')
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

    def get_info(module, bias_val=None, path_type=None, component_type=None, override_input_power=None, tag=None):
        info = {
            'tag': tag,
            'path_type': path_type,
            'component_type': component_type,
            'voltages': module.hook_data['mzi_voltages'].numpy() if module.hook_data['mzi_voltages'] is not None else None,
            'out_power': module.hook_data['optical_output'].numpy() if module.hook_data['optical_output'] is not None else None,
            'bias': bias_val.detach().cpu().numpy() if bias_val is not None else None
        }
        
        if override_input_power is not None:
            info['in_power'] = override_input_power.numpy()
        else:
            info['in_power'] = module.hook_data['optical_input'].numpy() if module.hook_data['optical_input'] is not None else None
                
        return info

    # Extract CNN Layers (拆分为正负路对)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, CNN_layer):
            layer_info = {'layer_idx': i, 'filters': []}
            for filt_idx, filt in enumerate(layer.filters):
                raw_input = filt.hook_data['optical_input']
                if raw_input is not None:
                    if raw_input.is_complex(): raw_input = raw_input.real
                    
                    # 正路: P_pos = ReLU(x)^2
                    pos_input = torch.abs(F.relu(raw_input)) ** 2
                    layer_info['filters'].append(get_info(
                        filt, bias_val=filt.bias, tag=f"filt{filt_idx}_pos",
                        path_type='positive', component_type='cnn_filter', override_input_power=pos_input
                    ))
                    
                    # 负路: P_neg = ReLU(-x)^2
                    neg_input = torch.abs(F.relu(-raw_input)) ** 2
                    layer_info['filters'].append(get_info(
                        filt, bias_val=filt.bias, tag=f"filt{filt_idx}_neg",
                        path_type='negative', component_type='cnn_filter', override_input_power=neg_input
                    ))
            results['layers'].append(layer_info)

    # Extract FC Layer (按照正/负路成对提取，方便硬件顺序测试)
    if hasattr(model, 'fc'):
        fc_res = {
            'global_bias': model.fc.bias.detach().cpu().numpy(), 
            'processors': [] # 改为列表结构
        }
        
        # 1. 提取 Encoders (成对: Pos0, Neg0, Pos1, Neg1...)
        num_enc = len(model.fc.pos_encoder_slices)
        for i in range(num_enc):
            p_pos = model.fc.pos_encoder_slices[i]
            p_neg = model.fc.neg_encoder_slices[i]
            
            fc_res['processors'].append(get_info(p_pos, path_type='positive', component_type='encoder', tag=f"enc{i}_pos"))
            fc_res['processors'].append(get_info(p_neg, path_type='negative', component_type='encoder', tag=f"enc{i}_neg"))
            
        # 2. 提取 Decoders (成对: Pos, Neg)
        fc_res['processors'].append(get_info(model.fc.pos_decoder_slice, path_type='positive', component_type='decoder', tag="dec_pos"))
        fc_res['processors'].append(get_info(model.fc.neg_decoder_slice, path_type='negative', component_type='decoder', tag="dec_neg"))
        
        results['fc'] = fc_res

    # 7. Save
    save_path = f"results/mzi_hardware_data_K{args.num_shared_weights}.npy"
    np.save(save_path, results)
    print(f"\n{'='*60}")
    print(f"Hardware snapshot saved to: {save_path}")
    print(f"Captured data for {len(results['layers'])} CNN layers.")
    print(f"CNN Layer 0: {len(results['layers'][0]['filters'])} entries (12 filters * 2 paths)")
    if hasattr(model, 'fc'):
        print(f"FC Layer: {len(results['fc']['processors'])} entries ({num_enc*2} encoders + 2 decoders)")
    print(f"{ '='*60}\n")

if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    collect_hardware_data()