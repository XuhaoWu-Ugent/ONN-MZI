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
    # Assuming the distilled model or the standard shared model
    checkpoint_path = f"distilled_shared_K{args.num_shared_weights}_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"optimized_shared_K{args.num_shared_weights}_best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found for K={args.num_shared_weights}")
        return
    
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    load_mzi_parameters_from_json(model, 'results/mzi_parameters.json')
    model.eval()

    # 3. Enable Hooks across the entire network
    print("Enabling hardware hooks...")
    # CNN Layers
    for layer in model.layers:
        if isinstance(layer, CNN_layer):
            for filt in layer.filters:
                filt.enable_hooks()
    # FC Layer
    if hasattr(model, 'fc'):
        model.fc.enable_hooks()

    # 4. Prepare a sample input
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True
    )
    data, target = next(iter(test_loader))
    data = data.to(device)
    # Resize for ONN
    data_onn = F.interpolate(data, size=(args.input_size, args.input_size), mode='bilinear')

    # 5. Run Forward Pass to excite the circuits
    print("Running inference to capture optical powers...")
    with torch.no_grad():
        _ = model(data_onn)

    # 6. Extract Data Structure
    results = {
        'metadata': {
            'K': args.num_shared_weights,
            'input_channels': args.input_channels,
            'hidden_channels': args.hidden_channels,
            'num_layers': args.num_layers
        },
        'layers': []
    }

    def get_info(module):
        return {
            'voltages': module.hook_data['mzi_voltages'].numpy() if module.hook_data['mzi_voltages'] is not None else None,
            'in_power': module.hook_data['optical_input'].numpy() if module.hook_data['optical_input'] is not None else None,
            'out_power': module.hook_data['optical_output'].numpy() if module.hook_data['optical_output'] is not None else None
        }

    # Extract CNN Layers
    for i, layer in enumerate(model.layers):
        if isinstance(layer, CNN_layer):
            layer_data = {'layer_idx': i, 'filters': []}
            for j, filt in enumerate(layer.filters):
                layer_data['filters'].append(get_info(filt))
            results['layers'].append(layer_data)

    # Extract FC Layer
    if hasattr(model, 'fc'):
        fc_data = {'type': 'FC_LoRA', 'processors': {}}
        for i, p in enumerate(model.fc.pos_encoder_slices):
            fc_data['processors'][f'pos_enc_k{i}'] = get_info(p)
        for i, p in enumerate(model.fc.neg_encoder_slices):
            fc_data['processors'][f'neg_enc_k{i}'] = get_info(p)
        fc_data['processors']['pos_dec'] = get_info(model.fc.pos_decoder_slice)
        fc_data['processors']['neg_dec'] = get_info(model.fc.neg_decoder_slice)
        results['fc'] = fc_data

    # 7. Save to NPY
    save_path = f"results/mzi_voltage_power_dump_K{args.num_shared_weights}.npy"
    np.save(save_path, results)
    
    print(f"\n{'='*60}")
    print(f"Hardware snapshot saved to: {save_path}")
    print(f"Captured data for {len(results['layers'])} CNN layers and 1 FC layer.")
    print(f"Total physical FC processors: {len(model.fc.pos_encoder_slices)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    collect_hardware_data()
