import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from module.ONN import OpticalNetwork
from module.CNN import CNN_layer
from module.channel import SingleChannelFilter
from module.MZI_array.mzi_row_array import MZIlayer_row
from module.MZI_array.mzi_column_array import MZIlayer_column
from train import train, test
from torch.cuda.amp import GradScaler
from args import get_args
import random
import numpy as np
import os
import json


def collect_and_save_filter_data(model, dataloader, device, save_dir="filter_data"):
    """
    收集并保存模型各层滤波器的特定数据，新版将明确区分吸收器前后的数据，并恢复所有参数的收集。
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    collected_filter_internals = {}
    hook_handles = []

    def get_filter_internal_data_hook(layer_idx, filter_idx_in_cnn_layer, cnn_layer_in_channels):
        def hook(module, input_args, output_tensor_overall_filter):
            input_for_intermediate_calc = input_args[0][0:1, :, :]

            _, mzi_array_input_patch, mzi_output_before_absorber_patch = module.forward(
                input_for_intermediate_calc, return_intermediate=True
            )

            with torch.no_grad():
                patches_for_weighting = mzi_output_before_absorber_patch.view(1, 1, -1)
                output_after_absorber_patch = patches_for_weighting * module.diagonal_matrix.view(1, 1, -1)

            key = f'layer_{layer_idx}_filter_{filter_idx_in_cnn_layer}'
            current_filter_data = {}

            current_filter_data['mzi_array_input'] = mzi_array_input_patch.detach().cpu().numpy()
            current_filter_data['mzi_output_before_absorber'] = mzi_output_before_absorber_patch.detach().cpu().numpy()
            current_filter_data['output_after_absorber'] = output_after_absorber_patch.detach().cpu().numpy()

            ks = module.kernel_size
            mzi_array_io_waveguide_desc = []
            num_elements_from_patch = ks * ks
            for i in range(10):
                if i < num_elements_from_patch:
                    patch_row = i // ks
                    patch_col = i % ks
                    mzi_array_io_waveguide_desc.append(
                        f"Waveguide {i}: From Patch Element [{patch_row},{patch_col}] (after F.unfold)"
                    )
                else:
                    mzi_array_io_waveguide_desc.append(f"Waveguide {i}: Padding")
            current_filter_data['mzi_array_io_waveguide_description'] = mzi_array_io_waveguide_desc

            raw_thetas_with_position = []
            for scf_layer_idx, mzi_array_layer_instance in enumerate(module.layers):
                mzi_array_type = 'unknown_mzi_array_layer'
                num_mzis_in_this_mzi_array_layer = 0

                if hasattr(mzi_array_layer_instance, 'MZI') and isinstance(mzi_array_layer_instance.MZI, torch.nn.ModuleList):
                    num_mzis_in_this_mzi_array_layer = len(mzi_array_layer_instance.MZI)

                    if isinstance(mzi_array_layer_instance, MZIlayer_row):
                        mzi_array_type = 'MZIlayer_row'
                    elif isinstance(mzi_array_layer_instance, MZIlayer_column):
                        mzi_array_type = 'MZIlayer_column'

                    for mzi_idx_in_array, mzi_instance in enumerate(mzi_array_layer_instance.MZI):
                        theta_value = mzi_instance.raw_sin_theta.detach().cpu().numpy().item()
                        theta_grad = mzi_instance.raw_sin_theta.grad.detach().cpu().numpy().item() if mzi_instance.raw_sin_theta.grad is not None else None

                        acting_on_waveguides_desc = "N/A"
                        if mzi_array_type == 'MZIlayer_row':
                            wg1 = 2 * mzi_idx_in_array
                            wg2 = 2 * mzi_idx_in_array + 1
                            acting_on_waveguides_desc = f"Acts on input waveguides [{wg1}, {wg2}] of the 10-WG bus entering this MZIlayer_row."
                        elif mzi_array_type == 'MZIlayer_column':
                            wg1 = (2 * mzi_idx_in_array) + 1
                            wg2 = (2 * mzi_idx_in_array) + 2
                            acting_on_waveguides_desc = f"Acts on input waveguides [{wg1}, {wg2}] of the 10-WG bus entering this MZIlayer_column."

                        position_info = {
                            'scf_mzi_array_layer_index': scf_layer_idx,
                            'mzi_array_layer_type': mzi_array_type,
                            'mzi_index_within_mzi_array_layer': mzi_idx_in_array,
                            'total_mzis_in_this_mzi_array_layer': num_mzis_in_this_mzi_array_layer,
                            'mzi_acting_on_waveguides_description': acting_on_waveguides_desc
                        }
                        raw_thetas_with_position.append({
                            'value': theta_value,
                            'gradient': theta_grad,
                            'position': position_info
                        })
                else:
                    print(f"Warning: Encountered unexpected layer type or structure in SingleChannelFilter.layers[{scf_layer_idx}]")

            current_filter_data['raw_sin_thetas_with_position'] = raw_thetas_with_position
            collected_filter_internals[key] = current_filter_data

        return hook

    try:
        data_iter = iter(dataloader)
        image_batch, _ = next(data_iter)
        single_input_image = image_batch[0:1].to(device)
    except StopIteration:
        print("错误：数据加载器为空，无法获取图像。" )
        return

    for cnn_layer_idx, cnn_layer_module in enumerate(model.layers):
        if isinstance(cnn_layer_module, CNN_layer):
            for filter_idx_in_cnn, single_channel_filter_module in enumerate(cnn_layer_module.filters):
                if isinstance(single_channel_filter_module, SingleChannelFilter):
                    handle = single_channel_filter_module.register_forward_hook(
                        get_filter_internal_data_hook(cnn_layer_idx, filter_idx_in_cnn, cnn_layer_module.in_channels)
                    )
                    hook_handles.append(handle)

    if not hook_handles:
        print("警告: 没有为任何 SingleChannelFilter 注册钩子。" )
        return

    # Run a forward and dummy backward pass to populate gradients for collection
    final_model_output = model(single_input_image)
    if final_model_output.requires_grad:
        dummy_loss = final_model_output.sum()
        dummy_loss.backward()
        model.zero_grad()

    for handle in hook_handles:
        handle.remove()

    complete_data_to_save = {
        'input_image_to_model': single_input_image.cpu().numpy(),
        'final_model_output': final_model_output.detach().cpu().numpy(),
        'filter_internals': collected_filter_internals
    }

    save_path = os.path.join(save_dir, 'hook_data_verified.npy')
    np.save(save_path, complete_data_to_save)
    print(f"已验证的、包含明确分段的Hook数据已保存至: {save_path}")


def print_memory_stats():
    """
    Print current CUDA memory usage statistics
    """
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

def main():
    """
    Main function: Implements the training and testing pipeline on MNIST dataset
    """
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.empty_cache()
    
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    collect_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=train_transform)
    collect_dataset = datasets.MNIST('../data', train=False, transform=collect_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    collect_loader = DataLoader(collect_dataset, batch_size=1, shuffle=False)
    
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
        detection_mode=args.detection_mode
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler()
    
    if args.wandb:
        wandb.init(project="optical-neural-network")
        wandb.config.update(args)

    # Initialize training log
    training_log = {
        'train_epochs': [],
        'train_losses': [],
        'train_accuracies': [],
        'test_epochs': [],
        'test_losses': [],
        'test_accuracies': []
    }

    for epoch in range(1, args.epochs + 1):
        # Train and get epoch metrics
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, scaler,
                                      clip_value=args.grad_clip, log_interval=args.log_interval, args=args)

        # Test and get epoch metrics
        test_loss, test_acc = test(model, device, test_loader)

        # Log epoch metrics
        training_log['train_epochs'].append(epoch)
        training_log['train_losses'].append(train_loss)
        training_log['train_accuracies'].append(train_acc)
        training_log['test_epochs'].append(epoch)
        training_log['test_losses'].append(test_loss)
        training_log['test_accuracies'].append(test_acc)

        if device.type == 'cuda':
            print_memory_stats()
            torch.cuda.empty_cache()

    # Save training log to JSON
    log_path = 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"\nTraining log saved to: {log_path}")

    print("\n训练完成。开始收集和保存已验证的Hook数据...")
    # 在这里，你需要确保模型已经加载了你想要分析的权重
    # 例如: model.load_state_dict(torch.load('optical_network.pt'))
    collect_and_save_filter_data(model, collect_loader, device)

    if args.save_model:
        # Generate descriptive filename with configuration info
        model_filename = (
            f"optical_network_{args.detection_mode}_"
            f"ch{args.hidden_channels}_"
            f"layers{args.num_layers}_"
            f"ep{args.epochs}_"
            f"lr{args.lr}.pt"
        )
        torch.save(model.state_dict(), model_filename)
        print(f"\n[Model Saved] {model_filename}")
        print(f"  - Detection mode: {args.detection_mode}")
        print(f"  - Hidden channels: {args.hidden_channels}")
        print(f"  - Num layers: {args.num_layers}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Learning rate: {args.lr}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
