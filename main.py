import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from module.ONN import OpticalNetwork
from module.CNN import CNN_layer # 需要导入CNN_layer来检查类型
from module.channel import SingleChannelFilter # 需要导入SingleChannelFilter来检查类型
from module.MZI_array.mzi_row_array import MZIlayer_row
from module.MZI_array.mzi_column_array import MZIlayer_column
from train import train, test
from torch.cuda.amp import GradScaler
from args import get_args
import random
import numpy as np
import os


def collect_and_save_filter_data(model, dataloader, device, save_dir="filter_data"):
    """
    收集并保存模型各层滤波器的特定数据，并包含详细的位置信息：
    - 单个滤波器的MZI阵列输入 (第一个patch, 1x10) 及其端口映射描述
    - 单个滤波器的MZI阵列输出 (第一个patch, 1x10) 及其端口映射描述
    - 单个滤波器内所有MZI的raw_sin_theta值及其详细位置信息
    - 单个滤波器内所有MZI的raw_sin_theta值的梯度
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    collected_filter_internals = {}
    hook_handles = []

    def get_filter_internal_data_hook(layer_idx, filter_idx_in_cnn_layer, cnn_layer_in_channels):
        def hook(module, input_args, output_tensor_overall_filter):
            # module: 当前被hook的 SingleChannelFilter 实例
            input_for_intermediate_calc = input_args[0][0:1, :, :]

            _, mzi_array_input_patch, mzi_array_processed_patch = module.forward(
                input_for_intermediate_calc, return_intermediate=True
            )

            key = f'layer_{layer_idx}_filter_{filter_idx_in_cnn_layer}'
            current_filter_data = {}

            # MZI Array Input/Output
            current_filter_data['mzi_array_input'] = mzi_array_input_patch.detach().cpu().numpy()
            current_filter_data['mzi_array_output'] = mzi_array_processed_patch.detach().cpu().numpy()

            # 为MZI阵列的10个输入/输出波导生成描述
            ks = module.kernel_size
            mzi_array_io_waveguide_desc = []
            num_elements_from_patch = ks * ks
            for i in range(10):  # 假设MZI阵列核心部分总是10路波导
                if i < num_elements_from_patch:
                    patch_row = i // ks
                    patch_col = i % ks
                    mzi_array_io_waveguide_desc.append(
                        f"Waveguide {i}: From Patch Element [{patch_row},{patch_col}] (after F.unfold)"
                    )
                else:
                    mzi_array_io_waveguide_desc.append(f"Waveguide {i}: Padding")
            current_filter_data['mzi_array_io_waveguide_description'] = mzi_array_io_waveguide_desc

            # 收集此滤波器中所有MZI的raw_sin_theta及其详细位置和梯度
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
        print("错误：数据加载器为空，无法获取图像。")
        return
    except Exception as e:
        print(f"获取输入数据时发生错误: {e}")
        return

    for cnn_layer_idx, cnn_layer_module in enumerate(model.layers):
        if isinstance(cnn_layer_module, CNN_layer):
            for filter_idx_in_cnn, single_channel_filter_module in enumerate(cnn_layer_module.filters):
                if isinstance(single_channel_filter_module, SingleChannelFilter):
                    handle = single_channel_filter_module.register_forward_hook(
                        get_filter_internal_data_hook(
                            cnn_layer_idx,
                            filter_idx_in_cnn,
                            cnn_layer_module.in_channels
                        )
                    )
                    hook_handles.append(handle)
                else:
                    print(f"警告: 在 CNN 层 {cnn_layer_idx} 中找到非 SingleChannelFilter 模块: {type(single_channel_filter_module)}")

    if not hook_handles:
        print("警告: 没有为任何 SingleChannelFilter 注册钩子。请检查模型结构。")
        return

    with torch.no_grad():
        final_model_output = model(single_input_image)

    for handle in hook_handles:
        handle.remove()

    complete_data_to_save = {
        'input_image_to_model': single_input_image.cpu().numpy(),
        'final_model_output': final_model_output.cpu().numpy(),
        'filter_internals': collected_filter_internals
    }

    save_path = os.path.join(save_dir, 'collected_filter_simulation_data_with_position.npy')
    np.save(save_path, complete_data_to_save)
    print(f"收集到的滤波器仿真数据（包含位置信息）已保存至: {save_path}")


def print_memory_stats():
    """
    Print current CUDA memory usage statistics
    
    Outputs:
        Allocated memory in MB
        Cached memory in MB
    """
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

def main():
    """
    Main function: Implements the training and testing pipeline on MNIST dataset
    """
    # Get command line arguments
    args = get_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set computation device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.empty_cache()
    
    # Define data preprocessing pipeline for training
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mapping [0, 1] to Normal distribution
    ])
    
    # Define data preprocessing pipeline for data collection (without normalization)
    collect_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=train_transform)
    collect_dataset = datasets.MNIST('../data', train=False, transform=collect_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False
    )
    collect_loader = DataLoader(
        collect_dataset,
        batch_size=1,
        shuffle=False
    )
    
    # Initialize model and move to device
    model = OpticalNetwork(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        output_size=args.output_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        mzi_repeat_num=args.mzi_repeat_num,
        mzi_row_num=args.mzi_row_num,
        mzi_column_num=args.mzi_column_num
    ).to(device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(project="optical-neural-network")
        wandb.config.update(args)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, scaler, 
              clip_value=args.grad_clip, log_interval=args.log_interval,args=args)
        test(model, device, test_loader)
        
        if use_cuda:
            print_memory_stats()
            torch.cuda.empty_cache()

    # 训练完成后，收集并保存滤波器数据（使用未归一化的数据）
    print("收集滤波器数据...")
    collect_and_save_filter_data(model, collect_loader, device)
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), "optical_network.pt")
    
    # Finish and close wandb logging
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
