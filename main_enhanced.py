import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from module.ONN_enhanced import OpticalNetworkEnhanced
from module.CNN_enhanced import CNN_layer_enhanced
from module.channel import SingleChannelFilter
from module.channel_power import SingleChannelFilterPower
from module.MZI_array.mzi_row_array import MZIlayer_row
from module.MZI_array.mzi_column_array import MZIlayer_column
from train import train, test
from torch.cuda.amp import GradScaler
from args_enhanced import get_args
import random
import numpy as np
import os


def collect_and_save_filter_data_enhanced(model, dataloader, device, save_dir="filter_data", filter_type='coherent'):
    """
    Enhanced version of data collection function that works with all filter types
    Collects and saves internal data from optical filters with proper handling for different detection modes
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    collected_filter_internals = {}
    hook_handles = []

    def get_filter_internal_data_hook_enhanced(layer_idx, filter_idx_in_cnn_layer, cnn_layer_in_channels, filter_type):
        def hook(module, input_args, output_tensor_overall_filter):
            input_for_intermediate_calc = input_args[0][0:1, :, :]

            _, mzi_array_input_patch, mzi_output_before_absorber_patch = module.forward(
                input_for_intermediate_calc, return_intermediate=True
            )

            with torch.no_grad():
                patches_for_weighting = mzi_output_before_absorber_patch.view(1, 1, -1)

                if filter_type == 'coherent':
                    # Coherent detection: complex amplitude processing
                    output_after_absorber_patch = patches_for_weighting * module.diagonal_matrix.view(1, 1, -1)
                elif filter_type == 'power':
                    # Power-domain detection: power superposition
                    power_patches = torch.abs(patches_for_weighting) ** 2
                    output_after_absorber_patch = power_patches * (module.diagonal_matrix.abs() ** 2).view(1, 1, -1)
                else:
                    output_after_absorber_patch = patches_for_weighting * module.diagonal_matrix.view(1, 1, -1)

            key = f'layer_{layer_idx}_filter_{filter_idx_in_cnn_layer}'
            current_filter_data = {}

            current_filter_data['mzi_array_input'] = mzi_array_input_patch.detach().cpu().numpy()
            current_filter_data['mzi_output_before_absorber'] = mzi_output_before_absorber_patch.detach().cpu().numpy()
            current_filter_data['output_after_absorber'] = output_after_absorber_patch.detach().cpu().numpy()
            current_filter_data['filter_type'] = filter_type

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
        print("Error: Dataloader is empty, cannot get image.")
        return

    # Get filter type from model
    model_filter_type = model.filter_type if hasattr(model, 'filter_type') else filter_type

    for cnn_layer_idx, cnn_layer_module in enumerate(model.layers):
        if isinstance(cnn_layer_module, CNN_layer_enhanced):
            for filter_idx_in_cnn, single_channel_filter_module in enumerate(cnn_layer_module.filters):
                if isinstance(single_channel_filter_module, (SingleChannelFilter, SingleChannelFilterPower)):
                    handle = single_channel_filter_module.register_forward_hook(
                        get_filter_internal_data_hook_enhanced(cnn_layer_idx, filter_idx_in_cnn, cnn_layer_module.in_channels, model_filter_type)
                    )
                    hook_handles.append(handle)

    if not hook_handles:
        print("Warning: No hooks registered for any SingleChannelFilter.")
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
        'filter_internals': collected_filter_internals,
        'model_filter_type': model_filter_type,
        'model_info': model.get_filter_info() if hasattr(model, 'get_filter_info') else {}
    }

    save_filename = f'hook_data_verified_{model_filter_type}.npy'
    save_path = os.path.join(save_dir, save_filename)
    np.save(save_path, complete_data_to_save)
    print(f"Enhanced hook data for {model_filter_type} mode saved to: {save_path}")


def print_memory_stats():
    """
    Print current CUDA memory usage statistics
    """
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")


def main():
    """
    Main function: Implements the training and testing pipeline on MNIST dataset with enhanced filter support
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

    # Create enhanced model
    # Supported filter types: 'coherent' (amplitude interference) or 'power' (power superposition)
    model = OpticalNetworkEnhanced(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        output_size=args.output_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        mzi_repeat_num=args.mzi_repeat_num,
        mzi_row_num=args.mzi_row_num,
        mzi_column_num=args.mzi_column_num,
        filter_type=args.filter_type
    ).to(device)

    # Print model configuration
    print(f"\n=== Enhanced Optical Network Configuration ===")
    print(f"Filter Type: {args.filter_type}")
    if hasattr(model, 'get_filter_info'):
        info = model.get_filter_info()
        print(f"Detection Principle: {info['detection_principle']}")
    print("=" * 50)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler()

    if args.wandb:
        wandb.init(project=f"optical-neural-network-{args.filter_type}")
        wandb.config.update(args)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, scaler,
              clip_value=args.grad_clip, log_interval=args.log_interval, args=args)
        test(model, device, test_loader)

        if device.type == 'cuda':
            print_memory_stats()
            torch.cuda.empty_cache()

    print(f"\nTraining completed. Starting data collection for {args.filter_type} mode...")
    collect_and_save_filter_data_enhanced(model, collect_loader, device, filter_type=args.filter_type)

    if args.save_model:
        model_filename = f"optical_network_{args.filter_type}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as: {model_filename}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()