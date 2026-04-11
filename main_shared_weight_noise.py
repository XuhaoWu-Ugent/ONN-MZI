import torch
import torch.cuda
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import wandb
from module.ONN_shared import OpticalNetwork
from module.CNN import CNN_layer
from module.channel import SingleChannelFilter
from module.MZI_array.mzi_row_array import MZIlayer_row
from module.MZI_array.mzi_column_array import MZIlayer_column
from module.chaotic_source import chaotic_source
from module.calibration_loader import load_mzi_calibration
from train import test  # Keep test, but we will redefine train
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from args import get_args
import random
import numpy as np
import os
import json


from torch.amp import autocast

def load_mzi_parameters_from_json(model, json_path):
    """Thin wrapper kept for backward compatibility."""
    load_mzi_calibration(model, json_path)

# === New Training Function with Weight Noise Injection ===
def train_with_weight_noise(model, device, train_loader, optimizer, epoch, scaler, args,
                            clip_value=1.0, log_interval=5, scheduler=None, rank=0):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    sigma = args.weight_noise_sigma
    if rank == 0 and epoch == 1 and sigma > 0:
        print(f"[Training] Weight Noise Regularization Enabled: sigma={sigma}")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 1. Inject Noise (Forward)
        saved_params = {}
        if sigma > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    saved_params[name] = param.data.clone()
                    noise = torch.randn_like(param) * sigma
                    param.data.add_(noise)
        
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        with autocast("cuda" if use_cuda else "cpu"):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        
        # 2. Restore Weights (Backward)
        if sigma > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(saved_params[name])

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()

        epoch_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        epoch_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)

        if batch_idx % log_interval == 0 and rank == 0:
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = 100. * correct / len(data)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}\tAcc: {acc:.2f}%')

    return epoch_loss / total_samples, 100. * epoch_correct / total_samples

# === New Robustness Test Function with Dynamic Noise & Ensemble ===
def test_robustness(model, device, test_loader, args, rank=0, num_repeats=5):
    model.eval()
    test_loss = 0
    correct = 0
    sigma = args.weight_noise_sigma
    
    # Cache clean weights
    original_weights = {name: p.data.clone() for name, p in model.named_parameters()}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_sum = None
            
            # Ensemble Loop
            for _ in range(num_repeats):
                # Inject Dynamic Noise for THIS inference pass
                if sigma > 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            noise = torch.randn_like(param) * sigma
                            param.data.copy_(original_weights[name] + noise)
                
                output = model(data)
                if output_sum is None: output_sum = output
                else: output_sum += output
            
            # Restore clean weights for next batch
            if sigma > 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(original_weights[name])

            output_avg = output_sum / num_repeats
            test_loss += F.cross_entropy(output_avg, target, reduction='sum').item()
            pred = output_avg.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    if dist.is_initialized():
        l_tensor = torch.tensor([test_loss], device=device)
        c_tensor = torch.tensor([correct], device=device)
        dist.all_reduce(l_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(c_tensor, op=dist.ReduceOp.SUM)
        test_loss, correct = l_tensor.item(), int(c_tensor.item())
        
    total_samples = len(test_loader.dataset)
    test_loss /= total_samples
    accuracy = 100. * correct / total_samples

    if rank == 0:
        print(f'\n[Robust Test] Sigma={sigma} | Ensemble={num_repeats}x')
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')

    return test_loss, accuracy



def train_with_weight_noise(model, device, train_loader, optimizer, epoch, scaler, args,
                            clip_value=1.0, log_interval=5, scheduler=None, rank=0):
    """
    Train the model for one epoch with Weight Noise Injection
    """
    model.train()

    epoch_loss = 0.0
    epoch_correct = 0
    total_samples = 0
    
    # Label Smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    sigma = args.weight_noise_sigma
    if rank == 0 and epoch == 1 and sigma > 0:
        print(f"[Weight Noise] Enabled with sigma={sigma}")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # --- Weight Noise Injection ---
        saved_params = {}
        if sigma > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Save original parameter
                    saved_params[name] = param.data.clone()
                    # Add Gaussian noise
                    noise = torch.randn_like(param) * sigma
                    param.data.add_(noise)
        # ------------------------------

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        with autocast("cuda" if use_cuda else "cpu"):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        
        # --- Restore Original Weights ---
        if sigma > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(saved_params[name])
        # --------------------------------

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        
        scaler.step(optimizer)
        scaler.update()

        # Step the scheduler every batch if provided
        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        epoch_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        epoch_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)

        # Log training progress
        if batch_idx % log_interval == 0 and rank == 0:
            # Calculate batch accuracy
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(data)
            loss_value = loss.item()

            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx * len(data),
                    "loss": loss_value,
                    "accuracy": accuracy,
                    "progress": 100. * batch_idx / len(train_loader)
                })

            if dist.is_initialized() and hasattr(train_loader, 'sampler'):
                dataset_size = len(train_loader.sampler)
            else:
                dataset_size = len(train_loader.dataset)

            print(f'Train Epoch: {epoch} ' 
                  f'[{batch_idx * len(data)}/{dataset_size} ' 
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t' 
                  f'Loss: {loss_value:.6f}\t' 
                  f'Accuracy: {accuracy:.2f}%')

    avg_loss = epoch_loss / total_samples
    avg_accuracy = 100. * epoch_correct / total_samples

    return avg_loss, avg_accuracy


def main():
    args = get_args()

    # Initialize distributed training
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    # Toggle MZI insertion loss (default: lossless per args.py)
    os.environ["MZI_LOSSLESS"] = "1" if args.lossless_mzi else "0"

    # Set seeds for reproducibility
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)
    
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=train_transform)

    # Use DistributedSampler for distributed training
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler,
                                 num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
    
    model = OpticalNetwork(
        input_channels=args.input_channels, hidden_channels=args.hidden_channels,
        output_size=args.output_size, num_layers=args.num_layers, kernel_size=args.kernel_size,
        input_size=args.input_size, mzi_repeat_num=args.mzi_repeat_num,
        mzi_row_num=args.mzi_row_num, mzi_column_num=args.mzi_column_num,
        detection_mode=args.detection_mode, use_optical_fc=args.use_optical_fc,
        fc_activation_mode=args.fc_activation_mode, num_shared_weights=args.num_shared_weights,
        fc_pos_only=args.fc_pos_only
    ).to(device)

    load_mzi_calibration(model, 'results/mzi_parameters_multi.json')

    # Wrap model with DDP for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 优化学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), 
        epochs=args.epochs, pct_start=0.2
    )
    
    scaler = GradScaler()
    best_test_accuracy = 0.0

    # Only print from rank 0
    if rank == 0:
        print(f"\n[Starting Shared Training K={args.num_shared_weights}]")
        print(f"[Distributed Training] World Size: {world_size}, Rank: {rank}")
        if args.weight_noise_sigma > 0:
            print(f"[Weight Noise] Enabled with sigma={args.weight_noise_sigma}")

    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # 1. Train with Noise Regularization (Static Noise during Forward)
        train_loss, train_acc = train_with_weight_noise(
            model, device, train_loader, optimizer, epoch, scaler,
            clip_value=args.grad_clip, log_interval=args.log_interval,
            args=args, scheduler=scheduler, rank=rank
        )
        
        # 2. Test with Robustness Simulation (Dynamic Noise per Inference + Ensemble)
        test_loss, test_acc = test_robustness(
            model, device, test_loader, args, rank=rank, num_repeats=5
        )

        # Only save model from rank 0
        if rank == 0:
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                if args.save_model:
                    model_to_save = model.module if is_distributed else model
                    torch.save(model_to_save.state_dict(), f"robust_shared_K{args.num_shared_weights}_best.pt")
                    print(f"[Best Model Updated] Acc: {test_acc:.2f}%")

    if rank == 0 and args.wandb:
        wandb.finish()

    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
