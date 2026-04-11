import torch
import torch.nn.functional as F
import torch.cuda
import torch.distributed as dist
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
from train import train, test
from torch.cuda.amp import GradScaler
from args import get_args
import random
import numpy as np
import os
import json


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# === New Dynamic Noise Class for Annealing ===
class DynamicGaussianNoise(object):
    def __init__(self, mean=0., std=0.):
        self.std = std
        self.mean = mean
    def set_std(self, new_std):
        self.std = new_std
    def __call__(self, tensor):
        if self.std <= 0: return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# === New Ensemble Test Function (Distributed-aware) ===
def test_ensemble(model, device, test_loader, rank=0, num_repeats=5):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_sum = None
            for _ in range(num_repeats):
                output = model(data)
                if output_sum is None: output_sum = output
                else: output_sum += output
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
        print(f'\nTest set (Ensemble {num_repeats}x): Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


def load_mzi_parameters_from_json(model, json_path):
    """Thin wrapper kept for backward compatibility."""
    load_mzi_calibration(model, json_path)


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
    
    # === Initialize Dynamic Noise for Annealing ===
    noise_transform = DynamicGaussianNoise(0., 0.) # Starts at 0
    
    # 优化数据加载
    transform_list = [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        noise_transform
    ]
    
    # Test set always uses the target noise to monitor robustness improvement
    test_noise_transform = DynamicGaussianNoise(0., args.input_noise_sigma)
    test_transform_list = [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        test_noise_transform
    ]
    
    train_transform = transforms.Compose(transform_list)
    test_transform = transforms.Compose(test_transform_list)
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transform)

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
    target_sigma = args.input_noise_sigma
    if rank == 0:
        print(f"\n[Starting Optimized Shared Training with K={args.num_shared_weights}]")
        print(f"[Distributed Training] World Size: {world_size}, Rank: {rank}")
        print(f"[Noise Annealing] Target Input Noise Sigma: {target_sigma}")

    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # === Noise Annealing Strategy ===
        if epoch <= 5:
            curr_sigma = 0.0
        elif epoch <= 15:
            curr_sigma = target_sigma * ((epoch - 5) / 10.0)
        else:
            curr_sigma = target_sigma
        noise_transform.set_std(curr_sigma)
        
        if rank == 0:
            print(f"\n>>> Epoch {epoch} | Train Noise Sigma: {curr_sigma:.4f}")

        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, scaler,
                                      clip_value=args.grad_clip, log_interval=args.log_interval,
                                      args=args, scheduler=scheduler, rank=rank)
        
        # Upgrade to Ensemble Test (5x passes) to improve accuracy under noise
        test_loss, test_acc = test_ensemble(model, device, test_loader, rank=rank, num_repeats=5)

        # Only save model from rank 0
        if rank == 0:
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                if args.save_model:
                    model_to_save = model.module if is_distributed else model
                    torch.save(model_to_save.state_dict(), f"optimized_shared_K{args.num_shared_weights}_best.pt")
                    print(f"[Best Model Updated] Acc: {test_acc:.2f}%")

    if rank == 0 and args.wandb:
        wandb.finish()

    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
