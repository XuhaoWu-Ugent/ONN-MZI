import torch
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
    
    # 优化数据加载
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
        print(f"\n[Starting Optimized Shared Training with K={args.num_shared_weights}]")
        print(f"[Distributed Training] World Size: {world_size}, Rank: {rank}")

    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler
        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, scaler,
                                      clip_value=args.grad_clip, log_interval=args.log_interval,
                                      args=args, scheduler=scheduler, rank=rank)
        test_loss, test_acc = test(model, device, test_loader, rank=rank)

        # Only save model from rank 0
        if rank == 0:
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                if args.save_model:
                    model_to_save = model.module if is_distributed else model
                    save_suffix = os.environ.get('SAVE_SUFFIX', '')
                    torch.save(model_to_save.state_dict(), f"optimized_shared_K{args.num_shared_weights}_ch{args.hidden_channels}{save_suffix}_best.pt")
                    print(f"[Best Model Updated] Acc: {test_acc:.2f}%")

    if rank == 0 and args.wandb:
        wandb.finish()

    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
