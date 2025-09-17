import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from module.ONN import OpticalNetwork  
from train import train, test
from torch.cuda.amp import GradScaler
from args import get_args
import random
import numpy as np


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
    print("1111111111111111111111111111111111111111111111111111")

    # Get command line arguments
    args = get_args()
    print("1111111111111111111111111111111111111111111111111111")
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
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先随机裁剪
        transforms.RandomHorizontalFlip(),    # 随机水平翻转
        transforms.Resize((args.input_size, args.input_size)),  # 再调整大小
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    # 测试集数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  # 只调整大小
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    
    # # Load MNIST dataset
    # train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    # Load CIFAR10 dataset
    train_dataset = datasets.CIFAR10(
        '../data',
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        '../data',
        train=False,
        transform=test_transform
    )

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
    
    # Initialize model and move to device
    model = OpticalNetwork(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        output_size=args.output_size,
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
    print("1111111111111111111111111111111111111111111111111111")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, scaler, 
              clip_value=args.grad_clip, log_interval=args.log_interval,args=args)
        test(model, device, test_loader)
        print("1111111111111111111111111111111111111111111111111111")

        if use_cuda:
            print_memory_stats()
            torch.cuda.empty_cache()
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), "optical_network.pt")
    
    # Finish and close wandb logging
    if args.wandb:
        wandb.finish()

main()
