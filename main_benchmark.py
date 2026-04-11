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
    # Toggle insertion loss via CLI flag (defaults to lossless)
    os.environ["MZI_LOSSLESS"] = "1" if args.lossless_mzi else "0"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # 优化点 1 & 2: 数据加载增强
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=train_transform)
    
    # 使用 num_workers 和 pin_memory 加速
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
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
        detection_mode=args.detection_mode,
        use_optical_fc=args.use_optical_fc,
        fc_activation_mode=args.fc_activation_mode
    ).to(device)

    load_mzi_calibration(model, 'results/mzi_parameters_multi.json')

    # 优化点 3: 引入学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # OneCycleLR: 初期快速 Warmup，后期余弦退火平滑收敛
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs,
        pct_start=0.2
    )
    
    scaler = GradScaler()
    
    if args.wandb:
        wandb.init(project="optical-neural-network-optimized")
        wandb.config.update(args)

    best_test_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        # 训练并传入 scheduler (用于 per-batch step)
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, scaler,
                                      clip_value=args.grad_clip, log_interval=args.log_interval, 
                                      args=args, scheduler=scheduler)

        test_loss, test_acc = test(model, device, test_loader)

        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            if args.save_model:
                torch.save(model.state_dict(), f"optimized_benchmark_best.pt")
                print(f"\n[Best Model Updated] Acc: {test_acc:.2f}%")

        # 优化点 4: 删除了每个 Epoch 结尾的 empty_cache() 以提升速度

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
