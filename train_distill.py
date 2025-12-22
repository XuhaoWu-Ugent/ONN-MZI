import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models
import wandb
from module.ONN_shared import OpticalNetwork
from module.CNN import CNN_layer
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from args import get_args
import random
import numpy as np
import os
import json

# === 1. ResNet18 Teacher Wrapper (Modified for MNIST) ===
class ResNet18MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load standard ResNet18
        self.model = models.resnet18(weights=None) # We train from scratch for MNIST
        
        # Modify first layer: 1 channel input (instead of 3), kernel 3x3 (instead of 7x7 for ImageNet)
        # This helps with small images like MNIST/CIFAR
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove maxpool to preserve spatial dimensions for small images
        self.model.maxpool = nn.Identity()
        
        # Modify FC layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# === 2. Dynamic Noise Class ===
class DynamicGaussianNoise(object):
    def __init__(self, mean=0., std=0.):
        self.std = std
        self.mean = mean
    
    def set_std(self, new_std):
        self.std = new_std
        
    def __call__(self, tensor):
        if self.std <= 0: return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# === 3. Helper: Train the Teacher (if not exists) ===
def train_teacher_if_needed(device, train_loader, test_loader, rank=0):
    path = "teacher_resnet18_mnist.pt"
    model = ResNet18MNIST().to(device)
    
    if os.path.exists(path):
        if rank == 0:
            print(f"[Teacher] Loading pre-trained Teacher from {path}")
        # Map location is important for distributed
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    if rank == 0:
        print("[Teacher] No pre-trained teacher found. Training ResNet18 on MNIST...")
    
    # Simple training loop for Teacher
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 5 epochs (usually enough for ResNet18 on MNIST to hit 99%)
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # Resize to 32x32 for ResNet
            data = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            print(f"[Teacher] Epoch {epoch+1}/5 completed.")

    # Save
    if rank == 0:
        torch.save(model.state_dict(), path)
        print(f"[Teacher] Saved to {path}")
    
    # Sync processes to ensure file exists before others load
    if dist.is_initialized():
        dist.barrier()
        # Reload to ensure consistency
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
    model.eval()
    return model

# === 4. Distillation Loss ===
def distillation_loss(student_logits, teacher_logits, temperature=4.0, alpha=0.5):
    """
    alpha: weight for KD loss (0 means pure CE, 1 means pure KD)
    """
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    # Log softmax from student
    student_log_softmax = F.log_softmax(student_logits / temperature, dim=1)
    
    # KL Divergence
    # We REMOVE (temperature ** 2) scaling here to keep loss magnitude comparable to CE.
    # This is often more stable for hardware-constrained networks.
    distill_loss = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean')
    
    return distill_loss

# === 5. Main Distillation Function ===
def train_distill(student, teacher, device, train_loader, optimizer, epoch, scaler, args,
                  noise_transform, current_sigma, scheduler=None, rank=0):
    student.train()
    teacher.eval() # Teacher is always in eval mode
    
    epoch_loss = 0.0
    epoch_correct = 0
    total_samples = 0
    
    # Standard CE for Hard Labels
    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Distillation Hyperparams
    TEMP = 4.0
    # Read ALPHA from environment variable if available, else default to 0.5
    ALPHA = float(os.environ.get('DISTILL_ALPHA', 0.5))
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # --- Data Preparation ---
        # 1. Clean Data for Teacher (Resized to 32x32 for ResNet)
        with torch.no_grad():
            clean_input = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
            teacher_logits = teacher(clean_input)

        # 2. Noisy Data for Student (Resized to 14x14 for ONN + Noise)
        # Resize first
        student_input = F.interpolate(data, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
        # Add Noise (Manually apply transform since we need clean for teacher)
        if current_sigma > 0:
            student_input = student_input + torch.randn_like(student_input) * current_sigma
        
        optimizer.zero_grad()
        
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        with autocast("cuda" if use_cuda else "cpu"):
            student_logits = student(student_input)
            
            # Calculate Losses
            loss_ce = criterion_ce(student_logits, target)
            loss_kd = distillation_loss(student_logits, teacher_logits, temperature=TEMP)
            
            # Combined Loss
            loss = (1.0 - ALPHA) * loss_ce + ALPHA * loss_kd
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()

        epoch_loss += loss.item() * len(data)
        pred = student_logits.argmax(dim=1, keepdim=True)
        epoch_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)

        if batch_idx % args.log_interval == 0 and rank == 0:
            acc = 100. * epoch_correct / total_samples
            
            # Fix Distributed Logging: Show local dataset size (15000) instead of global (60000)
            if dist.is_initialized() and hasattr(train_loader, 'sampler'):
                dataset_size = len(train_loader.sampler)
            else:
                dataset_size = len(train_loader.dataset)
                
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{dataset_size} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f} (CE:{loss_ce.item():.2f}, KD:{loss_kd.item():.2f})\tAcc: {acc:.2f}%')

    return epoch_loss / total_samples, 100. * epoch_correct / total_samples

# === 6. Ensemble Test (Same as before) ===
def test_ensemble(model, device, test_loader, args, rank=0, num_repeats=5):
    model.eval()
    test_loss = 0
    correct = 0
    
    # Prepare resizing transform
    resize = transforms.Resize((args.input_size, args.input_size))
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Resize for ONN
            data_onn = F.interpolate(data, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
            
            output_sum = None
            for _ in range(num_repeats):
                output = model(data_onn)
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
        print(f'\nTest set (Ensemble {num_repeats}x): Avg loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

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
                # module might be ModuleList or Module
                if isinstance(module, nn.ModuleList):
                    for p in module: 
                        for ml in p.layers: load_mzi_layer_params(ml)
                elif hasattr(module, 'layers'):
                    for ml in module.layers: load_mzi_layer_params(ml)

def main():
    args = get_args()
    
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    os.environ["MZI_LOSSLESS"] = "1" if args.lossless_mzi else "0"
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # === Loaders (Raw 28x28 for Teacher, will resize in loop) ===
    # We use basic normalization here
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test set with target noise for monitoring
    test_noise_transform = DynamicGaussianNoise(0., args.input_noise_sigma)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        test_noise_transform
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transform)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # === 1. Setup Teacher ===
    teacher = train_teacher_if_needed(device, train_loader, test_loader, rank)
    
    # === 2. Setup Student (ONN) ===
    student = OpticalNetwork(
        input_channels=args.input_channels, hidden_channels=args.hidden_channels,
        output_size=args.output_size, num_layers=args.num_layers, kernel_size=args.kernel_size,
        input_size=args.input_size, mzi_repeat_num=args.mzi_repeat_num,
        mzi_row_num=args.mzi_row_num, mzi_column_num=args.mzi_column_num,
        detection_mode=args.detection_mode, use_optical_fc=args.use_optical_fc,
        fc_activation_mode=args.fc_activation_mode, num_shared_weights=args.num_shared_weights,
        fc_pos_only=args.fc_pos_only
    ).to(device)
    
    load_mzi_parameters_from_json(student, 'results/mzi_parameters.json')

    if is_distributed:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), 
        epochs=args.epochs, pct_start=0.2
    )
    scaler = GradScaler()
    best_test_accuracy = 0.0
    
    if rank == 0:
        print(f"\n[Distillation] Starting Training. Teacher: ResNet18, Student: ONN (K={args.num_shared_weights})")
        print(f"[Distillation] Input Noise Sigma Target: {args.input_noise_sigma}")

    # === Training Loop ===
    for epoch in range(1, args.epochs + 1):
        if is_distributed: train_sampler.set_epoch(epoch)
        
        # Annealing Sigma
        if epoch <= 5: curr_sigma = 0.0
        elif epoch <= 15: curr_sigma = args.input_noise_sigma * ((epoch - 5) / 10.0)
        else: curr_sigma = args.input_noise_sigma
        
        if rank == 0:
            print(f"\n>>> Epoch {epoch} | Distilling with Input Noise: {curr_sigma:.4f}")

        train_distill(student, teacher, device, train_loader, optimizer, epoch, scaler, args,
                      test_noise_transform, curr_sigma, scheduler, rank)
        
        # Test (using Ensemble)
        test_loss, test_acc = test_ensemble(student, device, test_loader, args, rank, num_repeats=5)

        if rank == 0 and test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            model_to_save = student.module if is_distributed else student
            
            # Allow custom suffix for parameter sweeps
            suffix = os.environ.get('SAVE_SUFFIX', '')
            save_name = f"distilled_shared_K{args.num_shared_weights}{suffix}_best.pt"
            
            torch.save(model_to_save.state_dict(), save_name)
            print(f"[Best Student Updated] Acc: {test_acc:.2f}% (Saved to {save_name})")

    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
