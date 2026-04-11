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
from module.calibration_loader import load_mzi_calibration
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from args import get_args
import random
import numpy as np
import os
import json

# Hardware electrical constraint: |V| on every MZI heater must stay
# below this value (volts). The chip cannot be driven beyond ~5V due
# to thermal-power limits, so trained voltages are clamped after every
# optimiser step to keep them deployable to the physical heaters.
VOLTAGE_CLAMP_V = float(os.environ.get("VOLTAGE_CLAMP_V", "5.0"))

# Per-processor thermal power budget (mW). Each voltage configuration
# loaded onto the 50-MZI chip must not exceed this total heater power,
# so the on-chip thermal equilibrium matches the conditions under which
# mzi_parameters_multi.json was calibrated (17-37 mW typical, <=50 mW
# acceptable). Set to 0 to disable the constraint.
POWER_BUDGET_MW = float(os.environ.get("POWER_BUDGET_MW", "50.0"))


def _constrain_power_per_processor(model, budget_mw):
    """Project voltages so each processor's total heater power <= budget.

    After the per-MZI voltage clamp, this function checks every
    SingleChannelFilter / OpticalSliceProcessor in the model. If the
    sum P = Σ V_i² / R_meas_i exceeds `budget_mw`, all voltages in
    that processor are scaled by sqrt(budget/P), preserving the
    relative voltage pattern while bringing total power within budget.
    """
    if budget_mw <= 0:
        return
    from module.channel import SingleChannelFilter
    from module.optical_linear_shared import OpticalSliceProcessor

    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, (SingleChannelFilter, OpticalSliceProcessor)):
                continue
            v_list = []
            r_list = []
            for layer in module.layers:
                if hasattr(layer, "MZI"):
                    for mzi in layer.MZI:
                        v_list.append(mzi._voltage)
                        r_list.append(mzi.nominal_resistance)
            if not v_list:
                continue
            p_total_mw = sum(
                (v.item() ** 2) / r * 1000.0 for v, r in zip(v_list, r_list)
            )
            if p_total_mw > budget_mw:
                scale = (budget_mw / p_total_mw) ** 0.5
                for v in v_list:
                    v.data.mul_(scale)


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

    def forward(self, x, return_features=False):
        # Step through the model to capture features before FC
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        features = self.model.avgpool(x)
        
        x = torch.flatten(features, 1)
        logits = self.model.fc(x)
        
        if return_features:
            return logits, features
        return logits

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
    
    # Train for 5 epochs
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # Downsample strategy for training
            data_small = F.interpolate(data, size=(14, 14), mode='bilinear', align_corners=False)
            data_upscaled = F.interpolate(data_small, size=(32, 32), mode='bilinear', align_corners=False)
            
            optimizer.zero_grad()
            output = model(data_upscaled)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            print(f"[Teacher] Epoch {epoch+1}/5 training completed.")

    # Evaluate Teacher Quality
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Use same downsample strategy for evaluation
            data_small = F.interpolate(data, size=(14, 14), mode='bilinear', align_corners=False)
            data_upscaled = F.interpolate(data_small, size=(32, 32), mode='bilinear', align_corners=False)
            
            output = model(data_upscaled)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    # Sync metrics
    if dist.is_initialized():
        c_tensor = torch.tensor([correct], device=device)
        t_tensor = torch.tensor([total], device=device)
        dist.all_reduce(c_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)
        correct = c_tensor.item()
        total = t_tensor.item()

    acc = 100. * correct / total
    if rank == 0:
        print(f"===================================================")
        print(f"[Teacher] Final Test Accuracy (on 14x14 blurry inputs): {acc:.2f}%")
        print(f"===================================================")

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
    distill_loss = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean')
    return distill_loss

# === 5. Main Distillation Function (Updated for Weight Noise) ===
def train_distill(student, teacher, device, train_loader, optimizer, epoch, scaler, args,
                  noise_transform, current_sigma, feature_adapter, scheduler=None, rank=0):
    student.train()
    feature_adapter.train() 
    teacher.eval()
    
    epoch_loss = 0.0
    epoch_correct = 0
    total_samples = 0
    
    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_feat = torch.nn.MSELoss() 
    
    TEMP = 4.0
    KD_ALPHA = float(os.environ.get('DISTILL_ALPHA', 0.5))
    KD_BETA = float(os.environ.get('DISTILL_BETA', 1.0))
    
    # === Weight Noise Config ===
    sigma_weight = args.weight_noise_sigma
    if rank == 0 and epoch == 1 and sigma_weight > 0:
        print(f"[Distill] Weight Noise Injection Enabled: sigma={sigma_weight}")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # --- Data Preparation ---
        with torch.no_grad():
            low_res_data = F.interpolate(data, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
            teacher_input = F.interpolate(low_res_data, size=(32, 32), mode='bilinear', align_corners=False)
            
            # Get Teacher Logits AND Features
            teacher_logits, teacher_feat = teacher(teacher_input, return_features=True)
            teacher_feat = torch.flatten(teacher_feat, 1)

        student_input = low_res_data.clone()
        if current_sigma > 0:
            student_input = student_input + torch.randn_like(student_input) * current_sigma
        
        optimizer.zero_grad()
        
        # === 1. Inject Voltage Noise (Physical MZI Channels) ===
        saved_params = {}
        if sigma_weight > 0:
            for name, param in student.named_parameters():
                if param.requires_grad and "_voltage" in name:
                    saved_params[name] = param.data.clone()
                    noise = torch.randn_like(param) * sigma_weight
                    param.data.add_(noise)
        # ======================================
        
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        with autocast("cuda" if use_cuda else "cpu"):
            # Get Student Logits AND Features
            student_logits, student_feat = student(student_input, return_features=True)
            student_feat = torch.flatten(student_feat, 1)

            # Align Features
            student_feat_adapted = feature_adapter(student_feat)
            
            # Calculate Losses
            loss_ce = criterion_ce(student_logits, target)
            loss_kd = distillation_loss(student_logits, teacher_logits, temperature=TEMP)
            loss_feat = criterion_feat(student_feat_adapted, teacher_feat)
            
            loss = (1.0 - KD_ALPHA) * loss_ce + KD_ALPHA * loss_kd + KD_BETA * loss_feat
            
        scaler.scale(loss).backward()
        
        # === 2. Restore Clean Weights (ADDED) ===
        if sigma_weight > 0:
            for name, param in student.named_parameters():
                if param.requires_grad and "_voltage" in name:
                    param.data.copy_(saved_params[name])
        # ========================================

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()

        # === Hardware constraint: clamp |V| <= 5V ===
        # The chip's heaters cannot be driven beyond ~5V (thermal limit).
        # Apply the clamp after every optimiser step so the trained
        # voltages can be deployed directly without saturation clipping.
        with torch.no_grad():
            for name, param in student.named_parameters():
                if param.requires_grad and "_voltage" in name:
                    param.data.clamp_(-VOLTAGE_CLAMP_V, VOLTAGE_CLAMP_V)

        # === Thermal power constraint ===
        # Each 50-MZI processor must not exceed POWER_BUDGET_MW total
        # heater power, so the chip operates within the thermal envelope
        # of the multi-MZI calibration.
        if POWER_BUDGET_MW > 0:
            student_model = student.module if hasattr(student, "module") else student
            _constrain_power_per_processor(student_model, POWER_BUDGET_MW)

        epoch_loss += loss.item() * len(data)
        pred = student_logits.argmax(dim=1, keepdim=True)
        epoch_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)

        if batch_idx % args.log_interval == 0 and rank == 0:
            acc = 100. * epoch_correct / total_samples
            # ... (log code) ...
            if batch_idx % 100 == 0:
                 print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\t'
                  f'Loss: {loss.item():.4f}\tAcc: {acc:.2f}%')

    return epoch_loss / total_samples, 100. * epoch_correct / total_samples

# === 6. Ensemble Test (Updated for Weight Noise) ===
def test_ensemble(model, device, test_loader, args, rank=0, num_repeats=5):
    model.eval()
    test_loss = 0
    correct = 0
    
    # Prepare resizing transform
    resize = transforms.Resize((args.input_size, args.input_size))
    
    # Weight Noise Config for Test
    sigma_weight = args.weight_noise_sigma
    original_weights = {}
    if sigma_weight > 0:
        for name, p in model.named_parameters():
             if p.requires_grad:
                original_weights[name] = p.data.clone()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Resize for ONN
            data_onn = F.interpolate(data, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
            
            output_sum = None
            for _ in range(num_repeats):
                # === Inject Voltage Noise for Ensemble (Physical MZI Channels) ===
                if sigma_weight > 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad and "_voltage" in name:
                            noise = torch.randn_like(param) * sigma_weight
                            param.data.copy_(original_weights[name] + noise)
                # ========================================================
                
                output = model(data_onn)
                if output_sum is None: output_sum = output
                else: output_sum += output

            # Restore clean weights for next batch (or end of test)
            if sigma_weight > 0:
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
        print(f'\nTest set (Ensemble {num_repeats}x): Avg loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def load_mzi_parameters_from_json(model, json_path):
    """Thin wrapper kept for backward compatibility."""
    load_mzi_calibration(model, json_path)

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
    
    # === Loaders ===
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
        fc_pos_only=args.fc_pos_only,
        input_phase_noise_sigma=args.input_phase_noise_sigma
    ).to(device)
    
    load_mzi_calibration(student, 'results/mzi_parameters_multi.json')

    # === 3. Feature Adapter ===
    feature_adapter = nn.Linear(student.feature_size, 512).to(device)
    
    if is_distributed:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        feature_adapter = DDP(feature_adapter, device_ids=[local_rank], output_device=local_rank)

    # === Param groups: voltage gets a higher LR ===
    # Under the multi-MZI fit's working point (R*P_pi ~= 78.5) the voltage
    # parameter's loss-landscape gradient is intrinsically ~5-12x smaller
    # than under the old single-MZI default; we compensate by giving
    # `_voltage` its own param group at a higher LR. Other parameters
    # (diagonal_matrix, bias, output_proj, feature_adapter) stay at args.lr.
    voltage_params, other_params = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if "_voltage" in n:
            voltage_params.append(p)
        else:
            other_params.append(p)
    other_params += [p for p in feature_adapter.parameters() if p.requires_grad]

    voltage_lr_mult = float(os.environ.get("VOLTAGE_LR_MULT", "5.0"))
    other_lr = args.lr
    voltage_lr = args.lr * voltage_lr_mult

    # Voltage is a physical control signal, not a statistical weight;
    # it has no L2-prior interpretation. Applying Adam's weight decay
    # to voltages was observed to pull every MZI towards V≈0 (87% of
    # voltages ended up in |V|<0.5 after epoch 1), which collapses the
    # photonic cascade into a near-identity transform and reduces the
    # whole network to an electronic classifier — the opposite of the
    # hardware-aware training we want. We therefore disable weight
    # decay on the voltage group while keeping it on the electronic
    # parameters.
    optimizer = torch.optim.Adam(
        [
            {"params": other_params,   "lr": other_lr,   "weight_decay": 1e-5},
            {"params": voltage_params, "lr": voltage_lr, "weight_decay": 0.0},
        ],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[other_lr, voltage_lr],
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=float(os.environ.get("ONECYCLE_PCT_START", "0.1")),
    )

    if rank == 0:
        print(f"[Optimizer] voltage params: {sum(p.numel() for p in voltage_params)} "
              f"(lr={voltage_lr:.4f}); other params: "
              f"{sum(p.numel() for p in other_params)} (lr={other_lr:.4f})")

    if rank == 0:
        print(f"[Optimizer] hardware voltage clamp: |V| <= {VOLTAGE_CLAMP_V} V")
        if POWER_BUDGET_MW > 0:
            print(f"[Optimizer] per-processor power budget: {POWER_BUDGET_MW:.1f} mW")
    scaler = GradScaler()
    best_test_accuracy = 0.0
    
    KD_BETA = float(os.environ.get('DISTILL_BETA', 1.0))

    if rank == 0:
        print(f"\n[Distillation] Starting Training. Teacher: ResNet18, Student: ONN (K={args.num_shared_weights})")
        print(f"[Distillation] Input Noise Sigma Target: {args.input_noise_sigma}")
        if args.weight_noise_sigma > 0:
            print(f"[Distillation] Weight Noise Sigma: {args.weight_noise_sigma}")

    # === Training Loop ===
    for epoch in range(1, args.epochs + 1):
        if is_distributed: train_sampler.set_epoch(epoch)
        
        # Annealing Input Noise (unchanged)
        if epoch <= 5: curr_sigma = 0.0
        elif epoch <= 15: curr_sigma = args.input_noise_sigma * ((epoch - 5) / 10.0)
        else: curr_sigma = args.input_noise_sigma
        
        if rank == 0:
            print(f"\n>>> Epoch {epoch} | Distilling with Input Noise: {curr_sigma:.4f}")

        # Train (now with weight noise support)
        train_distill(student, teacher, device, train_loader, optimizer, epoch, scaler, args,
                      test_noise_transform, curr_sigma, feature_adapter, scheduler, rank)
        
        # Test (now with weight noise support)
        test_loss, test_acc = test_ensemble(student, device, test_loader, args, rank, num_repeats=5)

        if rank == 0 and test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            model_to_save = student.module if is_distributed else student
            suffix = os.environ.get('SAVE_SUFFIX', '')
            save_name = f"distilled_shared_K{args.num_shared_weights}_ch{args.hidden_channels}{suffix}_best.pt"
            torch.save(model_to_save.state_dict(), save_name)
            print(f"[Best Student Updated] Acc: {test_acc:.2f}% (Saved to {save_name})")

    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
