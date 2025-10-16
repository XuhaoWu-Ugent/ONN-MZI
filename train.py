import torch
import torch.nn.functional as F
import wandb
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import json
import os


def test(model, device, test_loader):
    """
    Evaluate model performance on test dataset

    Args:
        model: Neural network model
        device: Computing device (CPU/GPU)
        test_loader: DataLoader for test dataset

    Returns:
        tuple: (test_loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Calculate loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # Print results
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy

def train(model, device, train_loader, optimizer, epoch, scaler, args, clip_value=1.0, log_interval=5):
    """
    Train the model for one epoch

    Args:
        model: Neural network model
        device: Computing device (CPU/GPU)
        train_loader: DataLoader for training data
        optimizer: Optimization algorithm
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision training
        clip_value: Maximum gradient norm for clipping
        log_interval: Training log output interval

    Returns:
        tuple: (average_loss, average_accuracy) for the epoch
    """
    model.train()

    # Track epoch metrics
    epoch_loss = 0.0
    epoch_correct = 0
    total_samples = 0
    
    # Initialize timing statistics (currently disabled)
    #timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}
    #start = torch.cuda.Event(enable_timing=True)
    #alloc_end = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)
    #start.record()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Check for NaN values
        if torch.isnan(data).any():
            print("NaN in input data")
            
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        with autocast("cuda" if use_cuda else "cpu"):
            output = model(data)
            loss = F.cross_entropy(output, target)
            
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Print gradient information for debugging
        print(model.layers[0].filters[0].layers[0].MZI[1].raw_sin_theta.grad)
        
        # Debug code for parameter gradients (currently disabled)
        '''for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Parameter {name} has no gradient")
            else:
                print(f"Parameter {name} has gradient with norm: {param.grad.norm()}")'''
        
        # Gradient unscaling and clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Accumulate epoch metrics
        epoch_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        epoch_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)

        # Log training progress every 5 batches
        if batch_idx % log_interval == 0:
            # Calculate batch accuracy
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(data)

            # Get loss value
            loss_value = loss.item()

            # Log metrics to wandb
            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx * len(data),
                    "loss": loss_value,
                    "accuracy": accuracy,
                    "progress": 100. * batch_idx / len(train_loader)
                })

            # Print training progress
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss_value:.6f}\t'
                  f'Accuracy: {accuracy:.2f}%')

            # Timing statistics code (currently disabled)
            #child_timing_stats=model.get_all_time()
            #print("child_time: ",child_timing_stats)
            #end.record()
            #timing_stats['total'] = start.elapsed_time(end)
            #print("sum_time: ",timing_stats['total'])

            #print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
            #      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_value:.6f}')

    # Calculate epoch averages
    avg_loss = epoch_loss / total_samples
    avg_accuracy = 100. * epoch_correct / total_samples

    return avg_loss, avg_accuracy
