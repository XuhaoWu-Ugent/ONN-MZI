import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.ticker import MaxNLocator


def plot_training_curves(train_log_file='training_log.json', save_dir='training_plots'):
    """
    Plot training and testing curves from saved log file

    Args:
        train_log_file: Path to JSON file containing training logs
        save_dir: Directory to save plot figures
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load training log data
    if not os.path.exists(train_log_file):
        print(f"Error: Training log file '{train_log_file}' not found.")
        print("Please run training with logging enabled first.")
        return

    with open(train_log_file, 'r') as f:
        log_data = json.load(f)

    # Extract training data
    train_epochs = log_data.get('train_epochs', [])
    train_losses = log_data.get('train_losses', [])
    train_accuracies = log_data.get('train_accuracies', [])

    # Extract test data
    test_epochs = log_data.get('test_epochs', [])
    test_losses = log_data.get('test_losses', [])
    test_accuracies = log_data.get('test_accuracies', [])

    if not train_epochs:
        print("No training data found in log file.")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curves
    ax1.plot(train_epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    if test_epochs and test_losses:
        ax1.plot(test_epochs, test_losses, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot 2: Accuracy curves
    ax2.plot(train_epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
    if test_epochs and test_accuracies:
        ax2.plot(test_epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    # Save combined figure
    combined_save_path = os.path.join(save_dir, 'training_curves_combined.png')
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined training curves to: {combined_save_path}")
    plt.close()

    # Create separate detailed plots

    # Detailed Loss Plot
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(train_epochs, train_losses, 'b-o', label='Training Loss',
                 linewidth=2, markersize=4, alpha=0.8)
    if test_epochs and test_losses:
        ax_loss.plot(test_epochs, test_losses, 'r-s', label='Test Loss',
                     linewidth=2, markersize=4, alpha=0.8)
    ax_loss.set_xlabel('Epoch', fontsize=13)
    ax_loss.set_ylabel('Loss', fontsize=13)
    ax_loss.set_title('Loss vs Epoch', fontsize=15, fontweight='bold')
    ax_loss.legend(fontsize=12)
    ax_loss.grid(True, alpha=0.3, linestyle='--')
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

    loss_save_path = os.path.join(save_dir, 'training_loss_detailed.png')
    plt.savefig(loss_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed loss plot to: {loss_save_path}")
    plt.close()

    # Detailed Accuracy Plot
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    ax_acc.plot(train_epochs, train_accuracies, 'b-o', label='Training Accuracy',
                linewidth=2, markersize=4, alpha=0.8)
    if test_epochs and test_accuracies:
        ax_acc.plot(test_epochs, test_accuracies, 'r-s', label='Test Accuracy',
                    linewidth=2, markersize=4, alpha=0.8)
    ax_acc.set_xlabel('Epoch', fontsize=13)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=13)
    ax_acc.set_title('Accuracy vs Epoch', fontsize=15, fontweight='bold')
    ax_acc.legend(fontsize=12)
    ax_acc.grid(True, alpha=0.3, linestyle='--')
    ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))

    acc_save_path = os.path.join(save_dir, 'training_accuracy_detailed.png')
    plt.savefig(acc_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed accuracy plot to: {acc_save_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("Training Summary Statistics")
    print("="*60)
    print(f"Total Epochs: {len(train_epochs)}")
    print(f"\nTraining Loss:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final: {train_losses[-1]:.4f}")
    print(f"  Best: {min(train_losses):.4f}")
    print(f"\nTraining Accuracy:")
    print(f"  Initial: {train_accuracies[0]:.2f}%")
    print(f"  Final: {train_accuracies[-1]:.2f}%")
    print(f"  Best: {max(train_accuracies):.2f}%")

    if test_epochs and test_losses and test_accuracies:
        print(f"\nTest Loss:")
        print(f"  Initial: {test_losses[0]:.4f}")
        print(f"  Final: {test_losses[-1]:.4f}")
        print(f"  Best: {min(test_losses):.4f}")
        print(f"\nTest Accuracy:")
        print(f"  Initial: {test_accuracies[0]:.2f}%")
        print(f"  Final: {test_accuracies[-1]:.2f}%")
        print(f"  Best: {max(test_accuracies):.2f}%")
    print("="*60)


def plot_from_numpy(train_losses_file='training_losses.npy',
                   train_accuracies_file='training_accuracies.npy',
                   test_losses_file='test_losses.npy',
                   test_accuracies_file='test_accuracies.npy',
                   save_dir='training_plots'):
    """
    Plot training curves from individual numpy files

    Args:
        train_losses_file: Path to numpy file with training losses
        train_accuracies_file: Path to numpy file with training accuracies
        test_losses_file: Path to numpy file with test losses
        test_accuracies_file: Path to numpy file with test accuracies
        save_dir: Directory to save plot figures
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load data from numpy files
    try:
        train_losses = np.load(train_losses_file).tolist()
        train_accuracies = np.load(train_accuracies_file).tolist()
        train_epochs = list(range(1, len(train_losses) + 1))
    except FileNotFoundError as e:
        print(f"Error loading training data: {e}")
        return

    # Try to load test data (optional)
    try:
        test_losses = np.load(test_losses_file).tolist()
        test_accuracies = np.load(test_accuracies_file).tolist()
        test_epochs = list(range(1, len(test_losses) + 1))
    except FileNotFoundError:
        print("Test data not found, plotting training data only.")
        test_losses = []
        test_accuracies = []
        test_epochs = []

    # Create log_data dictionary and use the main plotting function
    log_data = {
        'train_epochs': train_epochs,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_epochs': test_epochs,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }

    # Save as JSON
    json_path = 'training_log.json'
    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"Converted numpy data to JSON: {json_path}")

    # Plot using the main function
    plot_training_curves(json_path, save_dir)


if __name__ == "__main__":
    # Try to plot from JSON first
    if os.path.exists('training_log.json'):
        print("Found training_log.json, plotting from JSON file...")
        plot_training_curves('training_log.json', save_dir='training_plots')

    # If JSON doesn't exist, try numpy files
    elif os.path.exists('training_losses.npy'):
        print("Found numpy files, plotting from numpy files...")
        plot_from_numpy(save_dir='training_plots')

    else:
        print("No training log files found.")
        print("Please run training with logging enabled first.")
        print("\nExpected files:")
        print("  - training_log.json (preferred)")
        print("  - or training_losses.npy + training_accuracies.npy")
