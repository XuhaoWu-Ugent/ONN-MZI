import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from module.ONN import OpticalNetwork
from args import get_args


def plot_confusion_matrix(model, dataloader, device, save_dir="confusion_matrix_output",
                         dataset_name="MNIST", num_classes=10, save_format='svg'):
    """
    Generate and plot confusion matrix for the model predictions

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation dataset
        device: Computing device (cuda/cpu)
        save_dir: Directory to save confusion matrix
        dataset_name: Name of the dataset for title
        num_classes: Number of classes in the dataset
        save_format: Format to save ('svg', 'png', 'pdf')
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_predictions = []
    all_labels = []

    print(f"Generating confusion matrix for {dataset_name}...")

    # Collect all predictions and labels
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=False)

            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Calculate accuracy
    accuracy = 100.0 * np.sum(all_predictions == all_labels) / len(all_labels)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Calculate per-class accuracy
    per_class_accuracy = 100.0 * cm.diagonal() / cm.sum(axis=1)

    # Normalize confusion matrix (percentage)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # ===== Plot 1: Raw Count Confusion Matrix =====
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11, fontweight='bold')

    # Set ticks
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes))

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')

    plt.xlabel('Predicted Labels', fontsize=13, fontweight='bold')
    plt.ylabel('True Labels', fontsize=13, fontweight='bold')
    plt.title(f'{dataset_name} Confusion Matrix (Counts)\nOverall Accuracy: {accuracy:.2f}%',
              fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save raw count confusion matrix
    count_save_path = os.path.join(save_dir, f'confusion_matrix_counts.{save_format}')
    plt.savefig(count_save_path, format=save_format, bbox_inches='tight')
    print(f"Saved count confusion matrix: {count_save_path}")
    plt.close()

    # ===== Plot 2: Normalized Confusion Matrix (Percentage) =====
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=100)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=11, fontweight='bold')

    # Set ticks
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes))

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text_color = 'white' if cm_normalized[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_normalized[i, j]:.1f}', ha='center', va='center',
                   color=text_color, fontsize=10, fontweight='bold')

    plt.xlabel('Predicted Labels', fontsize=13, fontweight='bold')
    plt.ylabel('True Labels', fontsize=13, fontweight='bold')
    plt.title(f'{dataset_name} Confusion Matrix (Percentage)\nOverall Accuracy: {accuracy:.2f}%',
              fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save normalized confusion matrix
    norm_save_path = os.path.join(save_dir, f'confusion_matrix_normalized.{save_format}')
    plt.savefig(norm_save_path, format=save_format, bbox_inches='tight')
    print(f"Saved normalized confusion matrix: {norm_save_path}")
    plt.close()

    # ===== Plot 3: Per-Class Accuracy Bar Chart =====
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(num_classes), per_class_accuracy, color='steelblue', alpha=0.8, edgecolor='black')

    # Color bars based on accuracy
    for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
        if acc >= 90:
            bar.set_color('green')
        elif acc >= 80:
            bar.set_color('orange')
        else:
            bar.set_color('red')

        # Add value labels on top of bars
        plt.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy: {accuracy:.2f}%')
    plt.xlabel('Class Label', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title(f'{dataset_name} Per-Class Accuracy', fontsize=15, fontweight='bold')
    plt.xticks(range(num_classes))
    plt.ylim(0, 105)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save per-class accuracy
    acc_save_path = os.path.join(save_dir, f'per_class_accuracy.{save_format}')
    plt.savefig(acc_save_path, format=save_format, bbox_inches='tight')
    print(f"Saved per-class accuracy chart: {acc_save_path}")
    plt.close()

    # ===== Print Detailed Statistics =====
    print("\n" + "="*60)
    print("Confusion Matrix Statistics")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nPer-Class Accuracy:")
    for i in range(num_classes):
        print(f"  Class {i}: {per_class_accuracy[i]:6.2f}% ({cm[i, i]:5d}/{cm[i].sum():5d})")

    # Find most confused pairs
    print(f"\nMost Confused Class Pairs (Top 5):")
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                confusion_pairs.append((i, j, cm[i, j]))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for idx, (true_label, pred_label, count) in enumerate(confusion_pairs[:5], 1):
        print(f"  {idx}. True: {true_label}, Predicted as: {pred_label} -> {count} times "
              f"({100.0*count/cm[true_label].sum():.1f}% of class {true_label})")

    print("="*60)

    return cm, accuracy, per_class_accuracy


def main():
    """
    Main function to load model and generate confusion matrix
    """
    args = get_args()
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Initialize model
    model = OpticalNetwork(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        output_size=args.output_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        mzi_repeat_num=args.mzi_repeat_num,
        mzi_row_num=args.mzi_row_num,
        mzi_column_num=args.mzi_column_num
    ).to(device)

    # Load trained model weights
    model_path = "optical_network.pt"
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model file {model_path} not found. Using random initialization.")
        print("Please train the model first or specify the correct model path.")

    # Generate confusion matrix
    plot_confusion_matrix(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir="confusion_matrix_output",
        dataset_name="MNIST",
        num_classes=args.output_size,
        save_format='svg'  # Change to 'png' or 'pdf' if needed
    )


if __name__ == "__main__":
    main()
