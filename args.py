# args.py
# Unified configuration for Optical Neural Network

import argparse


def get_args():
    """
    Parse command line arguments for Optical Neural Network training

    The unified architecture uses a balanced depthwise-style design where
    all layers have the same number of filters (hidden_channels).

    Detection modes:
        - coherent: Complex amplitude interference (phase-preserving)
        - power: Power-domain linear superposition (incoherent)
    """
    parser = argparse.ArgumentParser(
        description='Unified Optical Neural Network Configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Network Architecture Parameters
    arch_group = parser.add_argument_group('Network Architecture')
    arch_group.add_argument('--input-size', type=int, default=14,
                            help='Input image size')
    arch_group.add_argument('--input-channels', type=int, default=1,
                            help='Number of input channels (1 for grayscale)')
    arch_group.add_argument('--hidden-channels', type=int, default=12,
                            help='Number of channels in all layers')
    arch_group.add_argument('--num-layers', type=int, default=1,
                            help='Number of CNN layers')
    arch_group.add_argument('--kernel-size', type=int, default=3,
                            help='Convolution kernel size')
    arch_group.add_argument('--output-size', type=int, default=10,
                            help='Number of output classes')

    # MZI Array Parameters
    mzi_group = parser.add_argument_group('MZI Array Configuration')
    mzi_group.add_argument('--mzi-repeat-num', type=int, default=5,
                           help='Number of MZI layer repetitions in each filter')
    mzi_group.add_argument('--mzi-row-num', type=int, default=5,
                           help='Number of MZIs per row in MZI array')
    mzi_group.add_argument('--mzi-column-num', type=int, default=4,
                           help='Number of MZIs per column in MZI array')

    # Detection Mode
    detection_group = parser.add_argument_group('Detection Mode')
    detection_group.add_argument('--detection-mode', type=str, default='power',
                                 choices=['coherent', 'power'],
                                 help='Detection mode: '
                                      'coherent (complex amplitude interference) or '
                                      'power (power-domain linear superposition)')
    # Backward compatibility alias
    detection_group.add_argument('--filter-type', type=str, dest='detection_mode',
                                 choices=['coherent', 'power'],
                                 help='DEPRECATED: Use --detection-mode instead')

    # Training Parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch-size', type=int, default=200,
                            help='Training batch size')
    train_group.add_argument('--test-batch-size', type=int, default=200,
                            help='Testing batch size')
    train_group.add_argument('--epochs', type=int, default=7,
                            help='Number of training epochs')
    train_group.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate')
    train_group.add_argument('--grad-clip', type=float, default=10.0,
                            help='Gradient clipping threshold')

    # System Parameters
    sys_group = parser.add_argument_group('System Configuration')
    sys_group.add_argument('--no-cuda', action='store_true', default=False,
                          help='Disable CUDA training')
    sys_group.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')
    sys_group.add_argument('--log-interval', type=int, default=5,
                          help='Training log output interval (batches)')

    # Output Parameters
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--save-model', action='store_true', default=True,
                             help='Save the trained model')
    output_group.add_argument('--wandb', action='store_true', default=False,
                             help='Enable Weights & Biases logging')

    # Chaotic Noise Parameters
    chaos_group = parser.add_argument_group('Chaotic Noise Configuration')
    chaos_group.add_argument('--chaotic-noise', action='store_true', default=False,
                            help='Enable chaotic noise injection from external signal')
    chaos_group.add_argument('--chaotic-noise-scale', type=float, default=1.0,
                            help='Scaling factor for chaotic noise')
    chaos_group.add_argument('--chaotic-data-path', type=str, 
                            default='Chaotic_feedback_OSC/TimeSeriesSignal.csv',
                            help='Path to the chaotic signal CSV file')
    chaos_group.add_argument('--chaotic-mode', type=str, default='replace',
                            choices=['replace', 'add'],
                            help="Mode of noise injection: 'replace' (source = noise) or 'add' (source = 1 + noise)")

    args = parser.parse_args()

    # Print configuration summary
    print("\n" + "="*70)
    print("Configuration Summary")
    print("="*70)
    print(f"Architecture: {args.input_channels} -> " +
          f"{' -> '.join([str(args.hidden_channels)]*args.num_layers)} -> {args.output_size}")
    print(f"Detection Mode: {args.detection_mode.upper()}")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}")
    print(f"MZI Config: {args.mzi_repeat_num} repeats, {args.mzi_row_num}x{args.mzi_column_num}")
    print("="*70 + "\n")

    return args
