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
    arch_group.add_argument('--hidden-channels', type=int, default=4,
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
    mzi_group.add_argument('--fc-mzi-repeat-num', type=int, default=None,
                           help='FC-only MZI layer repetitions (default: same as --mzi-repeat-num). '
                                'Used to scale FC mesh independently of CNN mesh, '
                                'e.g. for the r=20+ mesh-size scaling outlook experiment.')
    mzi_group.add_argument('--fc-mzi-row-num', type=int, default=None,
                           help='FC-only MZIs per row (default: same as --mzi-row-num). '
                                'FC mesh port count r = 2*this. To scale to r=20 set 10, r=40 set 20.')
    mzi_group.add_argument('--fc-mzi-column-num', type=int, default=None,
                           help='FC-only MZIs per column (default: same as --mzi-column-num). '
                                'Must equal fc-mzi-row-num - 1 for port consistency.')
    mzi_group.add_argument('--lossless-mzi', action='store_true', default=True,
                           help='Disable MZI insertion loss during training/inference (default: lossless)')
    mzi_group.add_argument('--lossy-mzi', dest='lossless_mzi', action='store_false',
                           help='Enable calibrated MZI insertion loss')

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

    # Optical Fully Connected Layer
    fc_group = parser.add_argument_group('Optical Fully Connected Layer')
    fc_group.add_argument('--use-optical-fc', action='store_true', default=True,
                          help='Use OpticalSharedLinear instead of nn.Linear for final layer')
    fc_group.add_argument('--no-optical-fc', dest='use_optical_fc', action='store_false',
                          help='Use electronic nn.Linear for final layer (for comparison)')
    fc_group.add_argument('--fc-activation-mode', type=str, default='linear',
                          choices=['linear', 'nonlinear'],
                          help='Optical FC accumulation mode: '
                               'linear (pure accumulation) or '
                               'nonlinear (activation after each slice)')
    fc_group.add_argument('--num-shared-weights', type=int, default=None,
                           help='Number of shared weight matrices (K) for the Optical FC layer. '
                                'If not set, uses full independent weights (N).')
    fc_group.add_argument('--fc-pos-only', action='store_true', default=False,
                           help='Optical FC仅使用正路径(禁用差分负路径)用于定位/调试')

    # Training Parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch-size', type=int, default=200,
                            help='Training batch size')
    train_group.add_argument('--test-batch-size', type=int, default=200,
                            help='Testing batch size')
    train_group.add_argument('--epochs', type=int, default=7,
                            help='Number of training epochs')
    train_group.add_argument('--val-split', type=int, default=0,
                            help='Hold out this many MNIST training images as a validation set '
                                 '(fixed split, seeded by --seed). When > 0, model selection '
                                 '(best checkpoint) uses validation accuracy and the test set is '
                                 'only reported. 0 = legacy behaviour (select on test set).')
    train_group.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate')
    train_group.add_argument('--grad-clip', type=float, default=10.0,
                            help='Gradient clipping threshold')
    train_group.add_argument('--early-stopping', action='store_true', default=False,
                            help='Enable early stopping')
    train_group.add_argument('--patience', type=int, default=5,
                            help='Early stopping patience (epochs without improvement)')
    train_group.add_argument('--weight-noise-sigma', type=float, default=0.0,
                            help='Standard deviation of Gaussian noise added to weights during training')
    train_group.add_argument('--input-noise-sigma', type=float, default=0.0,
                            help='Standard deviation of Gaussian noise added to input data during training')
    train_group.add_argument('--input-phase-noise-sigma', type=float, default=0.0,
                            help='Standard deviation of input phase noise in radians (simulates fiber-induced phase fluctuations)')

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
    fc_type = f"Optical LoRA ({args.fc_activation_mode})" if args.use_optical_fc else "Electronic (nn.Linear)"
    print(f"Fully Connected: {fc_type}")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}")
    print(f"MZI Config: {args.mzi_repeat_num} repeats, {args.mzi_row_num}x{args.mzi_column_num}")
    print("="*70 + "\n")

    return args
