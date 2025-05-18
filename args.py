# arg.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Optical Neural Network Configuration')
    
    # Network Architecture Parameters
    parser.add_argument('--input-size', type=int, default=14,
                        help='Input image size (default: 14)')
    parser.add_argument('--input-channels', type=int, default=1,
                        help='Number of input channels (default: 1)')
    parser.add_argument('--hidden-channels', type=int, default=16,
                        help='Number of hidden channels (default: 16)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of network layers (default: 4)')
    parser.add_argument('--kernel-size', type=int, default=3,
                        help='Convolution kernel size (default: 3)')
    parser.add_argument('--output-size', type=int, default=10,
                        help='Number of output classes (default: 10)')

    # MZI Layer Parameters
    parser.add_argument('--mzi-repeat-num', type=int, default=5,
                        help='Number of MZI repetitions in SingleChannelFilter (default: 5)')
    parser.add_argument('--mzi-row-num', type=int, default=5,
                        help='Number of MZIs per row (default: 5)')
    parser.add_argument('--mzi-column-num', type=int, default=4,
                        help='Number of MZIs per column (default: 4)')

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Training batch size (default: 200)')
    parser.add_argument('--test-batch-size', type=int, default=200,
                        help='Testing batch size (default: 200)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping threshold (default: 10.0)')

    # Other Parameters
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='Training log output interval (default: 5)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Save the trained model')
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Enable wandb logging for training process')
    
    args = parser.parse_args()
    return args
