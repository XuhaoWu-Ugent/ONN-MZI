# Neural Network with MZI Parameters

This project implements a neural network training framework with customizable MZI (Mach-Zehnder Interferometer) parameters and various architectural configurations.

## Requirements

- Python 3.8+
- PyTorch
- CUDA (optional but recommended)
- Weights & Biases (wandb)
- numpy
- Other dependencies (list in requirements.txt)

## Installation

```bash
git clone [repository-url]
cd [project-directory]
pip install -r requirements.txt
```

## Usage

### Basic Command Line Execution

Run the training script with default parameters:
```bash
python main.py
```

Run with custom parameters:
```bash
python main.py --batch-size 512 --epochs 50 --lr 0.001
```

### Available Parameters

- Network Architecture:
  - `--input-size`: Input image size (default: 14)
  - `--input-channels`: Number of input channels (default: 1)
  - `--hidden-channels`: Number of hidden channels (default: 16)
  - `--num-layers`: Number of network layers (default: 3)
  - `--kernel-size`: Convolution kernel size (default: 3)
  - `--output-size`: Number of output classes (default: 10)

- MZI Configuration:
  - `--mzi-repeat-num`: MZI repeat number (default: 5)
  - `--mzi-row-num`: Number of MZI rows (default: 5)
  - `--mzi-column-num`: Number of MZI columns (default: 4)

- Training Parameters:
  - `--batch-size`: Training batch size (default: 1000)
  - `--test-batch-size`: Test batch size (default: 1000)
  - `--epochs`: Number of training epochs (default: 100)
  - `--lr`: Learning rate (default: 0.005)
  - `--grad-clip`: Gradient clipping threshold (default: 1.0)

- Other Options:
  - `--no-cuda`: Disable CUDA training
  - `--seed`: Random seed (default: 1)
  - `--log-interval`: Logging interval (default: 5)
  - `--save-model`: Save the trained model
  - `--wandb`: Enable Weights & Biases logging

### Batch Experiment Script

The project includes a shell script (`experiment.sh`) for running multiple experiments with different parameter combinations:

```bash
chmod +x experiment.sh
./experiment.sh
```

The script automatically:
- Creates experiment logs directory
- Tests various parameter combinations
- Saves individual experiment logs
- Generates experiment summary
- Monitors GPU status
- Shows progress information

## Directory Structure

```
project/
│
├── main.py              # Main training script
├── experiment.sh        # Batch experiment script
├── requirements.txt     # Project dependencies
├── models/             # Model definitions
├── utils/              # Utility functions
└── experiment_logs/    # Generated experiment logs
```

## Experiment Logging

Logs are saved in the `experiment_logs` directory with the following format:
- Individual experiment logs: `[timestamp]_[experiment_id].log`
- Experiment summary: `experiment_summary_[timestamp].txt`

## GPU Monitoring

The experiment script includes GPU monitoring capabilities:
- Checks GPU status before each experiment
- Records GPU utilization in experiment logs
- Includes cooling periods between experiments

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
