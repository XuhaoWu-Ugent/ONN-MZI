import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row


class FabricationDataset(Dataset):
    """
    Dataset wrapping the on-chip measurement logs.

    Each entry describes a single experiment where one input port is excited and
    exactly one MZI receives a non-zero heater voltage. The measurement records
    the optical power observed on a specific output port.
    """

    def __init__(self, data_dir: str) -> None:
        self.samples: List[dict] = []
        data_path = Path(data_dir)
        files = sorted(data_path.glob("MZI_array_output_in*.txt"))
        if not files:
            raise FileNotFoundError(
                f"No measurement files found in {data_path.resolve()}."
            )

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 6:
                        raise ValueError(
                            f"Unexpected line in {file_path.name}: {line.strip()}"
                        )
                    input_port, input_power, mzi_index, voltage, output_port, output_power = parts
                    self.samples.append(
                        {
                            "input_port": int(input_port),
                            "input_power": float(input_power),
                            "mzi_index": int(mzi_index),
                            "voltage": float(voltage),
                            "output_port": int(output_port),
                            "output_power": float(output_power),
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["input_port"], dtype=torch.long),
            torch.tensor(sample["input_power"], dtype=torch.float32),
            torch.tensor(sample["mzi_index"], dtype=torch.long),
            torch.tensor(sample["voltage"], dtype=torch.float32),
            torch.tensor(sample["output_port"], dtype=torch.long),
            torch.tensor(sample["output_power"], dtype=torch.float32),
        )


class MZIArray(nn.Module):
    """
    Full six-row by five-column MZI array operating on complex amplitudes.
    """

    def __init__(
        self,
        *,
        row_mzis: int = 5,
        column_mzis: int = 4,
        row_layers: int = 6,
    ) -> None:
        super().__init__()

        self.port_count = row_mzis * 2
        self.layers = nn.ModuleList()
        next_index = 0
        mzi_kwargs = {}
        for row_id in range(row_layers):
            row_layer = MZIlayer_row(
                num=row_mzis,
                start_index=next_index,
                mzi_kwargs=mzi_kwargs,
            )
            self.layers.append(row_layer)
            next_index += row_mzis

            # Insert a column layer between consecutive row layers.
            if row_id < row_layers - 1:
                column_layer = MZIlayer_column(
                    num=column_mzis,
                    start_index=next_index,
                    mzi_kwargs=mzi_kwargs,
                )
                self.layers.append(column_layer)
                next_index += column_mzis

        self.total_mzis = next_index

    def forward(
        self, input_powers: torch.Tensor, voltages: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate powers through the full array.

        Args:
            input_powers: Tensor with shape (batch, port_count).
            voltages: Tensor with shape (batch, total_mzis).
        """
        if input_powers.dim() != 2:
            raise ValueError("input_powers must have shape (batch, port_count).")
        if voltages.dim() != 2:
            raise ValueError("voltages must have shape (batch, total_mzis).")

        if input_powers.dtype.is_complex:
            state = input_powers
        else:
            state = torch.sqrt(torch.clamp(input_powers, min=0.0)).to(torch.complex64)
        for layer in self.layers:
            state = layer(state, voltages)
        return state

    def iter_mzis(self) -> Iterable:
        for layer in self.layers:
            if hasattr(layer, "mzis"):
                for mzi in layer.mzis:
                    yield mzi


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate fabrication errors of the MZI array using measured data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing MZI_array_output_in*.txt files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/mzi_parameters.json",
        help="Destination JSON file for the calibrated parameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the calibration on (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def ensure_output_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def train_model(args: argparse.Namespace) -> Tuple[MZIArray, FabricationDataset]:
    torch.manual_seed(args.seed)

    dataset = FabricationDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    device = torch.device(args.device)
    model = MZIArray().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    mse = nn.MSELoss()
    dataset_size = len(dataset)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            (
                input_ports,
                input_powers,
                mzi_indices,
                voltages,
                output_ports,
                output_powers,
            ) = batch
            input_ports = input_ports.to(device)
            input_powers = input_powers.to(device)
            mzi_indices = mzi_indices.to(device)
            voltages_batch = voltages.to(device)
            output_ports = output_ports.to(device)
            output_powers = output_powers.to(device)

            batch_size = input_ports.size(0)

            inputs = torch.zeros(
                batch_size, model.port_count, dtype=torch.float32, device=device
            )
            inputs.scatter_(
                1, input_ports.unsqueeze(1), input_powers.unsqueeze(1)
            )

            voltage_matrix = torch.zeros(
                batch_size, model.total_mzis, dtype=torch.float32, device=device
            )
            voltage_matrix.scatter_(
                1, mzi_indices.unsqueeze(1), voltages_batch.unsqueeze(1)
            )

            predictions = model(inputs, voltage_matrix)
            port_fields = predictions.gather(
                1, output_ports.unsqueeze(1)
            ).squeeze(1)
            predicted_power = port_fields.abs() ** 2

            loss = mse(predicted_power, output_powers)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        average_loss = total_loss / dataset_size
        print(f"Epoch {epoch:03d}: loss={average_loss:.6e}")

    return model, dataset


def export_parameters(model: MZIArray, output_file: str) -> None:
    ensure_output_directory(output_file)
    result = {}
    for mzi in model.iter_mzis():
        if mzi.index is None:
            continue
        params = mzi.physical_parameters()
        result[mzi.index] = {
            "a": float(params["a"].item()),
            "b": float(params["b"].item()),
            "delta_r": float(params["delta_r"].item()),
            "phi0": float(params["phi0"].item()),
        }

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Calibrated parameters written to {output_file}")

def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    model, _ = train_model(args)
    model.eval()
    export_parameters(model, args.output_file)

if __name__ == "__main__":
    main()
