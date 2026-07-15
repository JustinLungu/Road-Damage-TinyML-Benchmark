from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.load_model import load_model  # noqa: E402


MODEL_NAME = "tiny_cnn"
IMAGE_SIZE = 224
BATCH_SIZE = 1


def main() -> None:
    model = load_model(MODEL_NAME).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    fp32_size_mib = parameter_count * 4 / (1024 * 1024)

    dummy_input = torch.zeros(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"model: {MODEL_NAME}")
    print(f"type: {type(model).__name__}")
    print(f"parameters: {parameter_count:,}")
    print(f"fp32 parameter size: {fp32_size_mib:.2f} MiB")
    print(f"input shape: {tuple(dummy_input.shape)}")
    print(f"output logits shape: {tuple(logits.shape)}")
    print("note: logits are not meaningful until the model is fine-tuned.")


if __name__ == "__main__":
    main()
