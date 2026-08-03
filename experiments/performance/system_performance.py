import argparse
import subprocess
import sys
from pathlib import Path

import torch

# Allow this script to be run directly from the repo root with `uv run python ...`.
REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from src.constants import (  # noqa: E402
    COCO_IMAGES_DIR,
)
from experiments.performance.constants import (  # noqa: E402
    ALL_LOADED,
    ALL_LOADED_EXCLUDED_MODELS,
    PERFORMANCE_RESULT_KEYS,
    RESULTS_CSV,
)
from src.load_model import (  # noqa: E402
    MODEL_LOADERS,
    get_downloaded_model_names,
    load_model,
)
from src.performance_benchmark import (  # noqa: E402
    PerformanceBenchmark,
    list_coco_images,
    resolve_device,
)
from src.result_csv import save_result_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one or more loaded models on COCO validation images."
    )
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=list(MODEL_LOADERS.keys()) + [ALL_LOADED],
        help="One or more model names, or all-loaded for every downloaded model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, such as cpu, cuda:0, or auto.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of COCO validation images to use. Defaults to all images.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite the results CSV before writing rows for this run.",
    )
    args = parser.parse_args()

    model_names = resolve_model_names(args.model, parser)
    device = resolve_device(args.device)

    if args.overwrite:
        # Remove the old CSV once before this run starts writing new rows.
        RESULTS_CSV.unlink(missing_ok=True)

    if len(model_names) > 1:
        # Each model gets a fresh process so previous model memory is released.
        run_model_processes(model_names, args.device, args.num_images)
        return

    image_paths = list_coco_images(COCO_IMAGES_DIR, args.num_images)

    run_model_benchmark(model_names[0], image_paths, device)


def resolve_model_names(
    selected_models: list[str],
    parser: argparse.ArgumentParser,
) -> list[str]:
    if ALL_LOADED in selected_models:
        if len(selected_models) != 1:
            parser.error(f"{ALL_LOADED} cannot be combined with explicit model names.")

        # Select locally cached checkpoints and source-defined custom models.
        downloaded_models = [
            model_name
            for model_name in get_downloaded_model_names()
            if model_name not in ALL_LOADED_EXCLUDED_MODELS
        ]
        if not downloaded_models:
            parser.error("No downloaded model checkpoints were found under models/.")
        return downloaded_models

    # Drop duplicates while preserving the order given by the user.
    return list(dict.fromkeys(selected_models))


def run_model_processes(
    model_names: list[str],
    device_name: str,
    num_images: int | None,
) -> None:
    print(f"Selected models: {', '.join(model_names)}", flush=True)
    failed_models = []

    for model_name in model_names:
        # Re-run this script once per model, but with a single --model argument.
        # A fresh process gives cleaner RAM/GPU measurements because model state
        # and PyTorch/CUDA caches from previous models are released by the OS.
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            model_name,
            "--device",
            device_name,
        ]
        if num_images is not None:
            command.extend(["--num-images", str(num_images)])

        # check=False lets us continue and report all failed models together.
        completed_process = subprocess.run(command, check=False)
        if completed_process.returncode != 0:
            failed_models.append(model_name)

    if failed_models:
        raise RuntimeError(f"Benchmarks failed for: {', '.join(failed_models)}")


def run_model_benchmark(
    model_name: str,
    image_paths: list[Path],
    device: torch.device,
) -> None:
    print(f"\nLoading model: {model_name}")
    model = load_model(model_name)

    print(f"Benchmarking {len(image_paths)} images on {device}...")
    # PerformanceBenchmark returns one row worth of system metrics.
    benchmark = PerformanceBenchmark(model_name, model, image_paths, device)
    result = benchmark.run()
    save_result_csv(result, RESULTS_CSV, PERFORMANCE_RESULT_KEYS)

    print(f"Results saved to: {RESULTS_CSV}")
    for metric_name, metric_value in vars(result).items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
