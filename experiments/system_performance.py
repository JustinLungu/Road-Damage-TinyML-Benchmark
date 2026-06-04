import argparse
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.load_model import (  # noqa: E402
    MODEL_LOADERS,
    get_downloaded_model_names,
    load_model,
)
from src.performance_benchmark import (  # noqa: E402
    PerformanceBenchmark,
    append_result_csv,
    list_coco_images,
    resolve_device,
)


COCO_IMAGES_DIR = PROJECT_ROOT / "datasets" / "coco" / "images"
RESULTS_CSV = (
    PROJECT_ROOT / "results" / "system_metrics" / "system_performance_results.csv"
)
ALL_LOADED = "all-loaded"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one or more pretrained models on COCO validation images."
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
    args = parser.parse_args()

    model_names = resolve_model_names(args.model, parser)
    device = resolve_device(args.device)

    if len(model_names) > 1:
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

        downloaded_models = get_downloaded_model_names()
        if not downloaded_models:
            parser.error("No downloaded model checkpoints were found under models/.")
        return downloaded_models

    return list(dict.fromkeys(selected_models))


def run_model_processes(
    model_names: list[str],
    device_name: str,
    num_images: int | None,
) -> None:
    print(f"Selected models: {', '.join(model_names)}", flush=True)
    failed_models = []

    for model_name in model_names:
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
    benchmark = PerformanceBenchmark(model_name, model, image_paths, device)
    result = benchmark.run()
    append_result_csv(result, RESULTS_CSV)

    print(f"Results appended to: {RESULTS_CSV}")
    for metric_name, metric_value in vars(result).items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
