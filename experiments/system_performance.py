import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.load_model import MODEL_LOADERS, load_model  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one pretrained model on COCO validation images."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_LOADERS.keys()),
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

    device = resolve_device(args.device)
    image_paths = list_coco_images(COCO_IMAGES_DIR, args.num_images)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Benchmarking {len(image_paths)} images on {device}...")
    benchmark = PerformanceBenchmark(args.model, model, image_paths, device)
    result = benchmark.run()
    append_result_csv(result, RESULTS_CSV)

    print(f"Results appended to: {RESULTS_CSV}")
    for metric_name, metric_value in vars(result).items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
