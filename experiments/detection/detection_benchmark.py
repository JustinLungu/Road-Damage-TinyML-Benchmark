from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

# Allow direct execution from the repository root.
REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from experiments.detection.constants import (  # noqa: E402
    ALL_LOADED,
    IMAGE_CLASSIFICATION_MODELS,
    OBJECT_DETECTION_MODELS,
    RESULTS_CSV,
    SUPPORTED_MODELS,
)
from src.constants import (  # noqa: E402
    COCO_IMAGES_DIR,
    COCO_INSTANCES_VAL_ANNOTATIONS,
    IMAGENETTE_VALIDATION_LABELS,
)
from src.detection_benchmark import (  # noqa: E402
    ClassificationBenchmark,
    ObjectDetectionBenchmark,
    append_result_csv,
    load_classification_manifest,
)
from src.detection_benchmark.constants import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
)
from src.load_model import get_downloaded_model_names, load_model  # noqa: E402
from src.performance_benchmark import resolve_device  # noqa: E402


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_names = resolve_model_names(args.model, parser)
    validate_task_arguments(model_names, args.classification_labels, parser)

    if args.overwrite:
        RESULTS_CSV.unlink(missing_ok=True)

    if len(model_names) > 1:
        run_model_processes(model_names, args)
        return

    run_model_benchmark(
        model_name=model_names[0],
        device=resolve_device(args.device),
        num_images=args.num_images,
        classification_labels=args.classification_labels,
        classification_dataset_name=args.classification_dataset_name,
        classification_split=args.classification_split,
        coco_images_dir=args.coco_images_dir,
        coco_annotations=args.coco_annotations,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YOLO object detection or ImageNet-style classification "
            "accuracy."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=sorted(SUPPORTED_MODELS) + [ALL_LOADED],
        help="One or more supported models, or all-loaded.",
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
        help="Optional positive limit applied to the selected dataset.",
    )
    parser.add_argument(
        "--classification-labels",
        type=Path,
        default=IMAGENETTE_VALIDATION_LABELS,
        help=(
            "CSV with image_path and zero-based class_id columns. Defaults to "
            "the Imagenette validation manifest."
        ),
    )
    parser.add_argument(
        "--classification-dataset-name",
        default="imagenette",
        help="Dataset name recorded for classification rows.",
    )
    parser.add_argument(
        "--classification-split",
        default="validation",
        help="Dataset split recorded for classification rows.",
    )
    parser.add_argument(
        "--coco-images-dir",
        type=Path,
        default=COCO_IMAGES_DIR,
        help="Directory containing COCO validation JPEG files.",
    )
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        default=COCO_INSTANCES_VAL_ANNOTATIONS,
        help="COCO instances annotation JSON.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Detection confidence threshold for precision/recall/F1/IoU.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="Detection IoU threshold for precision/recall/F1/IoU.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite the result CSV before writing this run.",
    )
    return parser


def resolve_model_names(
    selected_models: list[str],
    parser: argparse.ArgumentParser,
) -> list[str]:
    if ALL_LOADED in selected_models:
        if len(selected_models) != 1:
            parser.error(f"{ALL_LOADED} cannot be combined with explicit models.")

        downloaded_models = [
            model_name
            for model_name in get_downloaded_model_names()
            if model_name in SUPPORTED_MODELS
        ]
        if not downloaded_models:
            parser.error("No supported downloaded checkpoints were found.")
        return downloaded_models

    return list(dict.fromkeys(selected_models))


def validate_task_arguments(
    model_names: list[str],
    classification_labels: Path | None,
    parser: argparse.ArgumentParser,
) -> None:
    selected_classifiers = [
        name for name in model_names if name in IMAGE_CLASSIFICATION_MODELS
    ]
    if selected_classifiers and (
        classification_labels is None or not classification_labels.is_file()
    ):
        parser.error(
            f"{', '.join(selected_classifiers)} is supported, but "
            "image-classification metrics require labeled classification "
            f"images. The manifest was not found at {classification_labels}. "
            "Run ./scripts/download_imagenette_val.sh to prepare the default "
            "dataset, or provide --classification-labels <manifest.csv>."
        )


def run_model_processes(
    model_names: list[str],
    args: argparse.Namespace,
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
            args.device,
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--iou-threshold",
            str(args.iou_threshold),
            "--coco-images-dir",
            str(args.coco_images_dir),
            "--coco-annotations",
            str(args.coco_annotations),
            "--classification-dataset-name",
            args.classification_dataset_name,
            "--classification-split",
            args.classification_split,
        ]
        if args.num_images is not None:
            command.extend(["--num-images", str(args.num_images)])
        if args.classification_labels is not None:
            command.extend(
                ["--classification-labels", str(args.classification_labels)]
            )

        completed_process = subprocess.run(command, check=False)
        if completed_process.returncode != 0:
            failed_models.append(model_name)

    if failed_models:
        raise RuntimeError(f"Benchmarks failed for: {', '.join(failed_models)}")


def run_model_benchmark(
    model_name: str,
    device: torch.device,
    num_images: int | None,
    classification_labels: Path | None,
    classification_dataset_name: str,
    classification_split: str,
    coco_images_dir: Path,
    coco_annotations: Path,
    confidence_threshold: float,
    iou_threshold: float,
) -> None:
    print(f"\nLoading model: {model_name}")
    model = load_model(model_name)

    if model_name in OBJECT_DETECTION_MODELS:
        print("Evaluating object detection on COCO validation...")
        benchmark = ObjectDetectionBenchmark(
            model_name=model_name,
            model=model,
            images_dir=coco_images_dir,
            annotation_path=coco_annotations,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            num_images=num_images,
        )
    else:
        if classification_labels is None:
            raise ValueError(
                "classification_labels is required for image classifiers."
            )
        print(
            f"Evaluating image classification from {classification_labels}..."
        )
        samples = load_classification_manifest(
            classification_labels,
            num_images,
        )
        benchmark = ClassificationBenchmark(
            model_name=model_name,
            model=model,
            samples=samples,
            device=device,
            dataset_name=classification_dataset_name,
            split=classification_split,
        )

    result = benchmark.run()
    append_result_csv(result, RESULTS_CSV)

    print(f"Results appended to: {RESULTS_CSV}")
    for metric_name, metric_value in vars(result).items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
