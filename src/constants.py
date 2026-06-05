from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
COCO_DIR = DATASETS_DIR / "coco"
COCO_IMAGES_DIR = COCO_DIR / "images"

MODELS_DIR = PROJECT_ROOT / "models"
CNN_DIR = MODELS_DIR / "cnn"
VIT_DIR = MODELS_DIR / "vit"
VLM_DIR = MODELS_DIR / "vlm"
MODEL_STORAGE_DIRS = (CNN_DIR, VIT_DIR, VLM_DIR)

RESULTS_DIR = PROJECT_ROOT / "results"
SYSTEM_METRICS_RESULTS_DIR = RESULTS_DIR / "system_metrics"
SYSTEM_PERFORMANCE_RESULTS_CSV = (
    SYSTEM_METRICS_RESULTS_DIR / "system_performance_results.csv"
)

YOLO_MODEL_CHECKPOINTS = {
    "yolov5nu": "yolov5nu.pt",
    "yolov8n": "yolov8n.pt",
}

MOBILENET_MODEL_CHECKPOINTS = {
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
    "mobilenet_v3_large": "mobilenet_v3_large-5c1a4163.pth",
}

MOBILEVIT_MODEL_IDS = {
    "mobilevit_xxs": "apple/mobilevit-xx-small",
    "mobilevit_xs": "apple/mobilevit-x-small",
    "mobilevit_s": "apple/mobilevit-small",
}

EFFICIENTFORMER_MODEL_IDS = {
    "efficientformer_l1": "efficientformer_l1.snap_dist_in1k",
    "efficientformer_l3": "efficientformer_l3.snap_dist_in1k",
    "efficientformer_l7": "efficientformer_l7.snap_dist_in1k",
}

SMOLVLM_MODEL_IDS = {
    "smolvlm_256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm_2b": "HuggingFaceTB/SmolVLM-Instruct",
}

MODEL_CHECKPOINT_PATHS: dict[str, Path] = {
    **{
        model_name: CNN_DIR / checkpoint_name
        for model_name, checkpoint_name in YOLO_MODEL_CHECKPOINTS.items()
    },
    **{
        model_name: CNN_DIR / "hub" / "checkpoints" / checkpoint_name
        for model_name, checkpoint_name in MOBILENET_MODEL_CHECKPOINTS.items()
    },
}

MODEL_CACHE_DIRS: dict[str, Path] = {
    **{model_name: VIT_DIR / model_name for model_name in MOBILEVIT_MODEL_IDS},
    **{model_name: VIT_DIR / model_name for model_name in EFFICIENTFORMER_MODEL_IDS},
    **{model_name: VLM_DIR / model_name for model_name in SMOLVLM_MODEL_IDS},
}

CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.bin", "*.safetensors")
