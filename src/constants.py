from pathlib import Path


############
# Repo Paths
############

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
COCO_DIR = DATASETS_DIR / "coco"
COCO_IMAGES_DIR = COCO_DIR / "images"
COCO_INSTANCES_VAL_ANNOTATIONS = COCO_DIR / "annotations" / "instances_val2017.json"
IMAGENETTE_DIR = DATASETS_DIR / "imagenette"
IMAGENETTE_VALIDATION_LABELS = IMAGENETTE_DIR / "validation_labels.csv"
RDD2022_DIR = DATASETS_DIR / "rdd2022"
RDD2022_BINARY_POTHOLE_DIR = RDD2022_DIR / "binary_pothole"

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
DETECTION_METRICS_RESULTS_DIR = RESULTS_DIR / "detection_metrics"
DETECTION_BENCHMARK_RESULTS_CSV = (
    DETECTION_METRICS_RESULTS_DIR / "detection_benchmark_results.csv"
)
RDD_TRAINING_RESULTS_DIR = RESULTS_DIR / "rdd_trained_models"


######################
# Supported Model Names
######################

# YOLO models are stored as explicit local checkpoint files under models/cnn/.
YOLO_MODEL_CHECKPOINTS = {
    "yolov5nu": "yolov5nu.pt",
    "yolov8n": "yolov8n.pt",
}

# Torchvision downloads MobileNet weights under models/cnn/hub/checkpoints/.
MOBILENET_MODEL_CHECKPOINTS = {
    "mobilenet_v2": "mobilenet_v2-7ebf99e0.pth",
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
    "mobilenet_v3_large": "mobilenet_v3_large-5c1a4163.pth",
}

# Torchvision downloads EfficientNet weights under models/cnn/hub/checkpoints/.
EFFICIENTNET_MODEL_CHECKPOINTS = {
    "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
}

# Torchvision downloads ResNet weights under models/cnn/hub/checkpoints/.
RESNET_MODEL_CHECKPOINTS = {
    "resnet18": "resnet18-f37072fd.pth",
}

# Torchvision downloads Inception weights under models/cnn/hub/checkpoints/.
INCEPTION_MODEL_CHECKPOINTS = {
    "inception_v3": "inception_v3_google-0cc3c7bd.pth",
}

# Hugging Face model IDs used by transformers.
MOBILEVIT_MODEL_IDS = {
    "mobilevit_xxs": "apple/mobilevit-xx-small",
    "mobilevit_xs": "apple/mobilevit-x-small",
    "mobilevit_s": "apple/mobilevit-small",
}

# timm model IDs used by timm.create_model(..., pretrained=True).
EFFICIENTFORMER_MODEL_IDS = {
    "efficientformer_l1": "efficientformer_l1.snap_dist_in1k",
    "efficientformer_l3": "efficientformer_l3.snap_dist_in1k",
    "efficientformer_l7": "efficientformer_l7.snap_dist_in1k",
}

# Hugging Face model IDs for the VLM benchmark path.
SMOLVLM_MODEL_IDS = {
    "smolvlm_256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm_2b": "HuggingFaceTB/SmolVLM-Instruct",
}


###########################
# Download Detection Helpers
###########################

# Models with known single-file checkpoint locations.
MODEL_CHECKPOINT_PATHS: dict[str, Path] = {
    **{
        model_name: CNN_DIR / checkpoint_name
        for model_name, checkpoint_name in YOLO_MODEL_CHECKPOINTS.items()
    },
    **{
        model_name: CNN_DIR / "hub" / "checkpoints" / checkpoint_name
        for model_name, checkpoint_name in MOBILENET_MODEL_CHECKPOINTS.items()
    },
    **{
        model_name: CNN_DIR / "hub" / "checkpoints" / checkpoint_name
        for model_name, checkpoint_name in EFFICIENTNET_MODEL_CHECKPOINTS.items()
    },
    **{
        model_name: CNN_DIR / "hub" / "checkpoints" / checkpoint_name
        for model_name, checkpoint_name in RESNET_MODEL_CHECKPOINTS.items()
    },
    **{
        model_name: CNN_DIR / "hub" / "checkpoints" / checkpoint_name
        for model_name, checkpoint_name in INCEPTION_MODEL_CHECKPOINTS.items()
    },
}

# Models cached as directories, where the exact checkpoint filename can vary.
MODEL_CACHE_DIRS: dict[str, Path] = {
    **{model_name: VIT_DIR / model_name for model_name in MOBILEVIT_MODEL_IDS},
    **{model_name: VIT_DIR / model_name for model_name in EFFICIENTFORMER_MODEL_IDS},
    **{model_name: VLM_DIR / model_name for model_name in SMOLVLM_MODEL_IDS},
}

# File extensions that count as downloaded model weights.
CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.bin", "*.safetensors")
