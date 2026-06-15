from pathlib import Path

from src.constants import (
    EFFICIENTFORMER_MODEL_IDS,
    EFFICIENTNET_MODEL_CHECKPOINTS,
    INCEPTION_MODEL_CHECKPOINTS,
    MOBILENET_MODEL_CHECKPOINTS,
    MOBILEVIT_MODEL_IDS,
    RESNET_MODEL_CHECKPOINTS,
    SMOLVLM_MODEL_IDS,
    SYSTEM_PERFORMANCE_RESULTS_CSV,
    YOLO_MODEL_CHECKPOINTS,
)


###############
# CLI Keywords
###############

# Special --model value that expands to every checkpoint already present locally.
ALL_LOADED = "all-loaded"

# Supported explicitly, but skipped by all-loaded because it can exceed local GPU RAM.
ALL_LOADED_EXCLUDED_MODELS = {"smolvlm_2b"}


###############
# Result Paths
###############

RESULTS_CSV: Path = SYSTEM_PERFORMANCE_RESULTS_CSV
DEFAULT_CSV: Path = RESULTS_CSV


################
# Plot Folders
################

IMAGE_CLASSIFICATION_DIR = "image_classification"
OBJECT_DETECTION_DIR = "object_detection"
SEMANTIC_INTERPRETATION_DIR = "semantic_interpretation"
ALL_MODELS_DIR = "all_models"


#################
# Plot Model Groups
#################

IMAGE_CLASSIFICATION_MODELS = frozenset(
    {
        *MOBILENET_MODEL_CHECKPOINTS,
        *EFFICIENTNET_MODEL_CHECKPOINTS,
        *RESNET_MODEL_CHECKPOINTS,
        *INCEPTION_MODEL_CHECKPOINTS,
        *MOBILEVIT_MODEL_IDS,
        *EFFICIENTFORMER_MODEL_IDS,
    }
)
OBJECT_DETECTION_MODELS = frozenset(YOLO_MODEL_CHECKPOINTS)
SEMANTIC_INTERPRETATION_MODELS = frozenset(SMOLVLM_MODEL_IDS)

TASK_MODEL_GROUPS = (
    (IMAGE_CLASSIFICATION_DIR, IMAGE_CLASSIFICATION_MODELS),
    (OBJECT_DETECTION_DIR, OBJECT_DETECTION_MODELS),
    (SEMANTIC_INTERPRETATION_DIR, SEMANTIC_INTERPRETATION_MODELS),
)
