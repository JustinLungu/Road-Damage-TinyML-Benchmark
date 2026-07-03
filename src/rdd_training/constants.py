from src.constants import (
    EFFICIENTFORMER_MODEL_IDS,
    EFFICIENTNET_MODEL_CHECKPOINTS,
    INCEPTION_MODEL_CHECKPOINTS,
    MOBILENET_MODEL_CHECKPOINTS,
    MOBILEVIT_MODEL_IDS,
    RESNET_MODEL_CHECKPOINTS,
)


##########################
# Binary Pothole Manifests
##########################

# True regenerates datasets/rdd2022/binary_pothole/*.csv before the rest of main.
# Use this when SPLIT_COUNTRIES changed or the full-image manifests are missing.
RUN_RDD_FULL_IMAGE_PREPROCESSING = False
POTHOLE_LABEL = "D40"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
BINARY_CLASS_NAMES = ("non_pothole", "pothole")
NUM_BINARY_CLASSES = len(BINARY_CLASS_NAMES)
ID_TO_LABEL = dict(enumerate(BINARY_CLASS_NAMES))
LABEL_TO_ID = {label: index for index, label in ID_TO_LABEL.items()}

SPLIT_COUNTRIES = {
    "train": ("China_Drone", "China_MotorBike", "Czech", "India"),
    "validation": ("United_States",),
    "test": ("Japan", "Norway"),
}

MANIFEST_COLUMNS = [
    "image_path",
    "annotation_path",
    "country",
    "split",
    "label",
    "label_name",
    "has_pothole",
    "num_objects",
    "num_pothole_objects",
    "unique_object_labels",
    "object_labels",
    "image_width",
    "image_height",
]

SUMMARY_COLUMNS = [
    "split",
    "country",
    "images",
    "pothole_images",
    "non_pothole_images",
    "pothole_fraction",
    "objects",
    "pothole_objects",
]

PATCH_MANIFEST_COLUMNS = [
    "image_path",
    "annotation_path",
    "country",
    "split",
    "label",
    "label_name",
    "source_object_label",
    "patch_source",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "patch_xmin",
    "patch_ymin",
    "patch_xmax",
    "patch_ymax",
    "image_width",
    "image_height",
]

PATCH_SUMMARY_COLUMNS = [
    "split",
    "country",
    "patches",
    "pothole_patches",
    "non_pothole_patches",
    "pothole_fraction",
]

REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "annotation_path",
    "country",
    "split",
    "label",
    "label_name",
}


#################################
# Patch-Grid Experiment Settings
#################################

# True regenerates datasets/rdd2022/binary_pothole_patches/*.csv before training.
# Keep False after CSVs exist unless patch/country-split settings changed.
RUN_RDD_PATCH_PREPROCESSING = True
RDD_EXPERIMENT_NAME = "annotation_patch_grid3_smoke"

# Training modes:
# "full_image": train on datasets/rdd2022/binary_pothole/*.csv full images.
# "annotation_patch": train on annotation-centered patch crops from XML boxes.
RDD_TRAINING_INPUT_MODE = "annotation_patch"

# Evaluation modes:
# "full_image": evaluate each full test image directly.
# "grid_image": split each full image into RDD_GRID_SIZE x RDD_GRID_SIZE patches,
# score each patch, and use the max pothole score as the image score.
RDD_EVALUATION_INPUT_MODE = "grid_image"
RDD_SUPPORTED_TRAINING_INPUT_MODES = ("full_image", "annotation_patch")
RDD_SUPPORTED_EVALUATION_INPUT_MODES = ("full_image", "grid_image")

# Patch preprocessing controls for annotation-derived training patches.
RDD_PATCH_SIZE = 224
RDD_PATCH_PADDING = 0.15
RDD_PATCH_INCLUDE_DAMAGE_NEGATIVES = True
RDD_PATCH_INCLUDE_BACKGROUND_NEGATIVES = True
RDD_BACKGROUND_NEGATIVES_PER_IMAGE = 1
RDD_MIN_BOX_AREA = 400

# Grid-image evaluation controls. Overlap is reserved for later; only 0.0 is
# currently implemented.
RDD_GRID_SIZE = 3
RDD_GRID_OVERLAP = 0.0
RDD_PATCH_AGGREGATION = "max_threshold"
RDD_PATCH_DECISION_THRESHOLD = 0.5

# If True, validation full images are scored with grid inference and the
# threshold with best RDD_THRESHOLD_METRIC is saved to threshold.json.
RDD_TUNE_PATCH_THRESHOLD = True
RDD_THRESHOLD_METRIC = "f1"
RDD_THRESHOLD_VALUES = tuple(index / 100 for index in range(5, 96, 5))


######################
# Fine-Tuning Models
######################

RDD_IMAGE_CLASSIFICATION_MODELS = frozenset(
    {
        *MOBILENET_MODEL_CHECKPOINTS,
        *EFFICIENTNET_MODEL_CHECKPOINTS,
        *RESNET_MODEL_CHECKPOINTS,
        *INCEPTION_MODEL_CHECKPOINTS,
        *MOBILEVIT_MODEL_IDS,
        *EFFICIENTFORMER_MODEL_IDS,
    }
)

RDD_MODEL_MODE = "single"  # "all" or "single"
# Used when RDD_MODEL_MODE is "single" for both adaptation smoke and training.
RDD_SINGLE_MODEL = "mobilevit_xxs"
# Used when RDD_MODEL_MODE is "all" for both adaptation smoke and training.
# This is the run subset, not necessarily every model that the adapter supports.
RDD_MODEL_NAMES = (
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet18",
    "inception_v3",
    "mobilevit_xxs",
    "mobilevit_xs",
    "mobilevit_s",
    "efficientformer_l1",
    "efficientformer_l3",
    "efficientformer_l7",
)
# Available image-classification models for this list:
# "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
# "efficientnet_b0", "resnet18", "inception_v3",
# "mobilevit_xxs", "mobilevit_xs", "mobilevit_s",
# "efficientformer_l1", "efficientformer_l3", "efficientformer_l7".


################
# Training Loop
################

RUN_RDD_TRAINING = True  # skip training loop if False
RDD_TRAINING_BATCH_SIZE = 16
RDD_TRAINING_NUM_WORKERS = 2
RDD_TRAINING_EPOCHS = 1
RDD_TRAINING_LEARNING_RATE = 1e-4
RDD_TRAINING_WEIGHT_DECAY = 1e-4
RDD_TRAINING_BEST_METRIC = "f1"
RDD_TRAINING_USE_WEIGHTED_LOSS = True
RDD_TRAINING_PROGRESS_INTERVAL = 50
RDD_TRAINING_SAVE_PLOTS = True
RDD_TRAINING_EARLY_STOPPING_PATIENCE = 4
RDD_TRAINING_EARLY_STOPPING_MIN_DELTA = 1e-4
RDD_TRAINING_DROP_LAST_BATCH = True
RDD_TRAINING_SKIP_EXISTING_CHECKPOINTS = True
RDD_SKIP_FAILED_MODELS = True


################
# Evaluation
################

RUN_RDD_EVALUATION = True  # skip evaluation loop if False
RDD_EVALUATION_BATCH_SIZE = 32
RDD_EVALUATION_NUM_WORKERS = 2
RDD_EVALUATION_PROGRESS_INTERVAL = 50
RDD_EVALUATION_SAVE_PLOTS = True


################
# Comparison
################

RUN_RDD_COMPARISON = True
RDD_COMPARISON_RANKING_METRIC = "f1"
RDD_COMPARISON_TOP_K = 3

DEFAULT_IMAGE_SIZE = 224
MODEL_IMAGE_SIZES = {
    "inception_v3": 299,
}
