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

REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "annotation_path",
    "country",
    "split",
    "label",
    "label_name",
}


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

RDD_MODEL_MODE = "all"  # "all" or "single"
# Used when RDD_MODEL_MODE is "single" for both adaptation smoke and training.
RDD_SINGLE_MODEL = "mobilenet_v3_small"
# Used when RDD_MODEL_MODE is "all" for both adaptation smoke and training.
# This is the run subset, not necessarily every model that the adapter supports.
RDD_MODEL_NAMES = (
    "mobilenet_v3_small",
    "resnet18",
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
RDD_TRAINING_EPOCHS = 5
RDD_TRAINING_LEARNING_RATE = 1e-4
RDD_TRAINING_WEIGHT_DECAY = 1e-4
RDD_TRAINING_BEST_METRIC = "f1"
RDD_TRAINING_USE_WEIGHTED_LOSS = True
RDD_TRAINING_PROGRESS_INTERVAL = 50
RDD_TRAINING_SAVE_PLOTS = True


################
# Evaluation
################

RUN_RDD_EVALUATION = False  # skip evaluation loop if False
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
