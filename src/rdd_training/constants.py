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

MODEL_ADAPTATION_SMOKE_TEST_MODE = "all" # "all" or "single"
MODEL_ADAPTATION_SMOKE_TEST_MODEL = "mobilevit_xxs"
