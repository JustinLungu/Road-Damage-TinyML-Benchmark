##########################
# Binary Pothole Manifests
##########################

POTHOLE_LABEL = "D40"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

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
