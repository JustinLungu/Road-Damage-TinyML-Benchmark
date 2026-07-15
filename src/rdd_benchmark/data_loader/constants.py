# Columns written for full-image binary pothole manifests.
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

# Columns written for annotation-centered patch manifests.
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

# Minimal columns required by both full-image and patch dataset loaders.
REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "annotation_path",
    "country",
    "split",
    "label",
    "label_name",
}

# Dataloader sampling strategies used by the experiment registry.
RDD_SAMPLER_NONE = "none"
RDD_SAMPLER_WEIGHTED = "weighted_sampler"
RDD_SUPPORTED_SAMPLER_STRATEGIES = (
    RDD_SAMPLER_NONE,
    RDD_SAMPLER_WEIGHTED,
)
