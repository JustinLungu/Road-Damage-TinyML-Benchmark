# Countries available in the downloaded annotated RDD2022 train split.
RDD_AVAILABLE_COUNTRIES = (
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
)

# Split strategies implemented by prepare_binary_pothole.py.
RDD_SUPPORTED_SPLIT_MODES = ("stratified_by_country", "country_holdout")

# Used for stratified_by_country: each country is split into these ratios.
RDD_SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
# Keeps stratified split generation deterministic across runs.
RDD_SPLIT_RANDOM_SEED = 42

# Used only for country_holdout: whole countries are assigned to one split.
SPLIT_COUNTRIES = {
    "train": ("China_Drone", "China_MotorBike", "Czech", "India"),
    "validation": ("United_States",),
    "test": ("Japan", "Norway"),
}

# Columns written in binary_pothole/summary.csv.
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

# Skips very small annotation boxes that are likely too noisy for training.
RDD_MIN_BOX_AREA = 400

# Training augmentation strategies used by the experiment registry.
RDD_AUGMENTATION_NONE = "none"
RDD_AUGMENTATION_STANDARD = "standard"
RDD_AUGMENTATION_STRONG = "strong"
RDD_SUPPORTED_AUGMENTATION_STRATEGIES = (
    RDD_AUGMENTATION_NONE,
    RDD_AUGMENTATION_STANDARD,
    RDD_AUGMENTATION_STRONG,
)
