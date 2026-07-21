# True regenerates datasets/rdd2022/binary_pothole/*.csv before the rest of main.
# Keep False for normal runs after the manifests have been prepared.
RUN_RDD_FULL_IMAGE_PREPROCESSING = False

# RDD uses D40 for potholes. Other damage labels are treated as non-pothole.
POTHOLE_LABEL = "D40"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
BINARY_CLASS_NAMES = ("non_pothole", "pothole")
NUM_BINARY_CLASSES = len(BINARY_CLASS_NAMES)
ID_TO_LABEL = dict(enumerate(BINARY_CLASS_NAMES))
LABEL_TO_ID = {label: index for index, label in ID_TO_LABEL.items()}

# Split modes:
# "stratified_by_country": every country contributes train/validation/test rows,
# keeping pothole/non-pothole proportions closer across splits. This is the
# recommended development split for fine-tuning and threshold tuning.
# "country_holdout": whole countries are assigned to one split, useful later as
# a harder cross-country generalization benchmark.
RDD_SPLIT_MODE = "stratified_by_country"


########################
# Experiment Selection
########################

# True regenerates datasets/rdd2022/binary_pothole_patches/*.csv before training.
# Keep False after CSVs exist unless patch/country-split settings changed.
RUN_RDD_PATCH_PREPROCESSING = False

# Default output folder used only when trainer/evaluator are instantiated
# directly. The main experiment runner saves under each A-E experiment name.
RDD_EXPERIMENT_NAME = "stratified_by_country"

# 30-epoch tiny/small run. A30 uses the same clean natural full-image setup as
# A, but saves to a separate results folder so the previous 10-epoch A run stays
# intact.
RDD_ACTIVE_EXPERIMENT_IDS = ("A30",)

# Required only when running Experiment E. Set this to the experiment folder name
# that E should use as its synthetic-data base, usually the better of C or D.
RDD_BEST_PREVIOUS_EXPERIMENT_NAME = None


####################
# Input/Eval Modes
####################

# Training modes:
# "full_image": train on datasets/rdd2022/binary_pothole/*.csv full images.
# "annotation_patch": train on annotation-centered patch crops from XML boxes.
RDD_TRAINING_INPUT_MODE = "full_image"

# Evaluation modes:
# "full_image": evaluate each full test image directly.
# "grid_image": split each full image into RDD_GRID_SIZE x RDD_GRID_SIZE patches,
# score each patch, and use the max pothole score as the image score.
RDD_EVALUATION_INPUT_MODE = "full_image"
RDD_SUPPORTED_TRAINING_INPUT_MODES = ("full_image", "annotation_patch")
RDD_SUPPORTED_EVALUATION_INPUT_MODES = ("full_image", "grid_image")


#######################
# Patch/Grid Settings
#######################

# Patch preprocessing controls for annotation-derived training patches.
RDD_PATCH_SIZE = 224

# Number of rows/columns for full-image grid inference. 3 means 9 patches/image.
RDD_GRID_SIZE = 3

# Fallback image-level decision threshold when threshold tuning is disabled.
RDD_PATCH_DECISION_THRESHOLD = 0.5

# If True, validation full images are scored with grid inference and the
# threshold with best RDD_THRESHOLD_METRIC is saved to threshold.json.
RDD_TUNE_PATCH_THRESHOLD = True
RDD_THRESHOLD_METRIC = "f1"
RDD_THRESHOLD_VALUES = tuple(index / 100 for index in range(5, 96, 5))


# "single" runs RDD_SINGLE_MODEL. "all" runs every model in RDD_MODEL_NAMES.
RDD_MODEL_MODE = "all"
# Used when RDD_MODEL_MODE is "single" for both adaptation smoke and training.
RDD_SINGLE_MODEL = "tiny_cnn"
# Used when RDD_MODEL_MODE is "all" for both adaptation smoke and training.
# Tiny/small sweep models: four custom/source-defined models plus Torchvision
# ShuffleNetV2 0.5x.
RDD_MODEL_NAMES = (
    "ds_cnn_small",
    "mobilenet_v1_025",
    "resnet8",
    "tiny_cnn",
    "shufflenet_v2_x0_5",
)
# Available image-classification models for this list:
# "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
# "efficientnet_b0", "resnet18", "shufflenet_v2_x0_5", "inception_v3",
# "mobilevit_xxs", "mobilevit_xs", "mobilevit_s",
# "efficientformer_l1", "efficientformer_l3", "efficientformer_l7",
# "tiny_cnn", "resnet8", "ds_cnn_small", "mobilenet_v1_025".


################
# Training Loop
################

RUN_RDD_TRAINING = True  # set True to fine-tune selected models
RDD_TRAINING_BATCH_SIZE = 32
RDD_TRAINING_NUM_WORKERS = 2
RDD_TRAINING_EPOCHS = 30
RDD_TRAINING_LEARNING_RATE = 1e-4
RDD_TRAINING_WEIGHT_DECAY = 1e-4

# Best checkpoint is selected using this validation metric.
RDD_TRAINING_BEST_METRIC = "f1"

# Compensates for pothole/non-pothole imbalance in the training loss.
RDD_TRAINING_USE_WEIGHTED_LOSS = True
RDD_TRAINING_PROGRESS_INTERVAL = 50
RDD_TRAINING_SAVE_PLOTS = True

# Stop when validation metric stops improving for this many epochs.
RDD_TRAINING_EARLY_STOPPING_PATIENCE = 8
RDD_TRAINING_EARLY_STOPPING_MIN_DELTA = 1e-4

# Avoids single-image tail batches causing batch-norm issues in some models.
RDD_TRAINING_DROP_LAST_BATCH = True

# Lets interrupted all-model runs continue without retraining completed models.
RDD_TRAINING_SKIP_EXISTING_CHECKPOINTS = True
RDD_SKIP_FAILED_MODELS = True


################
# Evaluation
################

RUN_RDD_EVALUATION = True  # set True to evaluate selected checkpoints
RDD_EVALUATION_BATCH_SIZE = 32
RDD_EVALUATION_NUM_WORKERS = 2
RDD_EVALUATION_PROGRESS_INTERVAL = 50
RDD_EVALUATION_SAVE_PLOTS = True

# Lets interrupted all-model runs continue without re-evaluating completed models.
RDD_EVALUATION_SKIP_EXISTING_RESULTS = True


################
# Comparison
################

RUN_RDD_COMPARISON = True

# Rank model_comparison.csv/top_models.csv by this metric.
RDD_COMPARISON_RANKING_METRIC = "f1"
RDD_COMPARISON_TOP_K = 3
