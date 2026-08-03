# True regenerates the full-image train/validation/test CSVs before a run.
RUN_RDD_PREPROCESSING = False

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
# recommended development split for training and model selection.
# "country_holdout": whole countries are assigned to one split, useful later as
# a harder cross-country generalization benchmark.
RDD_SPLIT_MODE = "stratified_by_country"


########################
# Experiment Selection
########################

# Experiments differ only in sampling, augmentation, and train-set balancing.
RDD_ACTIVE_EXPERIMENT_IDS = ("A",)


# Put one model here for a smoke test or several models for a sweep.
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
RDD_TRAINING_BEST_METRIC = "balanced_accuracy"

RDD_TRAINING_PROGRESS_INTERVAL = 50

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

# Lets interrupted all-model runs continue without re-evaluating completed models.
RDD_EVALUATION_SKIP_EXISTING_RESULTS = True


################
# Comparison
################

RUN_RDD_COMPARISON = True

# Rank model_comparison.csv by this metric.
RDD_COMPARISON_RANKING_METRIC = "f1"
