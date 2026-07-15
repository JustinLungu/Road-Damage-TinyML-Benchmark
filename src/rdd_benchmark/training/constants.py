from src.constants import (
    CUSTOM_IMAGE_CLASSIFICATION_MODELS,
    EFFICIENTFORMER_MODEL_IDS,
    EFFICIENTNET_MODEL_CHECKPOINTS,
    INCEPTION_MODEL_CHECKPOINTS,
    MOBILENET_MODEL_CHECKPOINTS,
    MOBILEVIT_MODEL_IDS,
    RESNET_MODEL_CHECKPOINTS,
)


# Only image-classification models are valid for binary RDD fine-tuning.
RDD_IMAGE_CLASSIFICATION_MODELS = frozenset(
    {
        *MOBILENET_MODEL_CHECKPOINTS,
        *EFFICIENTNET_MODEL_CHECKPOINTS,
        *RESNET_MODEL_CHECKPOINTS,
        *INCEPTION_MODEL_CHECKPOINTS,
        *MOBILEVIT_MODEL_IDS,
        *EFFICIENTFORMER_MODEL_IDS,
        *CUSTOM_IMAGE_CLASSIFICATION_MODELS,
    }
)

# Default input size for torchvision/Hugging Face image classifiers.
DEFAULT_IMAGE_SIZE = 224

# Model-specific image sizes that differ from DEFAULT_IMAGE_SIZE.
MODEL_IMAGE_SIZES = {
    "inception_v3": 299,
}
