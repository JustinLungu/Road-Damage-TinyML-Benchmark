import re


################
# VLM Inference
################

# SmolVLM is benchmarked with one fixed-prompt forward pass, not text generation.
SMOLVLM_PROMPT = "Describe the image briefly."


##################
# Unit Conversion
##################

BYTES_PER_MB = 1024 * 1024
MILLIWATTS_PER_WATT = 1000


####################
# Benchmark Defaults
####################

DEFAULT_SAMPLE_INTERVAL_S = 0.1
DEFAULT_WARMUP_RUNS = 5
PROGRESS_INTERVAL_IMAGES = 50
INFERENCE_BATCH_SIZE = 1
TIMING_SCOPE = "image_load_preprocess_inference"


#####################
# Jetson tegrastats
#####################

TEGRASTATS_COMMAND = "tegrastats"
TEGRASTATS_STOP_TIMEOUT_S = 2

# GR3D_FREQ is Jetson's GPU engine utilization field.
TEGRASTATS_GPU_UTILIZATION_PATTERN = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
# Jetson power field names differ by board/software version.
TEGRASTATS_POWER_PATTERN = re.compile(
    r"\b(?:POM_5V_IN|VDD_IN)\s+(\d+)(?:mW)?(?:/\d+(?:mW)?)?"
)
