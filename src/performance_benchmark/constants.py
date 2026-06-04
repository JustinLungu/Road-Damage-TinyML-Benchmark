import re


SMOLVLM_PROMPT = "Describe the image briefly."

BYTES_PER_MB = 1024 * 1024
MILLIWATTS_PER_WATT = 1000

DEFAULT_SAMPLE_INTERVAL_S = 0.1
DEFAULT_WARMUP_RUNS = 5
PROGRESS_INTERVAL_IMAGES = 50
TEGRASTATS_COMMAND = "tegrastats"
TEGRASTATS_STOP_TIMEOUT_S = 2

TEGRASTATS_GPU_UTILIZATION_PATTERN = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
TEGRASTATS_POWER_PATTERN = re.compile(
    r"\b(?:POM_5V_IN|VDD_IN)\s+(\d+)(?:mW)?(?:/\d+(?:mW)?)?"
)

YOLO_MODELS = {"yolov5nu", "yolov8n"}
MOBILENET_MODELS = {"mobilenet_v3_small", "mobilenet_v3_large"}
MOBILEVIT_MODELS = {
    "mobilevit_xxs": "apple/mobilevit-xx-small",
    "mobilevit_xs": "apple/mobilevit-x-small",
    "mobilevit_s": "apple/mobilevit-small",
}
EFFICIENTFORMER_MODELS = {
    "efficientformer_l1",
    "efficientformer_l3",
    "efficientformer_l7",
}
SMOLVLM_MODELS = {
    "smolvlm_256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm_2b": "HuggingFaceTB/SmolVLM-Instruct",
}
