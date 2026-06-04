from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import psutil
import torch
from PIL import Image

from src.load_model import VIT_DIR, VLM_DIR


SMOLVLM_PROMPT = "Describe the image briefly."
BYTES_PER_MB = 1024 * 1024

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

InferenceFunction = Callable[[Path], None]


@dataclass
class BenchmarkResult:
    model_name: str
    fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cpu_ram_mb: float
    peak_cpu_ram_mb: float
    avg_gpu_ram_mb: float | None
    peak_gpu_ram_mb: float | None
    avg_gpu_utilization_pct: float | None
    avg_power_w: float | None
    energy_per_inference_j: float | None
    num_images: int


@dataclass
class MetricSamples:
    cpu_ram_mb: list[float] = field(default_factory=list)
    gpu_ram_mb: list[float] = field(default_factory=list)
    gpu_utilization_pct: list[float] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)


class NvmlMonitor:
    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        self.pynvml: Any | None = None
        self.handle: Any | None = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self.pynvml = None
            self.handle = None

    def read(self) -> tuple[float | None, float | None]:
        if self.pynvml is None or self.handle is None:
            return None, None

        gpu_utilization = None
        power_w = None

        try:
            gpu_utilization = float(
                self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            )
        except Exception:
            pass

        try:
            power_w = float(self.pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000
        except Exception:
            pass

        return gpu_utilization, power_w

    def stop(self) -> None:
        if self.pynvml is None:
            return

        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass


class TegrastatsMonitor:
    GPU_UTILIZATION_PATTERN = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
    POWER_PATTERN = re.compile(
        r"\b(?:POM_5V_IN|VDD_IN)\s+(\d+)(?:mW)?(?:/\d+(?:mW)?)?"
    )

    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.gpu_utilization: float | None = None
        self.power_w: float | None = None

    def start(self) -> None:
        try:
            self.process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.process = None
            return

        self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self.reader_thread.start()

    def _read_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return

        for line in self.process.stdout:
            gpu_match = self.GPU_UTILIZATION_PATTERN.search(line)
            power_match = self.POWER_PATTERN.search(line)

            with self.lock:
                if gpu_match:
                    self.gpu_utilization = float(gpu_match.group(1))
                if power_match:
                    self.power_w = float(power_match.group(1)) / 1000

    def read(self) -> tuple[float | None, float | None]:
        with self.lock:
            return self.gpu_utilization, self.power_w

    def stop(self) -> None:
        if self.process is None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2)


class SystemMetricsSampler:
    def __init__(self, device: torch.device, interval_s: float = 0.1) -> None:
        self.device = device
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.samples = MetricSamples()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        device_index = device.index if device.index is not None else 0
        if shutil.which("tegrastats"):
            self.hardware_monitor: NvmlMonitor | TegrastatsMonitor = TegrastatsMonitor(
                interval_ms=max(1, int(interval_s * 1000))
            )
        else:
            self.hardware_monitor = NvmlMonitor(device_index=device_index)

    def start(self) -> None:
        self.hardware_monitor.start()
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        cpu_ram_mb = self.process.memory_info().rss / BYTES_PER_MB
        self.samples.cpu_ram_mb.append(cpu_ram_mb)

        if self.device.type == "cuda":
            gpu_ram_mb = torch.cuda.memory_allocated(self.device) / BYTES_PER_MB
            self.samples.gpu_ram_mb.append(gpu_ram_mb)

        gpu_utilization, power_w = self.hardware_monitor.read()
        if gpu_utilization is not None:
            self.samples.gpu_utilization_pct.append(gpu_utilization)
        if power_w is not None:
            self.samples.power_w.append(power_w)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        self.hardware_monitor.stop()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")

    return device


def list_coco_images(images_dir: Path, num_images: int | None = None) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO images directory does not exist: {images_dir}")

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in: {images_dir}")

    if num_images is not None:
        if num_images <= 0:
            raise ValueError("num_images must be greater than zero.")
        image_paths = image_paths[:num_images]

    return image_paths


def create_inference_function(
    model_name: str,
    model: Any,
    device: torch.device,
) -> InferenceFunction:
    if model_name in YOLO_MODELS:
        return _create_yolo_inference(model, device)
    if model_name in MOBILENET_MODELS:
        return _create_mobilenet_inference(model_name, model, device)
    if model_name in MOBILEVIT_MODELS:
        return _create_mobilevit_inference(model_name, model, device)
    if model_name in EFFICIENTFORMER_MODELS:
        return _create_efficientformer_inference(model, device)
    if model_name in SMOLVLM_MODELS:
        return _create_smolvlm_inference(model_name, model, device)

    raise ValueError(f"No inference adapter is defined for model: {model_name}")


def _create_yolo_inference(model: Any, device: torch.device) -> InferenceFunction:
    def infer(image_path: Path) -> None:
        model.predict(source=str(image_path), device=str(device), verbose=False)

    return infer


def _create_mobilenet_inference(
    model_name: str,
    model: Any,
    device: torch.device,
) -> InferenceFunction:
    from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
    else:
        weights = MobileNet_V3_Large_Weights.DEFAULT

    transform = weights.transforms()
    model.to(device).eval()

    def infer(image_path: Path) -> None:
        image = _load_rgb_image(image_path)
        input_tensor = transform(image).unsqueeze(0).to(device)
        model(input_tensor)

    return infer


def _create_mobilevit_inference(
    model_name: str,
    model: Any,
    device: torch.device,
) -> InferenceFunction:
    from transformers import AutoImageProcessor

    model_id = MOBILEVIT_MODELS[model_name]
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        cache_dir=str(VIT_DIR / model_name),
        use_fast=False,
    )
    model.to(device).eval()

    def infer(image_path: Path) -> None:
        image = _load_rgb_image(image_path)
        inputs = processor(images=image, return_tensors="pt")
        model(**_move_inputs_to_device(inputs, device))

    return infer


def _create_efficientformer_inference(
    model: Any,
    device: torch.device,
) -> InferenceFunction:
    import timm

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    model.to(device).eval()

    def infer(image_path: Path) -> None:
        image = _load_rgb_image(image_path)
        input_tensor = transform(image).unsqueeze(0).to(device)
        model(input_tensor)

    return infer


def _create_smolvlm_inference(
    model_name: str,
    model: Any,
    device: torch.device,
) -> InferenceFunction:
    from transformers import AutoProcessor

    model_id = SMOLVLM_MODELS[model_name]
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(VLM_DIR / model_name),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SMOLVLM_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    model.to(device).eval()

    def infer(image_path: Path) -> None:
        image = _load_rgb_image(image_path)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        model(**_move_inputs_to_device(inputs, device))

    return infer


def _load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _move_inputs_to_device(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def run_benchmark(
    model_name: str,
    model: Any,
    image_paths: list[Path],
    device: torch.device,
    warmup_runs: int = 5,
) -> BenchmarkResult:
    if not image_paths:
        raise ValueError("At least one image is required for the benchmark.")

    infer = create_inference_function(model_name, model, device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            infer(image_paths[0])
            _synchronize_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sampler = SystemMetricsSampler(device)
    latencies_s: list[float] = []

    sampler.start()
    try:
        with torch.inference_mode():
            for image_path in image_paths:
                _synchronize_device(device)
                start_time = time.perf_counter()
                infer(image_path)
                _synchronize_device(device)
                latencies_s.append(time.perf_counter() - start_time)
    finally:
        sampler.stop()

    total_inference_time_s = sum(latencies_s)
    avg_latency_s = fmean(latencies_s)
    avg_power_w = _average_or_none(sampler.samples.power_w)

    if device.type == "cuda":
        peak_gpu_ram_mb = torch.cuda.max_memory_allocated(device) / BYTES_PER_MB
    else:
        peak_gpu_ram_mb = None

    return BenchmarkResult(
        model_name=model_name,
        fps=len(image_paths) / total_inference_time_s,
        avg_latency_ms=avg_latency_s * 1000,
        p95_latency_ms=_percentile(latencies_s, 95) * 1000,
        avg_cpu_ram_mb=fmean(sampler.samples.cpu_ram_mb),
        peak_cpu_ram_mb=max(sampler.samples.cpu_ram_mb),
        avg_gpu_ram_mb=_average_or_none(sampler.samples.gpu_ram_mb),
        peak_gpu_ram_mb=peak_gpu_ram_mb,
        avg_gpu_utilization_pct=_average_or_none(
            sampler.samples.gpu_utilization_pct
        ),
        avg_power_w=avg_power_w,
        energy_per_inference_j=(
            avg_power_w * avg_latency_s if avg_power_w is not None else None
        ),
        num_images=len(image_paths),
    )


def append_result_csv(result: BenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(asdict(result).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _average_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float:
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index

    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )
