import subprocess
import threading

from src.performance_benchmark.constants import (
    MILLIWATTS_PER_WATT,
    TEGRASTATS_COMMAND,
    TEGRASTATS_GPU_UTILIZATION_PATTERN,
    TEGRASTATS_POWER_PATTERN,
    TEGRASTATS_STOP_TIMEOUT_S,
)


class TegrastatsMonitor:
    """Read Jetson utilization and input power from a tegrastats process."""

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
                [TEGRASTATS_COMMAND, "--interval", str(self.interval_ms)],
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
            gpu_match = TEGRASTATS_GPU_UTILIZATION_PATTERN.search(line)
            power_match = TEGRASTATS_POWER_PATTERN.search(line)

            with self.lock:
                # Store the latest parsed values; sampler reads them periodically.
                if gpu_match:
                    self.gpu_utilization = float(gpu_match.group(1))
                if power_match:
                    self.power_w = float(power_match.group(1)) / MILLIWATTS_PER_WATT

    def read(self) -> tuple[float | None, float | None]:
        with self.lock:
            return self.gpu_utilization, self.power_w

    def stop(self) -> None:
        if self.process is None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=TEGRASTATS_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=TEGRASTATS_STOP_TIMEOUT_S)

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=TEGRASTATS_STOP_TIMEOUT_S)
