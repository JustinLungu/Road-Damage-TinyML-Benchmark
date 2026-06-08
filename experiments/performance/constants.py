from pathlib import Path

from src.constants import SYSTEM_PERFORMANCE_RESULTS_CSV


###############
# CLI Keywords
###############

# Special --model value that expands to every checkpoint already present locally.
ALL_LOADED = "all-loaded"


###############
# Result Paths
###############

RESULTS_CSV: Path = SYSTEM_PERFORMANCE_RESULTS_CSV
DEFAULT_CSV: Path = RESULTS_CSV


################
# Plot Folders
################

WITH_VLMS_DIR = "with_vlms"
WITHOUT_VLMS_DIR = "without_vlms"
