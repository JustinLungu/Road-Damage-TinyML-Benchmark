# Configurations

This directory will contain configuration files for experiments, models, datasets, and deployment.

Use this directory for values that a user should tune without editing Python
code, such as experiment parameter sweeps, deployment profiles, model
thresholds, or dataset variants.

Repo-wide paths, model IDs, and checkpoint/cache locations currently live in
`src/constants.py` because they are shared code constants rather than
user-editable experiment settings.
