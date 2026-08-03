# Results

This directory contains generated plots, benchmark tables, evaluation
summaries, and other output artifacts.

- `system_metrics/` contains runtime performance results.
- `detection_metrics/` contains object-detection and image-classification
  quality results.
- `rdd_trained_models/` contains one folder per RDD experiment. Each model
  folder stores `best.pt`, `history.csv`, combined training curves, realistic
  and balanced-test metrics, confusion matrices, and a ROC curve. Each
  experiment also stores the ranked `model_comparison.csv`.
