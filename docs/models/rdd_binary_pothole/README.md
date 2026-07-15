# RDD Binary Pothole Checkpoint Demo

Use this demo to load a fine-tuned RDD checkpoint from `results/` and run one
binary pothole inference.

Edit `CHECKPOINT_PATH` in `demo.py` to point at any trained checkpoint, for
example:

```python
CHECKPOINT_PATH = (
    REPO_ROOT
    / "results"
    / "rdd_trained_models"
    / "G_clean400_full_image_downsample_1to5_standard_aug"
    / "tiny_cnn"
    / "best.pt"
)
```

Then run:

```bash
.venv/bin/python docs/models/rdd_binary_pothole/demo.py
```

The checkpoint stores the model name, so the same demo works for `tiny_cnn`,
`resnet8`, `ds_cnn_small`, `mobilenet_v1_025`, `shufflenet_v2_x0_5`, and the
larger RDD models.
