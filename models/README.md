# Models

This directory stores pretrained model weights downloaded by `src/load_model.py`.

After running the loader, users should expect the following structure:

```text
models/
├── cnn/
│   ├── yolov5nu.pt
│   ├── yolov8n.pt
│   └── hub/checkpoints/
│       ├── mobilenet_v2-7ebf99e0.pth
│       ├── mobilenet_v3_small-047dcff4.pth
│       └── mobilenet_v3_large-5c1a4163.pth
├── vit/
│   ├── mobilevit_xxs/
│   ├── mobilevit_xs/
│   ├── mobilevit_s/
│   ├── efficientformer_l1/
│   ├── efficientformer_l3/
│   └── efficientformer_l7/
└── vlm/
    ├── smolvlm_256m/
    ├── smolvlm_500m/
    └── smolvlm_2b/
```

The YOLO checkpoints are stored directly in `cnn/`. MobileNetV2/V3 weights are
stored by Torchvision under `cnn/hub/checkpoints/`. MobileViT,
EfficientFormer, and SmolVLM use library-managed cache folders, so their
internal file layout may vary between library versions.

Only models selected by the user are downloaded. Running the loader with
`--model all` downloads all supported pretrained models.
