# Models

This directory stores pretrained model weights downloaded by `src/load_model.py`.

After running the loader, users should expect the following structure:

```text
models/
├── cnn/
│   ├── yolov5nu.pt
│   ├── yolov8n.pt
│   └── hub/checkpoints/
│       ├── efficientnet_b0_rwightman-7f5810bc.pth
│       ├── inception_v3_google-0cc3c7bd.pth
│       ├── mobilenet_v2-7ebf99e0.pth
│       ├── mobilenet_v3_small-047dcff4.pth
│       ├── mobilenet_v3_large-5c1a4163.pth
│       └── resnet18-f37072fd.pth
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

The YOLO checkpoints are stored directly in `cnn/`. EfficientNet-B0,
InceptionV3, MobileNetV2/V3, and ResNet18 weights are stored by Torchvision under
`cnn/hub/checkpoints/`. MobileViT, EfficientFormer, and SmolVLM use
library-managed cache folders, so their internal file layout may vary between
library versions.

Only models selected by the user are downloaded. Running the loader with
`--model all` downloads all supported pretrained models.

## Model Sizes

The table below summarizes the local pretrained model artifacts and whether
they fit the supervisor-requested size buckets:

- **tiny**: less than 1 MB
- **small**: 1-10 MB

For Torchvision and YOLO models, the size is the measured local checkpoint file
size. For MobileViT, the size is the measured Hugging Face weight file in the
local cache. For EfficientFormer, the local cache may be managed by `timm`/Xet
without a simple single checkpoint file, so the table uses the FP32 parameter
size estimate from the instantiated architecture.

| Model | Type | Local/pretrained size | Size bucket | Used for RDD binary image classification? | Notes |
| --- | --- | ---: | --- | --- | --- |
| `tiny_cnn` | Image classification | approx. 0.34 MiB / 0.35 MB FP32 | tiny | Yes | Custom local model with no pretrained checkpoint. |
| `yolov5nu` | Object detection | 5.31 MiB / 5.56 MB | small | No | YOLO detector, not image classification. |
| `yolov8n` | Object detection | 6.25 MiB / 6.55 MB | small | No | YOLO detector, not image classification. |
| `mobilevit_xxs` | Image classification | 4.91 MiB / 5.15 MB | small | Yes | Smallest current RDD image-classification candidate. |
| `mobilevit_xs` | Image classification | 8.92 MiB / 9.35 MB | small | Yes | Fits the 1-10 MB small bucket. |
| `mobilenet_v3_small` | Image classification | 9.83 MiB / 10.31 MB | borderline small | Yes | Under 10 MiB but slightly above 10 decimal MB. |
| `mobilenet_v2` | Image classification | 13.60 MiB / 14.26 MB | medium | Yes | Above the requested small bucket. |
| `efficientnet_b0` | Image classification | 20.45 MiB / 21.44 MB | medium | Yes | Above the requested small bucket. |
| `mobilenet_v3_large` | Image classification | 21.11 MiB / 22.13 MB | medium | Yes | Above the requested small bucket. |
| `mobilevit_s` | Image classification | 21.37 MiB / 22.48 MB | medium | Yes | Above the requested small bucket. |
| `resnet18` | Image classification | 44.66 MiB / 46.83 MB | medium | Yes | Above the requested small bucket. |
| `efficientformer_l1` | Image classification | approx. 46.88 MiB / 49.16 MB FP32 | medium | Yes | Estimate from parameter count. |
| `inception_v3` | Image classification | 103.90 MiB / 108.95 MB | large | Yes | Above the requested small bucket. |
| `efficientformer_l3` | Image classification | approx. 119.80 MiB / 125.62 MB FP32 | large | Yes | Best current RDD baseline by F1, but not small. |
| `efficientformer_l7` | Image classification | approx. 313.68 MiB / 328.92 MB FP32 | large | Attempted | Failed current laptop training due CUDA out-of-memory. |
| `smolvlm_256m` | Vision-language model | much larger than 10 MB | large | No | Excluded from RDD image-classification fine-tuning. |
| `smolvlm_500m` | Vision-language model | much larger than 10 MB | large | No | Excluded from RDD image-classification fine-tuning. |
| `smolvlm_2b` | Vision-language model | much larger than 10 MB | large | No | Excluded from RDD image-classification fine-tuning. |

Current conclusion for the RDD image-classification path:

- We now have a **true tiny custom image-classification model**:
  `tiny_cnn`, which is under 1 MB before any quantization.
- We **do have small image-classification candidates**: `mobilevit_xxs`,
  `mobilevit_xs`, and possibly `mobilenet_v3_small` if the 10 MB limit is
  interpreted as 10 MiB.
- The current pretrained YOLO files are small, but they are object detectors
  and do not satisfy the binary image-classification model requirement.
