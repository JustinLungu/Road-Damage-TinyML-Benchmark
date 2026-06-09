# SmolVLM Vision-Language Models

This folder documents the Hugging Face SmolVLM models used in the repository:

| Repo name | Hugging Face model ID | Local cache | Approx. local cache size | Default use |
| --- | --- | --- | --- | --- |
| `smolvlm_256m` | `HuggingFaceTB/SmolVLM-256M-Instruct` | `models/vlm/smolvlm_256m/` | 494M | Practical default for demos |
| `smolvlm_500m` | `HuggingFaceTB/SmolVLM-500M-Instruct` | `models/vlm/smolvlm_500m/` | 973M | Larger local variant |
| `smolvlm_2b` | `HuggingFaceTB/SmolVLM-Instruct` | `models/vlm/smolvlm_2b/` | 4.2G | Largest local variant |

SmolVLM is a vision-language model. It takes an image and a text prompt, then generates text. In this repo it is useful for understanding image-question answering and caption-style inference, and for measuring how VLM inference behaves compared with smaller CNN classifiers and detectors.

For definitions of terms such as vision-language model, prompt, chat template, processor, image token, `pixel_values`, autoregressive generation, and `max_new_tokens`, see [definitions.md](definitions.md).

The repo registry names are defined in `src/constants.py`:

```python
SMOLVLM_MODEL_IDS = {
    "smolvlm_256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm_2b": "HuggingFaceTB/SmolVLM-Instruct",
}
```

## What Makes SmolVLM Different

YOLO answers "what objects are present and where are their boxes?" MobileNetV3 answers "which ImageNet class best describes the whole image?" SmolVLM answers a text prompt about an image.

That makes its output more flexible:

- it can caption an image;
- it can answer a question about visible content;
- it can compare parts of a scene in natural language;
- it can produce a short explanation instead of a fixed label.

The tradeoff is that its output is less structured. SmolVLM does not directly return bounding boxes, detector confidences, or a fixed top-k class vector. If you ask it to count objects or describe locations, it answers in text and can be approximate.

The local SmolVLM models are instruct variants. They are meant to be used with a chat-style prompt, even for a single image question.

## Internal Flow

At inference time, the demo follows this broad path:

```text
image file + text prompt
  -> chat template
  -> processor
  -> token ids + image tensors
  -> vision encoder
  -> language decoder
  -> generated token ids
  -> decoded text response
```

### 1. Image and Prompt

The demo first chooses one image and one prompt:

```python
PROMPT = "Describe the image in one sentence."
```

The same selected image is reused for every model in `MODEL_NAMES`, so the generated answers are easy to compare.

### 2. Chat Message

SmolVLM instruct models expect a chat-like message rather than a raw caption string. The demo creates a user message with an image slot and a text instruction:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }
]
```

The image slot marks where the visual input belongs in the conversation.

### 3. Chat Template

The processor formats that message into the exact token sequence expected by the checkpoint:

```python
prompt_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
```

`add_generation_prompt=True` appends the assistant-start marker so the model knows it should generate the assistant response next.

### 4. Processor

The processor prepares both modalities:

```python
inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
```

For SmolVLM this produces tensors such as:

- `input_ids`: token IDs for the formatted prompt;
- `attention_mask`: which text tokens are real input;
- `pixel_values`: normalized image tensors for the vision encoder;
- image attention/padding tensors when needed by the processor.

The original image can have different dimensions. The processor handles resizing and normalization according to the checkpoint's preprocessing config. In the local `smolvlm_256m` config, the vision side uses 512-pixel image processing and the processor stores `image_seq_len = 64`, which controls how many image positions are represented in the text side.

### 5. Vision Encoder

The vision encoder turns image tensors into visual features.

These features are not human-readable boxes or labels. They are learned vectors that capture visual information useful for answering the prompt.

### 6. Language Decoder

SmolVLM then uses a language decoder to generate the response. The local 256M config loads as `Idefics3ForConditionalGeneration`; its text side is Llama-like, while the vision side supplies image features through the multimodal architecture.

The decoder sees the formatted prompt plus the visual features and predicts the next text token.

### 7. Autoregressive Generation

Generation is repeated next-token prediction:

```python
generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
```

Each generated token becomes part of the context for the next generated token. This is why generation is slower than a single classifier forward pass, especially when `MAX_NEW_TOKENS` is large.

### 8. Decoding the Answer

The raw output of `generate()` is token IDs. The demo decodes only the newly generated tokens:

```python
new_token_ids = generated_ids[:, prompt_token_count:]
response = processor.batch_decode(new_token_ids, skip_special_tokens=True)[0]
```

Decoding only the new tokens avoids printing the original prompt back as part of the answer.

## How This Repo Loads Them

`src/load_model.py` loads SmolVLM through Hugging Face Transformers:

```python
from src.load_model import load_model

model = load_model("smolvlm_256m")
print(type(model))
```

Internally the loader calls:

```python
AutoModelForImageTextToText.from_pretrained(
    model_id,
    cache_dir=str(VLM_DIR / local_name),
).eval()
```

The processor is loaded separately because image-text models need both model weights and preprocessing/tokenization assets:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    cache_dir="models/vlm/smolvlm_256m",
)
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model(model_name)
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> AutoProcessor.from_pretrained(...)
  -> processor(text=prompt, images=[image], return_tensors="pt")
  -> model(**inputs)
```

The benchmark intentionally calls a forward pass, not `generate()`. That measures a fixed amount of model work for latency/power comparison. The demo in this folder calls `generate()` because it is meant to show what VLM inference looks like to a user.

## Inference Example

```python
from pathlib import Path

import torch
from transformers import AutoProcessor

from src.constants import SMOLVLM_MODEL_IDS, VLM_DIR
from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image, move_inputs_to_device

model_name = "smolvlm_256m"
image_path = Path("datasets/coco/images/000000047585.jpg")
prompt = "Describe the image in one sentence."
device = torch.device("cpu")

processor = AutoProcessor.from_pretrained(
    SMOLVLM_MODEL_IDS[model_name],
    cache_dir=str(VLM_DIR / model_name),
)
model = load_model(model_name).to(device).eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }
]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

image = load_rgb_image(image_path)
inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
prompt_token_count = inputs["input_ids"].shape[1]
inputs = move_inputs_to_device(inputs, device)

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
    )

new_token_ids = generated_ids[:, prompt_token_count:]
response = processor.batch_decode(new_token_ids, skip_special_tokens=True)[0]
print(response.strip())
```

## Reading The Output

SmolVLM returns generated text. Read it as the model's natural-language answer to the prompt about the image.

For example, with a prompt like:

```text
Describe the image briefly.
```

the model may return a caption-style sentence.

With a prompt like:

```text
What vehicles are visible?
```

the model may answer with a short list or sentence about visible vehicles.

This is different from:

- YOLO output, which contains bounding boxes, class IDs, and confidences;
- MobileNetV3 output, which contains logits for 1000 fixed ImageNet classes.

SmolVLM output should be inspected like generated language. It can be useful and flexible, but it can also omit details, phrase uncertainty poorly, or answer approximately.

## Quick Demo

Run the local demo from the repository root:

```bash
uv run python docs/models/smolvlm/demo.py
```

The demo is intentionally configured by constants at the top of `demo.py`, so edit these values before running:

```python
MODEL_NAMES = ["smolvlm_256m"]
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"
RANDOM_SEED = None
DEVICE_NAME = "cpu"
PROMPT = "Describe the image in one sentence."
MAX_NEW_TOKENS = 40
DO_SAMPLE = False
TEMPERATURE = 0.7
LOCAL_FILES_ONLY = True
SAVE_RESPONSE_JSON = True
SAVE_INPUT_IMAGE = True
OUTPUT_DIR = MODEL_DOCS_DIR / "outputs"
```

Set `MODEL_NAMES` to one or more supported names. For example, `["smolvlm_256m"]` runs only the smallest model, while `["smolvlm_256m", "smolvlm_500m"]` runs both models on the same selected image.

Set `USE_RANDOM_IMAGE = False` to use `IMAGE_PATH`. Set `USE_RANDOM_IMAGE = True` to ignore `IMAGE_PATH` and sample a random `.jpg` from `RANDOM_IMAGE_DIR`. Set `RANDOM_SEED` to an integer if you want the same random image across runs.

When `SAVE_RESPONSE_JSON = True`, each model writes a response JSON file under:

```text
docs/models/smolvlm/outputs/
```

When `SAVE_INPUT_IMAGE = True`, the selected source image is also copied to:

```text
docs/models/smolvlm/outputs/selected_image.jpg
```

This image is the visual reference for the generated text response. It is not annotated with boxes because SmolVLM does not predict object locations as structured detector output.

Start with `smolvlm_256m` on CPU. The 500M and 2B variants are heavier and may be slow or memory-intensive depending on your machine.

## Benchmark Commands

The system-performance benchmark measures latency, FPS, RAM, GPU metrics, power, and energy per inference. For SmolVLM it measures a fixed-prompt forward pass, not full text generation.

```bash
uv run python experiments/performance/system_performance.py \
  --model smolvlm_256m \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model smolvlm_500m \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model smolvlm_2b \
  --device cuda:0 \
  --num-images 100
```
