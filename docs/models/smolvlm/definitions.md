# SmolVLM Definitions

This glossary explains the terms used in the SmolVLM model notes.

## Vision-Language Model

A vision-language model, often shortened to VLM, consumes visual input and text input together.

For this repo, SmolVLM receives an image plus a written prompt, then generates a text answer. That means its natural output is language, not bounding boxes like YOLO or fixed ImageNet class probabilities like MobileNetV3.

## Multimodal Input

Multimodal input means the model receives more than one kind of data.

In the demo, the two modalities are:

1. an RGB image;
2. a text instruction such as `Describe the image briefly.`

The processor converts both into tensors that the model can use in one generation call.

## Prompt

The prompt is the text instruction given to the model.

Changing the prompt changes the task. For example:

```text
Describe the image briefly.
```

asks for a caption, while:

```text
What objects are visible in this image?
```

asks the model to focus on visible objects.

## Chat Template

A chat template turns a user message into the exact text format expected by an instruct-tuned model.

SmolVLM instruct checkpoints expect role markers, an image placeholder, and a generation prompt. The demo uses:

```python
processor.apply_chat_template(messages, add_generation_prompt=True)
```

This creates the formatted text sequence that goes into the tokenizer.

## Processor

The processor is a Hugging Face object that combines image preprocessing and text tokenization.

For SmolVLM, the processor:

- tokenizes the formatted prompt;
- resizes and normalizes the image;
- inserts image-related tokens into the text stream;
- returns tensor inputs such as `input_ids`, `attention_mask`, and `pixel_values`.

## Token

A token is a small unit of text handled by the language model.

Tokens can be whole words, word pieces, punctuation, or special markers. The model generates text one token at a time.

## Image Token

An image token is a special token that tells the language model where visual information belongs in the prompt.

The raw image is not converted into normal words. Instead, the image is processed by the vision side of the model, and special image tokens connect that visual representation to the text decoder.

## `input_ids`

`input_ids` are integer token IDs for the formatted prompt.

For a VLM prompt, this sequence includes normal text tokens and special tokens associated with the image position.

## `attention_mask`

`attention_mask` tells the model which token positions are real input and which positions are padding.

This is important when inputs in a batch have different lengths.

## `pixel_values`

`pixel_values` are the preprocessed image tensors.

They are the result of resizing, rescaling, normalizing, and arranging the image into the tensor shape expected by the vision encoder.

## Vision Encoder

The vision encoder converts image pixels into learned visual features.

Those features represent visual patterns such as shapes, objects, scene layout, and textural cues. SmolVLM then makes those features available to the language model while it generates its answer.

## Language Decoder

The language decoder is the text-generation part of the model.

It reads the prompt and the visual features, then predicts the next text token. It repeats this next-token prediction until it reaches a stop token or the configured token limit.

## Autoregressive Generation

Autoregressive generation means the model produces one token at a time.

Each new token is appended to the sequence and becomes part of the context for predicting the following token. This is why longer answers take longer to generate.

## `max_new_tokens`

`max_new_tokens` limits how many output tokens the model may generate after the prompt.

Increasing it allows longer answers but increases latency and memory use.

## Greedy Decoding

Greedy decoding picks the highest-scoring next token at each generation step.

In the demo, `DO_SAMPLE = False` uses greedy decoding. It is deterministic for the same model, image, prompt, and environment.

## Sampling

Sampling picks the next token from a probability distribution instead of always taking the top token.

In the demo, setting `DO_SAMPLE = True` enables sampling. `TEMPERATURE` controls how random the choices are. Higher values make output more varied; lower values make it more conservative.

## Forward Pass

A forward pass computes model outputs for the supplied inputs.

The performance benchmark uses a forward pass for SmolVLM so latency is comparable across images without also timing text decoding. The demo uses generation because it is meant to show the actual text output a user would see.

## Generation

Generation repeatedly runs the model to create new output tokens.

This is the normal way to use an instruct VLM when you want a caption, answer, or explanation.

## Local Cache

The local cache is where Hugging Face model files are stored on disk.

For SmolVLM in this repo, the caches live under:

```text
models/vlm/smolvlm_256m/
models/vlm/smolvlm_500m/
models/vlm/smolvlm_2b/
```

The demo loads the processor from the matching local cache and the model through `src.load_model.load_model`.
