# Qwen3.5 on ARM CPU

This example supports the text towers of
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), including
single-image, ordered multi-image, and bounded short-video inference for both
model sizes.

| Model | Text | Images | Decoded RGB-frame API | Local H.264 MP4 | Hidden size | GDN / full-attention layers |
| --- | --- | --- | --- | --- | ---: | ---: |
| 0.8B | Yes | Yes | Yes | Yes, portable backend only | 1024 | 18 / 6 |
| 4B | Yes | Yes | Yes | Yes, portable backend only | 2560 | 24 / 8 |

All configurations support batch size 1 and a maximum cache length of 2048
tokens. `Qwen3_5ForCausalLM::resetState()` clears the full-attention KV cache
and every GDN recurrent/convolution state before each prompt.

The multimodal path accepts one or more ordered still images, or one local
H.264 MP4 plus a text prompt. Images and video are mutually exclusive in one
request. Visual inputs must have aspect ratio at most 200. Still images are
smart-resized to a 65,536--262,144 pixel budget with both dimensions divisible
by 32. Video sampling is deterministic and defaults to 2 fps, with hard upper
bounds of 64 sampled frames, 256 MiB of input, 4096 projected video tokens,
536,870,912 decoded source pixels, and 16,777,216 selected source-frame pixels.
Callers may lower these limits but cannot raise them. Audio is ignored; multiple
videos, URLs, cameras, streaming, deep-stack visual features, MTP, and
multimodal benchmark mode are not supported.

### Video decoder boundary

The Qwen3.5 model and preprocessing code do not depend on a container parser or
video codec. They consume decoded RGB frames, source-frame indices, and the
source frame rate, then apply Qwen3.5 temporal patching, video token types, and
multimodal RoPE. The default correctness tests construct decoded frames
directly, so they do not require codec dependencies.

The example runner currently provides one `portable` reference backend for
local H.264 MP4 files. It uses minimp4 plus OpenH264 and is disabled by default.
It is not the preferred production Android integration: a future Android
backend should use MediaExtractor/MediaCodec through Media NDK and pass the
same decoded-frame contract to the Qwen3.5 path.

### Optional portable MP4 decoder

Select the `portable` backend only with separately supplied, pinned minimp4
headers and matching OpenH264 headers/library:

The validated dependency revisions are minimp4
`5a212a18dba7dca09543bbc7d65619274fd2931a` and OpenH264
`35325f4040c2be0f86246c4a8923f7fc04c1a998`. CMake validates the supplied
paths, not their revisions; downstream builds are responsible for preserving
that source and header/library identity.

```bash
cmake -S . -B build-video \
  -DMLLM_QWEN35_VIDEO_DECODER_BACKEND=portable \
  -DMLLM_QWEN35_MINIMP4_INCLUDE_DIR=/path/to/minimp4 \
  -DMLLM_QWEN35_OPENH264_SOURCE_DIR=/path/to/openh264 \
  -DMLLM_QWEN35_OPENH264_LIBRARIES=/path/to/libopenh264.a
cmake --build build-video --target mllm-qwen3-5-runner
```

The build does not download or vendor either dependency. With the default
`MLLM_QWEN35_VIDEO_DECODER_BACKEND=none`, neither dependency is needed and a
`--video_path` request fails explicitly. Review the dependency licenses and
codec-distribution requirements for your product before shipping the portable
decoder binary.

When both `MLLM_ENABLE_TEST=ON` and the `portable` backend are selected, CTest
registers `Qwen35PortableVideoDecoderSmoke`. This backend-specific smoke checks
the committed MP4 fixture, exact sampled indices and RGB hash, and Qwen3.5 patch
geometry. It is separate from the default model correctness tests.

### Chat template backend

`config.json` selects how the runner renders the prompt through
`chat_template_backend`: `legacy` (default) keeps the byte-stable single-turn
runner prompt; `jinja_required` renders the checkpoint's own official `chat_template.jinja`,
discovered next to `tokenizer.json`, with the vendored `jinja.cpp` engine and
requires a build with `-DMLLM_ENABLE_JINJA_CHAT_TEMPLATE=ON`. A
`jinja_required` model fails at load time instead of falling back when Jinja
support or the template is missing. See
`mllm/preprocessor/chat_template/README.md` for the pipeline and the
Transformers parity gate.
Both backends emit one `<|vision_start|><|image_pad|><|vision_end|>` or
`<|vision_start|><|video_pad|><|vision_end|>` placeholder per media input;
the runner then expands video placeholders into timestamped frame markers and
image/video placeholders into per-patch tokens exactly as before.

## Quantization

The user-facing W4A8 configuration uses dynamic INT8 activations with INT4
weights while retaining FP32 operator inputs and outputs. The existing
`w4a32_kai` names and conversion pipeline remain for compatibility with mllm
tooling.

Text and vision `nn::Linear` weights use the KAI path. Vision Conv3D, learned
position embeddings, LayerNorm parameters, text embeddings, recurrent
parameters, and convolution weights remain FP32. The tied text embedding is
retained for token lookup and separately packed as `lm_head_out.weight`.

## Convert a checkpoint

Run commands from the repository root. Validate the checkpoint and conversion
recipe before creating a model file, then audit the complete V2 descriptor and
offset table after conversion.

### Qwen3.5-0.8B text-only

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_w4a32_kai.json

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-0.8B \
  --output_path /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  --model_name Qwen3.5-0.8B \
  --cfg_path examples/qwen3_5/quant_cfg_0.8B_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --format v2 \
  --verbose

python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_w4a32_kai.json \
  --model-name Qwen3.5-0.8B
```

### Qwen3.5-0.8B multimodal

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_multimodal_w4a32_kai.json \
  --runtime-config examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-0.8B \
  --output_path /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  --model_name Qwen3.5-0.8B-Multimodal \
  --cfg_path examples/qwen3_5/quant_cfg_0.8B_multimodal_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --include_prefix model.visual. \
  --format v2 \
  --verbose

python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_multimodal_w4a32_kai.json \
  --runtime-config examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json \
  --model-name Qwen3.5-0.8B-Multimodal
```

### Qwen3.5-4B text-only

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-4B \
  --quant-config examples/qwen3_5/quant_cfg_4B_w4a32_kai.json

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-4B \
  --output_path /path/to/qwen3.5-4b-w4a32-kai.mllm \
  --model_name Qwen3.5-4B \
  --cfg_path examples/qwen3_5/quant_cfg_4B_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --format v2 \
  --verbose

python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-4b-w4a32-kai.mllm \
  /path/to/Qwen3.5-4B \
  --quant-config examples/qwen3_5/quant_cfg_4B_w4a32_kai.json \
  --model-name Qwen3.5-4B
```

### Qwen3.5-4B multimodal

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-4B \
  --quant-config examples/qwen3_5/quant_cfg_4B_multimodal_w4a32_kai.json \
  --runtime-config examples/qwen3_5/config_4B_multimodal_w4a32_kai.json

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-4B \
  --output_path /path/to/qwen3.5-4b-multimodal-w4a32-kai.mllm \
  --model_name Qwen3.5-4B-Multimodal \
  --cfg_path examples/qwen3_5/quant_cfg_4B_multimodal_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --include_prefix model.visual. \
  --format v2 \
  --verbose

python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-4b-multimodal-w4a32-kai.mllm \
  /path/to/Qwen3.5-4B \
  --quant-config examples/qwen3_5/quant_cfg_4B_multimodal_w4a32_kai.json \
  --runtime-config examples/qwen3_5/config_4B_multimodal_w4a32_kai.json \
  --model-name Qwen3.5-4B-Multimodal
```

The 4B converted tensor data exceeds 4 GiB. Use model-file V2 so descriptor
sizes and offsets remain 64-bit, and provide at least 32 GiB of available host
memory plus working disk space for conversion.

## Run

```bash
# Qwen3.5-0.8B single-image
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-0.8B/tokenizer.json \
  --config_path examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json \
  --image_path /path/to/image.jpg \
  --prompt "Describe the image." \
  --max_new_tokens 32

# Qwen3.5-0.8B multi-image (image order is preserved)
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-0.8B/tokenizer.json \
  --config_path examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json \
  --image_path /path/to/first.jpg \
  --image_path /path/to/second.jpg \
  --prompt "Compare the first and second images." \
  --max_new_tokens 32

# Qwen3.5-0.8B bounded short video (decoder-enabled build)
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-0.8B/tokenizer.json \
  --config_path examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json \
  --video_path /path/to/video.mp4 \
  --prompt "Describe this video." \
  --max_new_tokens 48

# Qwen3.5-4B text-only
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-4b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-4B/tokenizer.json \
  --config_path examples/qwen3_5/config_4B_w4a32_kai.json \
  --prompt "Give a one-sentence introduction." \
  --max_new_tokens 32

# Qwen3.5-4B multimodal; repeat --image_path for ordered multi-image input,
# or replace it with --video_path in a decoder-enabled build.
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-4b-multimodal-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-4B/tokenizer.json \
  --config_path examples/qwen3_5/config_4B_multimodal_w4a32_kai.json \
  --image_path /path/to/image.jpg \
  --prompt "Describe the image." \
  --max_new_tokens 32
```

Omit `--prompt` for the interactive loop. Repeat `--image_path` to attach
multiple still images in order; the same ordered image list is used for each
independent prompt in that process. Omit `--image_path` to run text-only
inference with either the text-only or multimodal model of the selected size. `--video_path`
requires a decoder-enabled build and cannot be combined with `--image_path`.
Use `--video_fps`, `--video_max_frames`, `--video_max_bytes`,
`--video_max_tokens`, `--video_max_decoded_pixels`, and
`--video_max_selected_pixels` only to lower the bounded defaults above.
