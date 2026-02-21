<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Florence-2 with TensorRT-LLM

Florence-2 is a multimodal encoder-decoder model (DaViT vision encoder + BART language model) from Microsoft. This guide shows how to accelerate the BART text backbone with TRT-LLM, with an optional TRT engine for the DaViT vision encoder.

## Architecture

```
Image → DaViT → image_projection → LayerNorm → image_features [B, 577, 1024]
Task prompt → tokenizer → text_ids [B, N_text]

[virtual_image_ids, text_ids] → TRT Encoder (prompt_embedding_table=image_features) → encoder_output
[decoder_start, bos] → TRT Decoder (cross-attention to encoder_output) → generated text
```

Image features are injected into the encoder via the **prompt embedding table** (p-tuning mechanism): tokens with ID >= vocab_size are "virtual tokens" whose embeddings come from the DaViT output. No core TRT-LLM code modifications are needed.

## Quick Start

### 1. Convert weights

```bash
python convert_florence2.py \
    --model_dir /workspaces/florence2/Florence-2-large-ft \
    --output_dir /tmp/florence2_ckpt \
    --dtype float16
```

### 2. Build TRT-LLM engines

```bash
bash build_florence2.sh /tmp/florence2_ckpt /tmp/florence2_engine
```

Or manually:

```bash
# Encoder (max_prompt_embedding_table_size=577 for 768x768 images)
trtllm-build \
    --checkpoint_dir /tmp/florence2_ckpt/encoder \
    --output_dir /tmp/florence2_engine/encoder \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_prompt_embedding_table_size 577

# Decoder (max_input_len=2 for [decoder_start, bos] prefix)
trtllm-build \
    --checkpoint_dir /tmp/florence2_ckpt/decoder \
    --output_dir /tmp/florence2_engine/decoder \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 2 \
    --max_seq_len 1024 \
    --max_encoder_input_len 1024 \
    --max_beam_width 3
```

### 3. Run inference

```bash
python run_florence2.py \
    --model_dir /workspaces/florence2/Florence-2-large-ft \
    --engine_dir /tmp/florence2_engine \
    --task "<CAPTION>" \
    --image /workspaces/florence2/images/car.jpg \
    --compare_hf
```

Supported task tokens: `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`, `<OCR>`, etc. For non-text tasks like `<OD>`/`<OCR>`, pass `--post_process` to parse structured outputs (bboxes, polygons, etc.).

### 4. (Optional) Build vision TRT engine

Export the DaViT vision encoder as a TRT engine. This replaces the PyTorch DaViT path with a TRT engine that takes `pixel_values [B, 3, 768, 768]` and outputs `image_features [B, 577, 1024]`.

The engine is built with FP32 internal precision by default (ONNX I/O stays FP16). This matches PyTorch's behaviour where LayerNorm and softmax internally upcast to FP32, avoiding numerical divergence that can affect beam search output.

```bash
python build_florence2_vision.py \
    --model_dir /path/to/Florence-2-large-ft \
    --output_dir /tmp/florence2_engine/vision \
    --max_batch_size 1
```

Or build everything at once using the shell script:

```bash
MODEL_DIR=/path/to/Florence-2-large-ft BUILD_VISION=1 \
    bash build_florence2.sh /tmp/florence2_ckpt /tmp/florence2_engine
```

### 5. Run with vision TRT engine

When `--vision_engine_dir` is provided, the HF model is not loaded for vision encoding (saves ~2GB memory and avoids PyTorch overhead). The HF model is still loaded if `--compare_hf` is set.

```bash
python run_florence2.py \
    --model_dir /path/to/Florence-2-large-ft \
    --engine_dir /tmp/florence2_engine \
    --vision_engine_dir /tmp/florence2_engine/vision \
    --task "<CAPTION>" \
    --image /path/to/image.jpg
```

## Beam Search Configuration

Florence-2's BART decoder assigns ~69% probability to `<s>` (bos, token 0) at every step. Without mitigation, beam search degenerates into infinite bos sequences. The HF model handles this via `text_config`/`generation_config` settings that `run_florence2.py` replicates:

| HF Setting | Value | TRT-LLM Equivalent |
|------------|-------|---------------------|
| `forced_bos_token_id` | 0 | `decoder_input_ids = [decoder_start, bos]` |
| `no_repeat_ngram_size` | 3 | `SamplingConfig.no_repeat_ngram_size = 3` |
| `early_stopping` | False (from `generation_config.json`) | `SamplingConfig.early_stopping = 0` |

After generating 3 bos tokens, `no_repeat_ngram_size=3` prevents further bos, forcing the model to produce content tokens.

## Debugging / Known Issue

When building the decoder with `--gpt_attention_plugin` **and** `--kv_cache_type paged`, we observed a deterministic issue in beam search (`num_beams>1`) where one beam's logits become all-NaN starting at `step=2`, which can later manifest as repeated tokens (e.g., `' yellow'` twice with an abnormally high probability).

To reproduce and dump the internal state around the first NaN:

```bash
python run_florence2.py \
  --model_dir /path/to/Florence-2-large-ft \
  --engine_dir /tmp/florence2_engine \
  --task "<CAPTION>" --image /path/to/image.jpg \
  --num_beams 2 --max_new_tokens 4 --length_penalty 0 \
  --debug_mode \
  --tllm_debug_nan_logits --tllm_debug_nan_logits_dump_step 2 \
  --tllm_debug_step_tensors_step 2 \
  --tllm_debug_step_tensors_names sequence_length,host_past_key_value_lengths,cache_indirection,input_ids,last_token_ids,position_ids,kv_cache_block_offsets
```

Workaround (correctness): disable the GPT attention plugin when building the decoder:

```bash
GPT_ATTENTION_PLUGIN=disable REMOVE_INPUT_PADDING=disable KV_CACHE_TYPE=continuous \
  bash build_florence2.sh /tmp/florence2_ckpt /tmp/florence2_engine_no_gpt
```

Then run:

```bash
python run_florence2.py \
  --model_dir /path/to/Florence-2-large-ft \
  --engine_dir /tmp/florence2_engine_no_gpt \
  --task "<CAPTION>" --image /path/to/image.jpg \
  --compare_hf --debug_topk
```

## Files

| File | Description |
|------|-------------|
| `convert_florence2.py` | Convert HF Florence-2 weights to TRT-LLM checkpoint format |
| `build_florence2.sh` | Build encoder + decoder TRT engines (optionally vision) |
| `build_florence2_vision.py` | Build TRT engine for DaViT vision encoder |
| `run_florence2.py` | End-to-end inference (DaViT in PyTorch or TRT + BART in TRT-LLM) |

## Performance

Tested on a single GPU with Florence-2-large-ft, `<CAPTION>` task, `num_beams=3`, `max_new_tokens=128`:

| Runtime | Latency | Notes |
|---------|---------|-------|
| HF (FP16, eager attention) | ~820ms | Full PyTorch |
| TRT-LLM (FP16) | ~240ms | ~3.4x speedup |

Output match rate vs HF: 92-100% (minor differences from attention kernel implementations).
