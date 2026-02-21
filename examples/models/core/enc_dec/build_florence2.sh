#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build TRT-LLM engines for Florence2 (BART encoder + decoder, optional DaViT vision).
#
# Usage:
#   bash build_florence2.sh <checkpoint_dir> <engine_dir> [options]
#
# Example:
#   # 1. Convert weights
#   python convert_florence2.py \
#       --model_dir /workspaces/florence2/Florence-2-large-ft  \
#       --output_dir /tmp/florence2_ckpt --dtype float16
#
#   # 2. Build engines
#   bash build_florence2.sh /tmp/florence2_ckpt /tmp/florence2_engine
#
#   # 3. Build engines + vision TRT engine
#   MODEL_DIR=/path/to/Florence-2-large-ft BUILD_VISION=1 \
#       bash build_florence2.sh /tmp/florence2_ckpt /tmp/florence2_engine
#
#   # 4. Run inference
#   python run_florence2.py \
#       --model_dir /path/to/Florence-2-large-ft \
#       --engine_dir /tmp/florence2_engine \
#       --vision_engine_dir /tmp/florence2_engine/vision \
#       --task "<CAPTION>"

set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: build_florence2.sh <checkpoint_dir> <engine_dir> [--max_batch_size N] [--max_input_len N] [--max_prompt_embedding_table_size N]}"
ENGINE_DIR="${2:?Usage: build_florence2.sh <checkpoint_dir> <engine_dir>}"

# Defaults
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_ENCODER_INPUT_LEN="${MAX_ENCODER_INPUT_LEN:-1024}"
# 577 = image_seq_length for Florence-2 at 768x768 resolution
MAX_PROMPT_EMBEDDING_TABLE_SIZE="${MAX_PROMPT_EMBEDDING_TABLE_SIZE:-577}"
MAX_BEAM_WIDTH="${MAX_BEAM_WIDTH:-3}"
DTYPE="${DTYPE:-float16}"
GPT_ATTENTION_PLUGIN="${GPT_ATTENTION_PLUGIN:-${DTYPE}}"
REMOVE_INPUT_PADDING="${REMOVE_INPUT_PADDING:-enable}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-paged}"

echo "============================================"
echo "Building Florence2 TRT-LLM Engines"
echo "============================================"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Engine dir:     ${ENGINE_DIR}"
echo "Max batch size: ${MAX_BATCH_SIZE}"
echo "Max input len:  ${MAX_INPUT_LEN}"
echo "Max prompt embedding table size: ${MAX_PROMPT_EMBEDDING_TABLE_SIZE}"
echo "Max beam width: ${MAX_BEAM_WIDTH}"
echo "Dtype:          ${DTYPE}"
echo "GPT attention:  ${GPT_ATTENTION_PLUGIN}"
echo "Input padding:  ${REMOVE_INPUT_PADDING}"
echo "KV cache type:  ${KV_CACHE_TYPE}"
echo "============================================"

# Build encoder engine
echo ""
echo ">>> Building ENCODER engine..."
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/encoder" \
    --output_dir "${ENGINE_DIR}/encoder" \
    --gpt_attention_plugin "${GPT_ATTENTION_PLUGIN}" \
    --gemm_plugin "${DTYPE}" \
    --remove_input_padding "${REMOVE_INPUT_PADDING}" \
    --kv_cache_type "${KV_CACHE_TYPE}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_input_len "${MAX_INPUT_LEN}" \
    --max_prompt_embedding_table_size "${MAX_PROMPT_EMBEDDING_TABLE_SIZE}"

echo ""
echo ">>> Building DECODER engine..."
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/decoder" \
    --output_dir "${ENGINE_DIR}/decoder" \
    --gpt_attention_plugin "${GPT_ATTENTION_PLUGIN}" \
    --gemm_plugin "${DTYPE}" \
    --remove_input_padding "${REMOVE_INPUT_PADDING}" \
    --kv_cache_type "${KV_CACHE_TYPE}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_input_len 2 \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_encoder_input_len "${MAX_ENCODER_INPUT_LEN}" \
    --max_beam_width "${MAX_BEAM_WIDTH}"

# Optional: Build vision TRT engine for DaViT
# Requires MODEL_DIR to be set (path to HF Florence2 model)
if [ -n "${BUILD_VISION:-}" ]; then
    if [ -z "${MODEL_DIR:-}" ]; then
        echo "ERROR: MODEL_DIR must be set to build vision engine"
        echo "  MODEL_DIR=/path/to/Florence-2-large-ft BUILD_VISION=1 bash build_florence2.sh ..."
        exit 1
    fi
    echo ""
    echo ">>> Building VISION engine..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "${SCRIPT_DIR}/build_florence2_vision.py" \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${ENGINE_DIR}/vision" \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --dtype "${DTYPE}"
fi

echo ""
echo "============================================"
echo "Build complete!"
echo "  Encoder: ${ENGINE_DIR}/encoder"
echo "  Decoder: ${ENGINE_DIR}/decoder"
if [ -n "${BUILD_VISION:-}" ]; then
echo "  Vision:  ${ENGINE_DIR}/vision"
fi
echo "============================================"
echo ""
echo "Run inference with:"
echo "  python run_florence2.py \\"
echo "      --model_dir <florence2_hf_dir> \\"
echo "      --engine_dir ${ENGINE_DIR} \\"
if [ -n "${BUILD_VISION:-}" ]; then
echo "      --vision_engine_dir ${ENGINE_DIR}/vision \\"
fi
echo "      --task \"<CAPTION>\" --compare_hf"
