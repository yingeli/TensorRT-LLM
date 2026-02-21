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
"""Convert Florence2 HF weights to TRT-LLM EncoderModel/DecoderModel format.

Florence2 uses a BART-based language model (encoder + decoder) with the same
architecture as standard BART.  The main difference is the weight prefix:
  HF BART:      model.{encoder,decoder}.*
  Florence2 HF: language_model.model.{encoder,decoder}.*

The shared embedding lives at ``language_model.model.shared.weight`` and there
is no separate ``lm_head.weight`` in the checkpoint (weights are tied).

Usage:
    python convert_florence2.py \
        --model_dir /path/to/Florence-2-large-ft \
        --output_dir /tmp/florence2_ckpt \
        --dtype float16
"""
import argparse
import copy
import datetime
import json
import logging
import os
import pathlib

import safetensors
from safetensors import torch as safetensors_torch
import torch
import helper

from tensorrt_llm import _utils
from tensorrt_llm import functional
from tensorrt_llm.models import modeling_utils

LOGGER = logging.getLogger(__name__)

layernorm_type_map = {i.name: i.value for i in functional.LayerNormType}
layernorm_position_map = {
    i.name: i.value
    for i in functional.LayerNormPositionType
}
mlp_type_map = {i.name: i.value for i in functional.MLPType}


def parse_florence2_config(args):
    """Parse Florence2 HF config.json and return encoder/decoder configs."""
    config_path = pathlib.Path(args.model_dir) / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)

    text_config = hf_config["text_config"]

    # Florence2 text_config is a BART config with these fields:
    #   d_model, encoder_layers, decoder_layers, encoder_attention_heads,
    #   decoder_attention_heads, encoder_ffn_dim, decoder_ffn_dim,
    #   max_position_embeddings, vocab_size, activation_function,
    #   normalize_before, scale_embedding, etc.

    def make_component_config(component):
        import types
        cfg = types.SimpleNamespace()
        cfg.model_type = "bart"

        cfg.n_layer = text_config[f"{component}_layers"]
        cfg.n_head = text_config[f"{component}_attention_heads"]
        cfg.hidden_size = text_config["d_model"]
        cfg.head_size = cfg.hidden_size // cfg.n_head
        cfg.ffn_hidden_size = text_config[f"{component}_ffn_dim"]
        cfg.vocab_size = text_config["vocab_size"]
        cfg.n_positions = text_config["max_position_embeddings"]

        # BART-specific flags
        cfg.has_position_embedding = True
        cfg.has_token_type_embedding = False
        cfg.has_embedding_layernorm = True
        cfg.has_embedding_scale = text_config.get("scale_embedding", False)
        cfg.q_scaling = 1.0
        cfg.has_attention_qkvo_bias = True
        cfg.has_mlp_bias = True
        cfg.has_model_final_layernorm = False  # BART (not mBART)

        normalize_before = text_config.get("normalize_before", False)
        cfg.layernorm_position = layernorm_position_map[
            'pre_layernorm' if normalize_before else 'post_layernorm']
        cfg.layernorm_type = layernorm_type_map["LayerNorm"]
        cfg.layernorm_eps = text_config.get("layer_norm_epsilon", 1e-5)
        cfg.hidden_act = text_config.get("activation_function", "gelu")
        cfg.gated_act = False
        cfg.mlp_type = mlp_type_map["MLP"]
        cfg.relative_attention = False
        cfg.num_buckets = 0
        cfg.max_distance = 0
        cfg.position_embedding_type = "learned_absolute"
        cfg.logits_dtype = "float32"

        if component == "decoder":
            cfg.rescale_before_lm_head = False
            cfg.encoder_hidden_size = text_config["d_model"]
            cfg.encoder_num_heads = text_config["encoder_attention_heads"]
            cfg.encoder_head_size = (text_config["d_model"] //
                                     text_config["encoder_attention_heads"])
            cfg.decoder_start_token_id = text_config.get(
                "decoder_start_token_id", 2)
            cfg.eos_token_id = text_config.get("eos_token_id", 2)
            cfg.bos_token_id = text_config.get("bos_token_id", 0)
            cfg.pad_token_id = text_config.get("pad_token_id", 1)

        return cfg

    encoder_config = make_component_config("encoder")
    decoder_config = make_component_config("decoder")
    return encoder_config, decoder_config


def load_florence2_weights(model_dir):
    """Load Florence2 weights from safetensors (preferred) or pytorch_model.bin."""
    safetensors_path = pathlib.Path(model_dir) / "model.safetensors"
    bin_path = pathlib.Path(model_dir) / "pytorch_model.bin"

    if safetensors_path.exists():
        LOGGER.info(f"Loading weights from {safetensors_path}")
        params = safetensors_torch.load_file(str(safetensors_path))
    elif bin_path.exists():
        LOGGER.info(f"Loading weights from {bin_path}")
        params = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            "Expected model.safetensors or pytorch_model.bin")

    return params


def convert_florence2_weights(config, component, params):
    """Convert Florence2 BART weights to TRT-LLM format.

    This is nearly identical to ``convert_bart_weights_to_tllm_safetensors``
    but uses the ``language_model.model.{component}`` prefix and handles
    the tied shared embedding for lm_head.
    """
    weights = {}
    mapping = config.mapping

    hidden_size = config.hidden_size
    helper.convert_weight_to_dtype(params, config.dtype)
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    # Florence2 prefix differs from standard BART
    hf_param_prefix = f'language_model.model.{component}'
    trtllm_layer_name = 'transformer.layers'
    trtllm_attn_layer_name = ('attention'
                              if component == 'encoder' else 'self_attention')
    trtllm_attn_layernorm_name = ('self_attention_layernorm'
                                  if component == 'decoder' else
                                  'attention_layernorm')

    # --- Embedding weights ---
    embedding_layer_names = {
        'embed_positions.weight': {
            "name": 'transformer.position_embedding.weight',
            "shape": (config.max_position_embeddings, hidden_size)
        },
        'layernorm_embedding.weight': {
            "name": 'transformer.ln_embed.weight',
            "shape": None
        },
        'layernorm_embedding.bias': {
            "name": 'transformer.ln_embed.bias',
            "shape": None
        },
    }

    # For embed_tokens, Florence2 uses shared.weight (no per-component copy in safetensors)
    shared_embedding_key = 'language_model.model.shared.weight'

    for hf_weight_name, weight_info in embedding_layer_names.items():
        full_key = f'{hf_param_prefix}.{hf_weight_name}'
        if 'position' in hf_weight_name:
            # Skip first 2 entries (BART position embedding offset)
            weights[weight_info["name"]] = params[full_key][2:].clone()
        else:
            weights[weight_info["name"]] = params[full_key].clone()
        weights[weight_info["name"]] = helper.reshape(
            weights[weight_info["name"]], weight_info["shape"])

    # vocab_embedding for the embedding layer (used by prompt_embedding_table lookup)
    weights["embedding.vocab_embedding.weight"] = helper.reshape(
        params[shared_embedding_key].clone(), (vocab_size, -1))

    # vocab_embedding for the transformer (used for regular token lookup)
    weights["transformer.vocab_embedding.weight"] = helper.reshape(
        params[shared_embedding_key].clone(), (vocab_size, -1))

    # --- Hidden layer weights ---
    hidden_layer_name_split = {
        'self_attn.out_proj.weight': {
            "name": f'{trtllm_attn_layer_name}.dense.weight',
            "shape": (hidden_size, hidden_size // mapping.tp_size),
            "split_dim": -1
        },
        'fc1.weight': {
            "name": 'mlp.fc.weight',
            "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
            "split_dim": 0
        },
        'fc1.bias': {
            "name": 'mlp.fc.bias',
            "shape": (ffn_hidden_size // mapping.tp_size),
            "split_dim": 0
        },
        'fc2.weight': {
            "name": 'mlp.proj.weight',
            "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
            "split_dim": -1
        },
    }

    hidden_layer_name_no_split = {
        'self_attn.out_proj.bias': {
            "name": f'{trtllm_attn_layer_name}.dense.bias',
            "shape": (hidden_size)
        },
        'self_attn_layer_norm.weight': {
            "name": f'{trtllm_attn_layernorm_name}.weight',
            "shape": None
        },
        'self_attn_layer_norm.bias': {
            "name": f'{trtllm_attn_layernorm_name}.bias',
            "shape": None
        },
        'fc2.bias': {
            "name": 'mlp.proj.bias',
            "shape": (hidden_size)
        },
        'final_layer_norm.weight': {
            "name": 'mlp_layernorm.weight',
            "shape": None
        },
        'final_layer_norm.bias': {
            "name": 'mlp_layernorm.bias',
            "shape": None
        },
    }

    if component == "decoder":
        hidden_layer_name_split.update({
            'encoder_attn.out_proj.weight': {
                "name": 'cross_attention.dense.weight',
                "shape": (hidden_size, hidden_size // mapping.tp_size),
                "split_dim": -1
            }
        })
        hidden_layer_name_no_split.update({
            'encoder_attn.out_proj.bias': {
                "name": 'cross_attention.dense.bias',
                "shape": (hidden_size)
            },
            'encoder_attn_layer_norm.weight': {
                "name": 'cross_attention_layernorm.weight',
                "shape": None
            },
            'encoder_attn_layer_norm.bias': {
                "name": 'cross_attention_layernorm.bias',
                "shape": None
            },
        })

    def get_attn_module_name(component, layer, attn_type):
        return f'language_model.model.{component}.layers.{int(layer)}.{attn_type}'

    num_layers = config.num_hidden_layers
    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        hf_layer_prefix = f'{hf_param_prefix}.layers.{layer_idx}'
        trtllm_layer_prefix = f'{trtllm_layer_name}.{local_layer_idx}'

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            weights[
                f'{trtllm_layer_prefix}.{weight_info["name"]}'] = helper.reshape(
                    helper.split(params[f'{hf_layer_prefix}.{hf_weight_name}'],
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 dim=weight_info["split_dim"]),
                    weight_info["shape"])

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            trtllm_fullname = f'{trtllm_layer_prefix}.{weight_info["name"]}'
            hf_fullname = f'{hf_layer_prefix}.{hf_weight_name}'
            weights[trtllm_fullname] = helper.reshape(
                params[hf_fullname].clone(), shape=weight_info["shape"])

        # Fuse Q/K/V for self-attention
        self_attn_module_name = get_attn_module_name(component, layer_idx,
                                                     'self_attn')
        weights.update(
            helper.fuse_qkv_one_layer(
                params, self_attn_module_name,
                f'{trtllm_layer_prefix}.{trtllm_attn_layer_name}',
                mapping.tp_size, mapping.tp_rank, "bart",
                (hidden_size * 3 // mapping.tp_size, hidden_size),
                (hidden_size * 3 // mapping.tp_size)))

        # Fuse Q/K/V for cross-attention (decoder only)
        if component == 'decoder':
            cross_attn_module_name = get_attn_module_name(
                component, layer_idx, 'encoder_attn')
            weights.update(
                helper.fuse_qkv_one_layer(
                    params, cross_attn_module_name,
                    f'{trtllm_layer_prefix}.cross_attention', mapping.tp_size,
                    mapping.tp_rank, "bart",
                    (hidden_size * 3 // mapping.tp_size, hidden_size),
                    (hidden_size * 3 // mapping.tp_size)))

    # LM head (decoder only) — uses shared embedding (tied weights)
    if component == 'decoder':
        lm_head_weights = params[shared_embedding_key].clone().detach()
        if lm_head_weights.shape[0] % mapping.tp_size != 0:
            vocab_size_padded = _utils.pad_vocab_size(config.vocab_size,
                                                      mapping.tp_size)
            pad_width = vocab_size_padded - config.vocab_size
            lm_head_weights = torch.nn.functional.pad(lm_head_weights,
                                                      (0, 0, 0, pad_width),
                                                      'constant',
                                                      value=0)
            vocab_size = vocab_size_padded
        else:
            vocab_size = config.vocab_size
        weights['lm_head.weight'] = helper.reshape(
            helper.split(lm_head_weights,
                         mapping.tp_size,
                         mapping.tp_rank,
                         dim=0),
            (vocab_size // mapping.tp_size, hidden_size))

    return weights


def convert(worker_rank, world_size, args, model_config, convert_args,
            saved_dir):
    for rank in range(worker_rank, world_size, args.workers):
        rank_config = copy.deepcopy(
            modeling_utils.PretrainedConfig.from_dict(model_config))
        rank_config.set_rank(rank)
        weights = convert_florence2_weights(config=rank_config, **convert_args)
        safetensors_torch.save_file(weights,
                                    f'{saved_dir}/rank{rank}.safetensors')


def convert_checkpoint(args):
    params = load_florence2_weights(args.model_dir)
    encoder_config, decoder_config = parse_florence2_config(args)

    saved_dir = pathlib.Path(args.output_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    encoder_saved_dir = saved_dir / "encoder"
    encoder_saved_dir.mkdir(parents=True, exist_ok=True)
    decoder_saved_dir = saved_dir / "decoder"
    decoder_saved_dir.mkdir(parents=True, exist_ok=True)

    world_size = args.tp_size * args.pp_size

    # --- Encoder config ---
    tllm_encoder_config = {
        'architecture': "EncoderModel",
        'dtype': args.dtype,
        'logits_dtype': encoder_config.logits_dtype,
        'num_hidden_layers': encoder_config.n_layer,
        'num_attention_heads': encoder_config.n_head,
        'hidden_size': encoder_config.hidden_size,
        'norm_epsilon': encoder_config.layernorm_eps,
        'vocab_size': encoder_config.vocab_size,
        'position_embedding_type': encoder_config.position_embedding_type,
        'hidden_act': encoder_config.hidden_act,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'max_position_embeddings': encoder_config.n_positions,
        'num_key_value_heads': encoder_config.n_head,
        'head_size': encoder_config.head_size,
        'has_position_embedding': encoder_config.has_position_embedding,
        'layernorm_type': encoder_config.layernorm_type,
        'has_attention_qkvo_bias': encoder_config.has_attention_qkvo_bias,
        'has_mlp_bias': encoder_config.has_mlp_bias,
        'has_model_final_layernorm':
        encoder_config.has_model_final_layernorm,
        'has_embedding_layernorm': encoder_config.has_embedding_layernorm,
        'has_embedding_scale': encoder_config.has_embedding_scale,
        'intermediate_size': encoder_config.ffn_hidden_size,
        'q_scaling': encoder_config.q_scaling,
        'layernorm_position': encoder_config.layernorm_position,
        'mlp_type': encoder_config.mlp_type,
        'relative_attention': encoder_config.relative_attention,
        'max_distance': encoder_config.max_distance,
        'num_buckets': encoder_config.num_buckets,
        'model_type': encoder_config.model_type,
    }

    with (encoder_saved_dir / "config.json").open('w') as f:
        json.dump(tllm_encoder_config, f, indent=4)

    # --- Decoder config ---
    tllm_decoder_config = {
        'architecture': "DecoderModel",
        'dtype': args.dtype,
        'logits_dtype': decoder_config.logits_dtype,
        'num_hidden_layers': decoder_config.n_layer,
        'num_attention_heads': decoder_config.n_head,
        'hidden_size': decoder_config.hidden_size,
        'norm_epsilon': decoder_config.layernorm_eps,
        'vocab_size': decoder_config.vocab_size,
        'position_embedding_type': decoder_config.position_embedding_type,
        'hidden_act': decoder_config.hidden_act,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'max_position_embeddings': decoder_config.n_positions,
        'head_size': decoder_config.head_size,
        'has_position_embedding': decoder_config.has_position_embedding,
        'layernorm_type': decoder_config.layernorm_type,
        'has_attention_qkvo_bias': decoder_config.has_attention_qkvo_bias,
        'has_mlp_bias': decoder_config.has_mlp_bias,
        'has_model_final_layernorm':
        decoder_config.has_model_final_layernorm,
        'has_embedding_layernorm': decoder_config.has_embedding_layernorm,
        'has_embedding_scale': decoder_config.has_embedding_scale,
        'intermediate_size': decoder_config.ffn_hidden_size,
        'q_scaling': decoder_config.q_scaling,
        'layernorm_position': decoder_config.layernorm_position,
        'mlp_type': decoder_config.mlp_type,
        'relative_attention': decoder_config.relative_attention,
        'max_distance': decoder_config.max_distance,
        'num_buckets': decoder_config.num_buckets,
        'model_type': decoder_config.model_type,
        'rescale_before_lm_head': decoder_config.rescale_before_lm_head,
        'encoder_hidden_size': decoder_config.encoder_hidden_size,
        'encoder_num_heads': decoder_config.encoder_num_heads,
        'encoder_head_size': decoder_config.encoder_head_size,
        'skip_cross_kv': False,
        'use_implicit_relative_attention': False,
        'decoder_start_token_id': decoder_config.decoder_start_token_id,
        'eos_token_id': decoder_config.eos_token_id,
        'bos_token_id': decoder_config.bos_token_id,
        'pad_token_id': decoder_config.pad_token_id,
    }

    with (decoder_saved_dir / "config.json").open('w') as f:
        json.dump(tllm_decoder_config, f, indent=4)

    # --- Convert weights ---
    encoder_convert_args = dict(params=params, component="encoder")
    decoder_convert_args = dict(params=params, component="decoder")

    if args.workers == 1:
        convert(0, world_size, args, tllm_encoder_config,
                encoder_convert_args, encoder_saved_dir)
        convert(0, world_size, args, tllm_decoder_config,
                decoder_convert_args, decoder_saved_dir)
    else:
        if args.workers > world_size:
            args.workers = world_size
        LOGGER.info(f'Convert checkpoint using {args.workers} workers.')
        import torch.multiprocessing as mp
        mp.spawn(convert,
                 nprocs=args.workers,
                 args=(world_size, args, tllm_encoder_config,
                       encoder_convert_args, encoder_saved_dir))
        mp.spawn(convert,
                 nprocs=args.workers,
                 args=(world_size, args, tllm_decoder_config,
                       decoder_convert_args, decoder_saved_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Florence2 HF checkpoint to TRT-LLM format",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_dir",
                        "-i",
                        type=str,
                        required=True,
                        help="Path to Florence2 HF model directory")
    parser.add_argument("--output_dir",
                        "-o",
                        type=str,
                        required=True,
                        help="Path to output TRT-LLM checkpoint directory")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Target inference dtype (default: float16)")
    parser.add_argument("--tp_size",
                        type=int,
                        default=1,
                        help="Tensor parallelism size (default: 1)")
    parser.add_argument("--pp_size",
                        type=int,
                        default=1,
                        help="Pipeline parallelism size (default: 1)")
    parser.add_argument("--workers",
                        type=int,
                        default=1,
                        help="Number of workers for conversion (default: 1)")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.datetime.now()
    run_time = stop_time - start_time
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
