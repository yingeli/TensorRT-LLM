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
"""Build a TensorRT engine for the Florence2 DaViT vision encoder.

This exports the full vision pipeline (DaViT + positional embeddings +
projection + LayerNorm) to ONNX, then builds a TRT engine. The engine
takes pixel_values [B, 3, 768, 768] and outputs image_features [B, 577, 1024].

Usage:
    python build_florence2_vision.py \
        --model_dir /path/to/Florence-2-large-ft \
        --output_dir /tmp/florence2_engine/vision \
        --max_batch_size 1
"""
import argparse
import hashlib
import importlib.util
import json
import math
import glob
import os
import sys
import types

import torch
import torch.nn as nn
import safetensors

from tensorrt_llm import _utils
from tensorrt_llm.tools import multimodal_builder


def _load_florence2_modeling_module(hf_model_dir: str):
    """Dynamically import Florence2 modeling code from the HF model dir."""
    hf_model_dir = os.path.abspath(hf_model_dir)
    module_suffix = hashlib.sha1(
        hf_model_dir.encode("utf-8")).hexdigest()[:10]
    pkg_name = f"_trtllm_florence2_{module_suffix}"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [hf_model_dir]
        sys.modules[pkg_name] = pkg

    config_mod_name = f"{pkg_name}.configuration_florence2"
    if config_mod_name not in sys.modules:
        config_path = os.path.join(hf_model_dir, "configuration_florence2.py")
        spec = importlib.util.spec_from_file_location(
            config_mod_name, config_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to load Florence2 configuration module from {config_path}"
            )
        config_mod = importlib.util.module_from_spec(spec)
        sys.modules[config_mod_name] = config_mod
        spec.loader.exec_module(config_mod)

    modeling_mod_name = f"{pkg_name}.modeling_florence2"
    if modeling_mod_name not in sys.modules:
        modeling_path = os.path.join(hf_model_dir, "modeling_florence2.py")
        spec = importlib.util.spec_from_file_location(
            modeling_mod_name, modeling_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to load Florence2 modeling module from {modeling_path}"
            )
        modeling_mod = importlib.util.module_from_spec(spec)
        sys.modules[modeling_mod_name] = modeling_mod
        spec.loader.exec_module(modeling_mod)

    return sys.modules[modeling_mod_name]


class _FP32LayerNorm(nn.Module):
    """LayerNorm that explicitly computes in FP32, matching PyTorch's internal
    behaviour.  When traced to ONNX this produces Cast→LayerNorm→Cast nodes,
    so TRT keeps FP32 for LayerNorm even when the builder uses FP16 precision.
    """

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps
        self.normalized_shape = ln.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(x.dtype)


def _replace_layernorm_with_fp32(module: nn.Module) -> int:
    """Recursively replace all nn.LayerNorm with _FP32LayerNorm."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, _FP32LayerNorm(child))
            count += 1
        else:
            count += _replace_layernorm_with_fp32(child)
    return count


class Florence2VisionWrapper(nn.Module):
    """Wraps the full Florence2 DaViT vision pipeline for ONNX/TRT export.

    Pipeline: pixel_values [B, 3, 768, 768]
      -> DaViT.forward_features_unpool()  -> [B, 576, 2048]
      -> + LearnedAbsolutePositionEmbedding2D
      -> + PositionalEmbeddingCosine1D (temporal)
      -> feature aggregation (spatial_avg_pool + temporal_avg_pool) -> [B, 577, 2048]
      -> @ image_projection [2048, 1024]
      -> LayerNorm(1024)
      -> image_features [B, 577, 1024]
    """

    def __init__(self, vision_tower, image_projection, image_proj_norm,
                 image_pos_embed, visual_temporal_embed, image_feature_source):
        super().__init__()
        self.vision_tower = vision_tower
        self.image_projection = nn.Parameter(image_projection)
        self.image_proj_norm = image_proj_norm
        self.image_pos_embed = image_pos_embed
        self.visual_temporal_embed = visual_temporal_embed
        self.image_feature_source = image_feature_source

    @classmethod
    def from_hugging_face(cls, hf_model_dir: str, dtype: str,
                          device: str = "cpu"):
        """Load the vision encoder from a Florence2 HF checkpoint."""
        hf_model_dir = os.path.abspath(hf_model_dir)

        with open(os.path.join(hf_model_dir, "config.json"), "r") as f:
            config = json.load(f)

        vision_config = config.get("vision_config", {})
        if vision_config.get("model_type") != "davit":
            raise ValueError(
                f"Expected davit vision model, got: "
                f"{vision_config.get('model_type')}")

        # Find safetensors checkpoint
        checkpoint_path = os.path.join(hf_model_dir, "model.safetensors")
        if not os.path.exists(checkpoint_path):
            shards = sorted(
                glob.glob(os.path.join(hf_model_dir, "*.safetensors")))
            if len(shards) != 1:
                raise FileNotFoundError(
                    f"Expected a single safetensors checkpoint in "
                    f"{hf_model_dir}, got: {shards}")
            checkpoint_path = shards[0]

        # Load Florence2 modeling module dynamically
        modeling_mod = _load_florence2_modeling_module(hf_model_dir)

        # Create DaViT vision tower
        davit_cfg = types.SimpleNamespace(**vision_config)
        vision_tower = modeling_mod.DaViT.from_config(config=davit_cfg)

        # Create positional embeddings
        image_dim_out = vision_config["dim_embed"][-1]

        image_pos_embed_cfg = vision_config.get("image_pos_embed", {})
        if image_pos_embed_cfg.get("type") != "learned_abs_2d":
            raise ValueError(
                f"Unsupported image_pos_embed type: "
                f"{image_pos_embed_cfg.get('type')}")
        num_pos = int(image_pos_embed_cfg.get("max_pos_embeddings", 50))
        image_pos_embed = modeling_mod.LearnedAbsolutePositionEmbedding2D(
            embedding_dim=image_dim_out, num_pos=num_pos)

        temporal_cfg = vision_config.get("visual_temporal_embedding", {})
        if temporal_cfg.get("type") != "COSINE":
            raise ValueError(
                f"Unsupported temporal embedding type: "
                f"{temporal_cfg.get('type')}")
        max_temporal = int(temporal_cfg.get("max_temporal_embeddings", 100))
        visual_temporal_embed = modeling_mod.PositionalEmbeddingCosine1D(
            embed_dim=image_dim_out, max_seq_len=max_temporal)

        image_feature_source = vision_config.get(
            "image_feature_source", ["spatial_avg_pool", "temporal_avg_pool"])

        projection_dim = config.get("projection_dim",
                                    config.get("text_config", {}).get("d_model", 1024))

        # Load weights from safetensors
        dtype_torch = _utils.str_dtype_to_torch(dtype)
        vision_tower = vision_tower.to(device=device, dtype=dtype_torch)
        image_pos_embed = image_pos_embed.to(device=device, dtype=dtype_torch)
        visual_temporal_embed = visual_temporal_embed.to(device=device,
                                                         dtype=dtype_torch)

        with safetensors.safe_open(checkpoint_path,
                                   framework="pt",
                                   device=device) as f:
            # Load vision tower weights
            vision_state_dict = {
                k.replace("vision_tower.", ""):
                f.get_tensor(k).to(device=device, dtype=dtype_torch)
                for k in f.keys() if k.startswith("vision_tower.")
            }
            missing, unexpected = vision_tower.load_state_dict(
                vision_state_dict, strict=False)
            if unexpected:
                print(f"WARNING: vision_tower unexpected keys: {unexpected}")
            if missing:
                print(f"WARNING: vision_tower missing keys: {missing}")

            # Load image projection
            image_projection = f.get_tensor("image_projection").to(
                device=device, dtype=dtype_torch)

            # Load image projection LayerNorm
            image_proj_norm = nn.LayerNorm(
                projection_dim, device=device, dtype=dtype_torch)
            image_proj_norm.weight.data.copy_(
                f.get_tensor("image_proj_norm.weight").to(
                    device=device, dtype=dtype_torch))
            image_proj_norm.bias.data.copy_(
                f.get_tensor("image_proj_norm.bias").to(
                    device=device, dtype=dtype_torch))

            # Load positional embedding weights
            image_pos_embed.row_embeddings.weight.data.copy_(
                f.get_tensor("image_pos_embed.row_embeddings.weight").to(
                    device=device, dtype=dtype_torch))
            image_pos_embed.column_embeddings.weight.data.copy_(
                f.get_tensor("image_pos_embed.column_embeddings.weight").to(
                    device=device, dtype=dtype_torch))

            # Load temporal embedding
            visual_temporal_embed.load_state_dict(
                {
                    "pos_idx_to_embed":
                    f.get_tensor(
                        "visual_temporal_embed.pos_idx_to_embed").to(
                            device=device, dtype=dtype_torch)
                },
                strict=False)

        # Delete unused sub-modules to avoid export issues
        if hasattr(vision_tower, "head"):
            del vision_tower.head
        if hasattr(vision_tower, "norms"):
            del vision_tower.norms

        return cls(
            vision_tower=vision_tower,
            image_projection=image_projection,
            image_proj_norm=image_proj_norm,
            image_pos_embed=image_pos_embed,
            visual_temporal_embed=visual_temporal_embed,
            image_feature_source=image_feature_source,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        num_frames = 1

        x = self.vision_tower.forward_features_unpool(pixel_values)
        # x: [B, num_tokens, dim_embed[-1]]
        x = x.view(batch_size * num_frames, -1, x.shape[-1])
        num_tokens = x.shape[-2]
        spatial_side = math.isqrt(int(num_tokens))
        if spatial_side * spatial_side != int(num_tokens):
            raise ValueError(
                f"Expected square feature map, got num_tokens={num_tokens}")

        x = x.view(batch_size * num_frames, spatial_side, spatial_side,
                    x.shape[-1])
        x = x + self.image_pos_embed(x)
        x = x.view(batch_size, num_frames * spatial_side * spatial_side,
                    x.shape[-1])

        # Temporal embedding
        temporal_embed = self.visual_temporal_embed(
            x.view(batch_size, num_frames, -1, x.shape[-1])[:, :, 0])
        x = (x.view(batch_size, num_frames, -1, x.shape[-1])
             + temporal_embed.view(1, num_frames, 1, x.shape[-1]))

        # Feature aggregation
        x_feat_dict = {
            'spatial_avg_pool': x.mean(dim=2),
            'temporal_avg_pool': x.mean(dim=1),
            'last_frame': x[:, -1],
        }
        selected = [x_feat_dict[src] for src in self.image_feature_source]
        image_features = torch.cat(selected, dim=1)

        # Projection + LayerNorm
        image_features = image_features @ self.image_projection
        image_features = self.image_proj_norm(image_features)
        return image_features


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build TRT engine for Florence2 DaViT vision encoder")
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to Florence2 HF model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Output directory for vision TRT engine")
    parser.add_argument("--max_batch_size",
                        type=int,
                        default=1,
                        help="Maximum batch size (default: 1)")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16",
                        choices=["float16", "float32"],
                        help="Data type (default: float16)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print(f"Loading Florence2 vision encoder from {args.model_dir}")
    dtype_torch = _utils.str_dtype_to_torch(args.dtype)
    wrapper = Florence2VisionWrapper.from_hugging_face(
        args.model_dir, dtype=args.dtype, device="cpu")
    wrapper.eval()

    # Replace nn.LayerNorm with FP32-computing variants.  PyTorch's
    # LayerNorm always upcasts to FP32 internally, but TRT's FP16 mode
    # doesn't — the resulting numerical drift gets amplified by beam
    # search.  By inserting explicit Cast(FP16→FP32)→LN→Cast(FP32→FP16)
    # into the ONNX graph we can build the TRT engine in FP16 while
    # keeping LayerNorm in FP32 — best of both worlds.
    n_replaced = _replace_layernorm_with_fp32(wrapper)
    print(f"Replaced {n_replaced} LayerNorm modules with FP32 variants")

    # Export to ONNX
    # Florence2 always resizes to 768x768, so spatial dims are fixed.
    # Only the batch dimension is dynamic.
    dummy_input = torch.randn(1, 3, 768, 768, dtype=dtype_torch)

    onnx_dir = os.path.join(args.output_dir, "onnx")
    print(f"Exporting ONNX to {onnx_dir}")
    multimodal_builder.export_onnx(
        wrapper,
        dummy_input,
        onnx_dir,
        onnx_name="vision_encoder.onnx",
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={"pixel_values": {0: "batch"}},
    )

    # Build TRT engine in FP16 — the explicit FP32 Cast nodes around
    # LayerNorm in the ONNX graph ensure those ops stay in FP32.
    print(f"Building TRT engine in {args.output_dir}")
    multimodal_builder.build_trt_engine(
        "florence2_vision",
        [3, 768, 768],
        onnx_dir,
        args.output_dir,
        args.max_batch_size,
        dtype=dtype_torch,
        onnx_name="vision_encoder.onnx",
    )

    print(f"Vision TRT engine built successfully: {args.output_dir}")


if __name__ == "__main__":
    main()
