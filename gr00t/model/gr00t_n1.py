# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Source: https://github.com/NVIDIA/Isaac-GR00T/blob/b211007ed6698e6642d2fd7679dabab1d97e9e6c/gr00t/model/gr00t_n1.py
# Modified by the EggHand authors: EgoVideo integration and learned projection loading.
# License: Apache-2.0; see LICENSE and NOTICE.

from dataclasses import dataclass, field
from typing import Tuple
from accelerate import init_on_device
import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import os
from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone.egovideo_backbone import EgoVideoBackbone
from .backbone.eagle_backbone import EagleBackbone
import torch.nn as nn
BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3

# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        backbone_cfg = dict(config.backbone_cfg) if config.backbone_cfg is not None else {}
        backbone_type = backbone_cfg.pop("backbone_type", "egovideo")
        self.backbone_type = backbone_type
        if backbone_type == "eagle":
            self.backbone = EagleBackbone(**backbone_cfg)
        elif backbone_type == "egovideo":
            self.backbone = EgoVideoBackbone(compute_dtype=config.compute_dtype, **backbone_cfg)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs or "action_raw" in inputs:
            action = inputs["action_raw"] if "action_raw" in inputs else inputs["action"]
            expected_action_dim = getattr(self.action_head, "input_action_dim", self.action_dim)
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == expected_action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                error_msg += f"\n{expected_action_dim=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]

            type_ok = isinstance(video, torch.Tensor)
            dtype_ok = video.dtype == torch.float32 or video.dtype == torch.float16 or video.dtype == torch.bfloat16
            shape_ok = len(video.shape) == 5 and video.shape[1] == N_COLOR_CHANNELS
            
            if not type_ok:
                error_msg += f"\nEgoVideoBackbone expects a torch.Tensor; got {type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\nEgoVideoBackbone expects floating-point video; got {video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\nEgoVideoBackbone expects (B,C,T,H,W) video; got {video.shape=}"
                detected_error = True

        if self.backbone_type == "eagle":
            has_eagle_inputs = any(key.startswith("eagle_") for key in inputs.keys())
            if not has_eagle_inputs:
                error_msg += "\nEagleBackbone expects eagle_* inputs, but none were found."
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        expected_action_dim = getattr(self.action_head, "output_action_dim", self.action_dim)
        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == expected_action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:

        backbone_inputs, action_inputs = self.prepare_input(inputs)

        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs
    
    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:

        backbone_inputs, action_inputs = self.prepare_input(inputs)

        backbone_outputs = self.backbone(backbone_inputs)

        # This prevents dtype mismatch errors (e.g., BFloat16 vs Float32) during mixed-precision training.
        backbone_outputs = tree.map_structure(lambda x: x.to(self.action_head.dtype) if torch.is_floating_point(x) else x, backbone_outputs)

        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        if "video" in inputs and inputs["video"].dim() == 5:

            # [8, 20, 3, 224, 224] -> [8, 3, 20, 224, 224]
            inputs["video"] = inputs["video"].permute(0, 2, 1, 3, 4)
        self.validate_inputs(inputs)

        #if "eagle_pixel_values" in inputs:

        #else:

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if not isinstance(x, torch.Tensor):
                return x

            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        if "backbone_cfg" not in kwargs:
            kwargs["backbone_cfg"] = {}

        backbone_cfg = kwargs.get("backbone_cfg") or {}
        backbone_type = backbone_cfg.get("backbone_type", "egovideo")

        projection_ckpt_path = os.path.join(local_model_path, "projection_weights.pth")
        print(f"Checking for projection weights at: {projection_ckpt_path}")
        if backbone_type == "egovideo" and "backbone_cfg" in kwargs and os.path.exists(projection_ckpt_path):
            kwargs["backbone_cfg"]["projection_ckpt_path"] = projection_ckpt_path
            print(f"Found projection weights at: {projection_ckpt_path}")

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )
        #ddddd

        if hasattr(pretrained_model, "backbone") and hasattr(pretrained_model.backbone, "ensure_checkpoints_loaded"):
            pretrained_model.backbone.ensure_checkpoints_loaded()

        if hasattr(pretrained_model, "backbone"):
            backbone = pretrained_model.backbone

            should_init_manually = not getattr(backbone, 'projection_ckpt_path', None)
            if should_init_manually and hasattr(backbone, 'vision_projection') and hasattr(backbone, 'text_projection'):
                print("GR00T_N1_5: No projection checkpoint found. Manually initializing projection layers...")
                import math
                with torch.no_grad():
                    for proj_layer in [backbone.vision_projection, backbone.text_projection]:

                        if isinstance(proj_layer, nn.Sequential):

                            for m in proj_layer:
                                if isinstance(m, nn.Linear):
                                    torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                                    if m.bias is not None:
                                        torch.nn.init.zeros_(m.bias)
                        elif isinstance(proj_layer, nn.Linear):

                            torch.nn.init.kaiming_uniform_(proj_layer.weight, a=math.sqrt(5))
                            if proj_layer.bias is not None:
                                torch.nn.init.zeros_(proj_layer.bias)

        if hasattr(pretrained_model, "action_head"):
            action_head = pretrained_model.action_head

            action_proj_ckpt_path = os.path.join(local_model_path, "action_projection_weights.pth")
            print(f"Checking for action projection weights at: {action_proj_ckpt_path}")
            if os.path.exists(action_proj_ckpt_path):
                print(f"Found action projection weights at: {action_proj_ckpt_path}")
                try:
                    proj_weights = torch.load(action_proj_ckpt_path, map_location="cpu")
                    if isinstance(proj_weights, dict):
                        if isinstance(proj_weights.get("state_dict"), dict):
                            proj_weights = proj_weights["state_dict"]
                        normalized = {}
                        for key, value in proj_weights.items():
                            if not isinstance(key, str):
                                continue
                            norm_key = key
                            if norm_key.startswith("module."):
                                norm_key = norm_key[len("module."):]
                            if norm_key.startswith("action_head."):
                                norm_key = norm_key[len("action_head."):]
                            normalized[norm_key] = value
                        proj_weights = normalized

                    action_state = action_head.state_dict()
                    allowed_prefixes = ("state_proj_in.", "action_proj_in.", "action_proj_out.")
                    loadable_weights = {
                        k: v
                        for k, v in proj_weights.items()
                        if k in action_state
                        and k.startswith(allowed_prefixes)
                        and isinstance(v, torch.Tensor)
                        and action_state[k].shape == v.shape
                    }
                    action_head.load_state_dict(loadable_weights, strict=False)
                    print(f"  - Loaded {len(loadable_weights)} action projection parameters.")
                except Exception as e:
                    print(f"  [ERROR] Failed to load action projection weights: {e}")

            def _proj_needs_init(layer):
                if not isinstance(layer, nn.Linear):
                    return False
                with torch.no_grad():
                    weight = layer.weight
                    if not torch.isfinite(weight).all():
                        return True
                    if weight.abs().max().item() > 1e4:
                        return True
                    if layer.bias is not None:
                        bias = layer.bias
                        if not torch.isfinite(bias).all():
                            return True
                        if bias.abs().max().item() > 1e4:
                            return True
                return False

            proj_layers = [
                ("state_proj_in", getattr(action_head, "state_proj_in", None)),
                ("action_proj_in", getattr(action_head, "action_proj_in", None)),
                ("action_proj_out", getattr(action_head, "action_proj_out", None)),
            ]
            bad_layers = [name for name, layer in proj_layers if _proj_needs_init(layer)]
            if bad_layers:
                print(
                    "GR00T_N1_5: Reinitializing action head projection layers due to invalid weights: "
                    + ", ".join(bad_layers)
                )
                action_head._init_projection_layers()

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model

# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
