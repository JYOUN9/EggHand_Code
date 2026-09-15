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

# Source: https://github.com/NVIDIA/Isaac-GR00T/blob/b211007ed6698e6642d2fd7679dabab1d97e9e6c/gr00t/model/transforms.py
# Modified by the EggHand authors: camera alignment and raw state/action handling for forecasting.
# License: Apache-2.0; see LICENSE and NOTICE.

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH

def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language

def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor

def collate(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            video_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                curr_video_inputs = v["video_inputs"]
                text_list += curr_text_list
                if curr_image_inputs is not None:
                    image_inputs += curr_image_inputs
                if curr_video_inputs is not None:
                    video_inputs += curr_video_inputs
            eagle_inputs = eagle_processor(
                text=text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True
            )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
        elif key == "text":
            # Keep raw text strings so downstream tokenizers can consume them.
            batch[key] = values
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch

class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)

class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1

        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """

        video_frames = batch["images"]  # [V, T, C, H, W]

        np_video = rearrange(video_frames, "1 t c h w -> t c h w")
        
        text_content = []

        # handle language
        lang = batch["language"]
        text_content.append({"type": "text", "text": lang})

        eagle_video_frames = [Image.fromarray(np.transpose(frame, (1, 2, 0))) for frame in np_video]

        eagle_video = [{"type": "video", "video": eagle_video_frames}]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_video + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""

        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )

        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        if not isinstance(raw_language, str):
            raw_language = str(raw_language)

        if self.formalize_language:
            raw_language = formalize_language(raw_language)

        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        if isinstance(state, torch.Tensor):
            state_mask = torch.zeros_like(state, dtype=torch.bool)
            state_mask[:, :n_state_dims] = True
        else:
            state_mask = np.zeros_like(state).astype(bool)
            state_mask[:, :n_state_dims] = True

        # One state token per observed timestep.
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        if n_action_dims > self.max_action_dim:
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            if "action_mask" in data:
                actions_mask = data["action_mask"]
                if isinstance(actions_mask, torch.Tensor):
                    actions_mask = actions_mask.cpu().numpy()
            else:
                actions_mask = actions != 0
            return actions, actions_mask, n_action_tokens

        if n_action_dims < self.max_action_dim:
            # Pad the channel dimension
            actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")
        else:
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()

        if "action_mask" in data:
            actions_mask = data["action_mask"]
            if n_action_dims < self.max_action_dim:
                # Pad it as well
                actions_mask = np.pad(
                    actions_mask,
                    ((0, 0), (0, self.max_action_dim - n_action_dims)),
                    "constant",
                    constant_values=False,
                )
            elif isinstance(actions_mask, torch.Tensor):
                actions_mask = actions_mask.cpu().numpy()
        else:
            # Missing coordinates remain excluded from the action loss.

            actions_mask = actions != 0

        return actions, actions_mask, n_action_tokens

    def _rz(self, angle):
        """
        Returns a 3x3 rotation matrix around the Z-axis.
        """
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def align_sequences_to_first_frame(self, state_data, action_data, num_joints=42):
        """Align state (20, 138) and action (10, 138) to the first observed camera pose."""
        # Determine if input is tensor to convert back later
        is_state_tensor = isinstance(state_data, torch.Tensor)
        is_action_tensor = isinstance(action_data, torch.Tensor)
        device = state_data.device if is_state_tensor else None

        # Convert to numpy for consistent operations
        state_data_np = state_data.cpu().numpy() if is_state_tensor else state_data.copy()
        action_data_np = action_data.cpu().numpy() if is_action_tensor else action_data.copy()

        # Extrinsics are 3x4 matrix (Rotation and Translation from World to Camera)
        # Reference is the oldest observed frame (offset -19), after edge padding.
        extrinsic_t0_flat = state_data_np[0, 126:138]
        T_world_to_camera_t0 = np.eye(4)
        T_world_to_camera_t0[:3, :] = extrinsic_t0_flat.reshape(3, 4)

        # T_camera_to_world_t0: Transformation from Camera frame at t=0 to World frame
        T_camera_to_world_t0 = np.linalg.inv(T_world_to_camera_t0)

        # Camera position in world coordinates at t=0
        t_camera_in_world_t0 = T_camera_to_world_t0[:3, 3]
        # Camera rotation in world coordinates at t=0
        R_camera_in_world_t0 = T_camera_to_world_t0[:3, :3]

        # Camera's forward vector in world coordinates (assuming camera's forward is +Z in its own frame)
        forward_t0_world = np.dot(R_camera_in_world_t0, np.array([0.0, 0.0, 1.0]).T)

        # Calculate rotation to align camera's forward vector (X-Y plane projection) with the global X-axis
        # This is a rotation around the Z-axis
        angle_to_align_forward_to_x = -np.arctan2(forward_t0_world[1], forward_t0_world[0])
        R_align_to_x = self._rz(angle_to_align_forward_to_x) # 3x3 rotation matrix

        # Construct the canonical transformation matrix (from original world to canonical world)
        # The canonical frame has its origin at t_camera_in_world_t0 and its X-axis aligned with the camera's forward direction.

        T_canonical_from_world = np.eye(4)
        T_canonical_from_world[:3, :3] = R_align_to_x
        T_canonical_from_world[:3, 3] = -np.dot(R_align_to_x, t_camera_in_world_t0)

        # The transform_joints function expects a 4x4 transformation matrix
        # that transforms points from the *original* coordinate system to the *new* coordinate system.
        T_ref_final = T_canonical_from_world

        def transform_joints_local(data_np_input, transform_matrix_local):

            T, D = data_np_input.shape

            joints = data_np_input[:, :126].reshape(T, num_joints, 3)
            # Missing zero joints are points too: retain the camera translation.
            ones = np.ones((T, num_joints, 1))
            joints_h = np.concatenate([joints, ones], axis=-1)

            joints_aligned = []
            for t in range(T):
                transformed = (transform_matrix_local @ joints_h[t].T).T
                joints_aligned.append(transformed[:, :3])
            
            res = data_np_input.copy()
            res[:, :126] = np.stack(joints_aligned).reshape(T, -1)
            return res

        new_state_np = transform_joints_local(state_data_np, T_ref_final)
        new_action_np = transform_joints_local(action_data_np, T_ref_final)

        # Convert back to torch.Tensor if original inputs were tensors
        if is_state_tensor:
            new_state = torch.from_numpy(new_state_np).to(device)
        else:
            new_state = new_state_np

        if is_action_tensor:
            new_action = torch.from_numpy(new_action_np).to(device)
        else:
            new_action = new_action_np
        return new_state, new_action

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}
        
        # 1) Prepare video and language with vlm processing.
        video_frames = self._prepare_video(data)
        video_frames_uint8 = video_frames.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": video_frames_uint8, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)
        
        # 2) Prepare state
        data["state"], data["action"] = self.align_sequences_to_first_frame(
            data["state"], data["action"]
        )
        data["state_raw"] = data["state"]
        data["action_raw"] = data["action"]
        if "action_mask" in data and "action_mask_raw" not in data:
            data["action_mask_raw"] = data["action_mask"]
        
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask
        if "state_raw" in data:
            transformed_data["state_raw"] = data["state_raw"]
        transformed_data["text"] = language
        video_for_backbone = video_frames_uint8.astype(np.float32) / 255.0
        if video_for_backbone.shape[0] == 1:
            video_for_backbone = video_for_backbone[0]
        transformed_data["video"] = video_for_backbone

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask
            if "action_raw" in data:
                transformed_data["action_raw"] = data["action_raw"]
            if "action_mask_raw" in data:
                transformed_data["action_mask_raw"] = data["action_mask_raw"]

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)

        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
