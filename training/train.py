# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 EggHand authors.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/NVIDIA/Isaac-GR00T/blob/b211007ed6698e6642d2fd7679dabab1d97e9e6c/getting_started/3_1_new_embodiment_finetuning.ipynb
# Modified by the EggHand authors: forecasting data, projections, losses and evaluation.
# License text: LICENSE.

from pathlib import Path
import os
import json
import shutil
import sys
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)
from utils.common import configure_training
RUN_ARGS = configure_training()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from gr00t.data.schema import EmbodimentTag

RESUME_TRAINING = RUN_ARGS.resume

dataset_path = os.environ["EGGHAND_DATASET_PATH"]
embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, set_seed
from accelerate.utils import convert_to_fp32

# Seed newly initialized projections as well as the Trainer data order.
set_seed(42)
# Keep FP32 parameters/activations and use TF32 tensor cores for supported CUDA operations.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import builtins
import types
import numpy as np
import numpy
import numpy.core.multiarray
import numpy.dtypes

SAFE_GLOBALS = [
    builtins.tuple,
    builtins.list,
    builtins.dict,
    builtins.set,
    builtins.frozenset,
    builtins.slice,
    builtins.range,
    builtins.bool,
    builtins.int,
    builtins.float,
    builtins.complex,
    builtins.str,
    builtins.bytes,
    builtins.bytearray,
    memoryview,
    type(None),
    type(Ellipsis),
    types.SimpleNamespace,
    types.MappingProxyType,
    numpy.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.generic,
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]

_dtype_names = [
    "ByteDType",
    "ShortDType",
    "Int32DType",
    "Int64DType",
    "UInt32DType",
    "UInt64DType",
    "Float32DType",
    "Float64DType",
    "Complex64DType",
    "Complex128DType",
]
for _dtype_name in _dtype_names:
    _dtype_cls = getattr(numpy.dtypes, _dtype_name, None)
    if _dtype_cls is not None:
        SAFE_GLOBALS.append(_dtype_cls)

try:
    from numpy.random import mtrand
    SAFE_GLOBALS.append(mtrand.RandomState)
except Exception:
    pass

try:
    from numpy.random import _pickle as numpy_pickle
    SAFE_GLOBALS.append(numpy_pickle.__RandomState_ctor)
except Exception:
    pass

try:
    from numpy.random import Generator, BitGenerator, PCG64, MT19937, Philox, SFC64
    SAFE_GLOBALS.extend([Generator, BitGenerator, PCG64, MT19937, Philox, SFC64])
except Exception:
    pass

torch.serialization.add_safe_globals(SAFE_GLOBALS)

from gr00t.data.dataset import ModalityConfig

state_keys = [
    "right_wrist",
    "right_thumb_1", "right_thumb_2", "right_thumb_3", "right_thumb_4",
    "right_index_1", "right_index_2", "right_index_3", "right_index_4",
    "right_middle_1", "right_middle_2", "right_middle_3", "right_middle_4",
    "right_ring_1", "right_ring_2", "right_ring_3", "right_ring_4",
    "right_pinky_1", "right_pinky_2", "right_pinky_3", "right_pinky_4",
    "left_wrist",
    "left_thumb_1", "left_thumb_2", "left_thumb_3", "left_thumb_4",
    "left_index_1", "left_index_2", "left_index_3", "left_index_4",
    "left_middle_1", "left_middle_2", "left_middle_3", "left_middle_4",
    "left_ring_1", "left_ring_2", "left_ring_3", "left_ring_4",
    "left_pinky_1", "left_pinky_2", "left_pinky_3", "left_pinky_4",
    "extrinsics",
]

action_keys = [k for k in state_keys if k != "extrinsics"]
action_dim = len(action_keys) * 3
EXTRINSICS_DIM = 12
raw_state_dim = action_dim + EXTRINSICS_DIM
model_state_dim = 64
model_action_dim = 32

# Four RGB observations span the 20-step state history.
video_modality = ModalityConfig(
    delta_indices=[-18, -12, -6, 0],
    modality_keys=["video.ego_view"]
)

state_modality = ModalityConfig(
    delta_indices=list(range(-19, 1)),
    modality_keys=[f"state.{k}" for k in state_keys]
)

# The target window includes the current timestep and nine future timesteps.
action_modality = ModalityConfig(
    delta_indices=list(range(10)),  # [0, 1, ..., 9]
    modality_keys=[f"action.{k}" for k in action_keys]
)

language_modality = ModalityConfig(
    delta_indices=[0],

    modality_keys=["annotation.task_description"]
)

modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "language": language_modality,
}

from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import (VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy)
from gr00t.data.transform.state_action import (StateActionToTensor, StateActionTransform)
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.data.transform.base import ModalityTransform
from pydantic import Field
from typing import Any, Dict, Tuple

class ScalarActionMaskOverrideTransform(ModalityTransform):
    """
    Overrides the action_mask based on whether the action value is 0.0.
    This is useful when the dataset's default mask is unreliable.
    This transform should be placed AFTER ConcatTransform.
    """
    apply_to: list[str] = Field(default_factory=lambda: ["action"])

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if "action" not in data:
            return data
        
        action_tensor = data["action"]
        # The action tensor is already concatenated, so we create a single mask.
        mask = (action_tensor != 0.0)
        if isinstance(mask, torch.Tensor):
            if mask.sum().item() == 0:
                mask = torch.ones_like(mask, dtype=torch.bool)
        else:
            if mask.sum() == 0:
                mask = np.ones_like(mask, dtype=bool)
        data["action_mask"] = mask
        
        return data

class PostGR00TNormalize(ModalityTransform):
    """Normalize each joint after camera alignment, using its slice in the concatenated tensor."""
    state_concat_order: list[str]
    action_concat_order: list[str]
    normalization_modes: dict[str, str]

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for modality in ["state", "action"]:
            if modality not in data:
                continue
            
            # (B, T, D) or (T, D)
            tensor = data[modality]
            concat_order = self.state_concat_order if modality == "state" else self.action_concat_order
            
            # Metadata statistics
            stats_dict = self.dataset_metadata.statistics.state if modality == "state" else self.dataset_metadata.statistics.action
            modality_meta = self.dataset_metadata.modalities.state if modality == "state" else self.dataset_metadata.modalities.action

            is_tensor = isinstance(tensor, torch.Tensor)

            start_idx = 0
            for key in concat_order:
                # key e.g. "state.right_wrist" -> subkey "right_wrist"
                subkey = key.split(".", 1)[1]
                
                if subkey not in modality_meta:
                    continue
                    
                dim = modality_meta[subkey].shape[0]
                end_idx = start_idx + dim

                if start_idx >= tensor.shape[-1] or end_idx > tensor.shape[-1]:
                    break

                if key in self.normalization_modes and self.normalization_modes[key] == "min_max":
                    if subkey in stats_dict:
                        stat = stats_dict[subkey]
                        
                        if is_tensor:
                            min_val = torch.tensor(stat.min, device=tensor.device, dtype=tensor.dtype)
                            max_val = torch.tensor(stat.max, device=tensor.device, dtype=tensor.dtype)

                            scale = max_val - min_val
                            scale[scale == 0] = 1.0
                            
                            # In-place update: (x - min) / (max - min) * 2 - 1
                            tensor[..., start_idx:end_idx] = (tensor[..., start_idx:end_idx] - min_val) / scale * 2 - 1
                        else:
                            min_val = np.array(stat.min, dtype=tensor.dtype)
                            max_val = np.array(stat.max, dtype=tensor.dtype)
                            
                            scale = max_val - min_val
                            scale[scale == 0] = 1.0
                            tensor[..., start_idx:end_idx] = (tensor[..., start_idx:end_idx] - min_val) / scale * 2 - 1
                                
                start_idx = end_idx

            mask_key = f"{modality}_mask"
            if mask_key in data:
                mask = data[mask_key]
                if is_tensor:
                    if not isinstance(mask, torch.Tensor):
                        mask = torch.from_numpy(mask).to(tensor.device)
                    tensor = tensor * mask.to(tensor.dtype)
                else:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    tensor = tensor * mask.astype(tensor.dtype)
                
            data[modality] = tensor
            
        return data

class RelativePoseLossTrainer(Trainer):
    """Compute the configured absolute, wrist-relative and pairwise hand-pose losses."""

    lambda_abs: float = 0.6
    lambda_rel: float = 0.2
    lambda_pair: float = 0.2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.joint_keys = [k for k in state_keys if k != "extrinsics"]
        self.num_joints = len(self.joint_keys)

        self.wrist_indices = {
            "right": self.joint_keys.index("right_wrist"),
            "left": self.joint_keys.index("left_wrist"),
        }

        self.hand_indices = {
            "right": [i for i, k in enumerate(self.joint_keys) if k.startswith("right_")],
            "left": [i for i, k in enumerate(self.joint_keys) if k.startswith("left_")],
        }

        self.hand_indices["right"].remove(self.wrist_indices["right"])
        self.hand_indices["left"].remove(self.wrist_indices["left"])
        joint_hands = []
        for k in self.joint_keys:
            if k.startswith("right_"):
                joint_hands.append(0)
            elif k.startswith("left_"):
                joint_hands.append(1)
            else:
                joint_hands.append(-1)
        same_hand = []
        for i in range(self.num_joints):
            row = []
            for j in range(self.num_joints):
                row.append(joint_hands[i] == joint_hands[j] and joint_hands[i] != -1)
            same_hand.append(row)
        self.same_hand_mask = torch.tensor(same_hand, dtype=torch.bool)

        self.lambda_abs = kwargs.pop("lambda_abs", self.lambda_abs)
        self.lambda_rel = kwargs.pop("lambda_rel", self.lambda_rel)
        self.lambda_pair = kwargs.pop("lambda_pair", self.lambda_pair)

    def to_relative_pose(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, J * 3) actions to wrist-relative coordinates for each hand."""
        B, T, D = actions.shape
        actions_reshaped = actions.view(B, T, self.num_joints, 3)

        relative_actions = actions_reshaped.clone()

        for hand_name, wrist_idx in self.wrist_indices.items():

            wrist_coords = actions_reshaped[:, :, wrist_idx, :].unsqueeze(2)

            joint_indices = self.hand_indices[hand_name]

            relative_actions[:, :, joint_indices, :] = relative_actions[:, :, joint_indices, :] - wrist_coords

        return relative_actions.view(B, T, D)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the configured hand-pose loss from flow-matching predictions."""

        is_training = model.training

        unwrapped_model = model.module if hasattr(model, "module") else model
        # Accelerate wraps forward(), so the custom action path needs its own AMP context.
        with self.accelerator.autocast():
            outputs = unwrapped_model.get_action(inputs)
        outputs = convert_to_fp32(outputs)

        model.train(is_training)

        pred_actions_full = outputs["action_pred"] # (B, T, D_total)
        gt_actions_full = inputs["action"]         # (B, T, D_total)
        action_mask_full = inputs["action_mask"]   # (B, T, D_total)

        dim_joints = self.num_joints * 3
        pred_actions_abs = pred_actions_full[..., :dim_joints]
        gt_actions_abs = gt_actions_full[..., :dim_joints]
        action_mask = action_mask_full[..., :dim_joints]

        pred_actions_rel = self.to_relative_pose(pred_actions_abs)
        gt_actions_rel = self.to_relative_pose(gt_actions_abs)

        loss_abs = F.l1_loss(pred_actions_abs, gt_actions_abs, reduction='none')
        loss_abs = (loss_abs * action_mask).sum() / action_mask.sum().clamp(min=1)

        B, T, D = gt_actions_abs.shape
        gt_abs_reshaped = gt_actions_abs.view(B, T, self.num_joints, 3)

        validity_mask_per_axis = torch.ones_like(gt_abs_reshaped, dtype=torch.bool)

        for hand_name, wrist_idx in self.wrist_indices.items():

            wrist_abs_gt = gt_abs_reshaped[:, :, wrist_idx, :].unsqueeze(2)

            finger_indices = self.hand_indices[hand_name]
            finger_abs_gt = gt_abs_reshaped[:, :, finger_indices, :]

            invalid_mask = (wrist_abs_gt == 0.0) | (finger_abs_gt == 0.0)
            validity_mask_per_axis[:, :, finger_indices, :] = ~invalid_mask

        final_mask = action_mask & validity_mask_per_axis.view(B, T, D)
        loss_rel = F.l1_loss(pred_actions_rel, gt_actions_rel, reduction='none')
        loss_rel = (loss_rel * final_mask).sum() / final_mask.sum().clamp(min=1)

        N = self.num_joints
        pred_actions_reshaped = pred_actions_abs.view(B, T, N, 3)
        gt_actions_reshaped = gt_actions_abs.view(B, T, N, 3)
        d_ij_t = torch.cdist(gt_actions_reshaped, gt_actions_reshaped, p=2)
        d_hat_ij_t = torch.cdist(pred_actions_reshaped, pred_actions_reshaped, p=2)
        squared_diff_pair = (d_hat_ij_t - d_ij_t)**2

        joint_valid_mask = action_mask.view(B, T, N, 3).all(dim=-1)

        pairwise_valid_mask = joint_valid_mask.unsqueeze(-1) & joint_valid_mask.unsqueeze(-2)
        same_hand_mask = self.same_hand_mask.to(pairwise_valid_mask.device)
        pairwise_valid_mask = pairwise_valid_mask & same_hand_mask.view(1, 1, N, N)

        loss_pair = (squared_diff_pair * pairwise_valid_mask).sum() / pairwise_valid_mask.sum().clamp(min=1)

        final_loss = (self.lambda_abs * loss_abs +
                      self.lambda_rel * loss_rel +
                      self.lambda_pair * loss_pair)

        self.log({"loss_abs": loss_abs.item(), "loss_rel": loss_rel.item()})
        if self.lambda_pair > 0:
            self.log({"loss_pair": loss_pair.item()})

        return (final_loss, outputs) if return_outputs else final_loss

# select the transforms you want to apply to the data
to_apply_transforms = ComposedModalityTransform(
    transforms=[
        # video transforms
        VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=video_modality.modality_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
        VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
        VideoToNumpy(apply_to=video_modality.modality_keys),

        # state transforms
        StateActionToTensor(apply_to=state_modality.modality_keys),

        # action transforms
        StateActionToTensor(apply_to=action_modality.modality_keys),

        # ConcatTransform
        ConcatTransform(
            video_concat_order=video_modality.modality_keys,
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
        ),
        # Override action_mask to handle invalid 0.0 values after concatenation
        ScalarActionMaskOverrideTransform(),

        # model-specific transform
        GR00TTransform(
            state_horizon=len(state_modality.delta_indices),
            action_horizon=len(action_modality.delta_indices),
            max_state_dim=model_state_dim,
            max_action_dim=action_dim,
        ),

        PostGR00TNormalize(
            apply_to=["state", "action"],
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
            normalization_modes={
                **{k: "min_max" for k in state_modality.modality_keys if "extrinsics" not in k},
                **{k: "min_max" for k in action_modality.modality_keys if "extrinsics" not in k}
            }
        ),
    ]
)



from gr00t.data.dataset import LeRobotSingleDataset

train_dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag=embodiment_tag,
    video_backend="torchvision_av",
    transforms=to_apply_transforms,
)






import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gr00t.model.gr00t_n1 import GR00T_N1_5

BASE_MODEL_PATH = os.environ["GR00T_BASE_MODEL_PATH"]
TUNE_LLM = False            # Whether to tune the LLM
TUNE_VISUAL = False          # Whether to tune the visual encoder
TUNE_PROJECTOR = True       # Whether to tune the projector
TUNE_DIFFUSION_MODEL = True # Whether to tune the diffusion model
BACKBONE_TYPE = "egovideo"  # "egovideo" or "eagle"
BACKBONE_EMBED_DIM = 2048
EAGLE_PATH = None
EAGLE_SELECT_LAYER = -1
EAGLE_PROJECT_TO_DIM = BACKBONE_EMBED_DIM

action_head_cfg = {
    "action_dim": model_action_dim,
    "action_horizon": len(action_modality.delta_indices),
    "add_pos_embed": True,
    "backbone_embedding_dim": BACKBONE_EMBED_DIM,
    "diffusion_model_cfg": {
        "attention_head_dim": 48,
        "cross_attention_dim": BACKBONE_EMBED_DIM,
        "dropout": 0.2,
        "final_dropout": True,
        "interleave_self_attention": True,
        "norm_type": "ada_norm",
        "num_attention_heads": 32,
        "num_layers": 16,
        "output_dim": 1024,
        "positional_embeddings": None
    },
    "hidden_size": 1024,
    "input_embedding_dim": 1536,
    "max_action_dim": model_action_dim,
    "max_state_dim": model_state_dim,
    "input_state_dim": raw_state_dim,
    "input_action_dim": action_dim,
    "output_action_dim": action_dim,
    "model_dtype": "float32",
    "noise_beta_alpha": 1.5,
    "noise_beta_beta": 1.0,
    "noise_s": 0.999,
    "num_inference_timesteps": 4,
    "num_target_vision_tokens": 32,
    "num_timestep_buckets": 1000,
    "tune_diffusion_model": True,
    "tune_projector": True,
    "use_vlln": True,
    "vl_self_attention_cfg": {
        "attention_head_dim": 64,
        "dropout": 0.2,
        "final_dropout": True,
        "num_attention_heads": 32,
        "num_layers": 4,
        "positional_embeddings": None
    }
}

if BACKBONE_TYPE == "egovideo":

    EGOVIDEO_CKPT_PATH = os.environ["EGOVIDEO_CKPT_PATH"]
    backbone_num_frames = len(video_modality.delta_indices)
    print(f"[INFO] Setting EgoVideoBackbone num_frames to {backbone_num_frames} based on video_modality delta_indices.")
    backbone_cfg = {
        "backbone_type": "egovideo",
        "tune_llm": TUNE_LLM,
        "tune_visual": TUNE_VISUAL,
        "num_frames": backbone_num_frames,
        "vision_output_dim": 768,
        "project_to_dim": BACKBONE_EMBED_DIM,
        "egovideo_ckpt_path": EGOVIDEO_CKPT_PATH,
    }
elif BACKBONE_TYPE == "eagle":
    backbone_cfg = {
        "backbone_type": "eagle",
        "tune_llm": TUNE_LLM,
        "tune_visual": TUNE_VISUAL,
        "select_layer": EAGLE_SELECT_LAYER,
        "project_to_dim": EAGLE_PROJECT_TO_DIM,
    }
    if EAGLE_PATH is not None:
        backbone_cfg["eagle_path"] = EAGLE_PATH
else:
    raise ValueError(f"Unsupported BACKBONE_TYPE: {BACKBONE_TYPE}")

model = GR00T_N1_5.from_pretrained(
    action_horizon=len(action_modality.delta_indices),
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    action_dim=action_dim,

    backbone_cfg=backbone_cfg,
    action_head_cfg=action_head_cfg, 
    ignore_mismatched_sizes=True,
    tune_llm=TUNE_LLM,  # backbone's LLM
    tune_visual=TUNE_VISUAL,  # backbone's vision tower
    tune_projector=TUNE_PROJECTOR,  # action head's projector
    tune_diffusion_model=TUNE_DIFFUSION_MODEL,  # action head's DiT
)

model.to(device)

def _maybe_reinit_proj(layer, name):
    if not isinstance(layer, torch.nn.Linear):
        return
    with torch.no_grad():
        weight = layer.weight
        bad = (not torch.isfinite(weight).all())
        if weight.numel() > 0 and weight.abs().max().item() > 1e4:
            bad = True
        if layer.bias is not None:
            bias = layer.bias
            if not torch.isfinite(bias).all():
                bad = True
            if bias.numel() > 0 and bias.abs().max().item() > 1e4:
                bad = True
        if bad:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            print(f"[INFO] Reinitialized {name} due to invalid weights.")

if not RESUME_TRAINING and hasattr(model, "action_head"):
    _maybe_reinit_proj(model.action_head.state_proj_in, "action_head.state_proj_in")
    _maybe_reinit_proj(model.action_head.action_proj_in, "action_head.action_proj_in")
    _maybe_reinit_proj(model.action_head.action_proj_out, "action_head.action_proj_out")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Trainable ratio      : {100 * trainable / total:.2f}%")
    return total, trainable

count_parameters(model)



from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

output_dir = os.environ["EGGHAND_OUTPUT_DIR"]
per_device_train_batch_size = RUN_ARGS.batch_size
max_steps = 200000 
report_to = "tensorboard"
dataloader_num_workers = 8

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=None,
    remove_unused_columns=False,
    deepspeed="",
    gradient_checkpointing=False,
    fp16=False,
    bf16=False,
    tf32=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=RUN_ARGS.gradient_accumulation_steps,
    dataloader_num_workers=dataloader_num_workers,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=False,
    optim="adamw_torch",
    adam_beta1=0.90,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=3e-5,
    weight_decay=1e-7,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    logging_strategy="steps",
    logging_steps=100,
    num_train_epochs=10,
    save_strategy="steps", 
    save_steps=3230,
    save_total_limit=10,
    report_to=report_to,
    seed=42,
    disable_tqdm=False,
    do_eval=False,
    ddp_find_unused_parameters=False,
)

def save_experiment_config(config_path: str) -> None:
    cfg = {
        "script": os.path.basename(__file__),
        "dataset_path": dataset_path,
        "embodiment_tag": embodiment_tag.value,
        "resume_training": RESUME_TRAINING,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "model_dims": {
            "raw_state_dim": raw_state_dim,
            "model_state_dim": model_state_dim,
            "model_action_dim": model_action_dim,
            "action_dim": action_dim,
        },
        "modality_configs": {
            "video": {
                "delta_indices": list(video_modality.delta_indices),
                "modality_keys": list(video_modality.modality_keys),
            },
            "state": {
                "delta_indices": list(state_modality.delta_indices),
                "modality_keys": list(state_modality.modality_keys),
            },
            "action": {
                "delta_indices": list(action_modality.delta_indices),
                "modality_keys": list(action_modality.modality_keys),
            },
            "language": {
                "delta_indices": list(language_modality.delta_indices),
                "modality_keys": list(language_modality.modality_keys),
            },
        },
        "backbone_type": BACKBONE_TYPE,
        "backbone_cfg": backbone_cfg,
        "action_head_cfg": action_head_cfg,
        "training_args": training_args.to_dict(),
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    print(f"[INFO] Saved experiment config to {config_path}")

def save_experiment_metadata(exp_cfg_dir: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    os.makedirs(exp_cfg_dir, exist_ok=True)
    metadata_path = os.path.join(exp_cfg_dir, "metadata.json")
    metadata_json = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_json = json.load(f)
    metadata_json.update({train_dataset.tag: train_dataset.metadata.model_dump(mode="json")})
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=4)
    print(f"[INFO] Saved dataset metadata to {metadata_path}")

exp_cfg_dir = os.path.join(training_args.output_dir, "experiment_cfg")
save_experiment_config(os.path.join(exp_cfg_dir, "experiment_config.json"))
save_experiment_config(os.path.join(training_args.output_dir, "experiment_config.json"))
save_experiment_metadata(exp_cfg_dir)

class CopyExperimentCfgCallback(TrainerCallback):
    """Copy experiment_cfg into each checkpoint so inference can load metadata."""
    def __init__(self, exp_cfg_dir: str):
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_output_dir) and os.path.isdir(self.exp_cfg_dir):
            shutil.copytree(
                self.exp_cfg_dir,
                os.path.join(checkpoint_output_dir, "experiment_cfg"),
                dirs_exist_ok=True,
            )

class SaveProjectionCallback(TrainerCallback):
    """Save the learned vision, text, state and action projections alongside the model."""
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        if model is None:
            print("[SaveProjectionCallback] WARN: Model not found in kwargs.")
            return

        backbone = getattr(model, "backbone", None)
        if backbone is None:
            print("[SaveProjectionCallback] WARN: Model has no backbone.")
        else:
            proj_state_dict = {}
            for name in ("vision_projection", "text_projection", "eagle_linear"):
                layer = getattr(backbone, name, None)
                if layer is None:
                    continue
                for key, value in layer.state_dict().items():
                    proj_state_dict[f"{name}.{key}"] = value.detach().cpu()

            if proj_state_dict:
                output_path = os.path.join(
                    args.output_dir, f"checkpoint-{state.global_step}", "projection_weights.pth"
                )
                torch.save(proj_state_dict, output_path)
                print(f"[SaveProjectionCallback] Saved projection layer weights to {output_path}")

        action_head = getattr(model, "action_head", None)
        if action_head is None:
            print("[SaveProjectionCallback] WARN: Model has no action_head.")
        else:
            action_proj_state_dict = {}
            for name in ("state_proj_in", "action_proj_in", "action_proj_out"):
                layer = getattr(action_head, name, None)
                if layer is None:
                    continue
                for key, value in layer.state_dict().items():
                    action_proj_state_dict[f"{name}.{key}"] = value.detach().cpu()

            if action_proj_state_dict:
                output_path = os.path.join(
                    args.output_dir, f"checkpoint-{state.global_step}", "action_projection_weights.pth"
                )
                torch.save(action_proj_state_dict, output_path)
                print(f"[SaveProjectionCallback] Saved action projection weights to {output_path}")


from gr00t.model.transforms import DefaultDataCollator

trainer = RelativePoseLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DefaultDataCollator(),
    callbacks=[SaveProjectionCallback(), CopyExperimentCfgCallback(exp_cfg_dir)],
)

checkpoint_to_resume = None
if RESUME_TRAINING:

    last_checkpoint = get_last_checkpoint(output_dir)
    if os.path.isdir(output_dir) and last_checkpoint is not None:
        checkpoint_to_resume = get_last_checkpoint(output_dir)
        print(f"[TRAINING] resuming the latest saved state in {output_dir}: {checkpoint_to_resume}")
    else:
        print(f"[TRAINING] No saved state found; starting training.")
else:
    print("[TRAINING] Starting training.")

if checkpoint_to_resume:
    trainer.args = training_args

trainer.train(resume_from_checkpoint=checkpoint_to_resume)
