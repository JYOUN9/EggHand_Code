# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 EggHand authors.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/NVIDIA/Isaac-GR00T/blob/b211007ed6698e6642d2fd7679dabab1d97e9e6c/getting_started/1_gr00t_inference.ipynb
# Modified by the EggHand authors: forecasting data, projections, losses and evaluation.
# License text: LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 EggHand authors.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/NVIDIA/Isaac-GR00T/blob/b211007ed6698e6642d2fd7679dabab1d97e9e6c/getting_started/1_gr00t_inference.ipynb
# Modified by the EggHand authors: forecasting data, projections, losses and evaluation.
# License text: LICENSE.

# # GR00T Inference for Metrics Calculation (Pair-Rel-Loss)
# 
# This script calculates ADE, FDE, MPJPE, and MPJPE-F metrics over the entire validation dataset.
# It is based on `new_embooodiment_Inference.py` but removes visualization for performance.

import os
import json
import re
from pathlib import Path
from typing import List
CPU_MAX_THREADS = max(1, int(os.getenv("CPU_MAX_THREADS", str(8))))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(CPU_MAX_THREADS))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.common import configure_inference
configure_inference()
import torch
import numpy as np
from tqdm import tqdm

torch.set_num_threads(CPU_MAX_THREADS)
try:
    torch.set_num_interop_threads(max(1, CPU_MAX_THREADS // 2))
except RuntimeError:
    pass

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform #, ScalarActionMaskOverrideTransform, PostGR00TNormalize

# Configuration

dataset_path = os.environ["EGGHAND_DATASET_PATH"]
# Update to the checkpoint you want to evaluate.
model_path = os.environ["EGGHAND_MODEL_PATH"]
embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
device = "cuda" if torch.cuda.is_available() else "cpu"
LOG_PREFIX = "[EggHand]"
print(f"{LOG_PREFIX} PROJECT_ROOT: {PROJECT_ROOT}")
print(f"{LOG_PREFIX} CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
print(f"{LOG_PREFIX} CPU_MAX_THREADS: {CPU_MAX_THREADS}")
print(f"{LOG_PREFIX} dataset_path: {dataset_path}")
print(f"{LOG_PREFIX} model_path: {model_path}")

# Optional JSON dataset toggle (matches baseline JSON loader)
USE_PARSED_JSON = False
JSON_DATA_FILES = [
]

def _has_model_files(path: Path) -> bool:
    for name in ("config.json", "pytorch_model.bin", "model.safetensors"):
        if (path / name).exists():
            return True
    if list(path.glob("*.safetensors")) or list(path.glob("*.bin")):
        return True
    return False

def resolve_model_path(path_str: str) -> str:
    # Reproduction must never silently choose a different checkpoint.
    from utils.common import validate_checkpoint
    return str(validate_checkpoint(path_str))

# State/Action Keys

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
    "left_pinky_1", "left_pinky_2", "left_pinky_3", "left_pinky_4", "extrinsics",
]
# Keep extrinsics for transforms, but exclude it from joint-only metrics.
joint_keys = [k for k in state_keys if k != "extrinsics"]
action_keys = [k for k in state_keys if k != "extrinsics"]
action_dim = len(action_keys) * 3
EXTRINSICS_DIM = 12
raw_state_dim = action_dim + EXTRINSICS_DIM
model_state_dim = 64
model_action_dim = 32
JOINT_KEY_TO_INDEX = {name: idx for idx, name in enumerate(joint_keys)}
HAND_JOINT_ORDER = [
    "wrist",
    "thumb_1", "thumb_2", "thumb_3", "thumb_4",
    "index_1", "index_2", "index_3", "index_4",
    "middle_1", "middle_2", "middle_3", "middle_4",
    "ring_1", "ring_2", "ring_3", "ring_4",
    "pinky_1", "pinky_2", "pinky_3", "pinky_4",
]
wrist_indices = [idx for idx, name in enumerate(joint_keys) if "wrist" in name]
if not wrist_indices:
    raise ValueError("Expected at least one wrist joint in state_keys.")
ADE_FDE_JOINTS = [joint_keys[idx] for idx in wrist_indices]

try:
    RIGHT_WRIST_INDEX = joint_keys.index("right_wrist")
    LEFT_WRIST_INDEX = joint_keys.index("left_wrist")
except ValueError as exc:
    raise ValueError("Both right_wrist and left_wrist must exist in state_keys.") from exc

# Map each joint to the corresponding wrist so we can build wrist-relative coords.
joint_wrist_indices = np.array([
    RIGHT_WRIST_INDEX if name.startswith("right_") else LEFT_WRIST_INDEX
    for name in joint_keys
])
RIGHT_HAND_INDICES = [idx for idx, name in enumerate(joint_keys) if name.startswith("right_")]
LEFT_HAND_INDICES = [idx for idx, name in enumerate(joint_keys) if name.startswith("left_")]
if not RIGHT_HAND_INDICES or not LEFT_HAND_INDICES:
    raise ValueError("Expected both right_ and left_ joints in state_keys for per-hand alignment.")

# Modality and Transform Configs

# This configuration should match the one used during training.
# 20 state frames and 4 RGB frames.
video_modality = ModalityConfig(
    delta_indices=[-18, -12, -6, 0], 
    modality_keys=["video.ego_view"]
)
state_modality = ModalityConfig(
    delta_indices=list(range(-19, 1)), 
    modality_keys=[f"state.{k}" for k in state_keys]
)
action_modality = ModalityConfig(delta_indices=list(range(10)), modality_keys=[f"action.{k}" for k in action_keys])
language_modality = ModalityConfig(delta_indices=[0], modality_keys=["annotation.task_description"])
STATE_NORMALIZATION_MODES = {k: "min_max" for k in state_modality.modality_keys if "extrinsics" not in k}
ACTION_NORMALIZATION_MODES = {k: "min_max" for k in action_modality.modality_keys if "extrinsics" not in k}

modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "language": language_modality,
}

STATE_DELTAS = np.array(state_modality.delta_indices, dtype=np.int32)
ACTION_DELTAS = np.array(action_modality.delta_indices, dtype=np.int32)

VIDEO_KEY_SET = set(video_modality.modality_keys)
STATE_KEY_SET = set(state_modality.modality_keys)
ACTION_KEY_SET = set(action_modality.modality_keys)

from pydantic import Field
from typing import Any, Dict, Tuple
from gr00t.data.transform.base import ModalityTransform

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

to_apply_transforms = ComposedModalityTransform(
    transforms=[
        VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=video_modality.modality_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
        VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
        VideoToNumpy(apply_to=video_modality.modality_keys),
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionToTensor(apply_to=action_modality.modality_keys),
        ConcatTransform(
            video_concat_order=video_modality.modality_keys,
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
        ),
        # Keep the 20-frame state horizon used in training.
        ScalarActionMaskOverrideTransform(),
        GR00TTransform(
            state_horizon=20, 
            action_horizon=10,
            max_state_dim=model_state_dim, max_action_dim=action_dim),
        PostGR00TNormalize(
            apply_to=["state", "action"],
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
            normalization_modes={
                **STATE_NORMALIZATION_MODES,
                **ACTION_NORMALIZATION_MODES,
            }
        ),
    ]
)

# Load Policy and Dataset

policy = Gr00tPolicy(
    model_path=resolve_model_path(model_path),
    embodiment_tag=embodiment_tag,
    modality_config=modality_configs,
    modality_transform=to_apply_transforms,
    device=device,
)

def _is_frame_payload(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("right_hand_3d", "left_hand_3d"):
        if key in payload:
            return True
    return False

def _collect_frame_dicts(raw_data: dict) -> list[dict]:
    if not isinstance(raw_data, dict):
        return []

    frame_groups: list[dict] = []
    candidate_items = [(k, v) for k, v in raw_data.items() if isinstance(v, dict)]
    if candidate_items and all(_is_frame_payload(v) for _, v in candidate_items):
        frame_groups.append({k: v for k, v in candidate_items})
        return frame_groups

    for _, take_payload in candidate_items:
        if _is_frame_payload(take_payload):
            frame_groups.append(take_payload)
            continue
        inner_dicts = {k: v for k, v in take_payload.items() if isinstance(v, dict)}
        if inner_dicts and all(_is_frame_payload(v) for v in inner_dicts.values()):
            frame_groups.append(inner_dicts)
    return frame_groups

def _frame_payload_to_array(payload: dict) -> np.ndarray:
    frame = np.zeros((len(joint_keys), 3), dtype=np.float32)
    for hand_prefix in ("right", "left"):
        points = payload.get(f"{hand_prefix}_hand_3d")
        if not isinstance(points, list):
            continue
        for joint_name, coords in zip(HAND_JOINT_ORDER, points):
            state_key = f"{hand_prefix}_{joint_name}"
            joint_idx = JOINT_KEY_TO_INDEX.get(state_key)
            if joint_idx is None:
                continue
            if not isinstance(coords, (list, tuple)) or len(coords) != 3:
                continue
            vec = np.array(coords, dtype=np.float32)
            if not np.all(np.isfinite(vec)):
                continue
            frame[joint_idx] = vec
    return frame

def _gather_window_with_edge_padding(frames: np.ndarray, base_index: int, deltas: np.ndarray) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((len(deltas), frames.shape[1], frames.shape[2]), dtype=np.float32)
    indices = np.clip(base_index + deltas, 0, frames.shape[0] - 1)
    return frames[indices]

def _build_samples_from_frames(frame_arrays: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
    if not frame_arrays:
        return []
    frames = np.stack(frame_arrays, axis=0)
    samples: list[dict[str, np.ndarray]] = []
    for base_index in range(frames.shape[0]):
        state_window = _gather_window_with_edge_padding(frames, base_index, STATE_DELTAS)
        action_window = _gather_window_with_edge_padding(frames, base_index, ACTION_DELTAS)
        sample: dict[str, np.ndarray] = {}
        for joint_idx, joint_name in enumerate(joint_keys):
            sample[f"state.{joint_name}"] = state_window[:, joint_idx, :]
            sample[f"action.{joint_name}"] = action_window[:, joint_idx, :]
        state_len = state_window.shape[0]
        action_len = action_window.shape[0]
        sample["state.extrinsics"] = np.zeros((state_len, 12), dtype=np.float32)
        if "extrinsics" in action_keys:
            sample["action.extrinsics"] = np.zeros((action_len, 12), dtype=np.float32)
        samples.append(sample)
    return samples

def load_samples_from_json(json_paths: list[str]) -> list[dict[str, np.ndarray]]:
    aggregated: list[dict[str, np.ndarray]] = []
    for path_str in json_paths:
        path = Path(path_str).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"JSON annotation not found: {path}")
        raw = json.loads(path.read_text())
        frame_groups = _collect_frame_dicts(raw)
        for frames in frame_groups:
            ordered = sorted(frames.items(), key=lambda kv: float(kv[0]))
            frame_arrays = [_frame_payload_to_array(payload) for _, payload in ordered]
            aggregated.extend(_build_samples_from_frames(frame_arrays))
    return aggregated

class ParsedSequenceDataset:
    def __init__(self, samples: list[dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.samples[idx]

if USE_PARSED_JSON:
    if not JSON_DATA_FILES:
        raise ValueError("USE_PARSED_JSON=True but JSON_DATA_FILES is empty.")
    parsed_samples = load_samples_from_json(JSON_DATA_FILES)
    dataset = ParsedSequenceDataset(parsed_samples)
    print(f"[Parsed JSON] Loaded {len(dataset)} samples from {len(JSON_DATA_FILES)} file(s).")
else:
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend="torchvision_av",
        transforms=None,  # Transforms are applied inside the policy
        embodiment_tag=embodiment_tag,
    )

def print_tensor_stats(name: str, arr: np.ndarray) -> None:
    print(
        f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, "
        f"mean={arr.mean():.6f}, std={arr.std():.6f}"
    )

def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)

def _denormalize_action(
    pred_action: dict[str, np.ndarray | torch.Tensor],
    normalization_modes: dict[str, str],
    stats_dict: dict,
) -> dict[str, np.ndarray]:
    denormed: dict[str, np.ndarray] = {}
    for key, value in pred_action.items():
        value_np = _to_numpy(value)
        mode = normalization_modes.get(key)
        if mode != "min_max":
            denormed[key] = value_np
            continue
        subkey = key.split(".", 1)[1]
        if subkey not in stats_dict:
            denormed[key] = value_np
            continue
        stat = stats_dict[subkey]
        min_val = np.array(stat.min, dtype=value_np.dtype)
        max_val = np.array(stat.max, dtype=value_np.dtype)
        scale = max_val - min_val
        scale[scale == 0] = 1.0
        denormed[key] = (value_np + 1.0) * 0.5 * scale + min_val
    return denormed

def _build_action_slices(modality_meta, action_keys: list[str]) -> dict[str, tuple[int, int]]:
    start_idx = 0
    slices: dict[str, tuple[int, int]] = {}
    for key in action_keys:
        subkey = key.split(".", 1)[1]
        dim = modality_meta[subkey].shape[0]
        end_idx = start_idx + dim
        slices[key] = (start_idx, end_idx)
        start_idx = end_idx
    return slices

def umeyama_similarity_transform(
    source: np.ndarray, target: np.ndarray, eps: float = 1e-8
) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    cov = (source_centered.T @ target_centered) / source.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    det_sign = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1.0, 1.0, det_sign])
    R = U @ D @ Vt
    var_source = np.mean(np.sum(source_centered**2, axis=1))
    var_source = max(var_source, eps)
    scale = np.trace(np.diag(S) @ D) / var_source
    translation = target_mean - scale * (R @ source_mean)
    return float(scale), R, translation

def make_wrist_relative(sequence: np.ndarray) -> np.ndarray:
    """Convert absolute coordinates into wrist-relative coordinates per joint."""
    wrist_positions = sequence[:, joint_wrist_indices, :]
    return sequence - wrist_positions

# Alignment Helpers

aligner = None
for transform in policy._modality_transform.transforms:
    if isinstance(transform, GR00TTransform):
        aligner = transform
        break
if aligner is None:
    raise RuntimeError("GR00TTransform not found in modality transforms.")

ACTION_SLICES = _build_action_slices(policy.metadata.modalities.action, action_modality.modality_keys)

# Metrics Calculation

# Metric accumulators (match baseline logic)
metric_accumulators = {
    "mpjpe": [],
    "mpjpe_f": [],
    "r_mpjpe": [],
    "r_mpjpe_f": [],
    "ade": [],
    "fde": [],
}

num_valid_samples = 0
gt_valid_joint_counts = 0
gt_total_joint_counts = 0
DEBUG_SAMPLE_LIMIT = 0

def _mean_metric(name: str) -> float:
    values = metric_accumulators[name]
    return float(np.mean(values)) if values else float("nan")

def _print_progress_metrics(processed: int) -> None:
    if processed == 0:
        return
    print(
        f"[Progress] processed={processed}, valid_samples={num_valid_samples}, "
        f"MPJPE={_mean_metric('mpjpe'):.4f}, MPJPE-F={_mean_metric('mpjpe_f'):.4f}, "
        f"R-MPJPE={_mean_metric('r_mpjpe'):.4f}, R-MPJPE-F={_mean_metric('r_mpjpe_f'):.4f}, "
        f"ADE={_mean_metric('ade'):.4f}, FDE={_mean_metric('fde'):.4f}"
    )

from utils.common import seed_inference
seed_inference()
progress_bar = tqdm(range(len(dataset)), desc="Calculating Metrics")
for idx in progress_bar:
    try:
        step_data = dataset[idx]
        observations = {
            k: v for k, v in step_data.items()
            if k in VIDEO_KEY_SET or k in STATE_KEY_SET or k in ACTION_KEY_SET or k.startswith("annotation.")
        }
        predicted_action = policy.get_action(observations)
        predicted_action = _denormalize_action(
            predicted_action,
            ACTION_NORMALIZATION_MODES,
            policy.metadata.statistics.action,
        )

        pred_all_joints = np.stack([predicted_action[f"action.{k}"] for k in joint_keys], axis=1)
        gt_all_joints_raw = np.stack([_to_numpy(step_data[f"action.{k}"]) for k in joint_keys], axis=1)

        valid_mask = ~np.all(gt_all_joints_raw == 0, axis=2)
        gt_total_joint_counts += valid_mask.size
        gt_valid_joint_counts += np.count_nonzero(valid_mask)

        if not np.any(valid_mask):
            continue

        state_concat = np.concatenate(
            [_to_numpy(step_data[k]) for k in state_modality.modality_keys],
            axis=-1,
        )
        action_concat = np.concatenate(
            [_to_numpy(step_data[k]) for k in action_modality.modality_keys],
            axis=-1,
        )
        _, action_aligned = aligner.align_sequences_to_first_frame(state_concat, action_concat)

        gt_aligned = {}
        for key in action_modality.modality_keys:
            start_idx, end_idx = ACTION_SLICES[key]
            gt_aligned[key] = action_aligned[:, start_idx:end_idx]
        gt_all_joints = np.stack([gt_aligned[f"action.{k}"] for k in joint_keys], axis=1)

        num_valid_samples += 1

        error_per_jt = np.linalg.norm(pred_all_joints - gt_all_joints, axis=2)

        valid_mpjpe_errors = error_per_jt[valid_mask]
        if valid_mpjpe_errors.size > 0:
            metric_accumulators["mpjpe"].extend(valid_mpjpe_errors.tolist())

        valid_mpjpe_f_errors = error_per_jt[-1, valid_mask[-1, :]]
        if valid_mpjpe_f_errors.size > 0:
            metric_accumulators["mpjpe_f"].extend(valid_mpjpe_f_errors.tolist())

        wrist_valid_per_joint = valid_mask[:, joint_wrist_indices]
        p_valid_mask_template = valid_mask & wrist_valid_per_joint

        wrist_errors = error_per_jt[:, wrist_indices]
        wrist_valid_mask = valid_mask[:, wrist_indices]
        wrist_mean_error_per_timestep: List[float] = []
        for t in range(error_per_jt.shape[0]):
            frame_mask = wrist_valid_mask[t, :]
            if np.any(frame_mask):
                wrist_mean_error_per_timestep.append(float(np.mean(wrist_errors[t, frame_mask])))
        if wrist_mean_error_per_timestep:
            metric_accumulators["ade"].append(float(np.mean(wrist_mean_error_per_timestep)))

        wrist_final_mask = wrist_valid_mask[-1, :]
        if np.any(wrist_final_mask):
            wrist_final_errors = wrist_errors[-1, wrist_final_mask]
            metric_accumulators["fde"].append(float(np.mean(wrist_final_errors)))

        gt_wrist_relative = make_wrist_relative(gt_all_joints)
        pred_wrist_relative = make_wrist_relative(pred_all_joints)
        r_errors = np.linalg.norm(pred_wrist_relative - gt_wrist_relative, axis=2)
        r_valid_errors = r_errors[p_valid_mask_template]
        if r_valid_errors.size > 0:
            metric_accumulators["r_mpjpe"].extend(r_valid_errors.tolist())

        r_final_mask = p_valid_mask_template[-1, :]
        if np.any(r_final_mask):
            r_final_errors = r_errors[-1, r_final_mask]
            metric_accumulators["r_mpjpe_f"].extend(r_final_errors.tolist())

    except Exception as e:
        raise RuntimeError(f"Evaluation failed at sample {idx}") from e

# Final metrics (match baseline reporting)
print("\n" + "=" * 60)
print(f"Inference Metrics (valid samples: {num_valid_samples} / {len(dataset)})")
valid_pct = 100.0 * gt_valid_joint_counts / max(1, gt_total_joint_counts)
print(
    f"GT joint validity: {gt_valid_joint_counts} / {gt_total_joint_counts} "
    f"({valid_pct:.2f}% of joints had at least one non-zero axis)"
)
print(f"ADE/FDE joints: {ADE_FDE_JOINTS}")
print("=" * 60)

print(f"MPJPE      : {_mean_metric('mpjpe'):.4f}")
print(f"MPJPE-F    : {_mean_metric('mpjpe_f'):.4f}")
print(f"R-MPJPE    : {_mean_metric('r_mpjpe'):.4f}")
print(f"R-MPJPE-F  : {_mean_metric('r_mpjpe_f'):.4f}")
print(f"ADE        : {_mean_metric('ade'):.4f}")
print(f"FDE        : {_mean_metric('fde'):.4f}")
print("=" * 60)

from utils.common import save_report
save_report(dataset, policy, {"clean": {k: _mean_metric(k) for k in metric_accumulators}}, {"clean": num_valid_samples}, gt_valid_joint_counts, gt_total_joint_counts)
