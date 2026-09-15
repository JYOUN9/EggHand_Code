# SPDX-FileCopyrightText: Copyright (c) 2026 EggHand authors.
# SPDX-License-Identifier: Apache-2.0
# Source: EggHand, gr00t/model/backbone/egovideo_backbone.py (this repository).
# License text: LICENSE.

import os
import numpy as np
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
import gr00t

# 

try:

    from .egovideo_model.setup_model import build_model
except ImportError as exc:
    raise ImportError(
        "EgoVideo could not be imported. Run python scripts/bootstrap_sources.py "
        "and install requirements-lock.txt; see the original import error below."
    ) from exc


class EgoVideoBackbone(nn.Module):
    """Project frozen EgoVideo visual and text features to the GR00T embedding dimension."""

    def __init__(
        self,
        tune_visual: bool = False,
        tune_llm: bool = False,
        num_frames: int = 4,
        egovideo_ckpt_path: str = None,
        egovideo_vision_dim: int = 1408,
        egovideo_text_dim: int = 512,
        out_seq_len: int = 512,
        project_to_dim: int = 2048,
        projection_ckpt_path: str = None,
        **kwargs
    ):
        """Load the four-frame encoder and create visual (1408D) and text (512D) projections to 2048D."""
        super().__init__()

        self.num_frames = num_frames
        self.egovideo_vision_dim = egovideo_vision_dim
        self.egovideo_text_dim = egovideo_text_dim
        self.project_to_dim = project_to_dim
        self.egovideo_ckpt_path = egovideo_ckpt_path
        self.projection_ckpt_path = projection_ckpt_path

        print(f"EgoVideoBackbone: Loading model from {egovideo_ckpt_path} using build_model...")
        try:
            self.model, self.tokenizer = build_model(
                ckpt_path=egovideo_ckpt_path,
                num_frames=self.num_frames
            )
        except Exception as e:
            print(f"EgoVideoBackbone: build_model failed: {e}")
            raise

        self._checkpoint_path = egovideo_ckpt_path
        self._checkpoint_loaded = False

        if os.environ.get("EGOVIDEO_CHECK_PARAM_NAN", "0") == "1":
            self.check_model_numerics()

        self.vision_projection = nn.Linear(self.egovideo_vision_dim, self.project_to_dim)

        self.text_projection = nn.Linear(self.egovideo_text_dim, self.project_to_dim)

        self.set_trainable_parameters(tune_visual=False, tune_llm=False)

    def set_trainable_parameters(self, tune_visual: bool, tune_llm: bool):
        """Freeze both EgoVideo encoders and train the vision and text projections."""

        self.tune_visual = False
        self.tune_llm = False

        if self.projection_ckpt_path and os.path.exists(self.projection_ckpt_path):
            print(f"EgoVideoBackbone: Loading projection weights from {self.projection_ckpt_path}")
            try:
                proj_weights = torch.load(self.projection_ckpt_path, map_location="cpu")
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
                        if norm_key.startswith("backbone."):
                            norm_key = norm_key[len("backbone."):]
                        normalized[norm_key] = value
                    proj_weights = normalized

                combined_proj_state_dict = {
                    **{f"vision_projection.{k}": v for k, v in self.vision_projection.state_dict().items()},
                    **{f"text_projection.{k}": v for k, v in self.text_projection.state_dict().items()}
                }

                loadable_weights = {k: v for k, v in proj_weights.items() if k in combined_proj_state_dict}
                self.load_state_dict(loadable_weights, strict=False, assign=True)
                
                print(f"  - Loaded {len(loadable_weights)} projection parameters.")

            except Exception as e:
                print(f"  [ERROR] Failed to load projection weights: {e}")
        else:
            print("EgoVideoBackbone: No projection checkpoint found or specified. Using initialized projection layers.")

        if hasattr(self, 'model') and self.model is not None:
            for p in self.model.parameters():
                p.requires_grad = False
            print("EgoVideoBackbone: Frozen self.model (text/visual backbone).")

        for p in self.vision_projection.parameters():
            p.requires_grad = True
        for p in self.text_projection.parameters():
            p.requires_grad = True

    def set_frozen_modules_to_eval_mode(self):
        """Keep the frozen EgoVideo model in evaluation mode on the current device."""
        if self.training and hasattr(self, 'model') and self.model is not None:

            self.model.eval()
    
    def ensure_checkpoints_loaded(self):
        """Load the encoder weights after parameters leave the meta device."""
        if not self._checkpoint_path or self._checkpoint_loaded:
            return

        if any(p.is_meta for p in self.model.parameters()):
            print("EgoVideoBackbone: Model is still on meta device, deferring checkpoint loading.")
            return

        if not os.path.exists(self._checkpoint_path):
            print(f"EgoVideoBackbone: Checkpoint path does not exist -> {self._checkpoint_path}")
            self._checkpoint_path = None
            return

        print(f"EgoVideoBackbone: Loading weights from {self._checkpoint_path}")
        try:

            ckpt = torch.load(self._checkpoint_path, map_location="cpu")
            new_ckpt = {}
            for k, v in ckpt.items():
                new_k = k.replace('module.', '')
                new_ckpt[new_k] = v
            
            msg = self.model.load_state_dict(new_ckpt, strict=False, assign=True)
            print(f"EgoVideoBackbone: Main backbone checkpoint loaded.")

            if msg.missing_keys:
                print("\n--- [INFO] Missing Keys (not found in checkpoint, thus initialized randomly) ---")
                for key in msg.missing_keys[:20]: print(f"  - {key}")
                if len(msg.missing_keys) > 20: print(f"  ... and {len(msg.missing_keys) - 20} more")
            
            if msg.unexpected_keys:
                print("\n--- [INFO] Unexpected Keys (in checkpoint, but not in model) ---")
                for key in msg.unexpected_keys[:20]: print(f"  - {key}")
                if len(msg.unexpected_keys) > 20: print(f"  ... and {len(msg.unexpected_keys) - 20} more")

            self._checkpoint_loaded = True

        except Exception as e:
            print(f"EgoVideoBackbone: Failed to load checkpoint {self._checkpoint_path}: {e}")

            self._checkpoint_path = None

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """Concatenate the projected visual CLS token and text token sequence."""

        self.ensure_checkpoints_loaded()
        self.set_frozen_modules_to_eval_mode()

        video_tensor = vl_input["video"]
        text_ids = vl_input["text_input_ids"]
        text_mask = vl_input["text_attention_mask"]

        # (B, T, C, H, W) -> (B, C, T, H, W)
        if video_tensor.dim() == 5 and video_tensor.shape[2] == 3:

            video_tensor = video_tensor.permute(0, 2, 1, 3, 4).contiguous()
        elif video_tensor.dim() != 5 or video_tensor.shape[1] != 3:

            raise ValueError(f"EgoVideoBackbone: expected (B,C,T,H,W) video; got shape={video_tensor.shape}")

        try:
            image_features, text_features = self.model(
                video_tensor,
                text_ids,
                text_mask
            )
        except Exception as e:
            print(f"EgoVideoBackbone: model.forward failed: {e}")
            print(f"  video_tensor.shape={video_tensor.shape}, dtype={video_tensor.dtype}")
            print(f"  text_ids.shape={text_ids.shape}, dtype={text_ids.dtype}")
            print(f"  text_mask.shape={text_mask.shape}, dtype={text_mask.dtype}")
            raise

        def _check_nan_inf(label: str, tensor: torch.Tensor):
            if not isinstance(tensor, torch.Tensor):
                return
            has_nan = torch.isnan(tensor).any()
            has_inf = torch.isinf(tensor).any()
            status = "✅" if not (has_nan or has_inf) else "🚨"
            print(f"{status} [{label:^25s}] has_nan: {has_nan}, has_inf: {has_inf}, shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}")

        # (B, Dim) -> (B, 1, Dim)

        if image_features.dim() == 2: 
            image_features_seq = image_features.unsqueeze(1)
        elif image_features.dim() == 3:
            image_features_seq = image_features
        else:
            raise ValueError(f"EgoVideo: expected image features (B,D) or (B,1,D); got {image_features.shape}")

        projected_cls = self.vision_projection(image_features_seq) # (B, 1, project_to_dim)

        # (B, SeqLen_T, Dim)
        if text_features.dim() != 3:
            raise ValueError(f"EgoVideo: expected text features (B,L,D); got {text_features.shape}")

        projected_text_sequence = self.text_projection(text_features) # (B, SeqLen_T, project_to_dim)

        # (B, 1, ProjDim) + (B, SeqLen_T, ProjDim) => (B, 1 + SeqLen_T, ProjDim)
        backbone_features = torch.cat([projected_cls, projected_text_sequence], dim=1)

        video_mask = torch.ones(
            projected_cls.shape[0], 1, 
            device=projected_cls.device, dtype=torch.long
        )
        # (B, 1) + (B, SeqLen_T) => (B, 1 + SeqLen_T)
        backbone_attention_mask = torch.cat([video_mask, text_mask], dim=1)

        return BatchFeature(
            data={"backbone_features": backbone_features, 
                  "backbone_attention_mask": backbone_attention_mask}
        )

    @torch.no_grad()
    def check_model_numerics(self, check_grads: bool = False) -> bool:
        """Report non-finite encoder parameters, buffers and optionally gradients."""

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Tokenize task text and prepare the video tensor and attention mask."""
        prepared_data = {}

        if "video" in batch:
            video_tensor = batch["video"]
        elif "eagle_pixel_values" in batch:
            video_tensor = batch["eagle_pixel_values"]
        else:
            raise ValueError("EgoVideoBackbone: batch is missing video/eagle_pixel_values.")
        
        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.from_numpy(video_tensor)
        elif not isinstance(video_tensor, torch.Tensor):
            video_tensor = torch.tensor(video_tensor)
        video_tensor = video_tensor.float()
        prepared_data["video"] = video_tensor

        if "text" in batch:
            text_list = batch["text"]
            if not isinstance(text_list, (list, tuple)):
                text_list = [str(text_list)]
            else:
                text_list = [str(t) for t in text_list]
        else:
            batch_size = video_tensor.shape[0] if video_tensor.dim() >= 5 else 1
            text_list = [""] * batch_size
        
        prepared_data["text"] = text_list

        text_inputs = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=59,
            return_tensors="pt",
        )
        prepared_data["text_input_ids"] = text_inputs.input_ids
        prepared_data["text_attention_mask"] = text_inputs.attention_mask

        for key, value in batch.items():
            if key not in prepared_data:
                prepared_data[key] = value

        return BatchFeature(data=prepared_data)
