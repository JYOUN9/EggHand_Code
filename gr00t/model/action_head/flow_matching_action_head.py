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

# Source: https://github.com/NVIDIA/Isaac-GR00T/blob/b211007ed6698e6642d2fd7679dabab1d97e9e6c/gr00t/model/action_head/flow_matching_action_head.py
# Modified by the EggHand authors: hand-state/action projections and forecasting loss support.
# License: Apache-2.0; see LICENSE and NOTICE.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x

@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    input_state_dim: int = field(
        default=None, metadata={"help": "State input dim before projection."}
    )
    input_action_dim: int = field(
        default=None, metadata={"help": "Action input dim before projection."}
    )
    output_action_dim: int = field(
        default=None, metadata={"help": "Action output dim after projection."}
    )
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.input_state_dim = config.input_state_dim or config.max_state_dim
        self.input_action_dim = config.input_action_dim or config.action_dim
        self.output_action_dim = config.output_action_dim or self.input_action_dim

        self.state_proj_in = (
            nn.Linear(self.input_state_dim, config.max_state_dim)
            if self.input_state_dim != config.max_state_dim
            else nn.Identity()
        )
        self.action_proj_in = (
            nn.Linear(self.input_action_dim, config.action_dim)
            if self.input_action_dim != config.action_dim
            else nn.Identity()
        )
        self.action_proj_out = (
            nn.Linear(config.action_dim, self.output_action_dim)
            if self.output_action_dim != config.action_dim
            else nn.Identity()
        )

        self._init_projection_layers()

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)
        self.loss_fn = HandGeometricLoss() # Custom loss function

    def _init_projection_layers(self):
        for layer in (self.state_proj_in, self.action_proj_in, self.action_proj_out):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _proj_needs_init(self, layer):
        if not isinstance(layer, nn.Linear):
            return False
        weight = layer.weight
        if getattr(weight, "is_meta", False):
            return True
        if not torch.isfinite(weight).all():
            return True
        if weight.numel() > 0 and weight.abs().max().item() > 1e4:
            return True
        if layer.bias is not None:
            bias = layer.bias
            if getattr(bias, "is_meta", False):
                return True
            if not torch.isfinite(bias).all():
                return True
            if bias.numel() > 0 and bias.abs().max().item() > 1e4:
                return True
        return False

    def _ensure_projection_layers_initialized(self):
        if getattr(self, "_projection_init_checked", False):
            return
        self._projection_init_checked = True
        proj_layers = [
            ("state_proj_in", self.state_proj_in),
            ("action_proj_in", self.action_proj_in),
            ("action_proj_out", self.action_proj_out),
        ]
        bad_layers = [name for name, layer in proj_layers if self._proj_needs_init(layer)]
        if bad_layers:
            print(
                "FlowmatchingActionHead: Reinitializing projection layers due to invalid weights: "
                + ", ".join(bad_layers)
            )
            self._init_projection_layers()

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            self.state_proj_in.requires_grad_(False)
            self.action_proj_in.requires_grad_(False)
            self.action_proj_out.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        # if self.freeze_decode_layer:
        #     self.decode_layer.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.state_proj_in.eval()
                self.action_proj_in.eval()
                self.action_proj_out.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        # Visual-language self-attention is unmasked in the paper implementation.
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()
        self._ensure_projection_layers_initialized()
        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_input_raw = (
            action_input.state_raw if "state_raw" in action_input else action_input.state
        )
        state_input = self.state_proj_in(state_input_raw)
        state_features = self.state_encoder(state_input, embodiment_id)

        # Embed noised action trajectory.
        actions_raw = action_input.action_raw if "action_raw" in action_input else action_input.action
        actions = self.action_proj_in(actions_raw)
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        vl_embs = vl_embeds
        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = (
            action_input.action_mask_raw
            if "action_mask_raw" in action_input
            else action_input.action_mask
        )
        pred_actions_raw = self.action_proj_out(pred_actions)
        noise_raw = self.action_proj_out(noise)
        velocity_raw = actions_raw - noise_raw
        noisy_trajectory_raw = (1 - t) * noise_raw + t * actions_raw
        loss = F.mse_loss(pred_actions_raw, velocity_raw, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self._ensure_projection_layers_initialized()
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        
        # Embed state.
        state_input_raw = (
            action_input.state_raw if "state_raw" in action_input else action_input.state
        )
        state_input = self.state_proj_in(state_input_raw)
        state_features = self.state_encoder(state_input, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            vl_embs = vl_embeds

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            # Preserve unmasked DiT attention; do not forward state/padding masks.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        actions_out = self.action_proj_out(actions)
        return BatchFeature(data={"action_pred": actions_out})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

class HandGeometricLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # (Parent Start Index, Child Start Index)

        self.right_hand_bones = [
            (0, 3), (3, 6), (6, 9), (9, 12),       # Wrist -> Thumb chain
            (0, 15), (15, 18), (18, 21), (21, 24), # Wrist -> Index chain
            (0, 27), (27, 30), (30, 33), (33, 36), # Wrist -> Middle chain
            (0, 39), (39, 42), (42, 45), (45, 48), # Wrist -> Ring chain
            (0, 51), (51, 54), (54, 57), (57, 60)  # Wrist -> Pinky chain
        ]

        self.left_hand_bones = [(s + 63, e + 63) for s, e in self.right_hand_bones]
        self.all_bones = self.right_hand_bones + self.left_hand_bones

        self.wrist_indices = [0, 1, 2, 63, 64, 65]

    def compute_bone_lengths(self, flat_actions):
        """Return per-bone lengths (B, T, Num_Bones) from flattened joint coordinates."""
        lengths = []
        for p_idx, c_idx in self.all_bones:

            parent = flat_actions[..., p_idx : p_idx+3]
            child = flat_actions[..., c_idx : c_idx+3]

            dist = torch.norm(parent - child, p=2, dim=-1)
            lengths.append(dist)
            
        return torch.stack(lengths, dim=-1)

    def forward(self, pred_velocity, noisy_trajectory, t, target_velocity, actions, action_mask):

        if t.dim() == 1: t = t.view(-1, 1, 1)
        pred_x1 = noisy_trajectory + (1.0 - t) * pred_velocity
        target_x1 = actions 

        loss_tensor = F.mse_loss(pred_velocity, target_velocity, reduction="none") * action_mask
        fm_loss = loss_tensor.sum() / (action_mask.sum() + 1e-6)

        diff1 = pred_x1[:, 1:, :] - pred_x1[:, :-1, :]
        diff2 = diff1[:, 1:, :] - diff1[:, :-1, :] 

        dim_mask = torch.ones(pred_x1.shape[-1], device=pred_x1.device)
        dim_mask[self.wrist_indices] = 0.0

        time_mask = action_mask[:, 2:] * action_mask[:, 1:-1] * action_mask[:, :-2]
        if time_mask.dim() == 2:
            time_mask = time_mask.unsqueeze(-1)

        # (B, T-2, D) * (D,) -> (B, T-2, D)
        raw_smooth_loss = (diff2 ** 2) * dim_mask 

        masked_smooth_loss = raw_smooth_loss * time_mask
        
        loss_smooth = masked_smooth_loss.sum() / (time_mask.sum() + 1e-6)

        # Loss 3: Geometric Consistency (Bone Length Loss)

        pred_lengths = self.compute_bone_lengths(pred_x1)
        target_lengths = self.compute_bone_lengths(target_x1)
        
        bone_diff = torch.abs(pred_lengths - target_lengths)
        
        # bone_diff: (B, T, Num_Bones)

        if action_mask.dim() == 3:

            bone_mask = action_mask[..., 0:1] 
        else:
            bone_mask = action_mask.unsqueeze(-1)
            
        loss_bone = (bone_diff * bone_mask).sum() / (bone_mask.sum() * bone_diff.shape[-1] + 1e-6)

        lambda_fm_loss = 0.6
        lambda_smooth = 0.1
        lambda_bone = 0.3
        
        total_loss = (lambda_fm_loss * fm_loss) + (lambda_smooth * loss_smooth) + (lambda_bone * loss_bone)

        return total_loss, {
            "loss_fm": fm_loss,
            "loss_smooth": loss_smooth,
            "loss_bone": loss_bone
        }