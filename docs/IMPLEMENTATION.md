# Implementation

EggHand adapts [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/b211007ed6698e6642d2fd7679dabab1d97e9e6c) and [EgoVideo](https://github.com/OpenGVLab/EgoVideo/tree/e237b676d5da8d8220f24c80299026293adc8ae0). The modified GR00T runtime is included here. Run `python scripts/bootstrap_sources.py` to download and adapt the remaining external source; no manual editing of an installed package is needed.

| File | EggHand adaptation |
|---|---|
| `gr00t/data/dataset.py` | Episode task text, temporal sampling and statistics loading |
| `gr00t/model/transforms.py` | Camera alignment, raw state/action tensors and normalization |
| `gr00t/model/backbone/egovideo_backbone.py` | Frozen EgoVideo encoder; 1408D visual and 512D text features projected to 2048D |
| `gr00t/model/gr00t_n1.py` | EgoVideo integration and projection checkpoint loading |
| `gr00t/model/action_head/flow_matching_action_head.py` | State projection 138→64 and action projection 126↔32 |
| `gr00t/model/action_head/cross_attention_dit.py` | Unmasked self- and cross-attention |
| `gr00t/model/policy.py` | Float32 weights, bfloat16 inference autocast and checkpoint statistics |

`patches/egovideo-local.patch` modifies three downloaded EgoVideo files:

- `setup_model.py`: local BERT config, full text-token features, unprojected visual output, unnormalized features, disabled fused/FlashAttention paths and `assign=True` checkpoint loading.
- `vision_encoder.py`: removal of imports for the disabled fused/FlashAttention paths.
- `bert/tokenization_bert.py`: vocabulary access compatible with tokenizer initialization.

The Eagle tokenizer/processor is bundled from the fixed GR00T revision because it is used by the input transform, even with EgoVideo. Its file-level Apache/MIT notices are retained. No Eagle model weights are needed.

## Sampling and metrics

State offsets are `-19..0`, video offsets are `[-18, -12, -6, 0]`, and action offsets are `0..9`. Episode boundaries use edge padding. Alignment uses the first observed camera frame, followed by per-joint min-max normalization. The action window includes the base timestep, as in the checkpoint's training configuration.

BERT uses its text padding mask. The DiT and visual-language self-attention stack do not apply padding/state-token masks. Missing all-zero joints are transformed as points, including translation; they are not reset to zero after alignment. The loss mask is computed per coordinate before alignment. Evaluation excludes a joint only when all three raw coordinates are zero, and the relative metric additionally requires a valid wrist.

ADE averages valid wrists within a frame, then frames within a sample, then samples. FDE uses the final target frame. R-MPJPE pools wrist-relative joint errors, including the zero-error wrists; R-MPJPE-F uses the final target frame. These two relative metrics are the paper's MPJPE and MPJPE-F columns. Predictions are denormalized once using the checkpoint statistics.

The entrypoints accept command-line paths and seed inference after model loading. Evaluation stops on errors rather than silently omitting samples.

## Training execution

Seed 42 is set before model initialization. Training uses FP32 parameters and activations with TF32 enabled for supported CUDA operations; FP16 and BF16 are disabled. The default per-GPU batch is 8, with 8 gradient accumulation steps. The Trainer calls the four-step `get_action()` path directly and computes the geometric losses in float32.
