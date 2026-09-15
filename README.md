# EggHand

### A Multimodal Foundation Model for Egocentric Hand Pose Forecasting

[Jaeyoung Choi](https://jyoun9.github.io/)<sup>*</sup>,
[Hyeondong Kim](https://www.hyeondong.kim/)<sup>*</sup>,
[Yujin Kim](https://www.linkedin.com/in/yujin-kim-/),
[Daehee Park](https://sites.google.com/view/dhpark/)<sup>†</sup>

[ISL Lab](https://isl.dgist.ac.kr/), DGIST · **CVPR Findings 2026**  
<sup>*</sup> Equal contribution · <sup>†</sup> Corresponding author

**[Project Page](https://jyoun9.github.io/EggHand/)** · **[arXiv](https://arxiv.org/abs/2605.07642)** · **[Paper PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Choi_EggHand_A_Multimodal_Foundation_Model_for_Egocentric_Hand_Pose_Forecasting_CVPRF_2026_paper.pdf)** · **[Model Checkpoint](https://drive.google.com/file/d/1qZJB43FRQGn-8EkvZ_0U2_YpJrLLIWg-/view?usp=share_link)**

Official implementation of EggHand. The model forecasts 3D hand motion from egocentric video, past hand poses and task text, combining an EgoVideo encoder with a GR00T action decoder.

## Installation

Linux, Python 3.10 and a CUDA GPU with BF16/TF32 support. The reference environment uses PyTorch 2.5.1, CUDA 12.4 and an NVIDIA L40S (48 GB). Run all commands from the repository root.

```bash
conda create -n egghand python=3.10.19 -y
conda activate egghand
python -m pip install --no-deps torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/bootstrap_sources.py
```

The adapted GR00T code and Eagle processor are included. `bootstrap_sources.py` downloads the pinned EgoVideo source and applies the included patch; Git is required. See [source modifications](docs/IMPLEMENTATION.md) and [third-party licenses](docs/THIRD_PARTY.md).

Download the pretrained models:

```bash
python scripts/download_models.py bert
python scripts/download_models.py gr00t
python -m pip install gdown==5.2.0
python scripts/download_models.py egovideo
```

| Model | Source | Local path |
| --- | --- | --- |
| BERT-large-uncased | [Hugging Face](https://huggingface.co/google-bert/bert-large-uncased) | `assets/huggingface/` |
| GR00T-N1.5-3B | [Hugging Face](https://huggingface.co/nvidia/GR00T-N1.5-3B) | `assets/GR00T-N1.5-3B/` |
| EgoVideo, four-frame model | [EgoVideo](https://github.com/OpenGVLab/EgoVideo) | `assets/EgoVideo_model.pth` |

BERT is required for both training and evaluation. GR00T and EgoVideo pretrained weights are needed for training initialization; a trained EggHand checkpoint contains their model weights. Download revisions are pinned in `configs/assets/`.

`--video-root` points to the directory containing `takes/`. Each output directory must be new. The converter writes Parquet samples, RGB videos and metadata in LeRobot v2.1 format. See [data preprocessing](data_preprocessing/README.md) for input formats and encoding settings.

| Split | Episodes | Samples | Frame rate |
| --- | ---: | ---: | ---: |
| Train | 555 | 41,314 | 10 fps |
| Validation | 109 | 8,830 | 10 fps |

## Training

```bash
conda activate egghand
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --dataset data/egoh4_lerobot/train \
  --output-dir outputs/main \
  --batch-size 8 --gradient-accumulation-steps 8
```

Defaults: 10 epochs, seed 42, AdamW, learning rate `3e-5`, weight decay `1e-7`, cosine decay with 1% warm-up, and FP32 with TF32 enabled. The effective batch size is 64; adjust the batch and accumulation arguments for available GPU memory. Absolute, wrist-relative and pairwise loss weights are `(0.6, 0.2, 0.2)`.

Use `--gr00t` and `--egovideo` to override the pretrained paths. Add `--resume` to restore the latest checkpoint in `--output-dir`.

The warm-up follows the original training configuration (1%; the paper text states 5%). The default accumulation and TF32 settings support training on a 48 GB GPU; retraining can yield different values from the paper checkpoint. [Implementation notes](docs/IMPLEMENTATION.md) cover sampling, masking and normalization.

## Evaluation

Pass a trained EggHand checkpoint directory, including its configuration, projection weights and `experiment_cfg/metadata.json`:

```bash
conda activate egghand
CUDA_VISIBLE_DEVICES=0 python inference/evaluate.py \
  --dataset data/egoh4_lerobot/val \
  --model-path /path/to/checkpoint \
  --output results/evaluate.json
```

Evaluation uses seed 42, four denoising steps and BF16 autocast. Results are reported in meters. ADE/FDE measure wrist trajectories; the paper's MPJPE/MPJPE-F correspond to `r_mpjpe`/`r_mpjpe_f` in the output JSON.

## Citation

```bibtex
@InProceedings{Choi_2026_CVPR,
  author    = {Choi, Jaeyoung and Kim, Hyeondong and Kim, Yujin and Park, Daehee},
  title     = {EggHand: A Multimodal Foundation Model for Egocentric Hand Pose Forecasting},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026},
  pages     = {3521-3531}
}
```

## License and acknowledgments

The source code is distributed under [Apache-2.0](LICENSE), with retained MIT notices for the relevant Eagle files and diffusion-policy utility. Pretrained weights and datasets have separate terms; GR00T-N1.5-derived weights are limited to noncommercial research and evaluation. See [NOTICE](NOTICE) and [third-party terms](docs/THIRD_PARTY.md).

This work builds on [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [EgoVideo](https://github.com/OpenGVLab/EgoVideo), [EgoH4](https://github.com/masashi-hatano/EgoH4), [Ego-Exo4D](https://ego-exo4d-data.org/), [LeRobot](https://github.com/huggingface/lerobot) and [diffusion_policy](https://github.com/real-stanford/diffusion_policy).
