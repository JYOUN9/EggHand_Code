# Third-party terms

The root source license does not grant rights to external model weights or data.

| Component | Applicable terms |
|---|---|
| Isaac-GR00T source | [Apache-2.0](https://github.com/NVIDIA/Isaac-GR00T/blob/b211007ed6698e6642d2fd7679dabab1d97e9e6c/LICENSE). Original notices are retained; EggHand modifications are described in [IMPLEMENTATION.md](IMPLEMENTATION.md). |
| Eagle processor source | Included with its original Apache-2.0 notices; `image_processing_eagle2_5_vl_fast.py` and `modeling_eagle2_5_vl.py` explicitly carry MIT notices. Their MIT license text is retained in `licenses/NVIDIA-Eagle-MIT.txt`. Tokenizer/configuration assets come from the same pinned Isaac-GR00T source tree. |
| GR00T-N1.5-3B weights and their derivatives | [NVIDIA model license](https://huggingface.co/nvidia/GR00T-N1.5-3B/blob/869830fc749c35f34771aa5209f923ac57e4564e/LICENSE): noncommercial research/evaluation only, including derivatives (§3.3). Redistribution must retain the license and notices (§3.1–3.2). A copy is in `licenses/NVIDIA-GR00T-N1.5-MODEL.txt`. |
| EgoVideo backbone and four-frame weights | No applicable blanket redistribution license was found for the [backbone source](https://github.com/OpenGVLab/EgoVideo/tree/e237b676d5da8d8220f24c80299026293adc8ae0/backbone) or the checkpoint linked by its README. The license in the separate `eccv-2022/` directory does not establish the terms for this backbone. These assets are not bundled. |
| BERT-large-uncased | [Apache-2.0 model card](https://huggingface.co/google-bert/bert-large-uncased). Downloaded separately. |
| Video encoding adaptation | [Hugging Face LeRobot](https://github.com/huggingface/lerobot/blob/6a3d57031aab37adff8eec2e11049510654cc5bb/LICENSE), Apache-2.0; source and copyright are retained in `data_preprocessing/prepare.py`. |
| RotationTransform adaptation | [diffusion_policy MIT license](https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/LICENSE), retained in `licenses/diffusion-policy-MIT.txt`. |
| EgoH4 combined annotations | Downloaded from the [author's dataset release](https://huggingface.co/datasets/masashi-hatano/EgoH4/tree/main), separately from its model checkpoint. These are derived Ego-Exo4D data; the dataset agreement applies. The coordinate files and ViT features are not bundled. |
| Ego-Exo4D data and annotations | Access is governed by the [dataset agreement](https://docs.ego-exo4d-data.org/getting-started/). Raw and prepared datasets are not bundled. |

The download scripts and adaptation patch do not supply missing upstream rights. Dataset-derived split metadata retains the underlying dataset conditions. MANO assets are not used or included.
