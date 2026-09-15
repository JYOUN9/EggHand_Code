# SPDX-FileCopyrightText: Copyright (c) 2026 EggHand authors.
# SPDX-License-Identifier: Apache-2.0
# Source: EggHand, utils/common.py (this repository).
# License text: LICENSE.

"""Shared paths, checkpoint loading, inference seed and metric output."""
import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]

def configure_runtime():
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    os.environ.setdefault('HF_HUB_CACHE', str(ROOT / 'assets/huggingface/hub'))

def configure_training():
    parser = argparse.ArgumentParser(description='Train EggHand.')
    parser.add_argument('--dataset', default=os.getenv('EGGHAND_DATASET_PATH', str(ROOT / 'data/egoh4_lerobot/train')))
    parser.add_argument('--output-dir', default=os.getenv('EGGHAND_OUTPUT_DIR', str(ROOT / 'outputs/main')))
    parser.add_argument('--gr00t', default=os.getenv('GR00T_BASE_MODEL_PATH', str(ROOT / 'assets/GR00T-N1.5-3B')))
    parser.add_argument('--egovideo', default=os.getenv('EGOVIDEO_CKPT_PATH', str(ROOT / 'assets/EgoVideo_model.pth')))
    parser.add_argument('--batch-size', type=int, default=8, help='Samples per GPU per forward pass')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Forward/backward passes per optimizer update')
    parser.add_argument('--resume', action='store_true', help='Resume the latest saved state in --output-dir')
    args = parser.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error('Batch size and gradient accumulation must be positive.')
    for key, value in {'EGGHAND_DATASET_PATH': args.dataset, 'EGGHAND_OUTPUT_DIR': args.output_dir,
                       'GR00T_BASE_MODEL_PATH': args.gr00t, 'EGOVIDEO_CKPT_PATH': args.egovideo}.items():
        os.environ[key] = str(Path(value).expanduser().resolve())
    configure_runtime()
    return args

def configure_inference():
    parser = argparse.ArgumentParser(description='Run EggHand inference and compute hand forecasting metrics.')
    parser.add_argument('--dataset', default=os.getenv('EGGHAND_DATASET_PATH', str(ROOT / 'data/egoh4_lerobot/val')))
    parser.add_argument('--model-path', default=os.getenv('EGGHAND_MODEL_PATH'),
                        required='EGGHAND_MODEL_PATH' not in os.environ, help='Directory saved by training')
    parser.add_argument('--output', default=os.getenv('EGGHAND_RESULT_PATH', str(ROOT / 'results' / (Path(sys.argv[0]).stem + '.json'))))
    parser.add_argument('--seed', type=int, default=int(os.getenv('EGGHAND_SEED', '42')))
    args = parser.parse_args()
    os.environ['EGGHAND_DATASET_PATH'] = str(Path(args.dataset).expanduser().resolve())
    os.environ['EGGHAND_MODEL_PATH'] = str(Path(args.model_path).expanduser().resolve())
    os.environ['EGGHAND_RESULT_PATH'] = str(Path(args.output).expanduser().resolve())
    os.environ['EGGHAND_SEED'] = str(args.seed)
    configure_runtime()
    return args

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def validate_checkpoint(path):
    path = Path(path).expanduser().resolve()
    required = ['config.json', 'experiment_cfg/metadata.json',
                'projection_weights.pth', 'action_projection_weights.pth']
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        required += ['model.safetensors.index.json']
        required += sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:
        required += ['model.safetensors']
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Incomplete checkpoint {path}: {missing}')
    config = json.loads((path / 'config.json').read_text())
    if config['action_head_cfg'].get('input_state_dim') != 138 or config['action_horizon'] != 10:
        raise ValueError('Expected an EggHand checkpoint with 138D state and a 10-step horizon.')
    return path

def seed_inference():
    import numpy as np
    import torch
    seed = int(os.getenv('EGGHAND_SEED', '42'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_report(dataset, policy, metrics, valid_samples, valid_joints, total_joints):
    if not all(math.isfinite(value) for row in metrics.values() for value in row.values()):
        raise ValueError('Evaluation returned non-finite metrics.')
    result = {
        'checkpoint': Path(policy.model_path).name,
        'seed': int(os.getenv('EGGHAND_SEED', '42')),
        'samples': len(dataset),
        'valid_samples': valid_samples,
        'valid_joints': int(valid_joints),
        'total_joints': int(total_joints),
        'metrics_meters': metrics,
        'paper_columns': {'ADE': 'ade', 'FDE': 'fde', 'MPJPE': 'r_mpjpe', 'MPJPE-F': 'r_mpjpe_f'},
    }
    default = ROOT / 'results' / (Path(sys.argv[0]).stem + '.json')
    output = Path(os.environ.get('EGGHAND_RESULT_PATH', str(default)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Results: {output}')
