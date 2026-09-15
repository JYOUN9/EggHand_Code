# SPDX-FileCopyrightText: Copyright (c) 2026 EggHand authors.
# SPDX-License-Identifier: Apache-2.0
# Source: EggHand, scripts/download_models.py (this repository).
# License text: LICENSE.

"""Download pretrained dependencies at the revisions used by EggHand."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import ROOT, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=['bert', 'gr00t', 'egovideo'])
    args = parser.parse_args()
    if args.asset == 'bert':
        from huggingface_hub import snapshot_download
        # Own a separate cache: the BERT loader requests the legacy ID at 'main'.
        # A SHA-only download does not create that ref, so explicitly pin it here.
        cache = ROOT / 'assets/huggingface/hub'
        revision = '6da4b6a26a1877e173fca3225479512db81a5e5b'
        path = Path(snapshot_download('bert-large-uncased', revision=revision, cache_dir=cache,
                                     allow_patterns=['config.json', 'model.safetensors', 'vocab.txt', 'tokenizer*.json', 'README.md']))
        expected = json.loads((ROOT / 'configs/assets/bert.json').read_text())
        for name, record in expected['files'].items():
            if sha256(path / name) != record['sha256']:
                raise ValueError(f'BERT differs from pinned revision: {name}')
        ref = cache / 'models--bert-large-uncased/refs/main'
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision)
        print(f'BERT verified. Set HF_HUB_CACHE={cache} before running train/evaluation.')
    elif args.asset == 'gr00t':
        from huggingface_hub import snapshot_download
        manifest = json.loads((ROOT / 'configs/assets/gr00t_base.json').read_text())
        path = Path(snapshot_download(manifest['repository'], revision=manifest['revision'],
                                      local_dir=ROOT / 'assets/GR00T-N1.5-3B'))
        for name, record in manifest['files'].items():
            if sha256(path / name) != record['sha256']:
                raise ValueError(f'GR00T differs from the training asset: {name}')
        print(f'GR00T base weights verified: {path}')
    else:
        import gdown
        target = ROOT / 'assets/EgoVideo_model.pth'
        manifest = json.loads((ROOT / 'configs/assets/egovideo_checkpoint.json').read_text())
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id='1k6f1eRdcL17IvXtdX_J8WxNbju2Ms3AW', output=str(target), quiet=False)
        if sha256(target) != manifest['sha256']:
            raise ValueError('Official download differs from the pinned training asset. Do not silently substitute it.')
        print(f'EgoVideo checkpoint matches the training asset: {target}')

if __name__ == '__main__':
    main()
