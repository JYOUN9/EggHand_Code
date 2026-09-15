# SPDX-FileCopyrightText: Copyright (c) 2026 EggHand authors.
# SPDX-License-Identifier: Apache-2.0
# Source: EggHand, scripts/bootstrap_sources.py (this repository).
# License text: LICENSE.

"""Download the external source and apply the EggHand adaptations.

EgoVideo licensing is unresolved: see docs/THIRD_PARTY.md before use/distribution.
Downloaded files are excluded from Git.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import ROOT, sha256

def checkout(url, revision, destination):
    subprocess.run(['git', 'init', '-q', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'remote', 'add', 'origin', url], check=True)
    subtree = 'backbone/model'
    subprocess.run(['git', '-C', str(destination), 'sparse-checkout', 'set', subtree], check=True)
    subprocess.run(['git', '-C', str(destination), 'fetch', '--depth=1', '--filter=blob:none', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', 'FETCH_HEAD'], check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--egovideo-source', type=Path, help='Existing checkout at the exact pinned revision')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/assets/egovideo_source.json').read_text())
    destination = ROOT / 'gr00t/model/backbone/egovideo_model'
    if destination.exists():
        for name, record in manifest['files'].items():
            if sha256(destination / name) != record['patched_sha256']:
                raise ValueError(f'Existing EgoVideo source differs: {name}; use a clean destination.')
        print('EgoVideo source is ready.')
    else:
        with tempfile.TemporaryDirectory() as temp:
            source = args.egovideo_source or Path(temp) / 'EgoVideo'
            if args.egovideo_source is None:
                checkout(manifest['repository'], manifest['revision'], source)
            for name, record in manifest['files'].items():
                if sha256(source / 'backbone/model' / name) != record['upstream_sha256']:
                    raise ValueError(f'Unexpected EgoVideo upstream bytes: {name}')
            staging = Path(temp) / 'patched'
            shutil.copytree(source / 'backbone/model', staging)
            subprocess.run(['git', 'apply', '--no-index', '--unidiff-zero', '--whitespace=nowarn', str(ROOT / 'patches/egovideo-local.patch')], cwd=staging, check=True)
            for name, record in manifest['files'].items():
                if sha256(staging / name) != record['patched_sha256']:
                    raise ValueError(f'Patched EgoVideo differs: {name}')
            shutil.copytree(staging, destination)
        print('EgoVideo source is ready.')

if __name__ == '__main__':
    main()
