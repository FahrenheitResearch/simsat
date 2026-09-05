#!/usr/bin/env python3
"""Fetch pinned NASA BMNG unshaded monthly Earth-color inputs.

Example: python scripts/simsat-fetch-base-map.py --output-dir /data/bmng-base
Pass --months 3 4 or --verify-only to restrict months or verify existing data.
These contrast-enhanced 2004 composites are visual inputs, not ABI reflectance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def verify(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset['bytes']:
        return False
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest == asset['sha256']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--months', type=int, nargs='+', choices=range(1, 13), default=list(range(1, 13)))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / 'crates/simsat/assets/bluemarble_base_map_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not args.verify_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        if asset['month'] not in args.months:
            continue
        path = args.output_dir / asset['filename']
        if verify(path, asset):
            print(f"verified {path}", flush=True)
            continue
        if args.verify_only or path.exists():
            raise SystemExit(f"Missing or checksum-mismatched asset: {path}")
        fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=args.output_dir)
        try:
            count = 0
            with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(asset['url'], timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > asset['bytes']:
                        raise ValueError('Downloaded asset exceeds its pinned size')
                    output.write(chunk)
            if not verify(Path(temp), asset):
                raise ValueError(f"Downloaded asset failed its size/SHA-256 check: {asset['url']}")
            os.replace(temp, path)
            print(f"downloaded and verified {path}", flush=True)
        finally:
            Path(temp).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
