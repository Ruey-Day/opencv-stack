#!/usr/bin/env python3
"""
Prepare a FastCaMo-Real sequence for Structure-PLP-SLAM.

What this does:
  1. Creates rgb.txt and depth.txt (TUM format: float_seconds  path)
     from the associations.txt (FastCaMo format: color.png color/color.png depth.png depth/depth.png)
  2. Converts int32 depth PNGs to uint16 (Structure-PLP-SLAM reads uint16 depth)
  3. Converts RGBA color PNGs to RGB

The converted depth and rgb frames are written to:
    <seq>/depth_u16/  and  <seq>/rgb/

Usage:
    python scripts/prep_fastcamo_for_slam.py /mnt/crucial/rueyday/data/fastcamo/apartment_1
    python scripts/prep_fastcamo_for_slam.py /mnt/crucial/rueyday/data/fastcamo/apartment_1 --skip_convert
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def parse_associations(assoc_path):
    entries = []
    with open(assoc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # format: color.png  color/color.png  depth.png  depth/depth.png
            color_file = parts[1]   # e.g. color/1610782892362.png
            depth_file = parts[3]   # e.g. depth/1610782892361.png
            # timestamp = filename stem in milliseconds → convert to seconds
            color_ts = float(Path(parts[0]).stem) / 1000.0
            depth_ts = float(Path(parts[2]).stem) / 1000.0
            entries.append((color_ts, color_file, depth_ts, depth_file))
    return entries


def convert_sequence(seq_dir: Path, skip_convert: bool):
    assoc = seq_dir / 'associations.txt'
    if not assoc.exists():
        print(f'[ERROR] No associations.txt in {seq_dir}')
        sys.exit(1)

    entries = parse_associations(assoc)
    print(f'  {len(entries)} frames in {seq_dir.name}')

    if not skip_convert:
        print('  Converting in-place: color (RGBA→RGB), depth (int32→uint16) ...')
        bad = set()
        for i, (cts, cfile, dts, dfile) in enumerate(entries):
            if i % 500 == 0:
                print(f'    {i}/{len(entries)}  (skipped: {len(bad)})')

            # Color: RGBA → RGB in-place
            color_path = seq_dir / cfile
            try:
                img = Image.open(color_path)
                if img.mode == 'RGBA':
                    img.convert('RGB').save(color_path)
            except Exception:
                bad.add(i)
                continue

            # Depth: int32 → uint16 in-place
            depth_path = seq_dir / dfile
            try:
                arr = np.array(Image.open(depth_path))
                if arr.dtype != np.uint16:
                    arr16 = np.clip(arr, 0, 65535).astype(np.uint16)
                    Image.fromarray(arr16).save(depth_path)
            except Exception:
                bad.add(i)

        if bad:
            print(f'  WARNING: skipped {len(bad)} corrupted frames')
            entries = [e for i, e in enumerate(entries) if i not in bad]
        print('  Conversion done.')
    else:
        print('  Skipping conversion (--skip_convert)')

    # Write rgb.txt and depth.txt in TUM format pointing to original folders
    rgb_txt   = seq_dir / 'rgb.txt'
    depth_txt = seq_dir / 'depth.txt'

    with open(rgb_txt, 'w') as fr, open(depth_txt, 'w') as fd:
        fr.write('# timestamp filename\n')
        fd.write('# timestamp filename\n')
        for cts, cfile, dts, dfile in entries:
            fr.write(f'{cts:.6f} {cfile}\n')
            fd.write(f'{dts:.6f} {dfile}\n')

    print(f'  Wrote {rgb_txt.name} and {depth_txt.name}')
    return len(entries)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('seq_dirs', nargs='+', help='FastCaMo sequence directory/ies')
    ap.add_argument('--skip_convert', action='store_true',
                    help='Skip image conversion (if already done), only rewrite txt files')
    args = ap.parse_args()

    for seq_dir in args.seq_dirs:
        seq_dir = Path(seq_dir)
        print(f'\n=== {seq_dir.name} ===')
        n = convert_sequence(seq_dir, args.skip_convert)
        print(f'  Ready — {n} frames prepared')


if __name__ == '__main__':
    main()
