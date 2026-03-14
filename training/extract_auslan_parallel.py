#!/usr/bin/env python3
"""Extract Auslan features in parallel across multiple workers."""
import sys
import json
import time
import csv
import multiprocessing as mp
from pathlib import Path
from functools import partial

sys.path.insert(0, '/home/ubuntu')

def process_video(args):
    """Process a single video - must be top-level for pickling."""
    vpath_str, sign_name = args
    vpath = Path(vpath_str)
    try:
        import extract_and_train as et
        if vpath.stat().st_size < 1000:
            return None
        raw_frames = et.extract_landmarks_from_video(vpath)
        if not raw_frames:
            return None
        smoothed = et.smooth_snapshots(raw_frames)
        segment = et.segment_stroke(smoothed)
        features = et.compute_segment_features(segment)
        if features is None:
            return None
        return {
            "sign": sign_name,
            "video": f"auslan_{vpath.name}",
            "features": features.tolist(),
            "segment_frames": len(segment),
            "raw_frames": len(raw_frames),
            "source": "auslan",
            "segment": [
                {"dom": s["dom"], "non": s["non"], "body": s["body"],
                 "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                 "timestamp": s["timestamp"]}
                for s in segment
            ],
        }
    except Exception as e:
        return None

def main():
    auslan_dir = Path('/home/ubuntu/auslan')
    videos_dir = auslan_dir / 'videos'
    sign_map_file = auslan_dir / 'sign_map.tsv'

    sign_map = {}
    if sign_map_file.exists():
        with open(sign_map_file) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                sign_map[row['filename']] = row['sign_name'].upper().replace(' ', '-').replace('(', '').replace(')', '').strip()

    videos = sorted(videos_dir.glob('*.mp4'))
    print(f"Auslan: {len(videos)} videos, {len(sign_map)} mapped labels")

    # Build work items
    work = []
    for vpath in videos:
        sign_name = sign_map.get(vpath.name)
        if not sign_name:
            sign_name = vpath.stem.upper().replace('-', ' ').replace('_', ' ').strip().replace(' ', '-')
        if sign_name:
            work.append((str(vpath), sign_name))

    print(f"Processing {len(work)} videos with {mp.cpu_count()} workers")
    t_start = time.time()

    all_data = []
    n_workers = min(mp.cpu_count(), 4)

    with mp.Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_video, work, chunksize=10)):
            if result is not None:
                all_data.append(result)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(work) - i - 1) / rate / 60
                print(f"  [{i+1}/{len(work)}] {len(all_data)} extracted ({rate:.1f}/s, ETA {eta:.0f}m)")
            if (i + 1) % 1000 == 0:
                with open('/home/ubuntu/auslan_features.json', 'w') as f:
                    json.dump(all_data, f)
                print(f"  [checkpoint] {len(all_data)} saved")

    with open('/home/ubuntu/auslan_features.json', 'w') as f:
        json.dump(all_data, f)

    elapsed = time.time() - t_start
    signs = set(d['sign'] for d in all_data)
    print(f"\nAuslan: {len(all_data)} features, {len(signs)} unique signs ({elapsed:.0f}s)")

if __name__ == '__main__':
    main()
