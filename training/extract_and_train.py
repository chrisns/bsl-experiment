#!/usr/bin/env python3
"""
BSL Sign Recognition — End-to-end ML pipeline.

1. Extract MediaPipe landmarks from training videos
2. Segment strokes using velocity-based detection
3. Compute fixed-size feature vectors per segment
4. Augment with landmark-level perturbations
5. Train PyTorch MLP classifier
6. Export to ONNX for browser inference
7. Evaluate against held-out test videos

Usage:
    python3 extract_and_train.py                           # train on test-videos (original)
    python3 extract_and_train.py --bsldict                 # train on BSLDict multi-signer data
    python3 extract_and_train.py --bsldict --extract-only  # just extract BSLDict features
    python3 extract_and_train.py --bsldict --train-only    # train from cached BSLDict features
"""
import os
import sys
import json
import time
import argparse
import glob
import numpy as np
from pathlib import Path

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ─── Configuration ───────────────────────────────────────────────
VIDEO_DIR = Path(__file__).parent.parent.parent / "frontend" / "test-videos"
BSLDICT_DIR = Path(__file__).parent.parent / "bsldict" / "videos"
OUTPUT_DIR = Path(__file__).parent
FEATURES_CACHE = OUTPUT_DIR / "extracted_features.json"
BSLDICT_FEATURES_CACHE = OUTPUT_DIR / "bsldict_features.json"
MODEL_OUTPUT = OUTPUT_DIR / "bsl_classifier.onnx"
METADATA_OUTPUT = OUTPUT_DIR / "model_metadata.json"

# Feature extraction mirrors sign-recogniser.js extractDtwFrame
# but computed in Python from MediaPipe Holistic landmarks
SMOOTHING_ALPHA = 0.4
SEG_VELOCITY_THRESHOLD = 0.06
SEG_IDLE_FRAMES = 5
SEG_MIN_STROKE_FRAMES = 4

# Training
AUGMENTATIONS_PER_SAMPLE = 100
HIDDEN_DIMS = [256, 128, 64]
LARGE_HIDDEN_DIMS = [512, 256, 128]  # For 200-2000 class vocabularies
XLARGE_HIDDEN_DIMS = [1024, 512, 256]  # For 2000-5000 class vocabularies
MASSIVE_HIDDEN_DIMS = [2048, 1024, 512, 256]  # For 5000+ class vocabularies
LEARNING_RATE = 0.001
EPOCHS = 80
LARGE_EPOCHS = 120  # More epochs for larger vocabularies
XLARGE_EPOCHS = 150  # Even more epochs for massive vocabularies
MASSIVE_EPOCHS = 200  # Most epochs for huge vocabularies
BATCH_SIZE = 64
LARGE_BATCH_SIZE = 128  # Larger batches for more data
XLARGE_BATCH_SIZE = 256  # Even larger batches for massive data
MASSIVE_BATCH_SIZE = 512  # Huge batches for massive data
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1
N_FOLDS = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ─── MediaPipe Feature Extraction ────────────────────────────────

def extract_landmarks_from_video(video_path):
    """Process a video through MediaPipe Holistic, return per-frame landmark snapshots."""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        snapshot = build_snapshot(results, len(frames) / fps * 1000)
        if snapshot is not None:
            frames.append(snapshot)

    cap.release()
    holistic.close()
    return frames


def v2dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def v2sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def v2cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def v2normalize(v):
    m = np.sqrt(v[0] ** 2 + v[1] ** 2)
    if m < 1e-8:
        return (0.0, 0.0)
    return (v[0] / m, v[1] / m)


def build_snapshot(results, timestamp_ms):
    """Mirror of sign-recogniser.js extractSnapshot(), returns dict with same structure."""
    pose = results.pose_landmarks
    if pose is None:
        return None

    lm = pose.landmark
    if len(lm) < 25:
        return None

    l_shoulder = (lm[11].x, lm[11].y)
    r_shoulder = (lm[12].x, lm[12].y)
    mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
    shoulder_width = v2dist(l_shoulder, r_shoulder)
    if shoulder_width < 0.01:
        return None

    nose = (lm[0].x, lm[0].y)
    chin = (nose[0], nose[1] + shoulder_width * 0.35)
    forehead = (nose[0], nose[1] - shoulder_width * 0.30)
    l_hip = (lm[23].x, lm[23].y) if len(lm) > 23 else (l_shoulder[0], l_shoulder[1] + shoulder_width * 1.5)
    r_hip = (lm[24].x, lm[24].y) if len(lm) > 24 else (r_shoulder[0], r_shoulder[1] + shoulder_width * 1.5)
    waist = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

    def normalise(pt):
        return ((pt[0] - mid_shoulder[0]) / shoulder_width,
                (pt[1] - mid_shoulder[1]) / shoulder_width)

    # MediaPipe: left hand in image = signer's right (dominant) hand (mirrored)
    dom_hand = results.left_hand_landmarks
    non_hand = results.right_hand_landmarks
    dom_wrist_pose = (lm[15].x, lm[15].y)  # left wrist = signer's right
    non_wrist_pose = (lm[16].x, lm[16].y)

    def extract_hand(hand_landmarks, wrist_fallback):
        has_hand = hand_landmarks is not None and len(hand_landmarks.landmark) >= 21
        if has_hand:
            hl = hand_landmarks.landmark
            wrist = (hl[0].x, hl[0].y)
            center = (hl[9].x, hl[9].y)  # middle MCP

            thumb_tip = (hl[4].x, hl[4].y)
            index_tip = (hl[8].x, hl[8].y)
            middle_tip = (hl[12].x, hl[12].y)
            ring_tip = (hl[16].x, hl[16].y)
            pinky_tip = (hl[20].x, hl[20].y)

            index_mcp = (hl[5].x, hl[5].y)
            pinky_mcp = (hl[17].x, hl[17].y)

            thumb_ip = (hl[3].x, hl[3].y)
            index_pip = (hl[6].x, hl[6].y)
            middle_pip = (hl[10].x, hl[10].y)
            ring_pip = (hl[14].x, hl[14].y)
            pinky_pip = (hl[18].x, hl[18].y)

            finger_ext = [
                1 if v2dist(thumb_tip, wrist) > v2dist(thumb_ip, wrist) * 1.1 else 0,
                1 if v2dist(index_tip, wrist) > v2dist(index_pip, wrist) * 1.15 else 0,
                1 if v2dist(middle_tip, wrist) > v2dist(middle_pip, wrist) * 1.15 else 0,
                1 if v2dist(ring_tip, wrist) > v2dist(ring_pip, wrist) * 1.15 else 0,
                1 if v2dist(pinky_tip, wrist) > v2dist(pinky_pip, wrist) * 1.15 else 0,
            ]

            to_index = v2sub(index_mcp, wrist)
            to_pinky = v2sub(pinky_mcp, wrist)
            palm_cross = v2cross(to_index, to_pinky)

            openness = (v2dist(thumb_tip, wrist) + v2dist(index_tip, wrist) +
                        v2dist(middle_tip, wrist) + v2dist(ring_tip, wrist) +
                        v2dist(pinky_tip, wrist)) / 5 / shoulder_width

            spread = (v2dist(index_tip, middle_tip) + v2dist(middle_tip, ring_tip) +
                      v2dist(ring_tip, pinky_tip)) / 3 / shoulder_width

            pinch_dist = v2dist(thumb_tip, index_tip) / shoulder_width

            finger_dir = v2normalize(v2sub(index_tip, wrist))
        else:
            wrist = wrist_fallback
            center = wrist_fallback
            finger_ext = [0, 0, 0, 0, 0]
            palm_cross = 0.0
            openness = 0.0
            spread = 0.0
            pinch_dist = 1.0
            finger_dir = (0.0, 0.0)

        pos = normalise(center)
        return {
            "present": has_hand,
            "pos": pos,
            "fingerExtension": finger_ext,
            "palmCross": palm_cross,
            "openness": openness,
            "spread": spread,
            "pinchDist": pinch_dist,
            "fingerDir": finger_dir,
        }

    dom = extract_hand(dom_hand, dom_wrist_pose)
    non = extract_hand(non_hand, non_wrist_pose)

    n_nose = normalise(nose)
    n_chin = normalise(chin)
    n_forehead = normalise(forehead)
    n_chest = normalise(mid_shoulder)
    n_waist = normalise(waist)

    dom_to_nose = v2dist(dom["pos"], n_nose)
    dom_to_chin = v2dist(dom["pos"], n_chin)
    dom_to_forehead = v2dist(dom["pos"], n_forehead)
    dom_to_chest = v2dist(dom["pos"], n_chest)
    dom_to_waist = v2dist(dom["pos"], n_waist)
    hand_dist = v2dist(dom["pos"], non["pos"])

    return {
        "dom": dom,
        "non": non,
        "body": {"nose": n_nose, "chin": n_chin, "forehead": n_forehead,
                 "chest": n_chest, "waist": n_waist},
        "distances": {
            "domToNose": dom_to_nose, "domToChin": dom_to_chin,
            "domToForehead": dom_to_forehead, "domToChest": dom_to_chest,
            "domToWaist": dom_to_waist, "handDist": hand_dist,
        },
        "shoulderWidth": shoulder_width,
        "timestamp": timestamp_ms,
    }


def smooth_snapshots(raw_frames):
    """Apply exponential smoothing (mirrors sign-recogniser.js smoothSnapshot)."""
    if not raw_frames:
        return []
    smoothed = [raw_frames[0]]
    a = SMOOTHING_ALPHA
    for i in range(1, len(raw_frames)):
        curr = raw_frames[i]
        prev = smoothed[-1]
        s = {
            "dom": {**curr["dom"]},
            "non": {**curr["non"]},
            "body": curr["body"],
            "distances": curr["distances"],
            "shoulderWidth": curr["shoulderWidth"],
            "timestamp": curr["timestamp"],
        }
        for hand_key in ["dom", "non"]:
            ch = curr[hand_key]
            ph = prev[hand_key]
            s[hand_key]["pos"] = (
                ph["pos"][0] * a + ch["pos"][0] * (1 - a),
                ph["pos"][1] * a + ch["pos"][1] * (1 - a),
            )
            s[hand_key]["openness"] = ph["openness"] * a + ch["openness"] * (1 - a)
            s[hand_key]["spread"] = ph["spread"] * a + ch["spread"] * (1 - a)
            s[hand_key]["pinchDist"] = ph["pinchDist"] * a + ch["pinchDist"] * (1 - a)
            s[hand_key]["palmCross"] = ph["palmCross"] * a + ch["palmCross"] * (1 - a)
        smoothed.append(s)
    return smoothed


def segment_stroke(frames):
    """Velocity-based segmentation (mirrors sign-recogniser.js updateSegmentation)."""
    if len(frames) < SEG_MIN_STROKE_FRAMES:
        return frames

    # Compute velocities
    velocities = []
    for i in range(1, len(frames)):
        if frames[i]["dom"]["present"] and frames[i - 1]["dom"]["present"]:
            dx = frames[i]["dom"]["pos"][0] - frames[i - 1]["dom"]["pos"][0]
            dy = frames[i]["dom"]["pos"][1] - frames[i - 1]["dom"]["pos"][1]
            dt = max((frames[i]["timestamp"] - frames[i - 1]["timestamp"]) / 1000, 0.001)
            velocities.append(np.sqrt(dx * dx + dy * dy) / dt)
        else:
            velocities.append(0.0)

    # Simple segmentation: find the region with sustained velocity > threshold
    # Use middle 60% of hand-present frames as fallback
    hand_frames = [f for f in frames if f["dom"]["present"]]
    if len(hand_frames) < SEG_MIN_STROKE_FRAMES:
        return frames

    # Find stroke region: longest contiguous run above threshold
    above = [v > SEG_VELOCITY_THRESHOLD for v in velocities]
    best_start, best_end = 0, len(frames)
    run_start = None
    longest = 0
    for i, a in enumerate(above):
        if a:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len > longest:
                    longest = run_len
                    best_start = max(0, run_start - 2)
                    best_end = min(len(frames), i + 2)
                run_start = None
    if run_start is not None:
        run_len = len(above) - run_start
        if run_len > longest:
            best_start = max(0, run_start - 2)
            best_end = len(frames)

    if best_end - best_start >= SEG_MIN_STROKE_FRAMES:
        segment = frames[best_start:best_end]
    else:
        # Fallback: middle 60%
        start = int(len(hand_frames) * 0.2)
        end = int(len(hand_frames) * 0.8)
        segment = hand_frames[start:end] if end - start >= SEG_MIN_STROKE_FRAMES else hand_frames

    return segment


def extract_dtw_frame(snapshot):
    """Mirror of sign-recogniser.js extractDtwFrame — 17-dim feature vector."""
    nose_x = snapshot["body"]["nose"][0]
    nose_y = snapshot["body"]["nose"][1]
    dom = snapshot["dom"]
    non = snapshot["non"]
    return [
        dom["pos"][0] - nose_x,
        dom["pos"][1] - nose_y,
        (non["pos"][0] - nose_x) if non["present"] else 0,
        (non["pos"][1] - nose_y) if non["present"] else 0,
        *dom["fingerExtension"],
        dom["openness"],
        dom["spread"],
        dom["palmCross"],
        dom["fingerDir"][0],
        dom["fingerDir"][1],
        dom["pinchDist"],
        1.0 if non["present"] else 0.0,
        snapshot["distances"]["handDist"],
    ]


# ─── Segment-Level Feature Vector ────────────────────────────────

def compute_segment_features(segment):
    """Compute fixed-size statistical feature vector from a variable-length segment.

    Takes a list of smoothed snapshots (a segmented stroke), extracts per-frame
    DTW features, then computes temporal statistics for a fixed-size vector.
    """
    frames = [extract_dtw_frame(s) for s in segment if s.get("body")]
    if len(frames) < 3:
        return None

    arr = np.array(frames, dtype=np.float32)  # (T, 17)
    T, D = arr.shape

    features = []

    # 1. Per-dimension statistics (17 dims × 7 stats = 119 features)
    for d in range(D):
        col = arr[:, d]
        features.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
            np.max(col) - np.min(col),  # range
            np.median(col),
            col[-1] - col[0],  # delta (end - start)
        ])

    # 2. Trajectory features (dom hand position = dims 0,1)
    dom_x = arr[:, 0]
    dom_y = arr[:, 1]

    # Displacement
    disp_x = dom_x[-1] - dom_x[0]
    disp_y = dom_y[-1] - dom_y[0]
    displacement = np.sqrt(disp_x ** 2 + disp_y ** 2)

    # Path length
    dx = np.diff(dom_x)
    dy = np.diff(dom_y)
    step_dists = np.sqrt(dx ** 2 + dy ** 2)
    path_length = np.sum(step_dists)

    # Straightness
    straightness = displacement / max(path_length, 1e-6)

    # Speed stats
    speed_mean = np.mean(step_dists)
    speed_max = np.max(step_dists)
    speed_std = np.std(step_dists)

    # Direction angle
    direction = np.arctan2(disp_y, disp_x)

    # Winding angle (cumulative angular change)
    angles = np.arctan2(dy, dx)
    angle_diffs = np.diff(angles)
    # Wrap to [-pi, pi]
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    winding = np.sum(np.abs(angle_diffs))

    # Reversal counts
    x_reversals = np.sum(np.diff(np.sign(dx)) != 0)
    y_reversals = np.sum(np.diff(np.sign(dy)) != 0)

    features.extend([
        displacement, path_length, straightness,
        speed_mean, speed_max, speed_std,
        direction, winding,
        x_reversals / max(T, 1), y_reversals / max(T, 1),
    ])

    # 3. Non-dom hand features
    non_present_ratio = np.mean(arr[:, 15])  # non-present flag
    hand_dist_mean = np.mean(arr[:, 16])
    hand_dist_delta = arr[-1, 16] - arr[0, 16]
    features.extend([non_present_ratio, hand_dist_mean, hand_dist_delta])

    # 4. Body-relative distances from raw snapshots
    dom_to_nose = [s["distances"]["domToNose"] for s in segment if s.get("body")]
    dom_to_chin = [s["distances"]["domToChin"] for s in segment if s.get("body")]
    dom_to_chest = [s["distances"]["domToChest"] for s in segment if s.get("body")]
    for dist_list in [dom_to_nose, dom_to_chin, dom_to_chest]:
        if dist_list:
            features.extend([np.mean(dist_list), np.min(dist_list), np.std(dist_list)])
        else:
            features.extend([0, 0, 0])

    # 5. Segment metadata
    duration_frames = T
    features.append(duration_frames / 30.0)  # normalise to seconds

    return np.array(features, dtype=np.float32)


# ─── Data Augmentation ───────────────────────────────────────────

def augment_segment(segment, rng):
    """Apply random augmentation to a segment of snapshots.

    Operates on landmark-level data (not pixels), so it's instant.
    """
    aug = []
    # Spatial noise
    noise_scale = rng.uniform(0.005, 0.025)
    # Scale perturbation
    scale = rng.uniform(0.9, 1.1)
    # Translation offset
    tx = rng.uniform(-0.05, 0.05)
    ty = rng.uniform(-0.05, 0.05)
    # Speed warp: resample to different length
    speed_factor = rng.uniform(0.7, 1.3)

    # Resample (speed warp)
    n_frames = max(4, int(len(segment) * speed_factor))
    indices = np.linspace(0, len(segment) - 1, n_frames).astype(int)

    for idx in indices:
        s = segment[idx]
        # Deep copy the snapshot structure
        new_s = {
            "dom": {**s["dom"]},
            "non": {**s["non"]},
            "body": {**s["body"]},
            "distances": {**s["distances"]},
            "shoulderWidth": s["shoulderWidth"],
            "timestamp": s["timestamp"],
        }

        for hand_key in ["dom", "non"]:
            h = new_s[hand_key]
            if h["present"]:
                # Apply noise + scale + translation to position
                px = h["pos"][0] * scale + tx + rng.normal(0, noise_scale)
                py = h["pos"][1] * scale + ty + rng.normal(0, noise_scale)
                h["pos"] = (px, py)
                # Noise on continuous features
                h["openness"] = max(0, h["openness"] + rng.normal(0, 0.02))
                h["spread"] = max(0, h["spread"] + rng.normal(0, 0.01))
                h["palmCross"] = h["palmCross"] + rng.normal(0, 0.01)
                h["pinchDist"] = max(0, h["pinchDist"] + rng.normal(0, 0.02))
                fd = h["fingerDir"]
                h["fingerDir"] = (fd[0] + rng.normal(0, 0.05), fd[1] + rng.normal(0, 0.05))
                # Occasionally flip a finger extension
                fe = list(h["fingerExtension"])
                for fi in range(5):
                    if rng.random() < 0.05:
                        fe[fi] = 1 - fe[fi]
                h["fingerExtension"] = fe

        # Update body positions with same transform
        body = new_s["body"]
        for key in ["nose", "chin", "forehead", "chest", "waist"]:
            old = body[key]
            body[key] = (old[0] * scale + tx, old[1] * scale + ty)

        # Recompute distances
        dom_pos = new_s["dom"]["pos"]
        non_pos = new_s["non"]["pos"]
        new_s["distances"] = {
            "domToNose": v2dist(dom_pos, body["nose"]),
            "domToChin": v2dist(dom_pos, body["chin"]),
            "domToForehead": v2dist(dom_pos, body["forehead"]),
            "domToChest": v2dist(dom_pos, body["chest"]),
            "domToWaist": v2dist(dom_pos, body["waist"]),
            "handDist": v2dist(dom_pos, non_pos),
        }

        aug.append(new_s)

    # Random frame dropout (skip 1-3 frames)
    if len(aug) > 8 and rng.random() < 0.3:
        n_drop = rng.integers(1, 4)
        drop_indices = rng.choice(len(aug), n_drop, replace=False)
        aug = [f for i, f in enumerate(aug) if i not in drop_indices]

    return aug


# ─── Model ────────────────────────────────────────────────────────

class BSLClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Pipeline ─────────────────────────────────────────────────────

def extract_all_features(video_dir):
    """Extract features from all test videos using Python MediaPipe."""
    videos = sorted(glob.glob(str(video_dir / "*.mp4")))
    print(f"\nExtracting features from {len(videos)} videos...")

    all_data = []
    for i, vpath in enumerate(videos):
        sign_name = Path(vpath).stem.upper()
        t0 = time.time()

        # 1. Extract landmarks
        raw_frames = extract_landmarks_from_video(vpath)
        if not raw_frames:
            print(f"  [{i+1}/{len(videos)}] {sign_name}: SKIPPED (no landmarks)")
            continue

        # 2. Smooth
        smoothed = smooth_snapshots(raw_frames)

        # 3. Segment
        segment = segment_stroke(smoothed)

        # 4. Compute features
        features = compute_segment_features(segment)
        if features is None:
            print(f"  [{i+1}/{len(videos)}] {sign_name}: SKIPPED (insufficient frames)")
            continue

        elapsed = time.time() - t0
        all_data.append({
            "sign": sign_name,
            "features": features.tolist(),
            "segment_frames": len(segment),
            "raw_frames": len(raw_frames),
            # Store raw segment for augmentation
            "segment": [
                {
                    "dom": s["dom"],
                    "non": s["non"],
                    "body": s["body"],
                    "distances": s["distances"],
                    "shoulderWidth": s["shoulderWidth"],
                    "timestamp": s["timestamp"],
                }
                for s in segment
            ],
        })
        print(f"  [{i+1}/{len(videos)}] {sign_name}: {len(raw_frames)} raw → {len(segment)} seg → {len(features)} features ({elapsed:.1f}s)")

    print(f"\nExtracted features for {len(all_data)}/{len(videos)} signs")
    return all_data


def extract_bsldict_features(bsldict_dir, existing_cache=None):
    """Extract features from BSLDict videos (nested SIGN/SIGN_N.mp4 structure).

    If existing_cache path is provided, loads it and only extracts NEW videos
    not already in the cache (incremental extraction).
    """
    sign_dirs = sorted([d for d in bsldict_dir.iterdir() if d.is_dir()])
    print(f"\nExtracting features from BSLDict ({len(sign_dirs)} signs)...")

    # Load existing cache for incremental extraction
    all_data = []
    cached_videos = set()
    if existing_cache and existing_cache.exists():
        with open(existing_cache) as f:
            all_data = json.load(f)
        cached_videos = {d["video"] for d in all_data}
        cached_signs = len(set(d["sign"] for d in all_data))
        print(f"  Loaded {len(all_data)} cached features ({cached_signs} signs)")

    total_videos = 0
    new_videos = 0
    skipped_cached = 0
    for sign_dir in sign_dirs:
        sign_name = sign_dir.name.upper()
        videos = sorted(sign_dir.glob("*.mp4"))
        for j, vpath in enumerate(videos):
            total_videos += 1

            # Skip if already in cache
            if vpath.name in cached_videos:
                skipped_cached += 1
                continue

            t0 = time.time()

            raw_frames = extract_landmarks_from_video(vpath)
            if not raw_frames:
                print(f"  {sign_name}_{j}: SKIPPED (no landmarks)")
                continue

            smoothed = smooth_snapshots(raw_frames)
            segment = segment_stroke(smoothed)
            features = compute_segment_features(segment)
            if features is None:
                print(f"  {sign_name}_{j}: SKIPPED (insufficient frames)")
                continue

            elapsed = time.time() - t0
            new_videos += 1
            all_data.append({
                "sign": sign_name,
                "video": vpath.name,
                "features": features.tolist(),
                "segment_frames": len(segment),
                "raw_frames": len(raw_frames),
                "segment": [
                    {
                        "dom": s["dom"],
                        "non": s["non"],
                        "body": s["body"],
                        "distances": s["distances"],
                        "shoulderWidth": s["shoulderWidth"],
                        "timestamp": s["timestamp"],
                    }
                    for s in segment
                ],
            })
            print(f"  [{new_videos}] {sign_name}_{j}: {len(raw_frames)} raw → {len(segment)} seg → {len(features)} features ({elapsed:.1f}s)")

            # Save checkpoint every 500 new videos
            if existing_cache and new_videos % 500 == 0:
                with open(existing_cache, "w") as f:
                    json.dump(all_data, f)
                signs_so_far = len(set(d["sign"] for d in all_data))
                print(f"  --- Checkpoint: {len(all_data)} total ({signs_so_far} signs, {new_videos} new) ---")

    signs_with_data = len(set(d["sign"] for d in all_data))
    print(f"\nExtracted: {len(all_data)} videos for {signs_with_data}/{len(sign_dirs)} signs")
    if skipped_cached:
        print(f"  ({skipped_cached} already cached, {new_videos} newly extracted)")
    return all_data


def extract_additional_sources(base_dir, existing_data=None, cache_path=None):
    """Extract features from Dicta-Sign and SSC STEM videos.

    Scans for:
      - base_dir/../dicta-sign/videos/*.mp4 (concept_map.tsv for labels)
      - base_dir/../ssc-stem/videos/{subject}/*.mp4

    Returns list of feature dicts merged with existing_data.
    """
    research_dir = Path(base_dir).parent.parent  # research/
    all_data = list(existing_data) if existing_data else []
    cached_videos = {d["video"] for d in all_data}
    new_videos = 0

    # --- Dicta-Sign ---
    dicta_dir = research_dir / "dicta-sign"
    concept_map_file = dicta_dir / "concept_map.tsv"
    dicta_videos_dir = dicta_dir / "videos"

    if dicta_videos_dir.exists() and concept_map_file.exists():
        # Load concept ID → word mapping
        concept_map = {}
        with open(concept_map_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    concept_map[parts[0]] = parts[1].upper().replace(" ", "-")

        videos = sorted(dicta_videos_dir.glob("*.mp4"))
        print(f"\nDicta-Sign: {len(videos)} videos, {len(concept_map)} concepts")

        for vpath in videos:
            concept_id = vpath.stem  # e.g., "42"
            video_key = f"dicta_{vpath.name}"
            if video_key in cached_videos:
                continue

            sign_name = concept_map.get(concept_id, f"DICTA-{concept_id}")
            t0 = time.time()
            raw_frames = extract_landmarks_from_video(vpath)
            if not raw_frames:
                continue

            smoothed = smooth_snapshots(raw_frames)
            segment = segment_stroke(smoothed)
            features = compute_segment_features(segment)
            if features is None:
                continue

            elapsed = time.time() - t0
            new_videos += 1
            all_data.append({
                "sign": sign_name,
                "video": video_key,
                "features": features.tolist(),
                "segment_frames": len(segment),
                "raw_frames": len(raw_frames),
                "source": "dicta-sign",
                "segment": [
                    {"dom": s["dom"], "non": s["non"], "body": s["body"],
                     "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                     "timestamp": s["timestamp"]}
                    for s in segment
                ],
            })
            if new_videos % 100 == 0:
                print(f"  Dicta-Sign [{new_videos}]: {sign_name} ({elapsed:.1f}s)")

        print(f"  Dicta-Sign: {new_videos} new videos extracted")

    # --- SSC STEM ---
    ssc_dir = research_dir / "ssc-stem" / "videos"
    if ssc_dir.exists():
        ssc_subjects = sorted([d for d in ssc_dir.iterdir() if d.is_dir()])
        total_ssc = sum(len(list(s.glob("*.mp4"))) for s in ssc_subjects)
        print(f"\nSSC STEM: {total_ssc} videos across {len(ssc_subjects)} subjects")

        ssc_new = 0
        for subject_dir in ssc_subjects:
            subject = subject_dir.name
            videos = sorted(subject_dir.glob("*.mp4"))
            for vpath in videos:
                # Skip definition (d.mp4) and fingerspell (f.mp4) variants
                stem = vpath.stem
                if stem.endswith("d") or stem.endswith("f"):
                    # Check if base term also exists
                    base = vpath.parent / f"{stem[:-1]}.mp4"
                    if base.exists():
                        continue  # This is a d/f variant

                video_key = f"ssc_{subject}_{vpath.name}"
                if video_key in cached_videos:
                    continue

                sign_name = stem.upper().replace("-", " ").replace("_", " ").strip().replace(" ", "-")
                t0 = time.time()
                raw_frames = extract_landmarks_from_video(vpath)
                if not raw_frames:
                    continue

                smoothed = smooth_snapshots(raw_frames)
                segment = segment_stroke(smoothed)
                features = compute_segment_features(segment)
                if features is None:
                    continue

                elapsed = time.time() - t0
                ssc_new += 1
                new_videos += 1
                all_data.append({
                    "sign": sign_name,
                    "video": video_key,
                    "features": features.tolist(),
                    "segment_frames": len(segment),
                    "raw_frames": len(raw_frames),
                    "source": f"ssc-{subject}",
                    "segment": [
                        {"dom": s["dom"], "non": s["non"], "body": s["body"],
                         "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                         "timestamp": s["timestamp"]}
                        for s in segment
                    ],
                })
                if ssc_new % 100 == 0:
                    print(f"  SSC [{ssc_new}]: {subject}/{sign_name} ({elapsed:.1f}s)")

        print(f"  SSC STEM: {ssc_new} new videos extracted")

    # --- Christian BSL ---
    christian_dir = research_dir / "christian-bsl" / "videos"
    if christian_dir.exists():
        videos = sorted(christian_dir.glob("*.mp4"))
        print(f"\nChristian BSL: {len(videos)} videos")

        christian_new = 0
        for vpath in videos:
            video_key = f"christian_{vpath.name}"
            if video_key in cached_videos:
                continue

            # Filename is the sign label (e.g. ABRAHAM.mp4, AFRAID_-_SCARED_-_FEAR.mp4)
            # Take first term before " - " separator for multi-word entries
            raw_name = vpath.stem.split("_-_")[0]
            sign_name = raw_name.upper().replace("_", "-").replace("(", "").replace(")", "").strip()
            if not sign_name:
                continue

            t0 = time.time()
            raw_frames = extract_landmarks_from_video(vpath)
            if not raw_frames:
                continue

            smoothed = smooth_snapshots(raw_frames)
            segment = segment_stroke(smoothed)
            features = compute_segment_features(segment)
            if features is None:
                continue

            elapsed = time.time() - t0
            christian_new += 1
            new_videos += 1
            all_data.append({
                "sign": sign_name,
                "video": video_key,
                "features": features.tolist(),
                "segment_frames": len(segment),
                "raw_frames": len(raw_frames),
                "source": "christian-bsl",
                "segment": [
                    {"dom": s["dom"], "non": s["non"], "body": s["body"],
                     "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                     "timestamp": s["timestamp"]}
                    for s in segment
                ],
            })
            if christian_new % 50 == 0:
                print(f"  Christian [{christian_new}]: {sign_name} ({elapsed:.1f}s)")

        print(f"  Christian BSL: {christian_new} new videos extracted")

    # --- BKS (British Kinesiology Signs) ---
    bks_dir = research_dir / "bks" / "videos"
    bks_map_file = research_dir / "bks" / "sign_map.tsv"
    if bks_dir.exists():
        # Load label mapping if available
        bks_map = {}
        if bks_map_file.exists():
            with open(bks_map_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2 and parts[0] != "filename":
                        bks_map[parts[0]] = parts[1].upper().replace(" ", "-")

        videos = sorted(bks_dir.glob("*.mp4"))
        print(f"\nBKS: {len(videos)} videos, {len(bks_map)} mapped labels")

        bks_new = 0
        for vpath in videos:
            video_key = f"bks_{vpath.name}"
            if video_key in cached_videos:
                continue

            # Use sign_map.tsv label or derive from filename
            sign_name = bks_map.get(vpath.name)
            if not sign_name:
                sign_name = vpath.stem.upper().replace("-", " ").replace("_", " ").strip().replace(" ", "-")
            if not sign_name:
                continue

            t0 = time.time()
            raw_frames = extract_landmarks_from_video(vpath)
            if not raw_frames:
                continue

            smoothed = smooth_snapshots(raw_frames)
            segment = segment_stroke(smoothed)
            features = compute_segment_features(segment)
            if features is None:
                continue

            elapsed = time.time() - t0
            bks_new += 1
            new_videos += 1
            all_data.append({
                "sign": sign_name,
                "video": video_key,
                "features": features.tolist(),
                "segment_frames": len(segment),
                "raw_frames": len(raw_frames),
                "source": "bks",
                "segment": [
                    {"dom": s["dom"], "non": s["non"], "body": s["body"],
                     "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                     "timestamp": s["timestamp"]}
                    for s in segment
                ],
            })
            if bks_new % 50 == 0:
                print(f"  BKS [{bks_new}]: {sign_name} ({elapsed:.1f}s)")

        print(f"  BKS: {bks_new} new videos extracted")

    # --- NZSL (New Zealand Sign Language — 82% identical to BSL) ---
    nzsl_dir = research_dir / "nzsl"
    nzsl_data_dir = nzsl_dir / "data"
    nzsl_video_dir = nzsl_dir / "video"
    if nzsl_data_dir.exists() and nzsl_video_dir.exists():
        # Load metadata to get gloss→video mapping
        json_files = sorted(nzsl_data_dir.glob("*.json"))
        print(f"\nNZSL: {len(json_files)} signs with metadata")

        nzsl_new = 0
        for jf in json_files:
            try:
                with open(jf) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            nzsl_id = meta.get("nzsl_id", jf.stem)
            video_name = meta.get("video", "")
            if not video_name:
                continue

            video_key = f"nzsl_{video_name}"
            if video_key in cached_videos:
                continue

            # Use primary English gloss as label
            glosses = meta.get("gloss", {}).get("english", [])
            if not glosses:
                continue
            sign_name = glosses[0].upper().replace(" ", "-").replace("(", "").replace(")", "").strip()
            if not sign_name:
                continue

            # Find video file
            vpath = nzsl_video_dir / str(nzsl_id) / video_name
            if not vpath.exists():
                continue

            t0 = time.time()
            raw_frames = extract_landmarks_from_video(vpath)
            if not raw_frames:
                continue

            smoothed = smooth_snapshots(raw_frames)
            segment = segment_stroke(smoothed)
            features = compute_segment_features(segment)
            if features is None:
                continue

            elapsed = time.time() - t0
            nzsl_new += 1
            new_videos += 1
            all_data.append({
                "sign": sign_name,
                "video": video_key,
                "features": features.tolist(),
                "segment_frames": len(segment),
                "raw_frames": len(raw_frames),
                "source": "nzsl",
                "segment": [
                    {"dom": s["dom"], "non": s["non"], "body": s["body"],
                     "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                     "timestamp": s["timestamp"]}
                    for s in segment
                ],
            })
            if nzsl_new % 100 == 0:
                print(f"  NZSL [{nzsl_new}]: {sign_name} ({elapsed:.1f}s)")

            # Checkpoint every 500 new NZSL videos
            if cache_path and nzsl_new % 500 == 0:
                with open(cache_path, "w") as f:
                    json.dump(all_data, f)
                print(f"  [checkpoint] {len(all_data)} total features saved")

        print(f"  NZSL: {nzsl_new} new videos extracted")

    # --- BSL SignBank ---
    signbank_dir = research_dir / "bsl-signbank" / "videos"
    if signbank_dir.exists():
        videos = sorted(signbank_dir.glob("*.mp4"))
        print(f"\nBSL SignBank: {len(videos)} videos")

        signbank_new = 0
        for vpath in videos:
            video_key = f"signbank_{vpath.name}"
            if video_key in cached_videos:
                continue

            sign_name = vpath.stem.upper().replace("-", " ").replace("_", " ").strip().replace(" ", "-")
            if not sign_name:
                continue

            t0 = time.time()
            raw_frames = extract_landmarks_from_video(vpath)
            if not raw_frames:
                continue

            smoothed = smooth_snapshots(raw_frames)
            segment = segment_stroke(smoothed)
            features = compute_segment_features(segment)
            if features is None:
                continue

            elapsed = time.time() - t0
            signbank_new += 1
            new_videos += 1
            all_data.append({
                "sign": sign_name,
                "video": video_key,
                "features": features.tolist(),
                "segment_frames": len(segment),
                "raw_frames": len(raw_frames),
                "source": "bsl-signbank",
                "segment": [
                    {"dom": s["dom"], "non": s["non"], "body": s["body"],
                     "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                     "timestamp": s["timestamp"]}
                    for s in segment
                ],
            })
            if signbank_new % 50 == 0:
                print(f"  SignBank [{signbank_new}]: {sign_name} ({elapsed:.1f}s)")

        print(f"  BSL SignBank: {signbank_new} new videos extracted")

    # --- Any generic additional source directories ---
    # Scans for research/*/videos/*.mp4 that haven't been handled above
    known_sources = {"bsldict", "dicta-sign", "ssc-stem", "christian-bsl", "bks", "nzsl", "bsl-signbank", "ml-training"}
    for source_dir in sorted(research_dir.iterdir()):
        if not source_dir.is_dir() or source_dir.name in known_sources:
            continue
        vids_dir = source_dir / "videos"
        if not vids_dir.exists():
            continue
        videos = sorted(vids_dir.glob("*.mp4"))
        if not videos:
            continue

        print(f"\n{source_dir.name}: {len(videos)} videos")
        source_new = 0
        for vpath in videos:
            video_key = f"{source_dir.name}_{vpath.name}"
            if video_key in cached_videos:
                continue

            sign_name = vpath.stem.upper().replace("-", " ").replace("_", " ").strip().replace(" ", "-")
            if not sign_name:
                continue

            t0 = time.time()
            raw_frames = extract_landmarks_from_video(vpath)
            if not raw_frames:
                continue

            smoothed = smooth_snapshots(raw_frames)
            segment = segment_stroke(smoothed)
            features = compute_segment_features(segment)
            if features is None:
                continue

            elapsed = time.time() - t0
            source_new += 1
            new_videos += 1
            all_data.append({
                "sign": sign_name,
                "video": video_key,
                "features": features.tolist(),
                "segment_frames": len(segment),
                "raw_frames": len(raw_frames),
                "source": source_dir.name,
                "segment": [
                    {"dom": s["dom"], "non": s["non"], "body": s["body"],
                     "distances": s["distances"], "shoulderWidth": s["shoulderWidth"],
                     "timestamp": s["timestamp"]}
                    for s in segment
                ],
            })
            if source_new % 50 == 0:
                print(f"  {source_dir.name} [{source_new}]: {sign_name} ({elapsed:.1f}s)")

        print(f"  {source_dir.name}: {source_new} new videos extracted")

    if cache_path and new_videos > 0:
        with open(cache_path, "w") as f:
            json.dump(all_data, f)
        print(f"\nSaved combined cache: {len(all_data)} total ({new_videos} new from additional sources)")

    return all_data


def build_augmented_dataset(all_data, n_augmentations):
    """Generate augmented training data from raw segments."""
    print(f"\nAugmenting data ({n_augmentations} variations per sign)...")
    rng = np.random.default_rng(42)

    features_list = []
    labels_list = []
    label_map = sorted(set(d["sign"] for d in all_data))
    sign_to_idx = {s: i for i, s in enumerate(label_map)}

    for d in all_data:
        sign = d["sign"]
        idx = sign_to_idx[sign]
        segment = d["segment"]

        # Original
        orig_features = compute_segment_features(segment)
        if orig_features is not None:
            features_list.append(orig_features)
            labels_list.append(idx)

        # Augmented
        for _ in range(n_augmentations):
            aug_segment = augment_segment(segment, rng)
            aug_features = compute_segment_features(aug_segment)
            if aug_features is not None:
                features_list.append(aug_features)
                labels_list.append(idx)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(label_map)} classes")
    return X, y, label_map


def train_model(X, y, label_map, n_folds=N_FOLDS):
    """Train with stratified k-fold cross-validation."""
    print(f"\nTraining on {DEVICE} with {n_folds}-fold CV...")

    input_dim = X.shape[1]
    num_classes = len(label_map)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_top5_accuracies = []
    best_model_state = None
    best_val_acc = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardise features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # To tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = BSLClassifier(input_dim, num_classes, HIDDEN_DIMS, DROPOUT).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_fold_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            preds = logits.argmax(dim=1)
            top1_acc = (preds == y_val_t).float().mean().item()
            # Top-5
            _, top5_preds = logits.topk(min(5, num_classes), dim=1)
            top5_correct = (top5_preds == y_val_t.unsqueeze(1)).any(dim=1).float().mean().item()

        fold_accuracies.append(top1_acc)
        fold_top5_accuracies.append(top5_correct)
        print(f"  Fold {fold+1}: Top-1 = {top1_acc*100:.1f}%, Top-5 = {top5_correct*100:.1f}%")

        if top1_acc > best_val_acc:
            best_val_acc = top1_acc
            best_model_state = model.state_dict().copy()
            best_scaler = scaler

    mean_top1 = np.mean(fold_accuracies)
    mean_top5 = np.mean(fold_top5_accuracies)
    print(f"\n  Mean Top-1: {mean_top1*100:.1f}% ± {np.std(fold_accuracies)*100:.1f}%")
    print(f"  Mean Top-5: {mean_top5*100:.1f}% ± {np.std(fold_top5_accuracies)*100:.1f}%")

    # Train final model on ALL data
    print("\nTraining final model on all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = BSLClassifier(input_dim, num_classes, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(out := model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model, scaler, label_map, {"top1": mean_top1, "top5": mean_top5}


def export_onnx(model, scaler, label_map, input_dim, output_path, metadata_path):
    """Export PyTorch model to ONNX for browser inference."""
    model.eval()
    model.to("cpu")

    dummy = torch.randn(1, input_dim, dtype=torch.float32)
    # Use dynamo=False for compatibility with opset 13 and ONNX Runtime Web
    export_kwargs = dict(
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    try:
        torch.onnx.export(model, dummy, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older PyTorch doesn't have dynamo parameter
        torch.onnx.export(model, dummy, str(output_path), **export_kwargs)

    # Save metadata (label map + scaler params for JS)
    metadata = {
        "version": 1,
        "labels": label_map,
        "num_classes": len(label_map),
        "input_dim": input_dim,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "hidden_dims": HIDDEN_DIMS,
        "augmentations_per_sample": AUGMENTATIONS_PER_SAMPLE,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    model_size = os.path.getsize(output_path) / 1024
    print(f"\nExported: {output_path} ({model_size:.0f} KB)")
    print(f"Metadata: {metadata_path}")
    print(f"Labels: {len(label_map)} classes")


def run_leave_one_out_eval(all_data, label_map):
    """Leave-one-out evaluation matching the browser batch test methodology."""
    print("\n" + "=" * 60)
    print("Leave-One-Out Evaluation (matches browser batch test)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    sign_to_idx = {s: i for i, s in enumerate(label_map)}
    passes = []
    fails = []

    for i, test_sample in enumerate(all_data):
        test_sign = test_sample["sign"]
        test_features = np.array(test_sample["features"], dtype=np.float32).reshape(1, -1)

        # Train on everything except this sign
        train_data = [d for d in all_data if d["sign"] != test_sign]

        # Build augmented training set
        features_list = []
        labels_list = []
        for d in train_data:
            idx = sign_to_idx[d["sign"]]
            seg = d["segment"]
            orig = compute_segment_features(seg)
            if orig is not None:
                features_list.append(orig)
                labels_list.append(idx)
            for _ in range(AUGMENTATIONS_PER_SAMPLE):
                aug = augment_segment(seg, rng)
                af = compute_segment_features(aug)
                if af is not None:
                    features_list.append(af)
                    labels_list.append(idx)

        X_train = np.array(features_list, dtype=np.float32)
        y_train = np.array(labels_list, dtype=np.int64)

        # Standardise
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(test_features)

        # Train
        input_dim = X_train.shape[1]
        num_classes = len(label_map)
        model = BSLClassifier(input_dim, num_classes, HIDDEN_DIMS, DROPOUT).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        X_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Predict
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
            logits = model(X_test_t)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            pred_sign = label_map[pred_idx]
            pred_conf = probs[0, pred_idx].item()
            # Top-5
            top5_vals, top5_idxs = probs.topk(5, dim=1)
            top5 = [(label_map[idx.item()], val.item()) for idx, val in zip(top5_idxs[0], top5_vals[0])]

        correct = pred_sign == test_sign
        if correct:
            passes.append(test_sign)
        else:
            fails.append(test_sign)

        status = "✓ PASS" if correct else "✗ FAIL"
        top5_str = ", ".join(f"{s}={v*100:.0f}%" for s, v in top5[:3])
        print(f"  [{i+1}/{len(all_data)}] {test_sign}: {pred_sign} ({pred_conf*100:.0f}%) [{top5_str}] {status}")

    print(f"\n{'=' * 60}")
    print(f"Leave-One-Out Results: {len(passes)}/{len(all_data)} = {len(passes)/len(all_data)*100:.1f}%")
    print(f"Passes: {', '.join(passes) if passes else 'none'}")
    print(f"{'=' * 60}")
    return passes, fails


# ─── Main ─────────────────────────────────────────────────────────

def evaluate_on_test_videos(model, scaler, label_map, test_data):
    """Evaluate trained model on held-out test videos (our 119 benchmark)."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating on {len(test_data)} test videos...")
    print(f"{'=' * 60}")

    sign_to_idx = {s: i for i, s in enumerate(label_map)}
    passes = []
    fails = []
    not_in_train = []

    model.eval()
    model.to(DEVICE)

    for i, sample in enumerate(test_data):
        test_sign = sample["sign"]
        if test_sign not in sign_to_idx:
            not_in_train.append(test_sign)
            continue

        features = np.array(sample["features"], dtype=np.float32).reshape(1, -1)
        features_scaled = scaler.transform(features)
        X_t = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            pred_sign = label_map[pred_idx]
            pred_conf = probs[0, pred_idx].item()
            top5_vals, top5_idxs = probs.topk(min(5, len(label_map)), dim=1)
            top5 = [(label_map[idx.item()], val.item()) for idx, val in zip(top5_idxs[0], top5_vals[0])]

        correct = pred_sign == test_sign
        if correct:
            passes.append(test_sign)
        else:
            fails.append(test_sign)

        status = "PASS" if correct else "FAIL"
        top5_str = ", ".join(f"{s}={v*100:.0f}%" for s, v in top5[:3])
        print(f"  [{i+1}/{len(test_data)}] {test_sign}: {pred_sign} ({pred_conf*100:.0f}%) [{top5_str}] {status}")

    print(f"\n{'=' * 60}")
    total = len(passes) + len(fails)
    print(f"Test Results: {len(passes)}/{total} = {len(passes)/max(total,1)*100:.1f}%")
    print(f"Passes: {', '.join(sorted(passes)) if passes else 'none'}")
    if not_in_train:
        print(f"Skipped (not in training labels): {', '.join(not_in_train)}")
    print(f"{'=' * 60}")
    return passes, fails


def main():
    parser = argparse.ArgumentParser(description="BSL ML Training Pipeline")
    parser.add_argument("--extract-only", action="store_true", help="Only extract features")
    parser.add_argument("--train-only", action="store_true", help="Train from cached features")
    parser.add_argument("--loo", action="store_true", help="Run leave-one-out evaluation")
    parser.add_argument("--bsldict", action="store_true",
                        help="Train on BSLDict multi-signer videos, test on original 119 videos")
    parser.add_argument("--augmentations", type=int, default=None,
                        help=f"Augmentations per sample (default: 100 for single-video, 20 for BSLDict)")
    args = parser.parse_args()

    # Default augmentations: fewer for BSLDict since we have real variance
    if args.augmentations is not None:
        n_augmentations = args.augmentations
    else:
        n_augmentations = 20 if args.bsldict else AUGMENTATIONS_PER_SAMPLE

    if args.bsldict:
        return main_bsldict(args, n_augmentations)

    # ── Original single-video pipeline ──
    # Step 1: Extract features
    if args.train_only and FEATURES_CACHE.exists():
        print(f"Loading cached features from {FEATURES_CACHE}")
        with open(FEATURES_CACHE) as f:
            all_data = json.load(f)
        print(f"  Loaded {len(all_data)} signs")
    else:
        t0 = time.time()
        all_data = extract_all_features(VIDEO_DIR)
        elapsed = time.time() - t0
        print(f"  Feature extraction took {elapsed:.1f}s ({elapsed/len(all_data):.1f}s per video)")

        # Cache
        with open(FEATURES_CACHE, "w") as f:
            json.dump(all_data, f)
        print(f"  Cached to {FEATURES_CACHE}")

        if args.extract_only:
            return

    label_map = sorted(set(d["sign"] for d in all_data))

    # Leave-one-out evaluation
    if args.loo:
        run_leave_one_out_eval(all_data, label_map)
        return

    # Step 2: Augment
    X, y, label_map = build_augmented_dataset(all_data, n_augmentations)

    # Step 3: Train with cross-validation
    model, scaler, label_map, metrics = train_model(X, y, label_map)

    # Step 4: Export
    export_onnx(model, scaler, label_map, X.shape[1], MODEL_OUTPUT, METADATA_OUTPUT)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  CV Top-1: {metrics['top1']*100:.1f}%")
    print(f"  CV Top-5: {metrics['top5']*100:.1f}%")
    print(f"  Model: {MODEL_OUTPUT}")
    print(f"  Metadata: {METADATA_OUTPUT}")
    print(f"{'=' * 60}")


def main_bsldict(args, n_augmentations):
    """Train on BSLDict multi-signer videos, evaluate on test videos."""
    cache = BSLDICT_FEATURES_CACHE

    # Step 1: Extract BSLDict training features
    if args.train_only and cache.exists():
        print(f"Loading cached BSLDict features from {cache}")
        with open(cache) as f:
            train_data = json.load(f)
        print(f"  Loaded {len(train_data)} training samples")
    else:
        if not BSLDICT_DIR.exists():
            print(f"ERROR: BSLDict videos not found at {BSLDICT_DIR}")
            print(f"Run: python3 research/bsldict/download_videos.py")
            sys.exit(1)

        t0 = time.time()
        train_data = extract_bsldict_features(BSLDICT_DIR, existing_cache=cache)
        elapsed = time.time() - t0
        print(f"  Feature extraction took {elapsed:.1f}s ({elapsed/max(len(train_data),1):.1f}s per video)")

        with open(cache, "w") as f:
            json.dump(train_data, f)
        print(f"  Cached to {cache}")

        if args.extract_only:
            return

    # Step 1b: Add features from additional sources (Dicta-Sign, SSC STEM)
    if not args.extract_only:
        train_data = extract_additional_sources(
            BSLDICT_DIR, existing_data=train_data, cache_path=cache
        )

    # Step 2: Extract test video features (always needed for evaluation)
    test_cache = FEATURES_CACHE
    if test_cache.exists():
        print(f"\nLoading cached test features from {test_cache}")
        with open(test_cache) as f:
            test_data = json.load(f)
        print(f"  Loaded {len(test_data)} test samples")
    else:
        print("\nExtracting test video features...")
        t0 = time.time()
        test_data = extract_all_features(VIDEO_DIR)
        elapsed = time.time() - t0
        print(f"  Test feature extraction took {elapsed:.1f}s")
        with open(test_cache, "w") as f:
            json.dump(test_data, f)

    label_map = sorted(set(d["sign"] for d in train_data))
    n_signs = len(label_map)
    n_videos = len(train_data)
    videos_per_sign = {}
    for d in train_data:
        videos_per_sign[d["sign"]] = videos_per_sign.get(d["sign"], 0) + 1

    # Scale architecture for large vocabularies
    global HIDDEN_DIMS, EPOCHS, BATCH_SIZE, AUGMENTATIONS_PER_SAMPLE
    if n_signs > 5000:
        HIDDEN_DIMS = MASSIVE_HIDDEN_DIMS
        EPOCHS = MASSIVE_EPOCHS
        BATCH_SIZE = MASSIVE_BATCH_SIZE
        AUGMENTATIONS_PER_SAMPLE = min(n_augmentations, 10)  # Less augmentation needed with more real data
        print(f"\nMassive vocabulary mode: {n_signs} signs")
        print(f"  Architecture: {HIDDEN_DIMS}, {EPOCHS} epochs, batch {BATCH_SIZE}, {AUGMENTATIONS_PER_SAMPLE} augmentations")
    elif n_signs > 2000:
        HIDDEN_DIMS = XLARGE_HIDDEN_DIMS
        EPOCHS = XLARGE_EPOCHS
        BATCH_SIZE = XLARGE_BATCH_SIZE
        AUGMENTATIONS_PER_SAMPLE = min(n_augmentations, 15)
        print(f"\nExtra-large vocabulary mode: {n_signs} signs")
        print(f"  Architecture: {HIDDEN_DIMS}, {EPOCHS} epochs, batch {BATCH_SIZE}, {AUGMENTATIONS_PER_SAMPLE} augmentations")
    elif n_signs > 200:
        HIDDEN_DIMS = LARGE_HIDDEN_DIMS
        EPOCHS = LARGE_EPOCHS
        BATCH_SIZE = LARGE_BATCH_SIZE
        print(f"\nLarge vocabulary mode: {n_signs} signs")
        print(f"  Architecture: {HIDDEN_DIMS}, {EPOCHS} epochs, batch {BATCH_SIZE}")

    print(f"\nTraining set: {n_videos} videos, {n_signs} signs")
    print(f"  Videos/sign: min={min(videos_per_sign.values())}, max={max(videos_per_sign.values())}, avg={n_videos/n_signs:.1f}")
    print(f"Test set: {len(test_data)} videos")

    # Step 3: Augment training data (AUGMENTATIONS_PER_SAMPLE may have been reduced for massive vocab)
    X, y, label_map = build_augmented_dataset(train_data, AUGMENTATIONS_PER_SAMPLE)

    # Step 4: Train with cross-validation
    model, scaler, label_map, metrics = train_model(X, y, label_map)

    # Step 5: Evaluate on held-out test videos
    passes, fails = evaluate_on_test_videos(model, scaler, label_map, test_data)

    # Step 6: Export ONNX
    export_onnx(model, scaler, label_map, X.shape[1], MODEL_OUTPUT, METADATA_OUTPUT)

    print(f"\n{'=' * 60}")
    print(f"BSLDict Pipeline complete!")
    print(f"  Training: {n_videos} videos, {n_signs} signs, {n_augmentations} augmentations")
    print(f"  CV Top-1: {metrics['top1']*100:.1f}%")
    print(f"  CV Top-5: {metrics['top5']*100:.1f}%")
    print(f"  Test accuracy: {len(passes)}/{len(passes)+len(fails)} = {len(passes)/max(len(passes)+len(fails),1)*100:.1f}%")
    print(f"  Test passes: {', '.join(sorted(passes)) if passes else 'none'}")
    print(f"  Model: {MODEL_OUTPUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
