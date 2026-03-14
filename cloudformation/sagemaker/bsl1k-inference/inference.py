"""
SageMaker inference handler for BSL-1K I3D model.
Matches github.com/gulvarol/bsl1k demo/demo.py preprocessing exactly.
"""
import io
import json
import math
import base64
import logging
import pickle

import cv2
import numpy as np
import torch
from scipy.special import softmax

from model import InceptionI3d

logger = logging.getLogger(__name__)

# Preprocessing constants (from BSL-1K demo.py)
RESIZE_RES = 256
INP_RES = 224
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)
NUM_IN_FRAMES = 16
STRIDE = 4


def model_fn(model_dir):
    """Load I3D model from model_dir. Direct checkpoint loading — no key remapping."""
    logger.info(f"Loading BSL-1K model from {model_dir}")

    # Load vocabulary
    info_path = f"{model_dir}/info.pkl"
    with open(info_path, "rb") as f:
        info = pickle.load(f)
    classes = info["words"]
    num_classes = len(classes)
    logger.info(f"Loaded vocabulary with {num_classes} classes")

    # Create model matching original BSL-1K architecture exactly
    model = InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=NUM_IN_FRAMES,
    )

    # Load checkpoint — wrap in DataParallel then load (matching original demo.py)
    checkpoint_path = f"{model_dir}/bsl1k.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # The checkpoint was saved from DataParallel, so keys have 'module.' prefix.
    # Wrap model in DataParallel, load, then unwrap.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info(f"Loaded {len(state_dict)} parameters on {device}")
    return {"model": model, "classes": classes, "device": device}


def input_fn(request_body, content_type):
    """Parse input: JSON with base64-encoded JPEG frames array.
    Preprocess following BSL-1K demo.py exactly: resize 256x256, center crop 224, normalize.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)
    frames_b64 = data.get("frames", [])

    if not frames_b64:
        raise ValueError("No frames provided")

    frames = []
    for frame_b64 in frames_b64:
        img_bytes = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 256x256 (NOT preserving aspect ratio — matches original resize_generic)
        img = cv2.resize(img, (RESIZE_RES, RESIZE_RES), interpolation=cv2.INTER_LINEAR)

        # Center crop to 224x224
        ulx = (RESIZE_RES - INP_RES) // 2
        uly = (RESIZE_RES - INP_RES) // 2
        img = img[uly:uly + INP_RES, ulx:ulx + INP_RES]

        # Normalize to [0, 1] then apply color_normalize: (x - mean) / std
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD

        frames.append(img)

    if not frames:
        raise ValueError("No valid frames decoded")

    # Sliding window processing (matches demo.py sliding_windows function)
    n_frames = len(frames)

    # Pad if fewer frames than needed
    if n_frames < NUM_IN_FRAMES:
        while len(frames) < NUM_IN_FRAMES:
            frames.append(frames[-1].copy())
        n_frames = len(frames)

    # Stack all frames: (T, H, W, C) -> (C, T, H, W)
    all_frames = np.stack(frames)
    rgb = torch.from_numpy(all_frames).permute(3, 0, 1, 2)  # (C, T, H, W)

    # Create sliding window clips
    num_clips = math.ceil((n_frames - NUM_IN_FRAMES) / STRIDE) + 1
    clips = torch.zeros(num_clips, 3, NUM_IN_FRAMES, INP_RES, INP_RES)
    for j in range(num_clips):
        actual_clip_length = min(NUM_IN_FRAMES, n_frames - j * STRIDE)
        if actual_clip_length == NUM_IN_FRAMES:
            t_beg = j * STRIDE
        else:
            t_beg = n_frames - NUM_IN_FRAMES
        clips[j] = rgb[:, t_beg:t_beg + NUM_IN_FRAMES, :, :]

    return {"clips": clips, "num_clips": num_clips, "batch_id": data.get("batchId", "unknown")}


def predict_fn(input_data, model_dict):
    """Run I3D inference with sliding windows. Use max-per-class aggregation."""
    model = model_dict["model"]
    classes = model_dict["classes"]
    device = model_dict["device"]

    clips = input_data["clips"].to(device)
    batch_id = input_data["batch_id"]
    num_clips = input_data["num_clips"]

    with torch.no_grad():
        # Forward pass on all clips (batch)
        logits = model(clips)  # (num_clips, num_classes)

        # Apply softmax per window, then take MAX probability per class.
        # This prevents noisy/neutral windows from diluting strong predictions.
        per_window_probs = softmax(logits.cpu().numpy(), axis=1)  # (num_clips, num_classes)

        if num_clips == 1:
            probs = per_window_probs[0]
        else:
            probs = per_window_probs.max(axis=0)

        # Top 10 predictions (more options for frontend smoothing)
        top_indices = np.argsort(probs)[::-1][:10]
        predictions = []
        for idx in top_indices:
            predictions.append({
                "sign": classes[idx],
                "confidence": float(probs[idx]),
            })

    return {"predictions": predictions, "batchId": batch_id}


def output_fn(prediction, accept):
    """Return JSON response."""
    return json.dumps(prediction), "application/json"
