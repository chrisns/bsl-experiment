#!/usr/bin/env python3
"""Train v19 model: v18 features + Auslan features, on GPU."""
import sys
sys.path.insert(0, '/home/ubuntu')

import json
import os
import numpy as np
from collections import Counter
import extract_and_train as et

def main():
    # Load v18 merged features
    print("Loading v18 merged features...")
    with open('/home/ubuntu/bsldict_features.json') as f:
        all_data = json.load(f)
    print(f"  v18 data: {len(all_data)} features")

    # Load Auslan features
    print("Loading Auslan features...")
    with open('/home/ubuntu/auslan_features.json') as f:
        auslan_data = json.load(f)
    print(f"  Auslan: {len(auslan_data)} features")
    all_data.extend(auslan_data)

    # Deduplicate by video key
    seen = set()
    unique_data = []
    for d in all_data:
        key = d['video']
        if key not in seen:
            seen.add(key)
            unique_data.append(d)
    print(f"\nTotal: {len(unique_data)} unique features (removed {len(all_data) - len(unique_data)} duplicates)")
    all_data = unique_data

    signs = set(d['sign'] for d in all_data)
    source_counts = Counter(d.get('source', 'bsldict') for d in all_data)
    print(f"Unique signs: {len(signs)}")
    print("Sources:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    # Architecture
    n_signs = len(signs)
    if n_signs > 5000:
        hidden_dims = et.MASSIVE_HIDDEN_DIMS
        augmentations = 10
        epochs = et.MASSIVE_EPOCHS
        batch_size = et.MASSIVE_BATCH_SIZE
    elif n_signs > 2000:
        hidden_dims = et.XLARGE_HIDDEN_DIMS
        augmentations = 15
        epochs = et.XLARGE_EPOCHS
        batch_size = et.XLARGE_BATCH_SIZE
    else:
        hidden_dims = et.LARGE_HIDDEN_DIMS
        augmentations = 20
        epochs = et.LARGE_EPOCHS
        batch_size = et.LARGE_BATCH_SIZE

    print(f"\nArchitecture: {hidden_dims}")
    print(f"Augmentations: {augmentations}, Epochs: {epochs}, Batch: {batch_size}")

    et.HIDDEN_DIMS = hidden_dims
    et.EPOCHS = epochs
    et.BATCH_SIZE = batch_size
    et.AUGMENTATIONS_PER_SAMPLE = augmentations

    X, y, label_map = et.build_augmented_dataset(all_data, augmentations)
    model, scaler, label_map, metrics = et.train_model(X, y, label_map)

    output_onnx = '/home/ubuntu/bsl_classifier.onnx'
    output_meta = '/home/ubuntu/model_metadata.json'
    et.export_onnx(model, scaler, label_map, X.shape[1], output_onnx, output_meta)

    print("\n=== Uploading v19 results to S3 ===")
    os.system(f"aws s3 cp {output_onnx} s3://bsl-training-305137865866/output/bsl_classifier_v19.onnx")
    os.system(f"aws s3 cp {output_meta} s3://bsl-training-305137865866/output/model_metadata_v19.json")

    print(f"\n{'=' * 60}")
    print(f"V19 TRAINING COMPLETE!")
    print(f"  Total data: {len(all_data)} videos, {n_signs} signs")
    print(f"  Architecture: {hidden_dims}")
    print(f"  CV Top-1: {metrics['top1']*100:.1f}%")
    print(f"  CV Top-5: {metrics['top5']*100:.1f}%")
    print(f"  Model: {output_onnx}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
