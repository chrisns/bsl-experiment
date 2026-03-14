# BSL Sign Recognition — Lessons Learned

## Architecture
- **MediaPipe Holistic** extracts face (468), hand (21x2), pose (33) landmarks
- Signs scored on 6 phonological parameters: Movement (0.30), Location (0.20), Handshape (0.15), Orientation (0.12), Two-handed (0.10), Contact (0.13)
- **Multiplicative scoring** (weighted geometric mean) — poor match in ANY dimension kills total
- 270 BSL signs in dictionary (`frontend/sign-dictionary.js`)
- Velocity-based segmentation with fallback for idle videos
- Batch test harness: `frontend/video-test.html` with 119 test videos at 30fps
- Local test server: `python3 -m http.server 9876` from `frontend/`

## Best Results: v16-expanded (102/119 = 85.7%, 944 signs)
- **v16**: Trained on 2786 BSLDict multi-signer videos (944 signs, avg 3.0 videos/sign), tested on original 119 test videos
- 102/119 passes — only 1 regression from v15 despite 8x vocabulary expansion (119→944 signs)
- Scaled MLP architecture (512→256→128), 120 epochs, batch 128, 20 augmentations, 142-dim features
- CV accuracy: 99.8% Top-1, 100% Top-5
- Model size: 1419KB ONNX
- 17 failures: AFTERNOON, CORRECT, FINISH, HEARING, HELLO, HOSPITAL, MORNING, NICE-TO-MEET-YOU, NIGHT, NUMBER, PHONE, SAD, STAND, START, STOP, TRAIN, WHAT
- Notable: CORRECT→RIGHT (100%), NIGHT→DARK (97%), START→GET (98%) are semantically correct but label mismatches
- HOSPITAL and STAND still only have 1 BSLDict training video each

## v15-bsldict (103/119 = 86.6%, 119 signs)
- **v15**: Trained on 515 BSLDict multi-signer videos (avg 4.3 signers/sign), tested on original 119 test videos
- 103/119 passes — 7.3x improvement over v14 (14/119 = 11.8%)
- **Multi-signer training data was the key breakthrough** — real variance from different signers replaces synthetic augmentation
- MLP architecture (256→128→64), 20 augmentations, 142-dim features
- CV accuracy: 99.7% Top-1, 100% Top-5

## Previous Results
- **v14** (ML only, single video): 14/119 (11.8%) — BIG, CAR, CLEAN, COMPUTER, DOG, EVENING, HOSPITAL, HOUSE, HOW, PAIN, RUN, SIGN, SMALL, TEACHER
- **v9** (categorical): 6/119 (5.0%) — FRIEND, HAPPY, LIKE, OLD, PHONE, YES
- **v13** (cat+DTW): 6/119 (5.0%) — CHILD, HAPPY, LIKE, OLD, PHONE, YES

## What Doesn't Work (Proven)
- **Continuous handshape/location scoring**: Too much partial credit, compresses score range (v10, v11)
- **Parameter tweaks**: Diminishing returns on categorical scoring
- **DTW with full-buffer templates** (v10-v12): Representation mismatch
- **Single-video ML** (v14): 1 video per sign can't generalise — augmentation doesn't capture real signer variance

## What Works (Confirmed)
- **Multi-signer ML training** (v15→v16): BSLDict multi-signer data is the key — scales from 119 to 944 signs with minimal accuracy loss
- **Statistical feature vectors** (142-dim): Per-dim stats + trajectory + body-relative distances
- **MLP classifier**: Scales from 256→128→64 (119 signs) to 512→256→128 (944 signs) with BatchNorm, Dropout(0.3), label smoothing
- **Vocabulary scaling**: 8x vocabulary increase (119→944) only lost 1 test case (86.6%→85.7%)

## BSLDict Data Source
- **BSLDict** (Oxford VGG): 14,210 video clips, 9,283 signs, 124 signers
- v15: Downloaded 515 videos for 119 signs
- v16: Downloaded 2,803 videos for 947 signs from signbsl.com (2,747 wget, 56 youtube-dl, 56 failed)
- Metadata: `research/bsldict/bsldict_v1.pkl` (147MB)
- Videos: `research/bsldict/videos/<SIGN>/<SIGN>_<N>.mp4`
- Download script: `research/bsldict/download_videos.py` (expanded with 1000-sign conversational vocabulary)

## Other BSL Data Sources (researched, not yet used)
- **BOBSL** (BBC-Oxford): 1,400 hours, 37 signers, 2,281 sign classes — requires BBC approval
- **BSLCP CAVA** (UCL): 249 deaf signers, 8 UK regions, ELAN annotations — open access lexical data
- **BSL SignBank** (UCL/DCAL): ~2,500 signs with video clips
- **SpreadTheSign**: 610,000+ videos across all sign languages
- **Kaggle BSL datasets**: MediaPipe keypoints for alphabet/numbers

## ML Pipeline
- **Training script**: `research/ml-training/extract_and_train.py`
  - `--bsldict` flag: train on BSLDict, test on original 119 videos
  - `--train-only`: skip extraction, use cached features
  - `--extract-only`: just extract and cache features
  - Features cached to `bsldict_features.json` (training) and `extracted_features.json` (test)
  - Auto-scales to LARGE_HIDDEN_DIMS [512,256,128] for 200+ class vocabularies
- **Browser integration**: `frontend/ml-classifier.js` + ONNX Runtime Web
- **Model**: `frontend/bsl_classifier.onnx` + `frontend/model_metadata.json`

## Key Files
- `frontend/sign-recogniser.js` (~3700 lines) — core recognition engine
- `frontend/ml-classifier.js` — ONNX Runtime Web classifier
- `frontend/bsl_classifier.onnx` — trained ONNX model
- `frontend/model_metadata.json` — label map + scaler params
- `frontend/sign-dictionary.js` (~2844 lines) — 270 sign definitions (categorical matching)
- `frontend/video-test.html` — batch test harness
- `research/ml-training/extract_and_train.py` — Python ML pipeline
- `research/bsldict/download_videos.py` — BSLDict video downloader (1000-sign conversational vocabulary)
- `research/batch-results-v*.json` — saved test results (v1-v15)
