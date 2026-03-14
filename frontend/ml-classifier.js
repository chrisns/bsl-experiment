/**
 * ML-based BSL sign classifier using ONNX Runtime Web.
 *
 * Loads a trained ONNX model and performs inference on segment features
 * extracted from MediaPipe landmarks. Mirrors the Python feature extraction
 * pipeline in extract_and_train.py.
 *
 * Usage:
 *   await MLClassifier.init('bsl_classifier.onnx', 'model_metadata.json');
 *   const result = MLClassifier.classify(segment);  // segment = array of snapshots
 *   // result = { predictions: [{sign, confidence}, ...], topSign, topConfidence }
 */
const MLClassifier = (function() {
    'use strict';

    let session = null;
    let metadata = null;
    let ready = false;

    // ─── Initialization ──────────────────────────────────────────

    async function init(modelUrl, metadataUrl) {
        try {
            // Cache-bust to prevent stale model/metadata mismatch
            const cacheBust = `?v=${Date.now()}`;
            const [metaResp] = await Promise.all([
                fetch(metadataUrl + cacheBust).then(r => r.json()),
            ]);
            metadata = metaResp;

            session = await ort.InferenceSession.create(modelUrl + cacheBust);
            ready = true;
            console.log(`[MLClassifier] Ready: ${metadata.num_classes} classes, ${metadata.input_dim} features`);
            return true;
        } catch (e) {
            console.error('[MLClassifier] Failed to initialize:', e);
            ready = false;
            return false;
        }
    }

    function isReady() {
        return ready;
    }

    // ─── Feature Extraction (mirrors Python compute_segment_features) ──

    function v2dist(a, b) {
        return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
    }

    function extractDtwFrame(snapshot) {
        const noseX = snapshot.body ? snapshot.body.nose.x : 0;
        const noseY = snapshot.body ? snapshot.body.nose.y : 0;
        const dom = snapshot.dom;
        const non = snapshot.non;
        return [
            dom.pos.x - noseX,
            dom.pos.y - noseY,
            non.present ? (non.pos.x - noseX) : 0,
            non.present ? (non.pos.y - noseY) : 0,
            ...dom.fingerExtension,
            dom.openness,
            dom.spread,
            dom.palmCross,
            dom.fingerDir.x,
            dom.fingerDir.y,
            dom.pinchDist,
            non.present ? 1 : 0,
            snapshot.distances ? snapshot.distances.handDist : 0,
        ];
    }

    function computeSegmentFeatures(segment) {
        // Filter to frames with body landmarks (matches Python)
        const frames = segment
            .filter(s => s.body)
            .map(s => extractDtwFrame(s));

        if (frames.length < 3) return null;

        const T = frames.length;
        const D = frames[0].length; // 17

        const features = [];

        // 1. Per-dimension statistics (17 dims x 7 stats = 119 features)
        for (let d = 0; d < D; d++) {
            const col = frames.map(f => f[d]);
            col.sort((a, b) => a - b);
            const mean = col.reduce((s, v) => s + v, 0) / T;
            const std = Math.sqrt(col.reduce((s, v) => s + (v - mean) ** 2, 0) / T);
            const min = col[0];
            const max = col[col.length - 1];
            const range = max - min;
            const median = T % 2 === 1 ? col[Math.floor(T / 2)]
                : (col[T / 2 - 1] + col[T / 2]) / 2;
            // Delta uses original order, not sorted
            const origCol = frames.map(f => f[d]);
            const delta = origCol[origCol.length - 1] - origCol[0];

            features.push(mean, std, min, max, range, median, delta);
        }

        // 2. Trajectory features (dom hand position = dims 0, 1)
        const domX = frames.map(f => f[0]);
        const domY = frames.map(f => f[1]);

        const dispX = domX[domX.length - 1] - domX[0];
        const dispY = domY[domY.length - 1] - domY[0];
        const displacement = Math.sqrt(dispX ** 2 + dispY ** 2);

        let pathLength = 0;
        const stepDists = [];
        const dx = [];
        const dy = [];
        for (let i = 1; i < T; i++) {
            const ddx = domX[i] - domX[i - 1];
            const ddy = domY[i] - domY[i - 1];
            dx.push(ddx);
            dy.push(ddy);
            const sd = Math.sqrt(ddx ** 2 + ddy ** 2);
            stepDists.push(sd);
            pathLength += sd;
        }

        const straightness = displacement / Math.max(pathLength, 1e-6);
        const speedMean = stepDists.reduce((s, v) => s + v, 0) / stepDists.length;
        const speedMax = Math.max(...stepDists);
        const speedStd = Math.sqrt(stepDists.reduce((s, v) => s + (v - speedMean) ** 2, 0) / stepDists.length);
        const direction = Math.atan2(dispY, dispX);

        // Winding angle
        const angles = dx.map((ddx, i) => Math.atan2(dy[i], ddx));
        let winding = 0;
        for (let i = 1; i < angles.length; i++) {
            let diff = angles[i] - angles[i - 1];
            // Wrap to [-pi, pi]
            diff = ((diff + Math.PI) % (2 * Math.PI)) - Math.PI;
            if (diff < -Math.PI) diff += 2 * Math.PI;
            winding += Math.abs(diff);
        }

        // Reversal counts
        let xReversals = 0, yReversals = 0;
        for (let i = 1; i < dx.length; i++) {
            if (Math.sign(dx[i]) !== Math.sign(dx[i - 1]) && Math.sign(dx[i]) !== 0 && Math.sign(dx[i - 1]) !== 0) xReversals++;
            if (Math.sign(dy[i]) !== Math.sign(dy[i - 1]) && Math.sign(dy[i]) !== 0 && Math.sign(dy[i - 1]) !== 0) yReversals++;
        }

        features.push(
            displacement, pathLength, straightness,
            speedMean, speedMax, speedStd,
            direction, winding,
            xReversals / Math.max(T, 1), yReversals / Math.max(T, 1),
        );

        // 3. Non-dom hand features
        const nonPresentRatio = frames.reduce((s, f) => s + f[15], 0) / T;
        const handDistMean = frames.reduce((s, f) => s + f[16], 0) / T;
        const handDistDelta = frames[frames.length - 1][16] - frames[0][16];
        features.push(nonPresentRatio, handDistMean, handDistDelta);

        // 4. Body-relative distances from raw snapshots
        const bodyFrames = segment.filter(s => s.body);
        const distKeys = ['domToNose', 'domToChin', 'domToChest'];
        for (const key of distKeys) {
            const vals = bodyFrames.map(s => s.distances[key]).filter(v => v != null);
            if (vals.length > 0) {
                const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
                const min = Math.min(...vals);
                const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
                features.push(mean, min, std);
            } else {
                features.push(0, 0, 0);
            }
        }

        // 5. Segment metadata
        features.push(T / 30.0);

        return features;
    }

    // ─── Inference ───────────────────────────────────────────────

    function standardize(features) {
        const mean = metadata.scaler_mean;
        const std = metadata.scaler_std;
        return features.map((v, i) => (v - mean[i]) / (std[i] || 1));
    }

    async function classify(segment, topK = 5) {
        if (!ready || !session || !metadata) return null;

        const features = computeSegmentFeatures(segment);
        if (!features || features.length !== metadata.input_dim) {
            console.warn(`[MLClassifier] Feature dim mismatch: got ${features ? features.length : 0}, expected ${metadata.input_dim}`);
            return null;
        }

        const scaled = standardize(features);
        const inputTensor = new ort.Tensor('float32', new Float32Array(scaled), [1, metadata.input_dim]);

        const results = await session.run({ features: inputTensor });
        const logits = results.logits.data;

        // Softmax
        const maxLogit = Math.max(...logits);
        const exps = Array.from(logits).map(v => Math.exp(v - maxLogit));
        const sumExp = exps.reduce((s, v) => s + v, 0);
        const probs = exps.map(v => v / sumExp);

        // Top-K predictions
        const indexed = probs.map((p, i) => ({ sign: metadata.labels[i], confidence: p, index: i }));
        indexed.sort((a, b) => b.confidence - a.confidence);
        const predictions = indexed.slice(0, topK);

        return {
            predictions,
            topSign: predictions[0].sign,
            topConfidence: predictions[0].confidence,
        };
    }

    // Synchronous classification (for integration with recogniseSegment)
    function classifySync(segment, topK = 5) {
        if (!ready || !session || !metadata) return null;

        const features = computeSegmentFeatures(segment);
        if (!features || features.length !== metadata.input_dim) return null;

        // Return a promise that the caller can await
        const scaled = standardize(features);
        const inputTensor = new ort.Tensor('float32', new Float32Array(scaled), [1, metadata.input_dim]);

        return session.run({ features: inputTensor }).then(results => {
            const logits = results.logits.data;
            const maxLogit = Math.max(...logits);
            const exps = Array.from(logits).map(v => Math.exp(v - maxLogit));
            const sumExp = exps.reduce((s, v) => s + v, 0);
            const probs = exps.map(v => v / sumExp);

            const indexed = probs.map((p, i) => ({ sign: metadata.labels[i], confidence: p }));
            indexed.sort((a, b) => b.confidence - a.confidence);
            return indexed.slice(0, topK);
        });
    }

    // ─── Public API ──────────────────────────────────────────────

    return {
        init,
        isReady,
        classify,
        classifySync,
        computeSegmentFeatures,
    };
})();
