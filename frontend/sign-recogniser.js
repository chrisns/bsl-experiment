/**
 * BSL Sign Recogniser — motion-first recognition using MediaPipe Holistic landmarks.
 *
 * Recognises 100+ BSL signs by analysing hand trajectories, handshapes, locations,
 * orientations, and contact over a rolling temporal buffer. Sign definitions are
 * data-driven using BSL phonological parameters (location, handshape, movement,
 * orientation, two-handed configuration, contact).
 *
 * v2: Adds sign segmentation (velocity-based stroke detection), non-manual feature
 * (NMF) recognition (head nod, shake, eyebrow raise, mouth patterns), segment-based
 * recognition, sentence accumulation, and alternative recognition paths (e.g. YES
 * can be recognised as either a fist-knock or a head nod).
 *
 * Camera is assumed to be in selfie/mirror mode. MediaPipe Holistic returns
 * landmarks in the mirrored frame, so "left" in the data is the signer's right
 * hand (dominant hand for most right-handed BSL signers). We label from the
 * SIGNER's perspective throughout: dom = signer's dominant (right) hand,
 * non = signer's non-dominant (left) hand.
 *
 * MediaPipe hand landmarks: 0=wrist, 4=thumb tip, 8=index tip, 12=middle tip,
 * 16=ring tip, 20=pinky tip. Pose: 0=nose, 11/12=shoulders, 13/14=elbows,
 * 15/16=wrists, 23/24=hips.
 *
 * MediaPipe face landmarks (468 points): 1=nose tip, 13=upper lip, 14=lower lip,
 * 70/63=left brow, 300/293=right brow, 234=right ear tragion, 454=left ear tragion.
 */
const SignRecogniser = (() => {
    'use strict';

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════════

    const BUFFER_SIZE = 60;          // ~2s at 30fps
    const MIN_FRAMES = 12;           // minimum frames before recognition
    const SMOOTHING_ALPHA = 0.4;     // exponential smoothing (0 = no smooth, 1 = no change)
    const MATCH_THRESHOLD = 0.30;    // minimum total score to fire recognition (multiplicative scoring produces lower values)
    const COOLDOWN_MS = 1200;        // minimum ms between same-sign recognitions
    const TEMPORAL_CONSISTENCY = 8;  // frames the top sign must lead to fire
    const TOP_N = 5;                 // number of alternatives to return

    // Phonological parameter weights (must sum to ~1.0 for interpretability)
    const W_MOVEMENT = 0.30;
    const W_LOCATION = 0.20;
    const W_HANDSHAPE = 0.15;
    const W_ORIENTATION = 0.12;
    const W_TWO_HANDED = 0.10;
    const W_CONTACT = 0.13;

    // Segmentation configuration
    const SEG_VELOCITY_THRESHOLD = 0.06;  // lowered: smoothing reduces velocity to ~40% of actual
    const SEG_IDLE_FRAMES = 5;            // reduced from 8: many BSL signs are short
    const SEG_MIN_STROKE_FRAMES = 4;      // reduced from 6: catch shorter signs
    const SEG_MAX_SEGMENT_FRAMES = 90;    // ~3s max segment length
    const SEG_HOLD_MAX_FRAMES = 5;        // reduced from 12: don't wait 400ms in limbo

    // NMF (Non-Manual Feature) configuration
    const NMF_BUFFER_SIZE = 30;           // ~1s of face data
    const NMF_NOD_THRESHOLD = 0.010;      // vertical displacement for nod detection
    const NMF_SHAKE_THRESHOLD = 0.010;    // horizontal displacement for shake detection
    const NMF_NOD_MIN_REVERSALS = 2;      // minimum vertical reversals for a nod
    const NMF_SHAKE_MIN_REVERSALS = 2;    // minimum horizontal reversals for a shake
    const NMF_BROW_RAISE_THRESHOLD = 0.008; // relative brow-to-eye distance change
    const NMF_MOUTH_OPEN_THRESHOLD = 0.02;  // lip distance for mouth open
    const NMF_TILT_THRESHOLD = 0.015;       // ear height difference for head tilt

    // Sentence configuration
    const SENTENCE_TIMEOUT_MS = 3000;     // ms of silence before sentence is "complete"
    const SENTENCE_MAX_SIGNS = 50;        // maximum signs in one sentence

    // ═══════════════════════════════════════════════════════════════════════
    // State
    // ═══════════════════════════════════════════════════════════════════════

    let rawBuffer = [];              // raw landmark snapshots
    let smoothedBuffer = [];         // exponentially smoothed snapshots
    let trajectoryCache = null;      // cached trajectory classification
    let onRecognition = null;
    let lastRecognition = { sign: null, time: 0 };
    let recentTopSigns = [];         // for temporal consistency

    // Segmentation state
    let segState = 'IDLE';           // IDLE | ONSET | STROKE | HOLD | RETRACTION
    let segOnsetFrame = 0;           // frame index where onset began
    let segStrokeStart = 0;          // frame index where stroke began
    let segIdleCounter = 0;          // consecutive idle frames
    let segHoldCounter = 0;          // consecutive hold frames in HOLD state
    let segRetractionCounter = 0;    // consecutive decelerating frames in RETRACTION
    let segPeakVelocity = 0;        // peak velocity during current segment
    let segSegmentBuffer = [];       // frames collected for current segment

    // NMF state
    let nmfBuffer = [];              // rolling buffer of face landmark snapshots
    let currentNmf = {               // current non-manual feature state
        headNod: false,
        headShake: false,
        headTilt: 0,                 // negative = left tilt, positive = right tilt
        eyebrowRaise: false,
        mouthOpen: false,
        nodCount: 0,
        shakeCount: 0,
    };

    // Sentence state
    let sentenceBuffer = [];         // Array of { sign, confidence, timestamp, nmf, source }
    let lastSignTime = 0;            // timestamp of last recognised sign
    let onSentenceUpdate = null;     // callback for sentence changes

    // DTW template matching state
    let dtwTemplates = new Map();    // sign name → array of template sequences
    let dtwEnabled = false;          // whether DTW scoring is active
    let dtwExcludeSign = null;       // sign to exclude during leave-one-out testing
    let lastCompletedSegment = null; // last segmented stroke (for template recording)

    // ML classifier state
    let mlClassifier = null;         // MLClassifier module reference (set via setMlClassifier)
    let mlEnabled = false;           // whether ML scoring is active
    let mlMode = 'blend';            // 'blend' | 'override' | 'only'
    let mlBlendWeight = 0.5;         // weight for ML score in blend mode (0-1)

    // DTW feature weights: control relative importance of each feature dimension
    // 17 dimensions: position relative to nose, finger state, hand shape, orientation
    const DTW_WEIGHTS = [
        3.0, 3.0,                       // dom hand position relative to nose (x, y) — trajectory shape
        2.0, 2.0,                       // non-dom hand position relative to nose (x, y)
        0.8, 0.8, 0.8, 0.8, 0.8,       // finger extension (5 digits)
        1.5,                            // hand openness (continuous)
        1.0,                            // hand spread (continuous)
        1.5,                            // palm cross product (continuous orientation)
        1.0, 1.0,                       // finger direction (x, y)
        1.0,                            // pinch distance
        1.0,                            // non-dom hand present
        1.5,                            // inter-hand distance
    ];
    const DTW_FEATURE_DIM = DTW_WEIGHTS.length;
    const DTW_SIGMA = 4.0;             // exponential decay sensitivity (higher for richer features)
    const DTW_BLEND_CAT = 0.7;         // categorical weight in DTW blend
    const DTW_BLEND_DTW = 0.3;         // DTW weight in blend

    // ═══════════════════════════════════════════════════════════════════════
    // Geometry utilities
    // ═══════════════════════════════════════════════════════════════════════

    function v2dist(a, b) {
        const dx = a.x - b.x, dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function v2sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
    function v2add(a, b) { return { x: a.x + b.x, y: a.y + b.y }; }
    function v2scale(a, s) { return { x: a.x * s, y: a.y * s }; }
    function v2mag(a) { return Math.sqrt(a.x * a.x + a.y * a.y); }
    function v2dot(a, b) { return a.x * b.x + a.y * b.y; }

    function v2normalize(a) {
        const m = v2mag(a);
        return m < 1e-8 ? { x: 0, y: 0 } : { x: a.x / m, y: a.y / m };
    }

    function v2angle(a, b) {
        const d = v2dot(v2normalize(a), v2normalize(b));
        return Math.acos(Math.max(-1, Math.min(1, d)));
    }

    function v2cross(a, b) { return a.x * b.y - a.y * b.x; }

    function angleBetweenPoints(a, b, c) {
        const ab = v2sub(a, b), cb = v2sub(c, b);
        return v2angle(ab, cb);
    }

    function lerp(a, b, t) { return a + (b - a) * t; }

    function clamp01(v) { return Math.max(0, Math.min(1, v)); }

    function smoothValue(prev, curr, alpha) {
        return prev + alpha * (curr - prev);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Landmark extraction & smoothing
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Extract a normalised snapshot from MediaPipe results.
     * All positions are normalised relative to shoulder midpoint and width.
     * In selfie mode MediaPipe's "left" landmarks correspond to the signer's
     * right (dominant) hand.
     */
    function extractSnapshot(results, timestampMs) {
        const pose = results.poseLandmarks;
        if (!pose || !pose[11] || !pose[12]) return null;

        const lShoulder = pose[11];
        const rShoulder = pose[12];
        const midShoulder = {
            x: (lShoulder.x + rShoulder.x) / 2,
            y: (lShoulder.y + rShoulder.y) / 2,
        };
        const shoulderWidth = v2dist(lShoulder, rShoulder);
        if (shoulderWidth < 0.01) return null;

        const nose = pose[0] || midShoulder;
        const chin = { x: nose.x, y: nose.y + shoulderWidth * 0.35 };
        const forehead = { x: nose.x, y: nose.y - shoulderWidth * 0.30 };
        const lHip = pose[23] || { x: lShoulder.x, y: lShoulder.y + shoulderWidth * 1.5 };
        const rHip = pose[24] || { x: rShoulder.x, y: rShoulder.y + shoulderWidth * 1.5 };
        const waist = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };

        // Mirrored: MediaPipe leftHand = signer's dominant (right) hand
        const domHandRaw = results.leftHandLandmarks;
        const nonHandRaw = results.rightHandLandmarks;
        const domWristPose = pose[15]; // left wrist in MP = signer's right
        const nonWristPose = pose[16];

        function normalise(pt) {
            return {
                x: (pt.x - midShoulder.x) / shoulderWidth,
                y: (pt.y - midShoulder.y) / shoulderWidth,
            };
        }

        function extractHand(handLandmarks, wristFallback) {
            const hasHand = !!handLandmarks && handLandmarks.length >= 21;
            const lm = hasHand ? handLandmarks : null;
            const wrist = lm ? lm[0] : (wristFallback || midShoulder);
            const center = lm ? lm[9] : wrist; // MCP of middle finger

            // Fingertip landmarks
            const thumbTip = lm ? lm[4] : wrist;
            const indexTip = lm ? lm[8] : wrist;
            const middleTip = lm ? lm[12] : wrist;
            const ringTip = lm ? lm[16] : wrist;
            const pinkyTip = lm ? lm[20] : wrist;

            // Finger MCP (base) landmarks
            const indexMcp = lm ? lm[5] : wrist;
            const middleMcp = lm ? lm[9] : wrist;
            const ringMcp = lm ? lm[13] : wrist;
            const pinkyMcp = lm ? lm[17] : wrist;

            // Finger PIP (middle joint) landmarks
            const thumbIp = lm ? lm[3] : wrist;
            const indexPip = lm ? lm[6] : wrist;
            const middlePip = lm ? lm[10] : wrist;
            const ringPip = lm ? lm[14] : wrist;
            const pinkyPip = lm ? lm[18] : wrist;

            // Finger extension: tip further from wrist than PIP
            const fingerExtension = hasHand ? [
                v2dist(thumbTip, wrist) > v2dist(thumbIp, wrist) * 1.1 ? 1 : 0,
                v2dist(indexTip, wrist) > v2dist(indexPip, wrist) * 1.15 ? 1 : 0,
                v2dist(middleTip, wrist) > v2dist(middlePip, wrist) * 1.15 ? 1 : 0,
                v2dist(ringTip, wrist) > v2dist(ringPip, wrist) * 1.15 ? 1 : 0,
                v2dist(pinkyTip, wrist) > v2dist(pinkyPip, wrist) * 1.15 ? 1 : 0,
            ] : [0, 0, 0, 0, 0];

            const extendedCount = fingerExtension.reduce((a, b) => a + b, 0);

            // Palm direction: cross product of (index_mcp - wrist) x (pinky_mcp - wrist)
            // Positive = palm facing camera, negative = palm facing away
            const toIndex = v2sub(indexMcp, wrist);
            const toPinky = v2sub(pinkyMcp, wrist);
            const palmCross = v2cross(toIndex, toPinky);

            // Palm center (average of MCPs)
            const palmCenter = hasHand ? {
                x: (wrist.x + indexMcp.x + middleMcp.x + ringMcp.x + pinkyMcp.x) / 5,
                y: (wrist.y + indexMcp.y + middleMcp.y + ringMcp.y + pinkyMcp.y) / 5,
            } : wrist;

            // Hand spread: average distance between adjacent fingertips
            const spread = hasHand ? (
                v2dist(indexTip, middleTip) +
                v2dist(middleTip, ringTip) +
                v2dist(ringTip, pinkyTip)
            ) / 3 / shoulderWidth : 0;

            // Thumb-index pinch distance
            const pinchDist = hasHand ? v2dist(thumbTip, indexTip) / shoulderWidth : 1;

            // Openness: average tip-to-wrist distance normalised
            const openness = hasHand ? (
                v2dist(thumbTip, wrist) +
                v2dist(indexTip, wrist) +
                v2dist(middleTip, wrist) +
                v2dist(ringTip, wrist) +
                v2dist(pinkyTip, wrist)
            ) / 5 / shoulderWidth : 0;

            // Finger direction: index tip relative to wrist
            const fingerDir = hasHand ? v2normalize(v2sub(indexTip, wrist)) : { x: 0, y: 0 };

            return {
                present: hasHand,
                pos: normalise(center),
                rawPos: center,
                wrist: normalise(wrist),
                rawWrist: wrist,
                fingerExtension,
                extendedCount,
                palmCross,
                palmCenter: normalise(palmCenter),
                spread,
                pinchDist,
                openness,
                fingerDir,
                thumbTip: normalise(thumbTip),
                indexTip: normalise(indexTip),
                middleTip: normalise(middleTip),
                ringTip: normalise(ringTip),
                pinkyTip: normalise(pinkyTip),
            };
        }

        const dom = extractHand(domHandRaw, domWristPose);
        const non = extractHand(nonHandRaw, nonWristPose);

        // Body reference points (normalised)
        const nNose = normalise(nose);
        const nChin = normalise(chin);
        const nForehead = normalise(forehead);
        const nChest = normalise(midShoulder);
        const nWaist = normalise(waist);
        const nLShoulder = normalise(lShoulder);
        const nRShoulder = normalise(rShoulder);

        // Distance from dom hand to body parts
        const domToNose = v2dist(dom.pos, nNose);
        const domToChin = v2dist(dom.pos, nChin);
        const domToForehead = v2dist(dom.pos, nForehead);
        const domToChest = v2dist(dom.pos, nChest);
        const domToWaist = v2dist(dom.pos, nWaist);

        const nonToNose = v2dist(non.pos, nNose);
        const nonToChin = v2dist(non.pos, nChin);
        const nonToForehead = v2dist(non.pos, nForehead);
        const nonToChest = v2dist(non.pos, nChest);

        const handDist = v2dist(dom.pos, non.pos);

        return {
            dom, non,
            body: { nose: nNose, chin: nChin, forehead: nForehead, chest: nChest, waist: nWaist,
                    lShoulder: nLShoulder, rShoulder: nRShoulder },
            distances: {
                domToNose, domToChin, domToForehead, domToChest, domToWaist,
                nonToNose, nonToChin, nonToForehead, nonToChest,
                handDist,
            },
            shoulderWidth,
            timestamp: timestampMs != null ? timestampMs : performance.now(),
        };
    }

    /**
     * Apply exponential smoothing to a new snapshot against the previous smoothed value.
     */
    function smoothSnapshot(curr, prev) {
        if (!prev) return curr;
        const a = SMOOTHING_ALPHA;

        function smoothHand(cH, pH) {
            return {
                ...cH,
                pos: { x: smoothValue(pH.pos.x, cH.pos.x, a), y: smoothValue(pH.pos.y, cH.pos.y, a) },
                wrist: { x: smoothValue(pH.wrist.x, cH.wrist.x, a), y: smoothValue(pH.wrist.y, cH.wrist.y, a) },
                palmCenter: { x: smoothValue(pH.palmCenter.x, cH.palmCenter.x, a), y: smoothValue(pH.palmCenter.y, cH.palmCenter.y, a) },
                openness: smoothValue(pH.openness, cH.openness, a),
                spread: smoothValue(pH.spread, cH.spread, a),
                pinchDist: smoothValue(pH.pinchDist, cH.pinchDist, a),
                palmCross: smoothValue(pH.palmCross, cH.palmCross, a),
            };
        }

        return {
            ...curr,
            dom: smoothHand(curr.dom, prev.dom),
            non: smoothHand(curr.non, prev.non),
            distances: {
                domToNose: smoothValue(prev.distances.domToNose, curr.distances.domToNose, a),
                domToChin: smoothValue(prev.distances.domToChin, curr.distances.domToChin, a),
                domToForehead: smoothValue(prev.distances.domToForehead, curr.distances.domToForehead, a),
                domToChest: smoothValue(prev.distances.domToChest, curr.distances.domToChest, a),
                domToWaist: smoothValue(prev.distances.domToWaist, curr.distances.domToWaist, a),
                nonToNose: smoothValue(prev.distances.nonToNose, curr.distances.nonToNose, a),
                nonToChin: smoothValue(prev.distances.nonToChin, curr.distances.nonToChin, a),
                nonToForehead: smoothValue(prev.distances.nonToForehead, curr.distances.nonToForehead, a),
                nonToChest: smoothValue(prev.distances.nonToChest, curr.distances.nonToChest, a),
                handDist: smoothValue(prev.distances.handDist, curr.distances.handDist, a),
            },
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Trajectory analysis
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Trajectory types we can classify:
     *   linear-up, linear-down, linear-left, linear-right,
     *   linear-forward (towards camera = down+towards face area),
     *   arc-cw, arc-ccw, circular-cw, circular-ccw,
     *   oscillate-h, oscillate-v, oscillate-circular,
     *   tap-towards, tap-away,
     *   static
     */

    function computeVelocities(buffer, hand) {
        const velocities = [];
        for (let i = 1; i < buffer.length; i++) {
            const dt = (buffer[i].timestamp - buffer[i - 1].timestamp) / 1000;
            if (dt < 0.001) { velocities.push({ x: 0, y: 0 }); continue; }
            const p0 = buffer[i - 1][hand].pos;
            const p1 = buffer[i][hand].pos;
            velocities.push({ x: (p1.x - p0.x) / dt, y: (p1.y - p0.y) / dt });
        }
        return velocities;
    }

    function computeAccelerations(velocities) {
        const accels = [];
        for (let i = 1; i < velocities.length; i++) {
            accels.push({
                x: velocities[i].x - velocities[i - 1].x,
                y: velocities[i].y - velocities[i - 1].y,
            });
        }
        return accels;
    }

    /**
     * Count direction reversals along an axis.
     */
    function countReversals(velocities, axis) {
        let count = 0;
        for (let i = 1; i < velocities.length; i++) {
            if (velocities[i][axis] * velocities[i - 1][axis] < 0) count++;
        }
        return count;
    }

    /**
     * Compute net displacement, total path length, and average speed.
     */
    function pathStats(buffer, hand) {
        if (buffer.length < 2) return { displacement: { x: 0, y: 0 }, pathLength: 0, avgSpeed: 0, maxSpeed: 0 };
        const first = buffer[0][hand].pos;
        const last = buffer[buffer.length - 1][hand].pos;
        let pathLength = 0;
        let maxSpeed = 0;
        for (let i = 1; i < buffer.length; i++) {
            const d = v2dist(buffer[i][hand].pos, buffer[i - 1][hand].pos);
            pathLength += d;
            const dt = (buffer[i].timestamp - buffer[i - 1].timestamp) / 1000;
            if (dt > 0.001) {
                const spd = d / dt;
                if (spd > maxSpeed) maxSpeed = spd;
            }
        }
        const totalTime = (buffer[buffer.length - 1].timestamp - buffer[0].timestamp) / 1000;
        return {
            displacement: v2sub(last, first),
            pathLength,
            avgSpeed: totalTime > 0 ? pathLength / totalTime : 0,
            maxSpeed,
        };
    }

    /**
     * Compute the winding angle to detect arcs/circles.
     * Positive = clockwise in screen space (y-down).
     */
    function windingAngle(buffer, hand) {
        let total = 0;
        for (let i = 2; i < buffer.length; i++) {
            const a = buffer[i - 2][hand].pos;
            const b = buffer[i - 1][hand].pos;
            const c = buffer[i][hand].pos;
            const ab = v2sub(b, a);
            const bc = v2sub(c, b);
            const cross = v2cross(ab, bc);
            const dot = v2dot(ab, bc);
            total += Math.atan2(cross, dot);
        }
        return total;
    }

    /**
     * Classify the trajectory of one hand over the buffer.
     */
    function classifyTrajectory(buffer, hand) {
        if (buffer.length < MIN_FRAMES) return { type: 'static', confidence: 1 };

        // Use last ~30 frames for trajectory (1s window)
        const window = buffer.slice(-Math.min(30, buffer.length));
        const stats = pathStats(window, hand);
        const velocities = computeVelocities(window, hand);
        const revX = countReversals(velocities, 'x');
        const revY = countReversals(velocities, 'y');
        const winding = windingAngle(window, hand);
        const disp = v2mag(stats.displacement);
        const straightness = stats.pathLength > 0.01 ? disp / stats.pathLength : 1;
        const frames = window.length;

        // Thresholds — lower oscillation threshold for short segments to catch
        // brief BSL waves/shakes (a 2-cycle wave = 4 reversals in ~15 frames)
        const oscillateThreshold = Math.max(3, frames * 0.15);
        const movementThreshold = 0.08; // normalised units

        // Common fields included in all return values for trajectory feature scoring
        const base = { speed: stats.avgSpeed, displacement: stats.displacement, pathLength: stats.pathLength, straightness };

        // Static: very little movement
        if (stats.pathLength < movementThreshold * 0.5) {
            return { type: 'static', confidence: 1, ...base };
        }

        // Oscillating: many reversals — check full window first, then second half
        // (the second-half check catches signs that start with a hand raise
        // followed by oscillation, where the raise dilutes the reversal count)
        let isOscH = revX >= oscillateThreshold;
        let isOscV = revY >= oscillateThreshold;

        if (!isOscH && !isOscV && window.length >= 8) {
            const half = window.slice(Math.floor(window.length / 2));
            const halfVel = computeVelocities(half, hand);
            const halfRevX = countReversals(halfVel, 'x');
            const halfRevY = countReversals(halfVel, 'y');
            const halfThreshold = Math.max(2, half.length * 0.15);
            isOscH = halfRevX >= halfThreshold;
            isOscV = halfRevY >= halfThreshold;
        }

        if (isOscH && isOscV) {
            return { type: 'oscillate-circular', confidence: 0.8, ...base, revX, revY };
        }
        if (isOscH && !isOscV) {
            return { type: 'oscillate-h', confidence: 0.85, ...base, revX };
        }
        if (isOscV && !isOscH) {
            return { type: 'oscillate-v', confidence: 0.85, ...base, revY };
        }

        // Tap: short forward-back motion (2-5 reversals in Y near a reference)
        if (revY >= 2 && revY < oscillateThreshold && stats.pathLength > movementThreshold && disp < stats.pathLength * 0.3) {
            return { type: 'tap', confidence: 0.7, ...base, revY };
        }

        // Circular: significant winding with return to start
        const fullCircle = Math.abs(winding) > Math.PI * 1.3;
        if (fullCircle && disp < stats.pathLength * 0.4) {
            const cw = winding > 0;
            return { type: cw ? 'circular-cw' : 'circular-ccw', confidence: 0.8, ...base, winding };
        }

        // Arc: partial circle
        const hasArc = Math.abs(winding) > Math.PI * 0.4 && straightness < 0.7;
        if (hasArc) {
            const cw = winding > 0;
            return { type: cw ? 'arc-cw' : 'arc-ccw', confidence: 0.7, ...base, winding };
        }

        // Linear: mostly straight with significant displacement
        if (disp > movementThreshold && straightness > 0.5) {
            const dir = v2normalize(stats.displacement);
            let dirLabel;
            if (Math.abs(dir.y) > Math.abs(dir.x)) {
                dirLabel = dir.y < 0 ? 'linear-up' : 'linear-down';
            } else {
                dirLabel = dir.x < 0 ? 'linear-left' : 'linear-right';
            }
            return { type: dirLabel, confidence: straightness, ...base, direction: dir };
        }

        // Default: some movement but no clear pattern
        return { type: 'moving', confidence: 0.5, ...base, revX, revY, winding };
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Handshape classification
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Classify the handshape from finger extension data.
     * Returns: { shape, confidence }
     *
     * Shapes:
     *   open/flat — all fingers extended, spread
     *   closed/fist — no fingers extended
     *   point — index only
     *   V — index + middle
     *   C — curved open hand (moderate openness)
     *   pinch — thumb+index touching, others open or closed
     *   claw — fingers spread and curled
     *   horns — index + pinky extended
     *   thumbsUp — thumb extended, rest closed
     *   L — thumb + index
     */
    function classifyHandshape(handData) {
        if (!handData.present) return { shape: 'unknown', confidence: 0 };

        const [thumb, index, middle, ring, pinky] = handData.fingerExtension;
        const ext = handData.extendedCount;

        // Fist: nothing extended
        if (ext === 0) return { shape: 'fist', confidence: 0.9 };

        // Thumbs up: only thumb
        if (ext === 1 && thumb) return { shape: 'thumbsUp', confidence: 0.85 };

        // Point: only index (possibly with thumb)
        if (index && !middle && !ring && !pinky) {
            if (thumb) return { shape: 'L', confidence: 0.8 };
            return { shape: 'point', confidence: 0.9 };
        }

        // V shape: index + middle
        if (index && middle && !ring && !pinky) return { shape: 'V', confidence: 0.85 };

        // Horns: index + pinky
        if (index && !middle && !ring && pinky) return { shape: 'horns', confidence: 0.8 };

        // Y-hand: thumb + pinky (used for PHONE sign)
        if (thumb && !index && !middle && !ring && pinky) return { shape: 'Y-hand', confidence: 0.8 };

        // ILY: thumb + index + pinky (I Love You handshape)
        if (thumb && index && !middle && !ring && pinky) return { shape: 'ILY', confidence: 0.8 };

        // Pinch: thumb + index close together
        if (handData.pinchDist < 0.12) return { shape: 'pinch', confidence: 0.8 };

        // Full open / flat
        if (ext >= 4) {
            if (handData.spread > 0.06) return { shape: 'open', confidence: 0.85 };
            return { shape: 'flat', confidence: 0.85 };
        }

        // Three fingers (index + middle + ring)
        if (index && middle && ring && !pinky) return { shape: 'three', confidence: 0.75 };

        // C-shape: moderate openness, fingers curved
        if (ext >= 2 && handData.openness > 0.15 && handData.openness < 0.5) {
            return { shape: 'C', confidence: 0.65 };
        }

        // Claw: fingers spread but curled (low extension despite spread)
        if (handData.spread > 0.04 && ext <= 2) return { shape: 'claw', confidence: 0.6 };

        return { shape: 'open', confidence: 0.5 };
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Location classification
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Classify where the dominant hand is relative to the body.
     * Uses the smoothed snapshot distances.
     */
    function classifyLocation(snapshot, hand) {
        const d = snapshot.distances;
        const h = snapshot[hand];
        const prefix = hand === 'dom' ? 'dom' : 'non';

        const toNose = d[prefix + 'ToNose'];
        const toChin = d[prefix + 'ToChin'];
        const toForehead = d[prefix + 'ToForehead'];
        const toChest = d[prefix + 'ToChest'];

        // Relative Y to determine vertical zone
        const posY = h.pos.y;
        const foreheadY = snapshot.body.forehead.y;
        const chinY = snapshot.body.chin.y;
        const chestY = snapshot.body.chest.y;
        const waistY = snapshot.body.waist.y;

        // Side detection first — ear is face-height but far to the side
        const posX = h.pos.x;
        const isFaceHeight = posY >= foreheadY && posY <= chinY + 0.2;

        // Ear: face-height AND far to one side (near ear position)
        if (isFaceHeight && Math.abs(posX) > 0.55) return 'ear';

        // Cheek: between nose and chin, slightly to the side
        if (isFaceHeight && toNose < 0.6 && toChin < 0.6 && Math.abs(posX) > 0.15) return 'cheek';

        // Temple: forehead height but to the side
        if (posY < foreheadY + 0.15 && Math.abs(posX) > 0.3) return 'temple';

        if (toForehead < 0.5) return 'forehead';
        if (toNose < 0.45) return 'nose';
        if (toChin < 0.5) return 'chin';
        if (posY < foreheadY) return 'above-head';
        if (posY < chinY) return 'face';

        // Neck: between chin and upper chest, close to body center
        if (posY > chinY && posY < chestY - 0.1 && Math.abs(posX) < 0.4) return 'neck';

        // Shoulder: chest-height but far to the side
        if (posY >= chinY && posY < chestY + 0.3 && Math.abs(posX) > 0.5) return 'shoulder';

        if (toChest < 0.5) return 'chest';
        if (posY < chestY + 0.3) return 'upper-chest';
        if (posY < waistY) return 'stomach';
        if (posY < waistY + 0.4) return 'waist';

        if (Math.abs(posX) > 0.8) return posX > 0 ? 'far-right' : 'far-left';
        if (Math.abs(posX) > 0.4) return posX > 0 ? 'right-side' : 'left-side';

        return 'neutral';
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Orientation classification
    // ═══════════════════════════════════════════════════════════════════════

    function classifyOrientation(handData) {
        if (!handData.present) return 'unknown';
        // palmCross > 0 means palm facing camera (in mirrored mode)
        // fingerDir gives the pointing direction
        const pc = handData.palmCross;
        const fd = handData.fingerDir;

        if (pc > 0.005) return 'palm-forward';   // palm towards camera/viewer
        if (pc < -0.005) return 'palm-back';      // palm towards signer
        if (fd.y < -0.5) return 'palm-up';        // fingers pointing up generally = palm outward
        if (fd.y > 0.5) return 'palm-down';
        return 'palm-side';
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Scoring functions for sign parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Score how well a movement requirement matches the observed trajectory.
     * movReq: { type, speed?, direction?, minReversals? } or array of acceptable types
     * observed: trajectory classification object
     */
    function scoreMovement(movReq, domTraj, nonTraj) {
        if (!movReq) return 1.0; // no movement requirement = always matches

        // If the requirement specifies per-hand, route accordingly
        const req = movReq.dom || movReq;
        const traj = domTraj;

        let score = 0;

        // Type matching: can be a single string or array of acceptable types
        const acceptableTypes = Array.isArray(req.type) ? req.type : [req.type];
        const typeMatch = acceptableTypes.some(t => {
            if (t === traj.type) return true;
            // Partial matches
            if (t === 'linear' && traj.type.startsWith('linear')) return true;
            if (t === 'oscillate' && traj.type.startsWith('oscillate')) return true;
            if (t === 'circular' && traj.type.startsWith('circular')) return true;
            if (t === 'arc' && traj.type.startsWith('arc')) return true;
            if (t === 'any-movement' && traj.type !== 'static') return true;
            return false;
        });

        if (typeMatch) {
            score += 0.55;

            // TRAJECTORY FEATURE BONUS: When type matches, use raw trajectory
            // features for finer discrimination between signs sharing same type.
            const disp = traj.displacement ? v2mag(traj.displacement) : 0;
            const pathLen = traj.pathLength || disp || 0.01;
            const straightness = pathLen > 0.01 ? disp / pathLen : 1;

            // Oscillating signs should have low displacement-to-path ratio
            if (acceptableTypes.some(t => t.startsWith('oscillate') || t === 'tap')) {
                const ratio = disp / Math.max(pathLen, 0.01);
                if (ratio < 0.25) score += 0.20;       // very oscillatory — good match
                else if (ratio < 0.45) score += 0.12;   // somewhat oscillatory
                else score += 0.04;                       // more linear than oscillating
            }
            // Linear signs should have high straightness
            else if (acceptableTypes.some(t => t.startsWith('linear'))) {
                if (straightness > 0.65) score += 0.20;  // very straight — good match
                else if (straightness > 0.4) score += 0.12;
                else score += 0.04;                        // too curvy for linear
            }
            // Circular signs should have high winding
            else if (acceptableTypes.some(t => t.startsWith('circular'))) {
                const absWinding = Math.abs(traj.winding || 0);
                if (absWinding > Math.PI * 1.5) score += 0.20;
                else if (absWinding > Math.PI) score += 0.12;
                else score += 0.04;
            }
            // Arc signs should have medium winding
            else if (acceptableTypes.some(t => t.startsWith('arc'))) {
                const absWinding = Math.abs(traj.winding || 0);
                if (absWinding > Math.PI * 0.4 && absWinding < Math.PI * 1.5) score += 0.20;
                else score += 0.06;
            }
            // Static signs should have very little movement
            else if (acceptableTypes.includes('static')) {
                if (pathLen < 0.05) score += 0.20;
                else if (pathLen < 0.15) score += 0.10;
                else score += 0.02;
            }
            else {
                score += 0.10; // any-movement or unknown
            }
        } else {
            // Partial credit for related movement types — kept small for discrimination
            const isMoving = traj.type !== 'static';
            const wantsMoving = !acceptableTypes.includes('static');

            // Tight related groups — only genuinely similar movements
            const relatedGroups = [
                ['circular-cw', 'circular-ccw'],
                ['arc-cw', 'arc-ccw'],
                ['linear-up', 'linear-down'],
                ['linear-left', 'linear-right'],
                ['linear-up', 'linear-forward'],
                ['linear-down', 'linear-forward'],
            ];
            const isRelated = acceptableTypes.some(t => {
                return relatedGroups.some(group =>
                    group.includes(t) && group.includes(traj.type)
                );
            });

            if (isRelated) score += 0.25;
            else if (isMoving && wantsMoving) score += 0.04;
        }

        // Speed matching — non-overlapping ranges for discrimination
        if (req.speed) {
            const spd = traj.speed || 0;
            if (req.speed === 'fast' && spd > 2.0) score += 0.2;
            else if (req.speed === 'fast' && spd > 1.0) score += 0.1;
            else if (req.speed === 'moderate' && spd > 0.4 && spd < 2.5) score += 0.2;
            else if (req.speed === 'slow' && spd < 0.8) score += 0.2;
            else if (req.speed === 'slow' && spd < 1.5) score += 0.1;
            else score += 0.03;
        } else {
            score += 0.10; // no speed requirement = small partial credit
        }

        // Direction matching for linear movements
        if (req.direction && traj.displacement) {
            const dir = v2normalize(traj.displacement);
            let dirScore = 0;
            if (req.direction === 'up' && dir.y < -0.3) dirScore = 1;
            else if (req.direction === 'down' && dir.y > 0.3) dirScore = 1;
            else if (req.direction === 'left' && dir.x < -0.3) dirScore = 1;
            else if (req.direction === 'right' && dir.x > 0.3) dirScore = 1;
            else if (req.direction === 'forward-down' && dir.y > 0.2) dirScore = 0.8;
            else if (req.direction === 'forward-up' && dir.y < -0.2) dirScore = 0.8;
            else if (req.direction === 'away') dirScore = 0.6; // harder to detect in 2D
            else dirScore = 0.1;
            score += 0.25 * dirScore;
        } else {
            score += 0.1;
        }

        // Non-dominant hand movement requirement
        if (movReq.non && nonTraj) {
            const nonScore = scoreMovement({ dom: movReq.non }, nonTraj, null);
            score = score * 0.6 + nonScore * 0.4;
        }

        return clamp01(score);
    }

    /**
     * Score a location requirement against observed location.
     */
    function scoreLocation(locReq, snapshot) {
        if (!locReq) return 1.0;

        const observed = classifyLocation(snapshot, 'dom');
        const acceptable = Array.isArray(locReq) ? locReq : [locReq];

        if (acceptable.includes(observed)) return 1.0;

        // Partial credit for nearby locations
        const closeMap = {
            'forehead': ['face', 'temple', 'above-head'],
            'nose': ['face', 'chin', 'cheek'],
            'chin': ['face', 'nose', 'neck', 'cheek'],
            'face': ['forehead', 'nose', 'chin', 'cheek', 'ear', 'temple'],
            'cheek': ['face', 'nose', 'chin', 'ear'],
            'ear': ['face', 'cheek', 'temple'],
            'temple': ['forehead', 'ear', 'face'],
            'neck': ['chin', 'upper-chest'],
            'shoulder': ['upper-chest', 'ear'],
            'chest': ['upper-chest', 'neck'],
            'upper-chest': ['chest', 'neck', 'shoulder'],
            'stomach': ['chest', 'upper-chest', 'waist'],
            'waist': ['stomach'],
            'neutral': ['chest', 'upper-chest', 'right-side', 'left-side'],
            'right-side': ['neutral', 'far-right', 'shoulder'],
            'left-side': ['neutral', 'far-left', 'shoulder'],
            'above-head': ['forehead', 'face'],
        };
        const farMap = {
            'forehead': ['nose', 'cheek', 'ear'],
            'nose': ['forehead', 'ear', 'upper-chest'],
            'chin': ['upper-chest', 'forehead'],
            'face': ['neck', 'upper-chest', 'above-head'],
            'neck': ['face', 'chest', 'shoulder'],
            'chest': ['stomach', 'neutral', 'chin'],
            'upper-chest': ['chin', 'stomach', 'neutral'],
            'neutral': ['stomach', 'waist'],
        };

        for (const loc of acceptable) {
            const close = closeMap[loc] || [];
            if (close.includes(observed)) return 0.6;
        }
        for (const loc of acceptable) {
            const far = farMap[loc] || [];
            if (far.includes(observed)) return 0.25;
        }

        return 0.04;
    }

    /**
     * Score a handshape requirement.
     */
    function scoreHandshape(shapeReq, snapshot) {
        if (!shapeReq) return 1.0;

        const domReq = shapeReq.dom || shapeReq;
        const domObs = classifyHandshape(snapshot.dom);

        const acceptableDom = Array.isArray(domReq) ? domReq : [domReq];
        let domScore = 0;
        if (acceptableDom.includes(domObs.shape)) {
            domScore = domObs.confidence;
        } else {
            // Partial credit for similar shapes
            const similarMap = {
                'open': ['flat', 'C'],
                'flat': ['open'],
                'fist': ['thumbsUp', 'S-hand'],
                'point': ['L'],
                'L': ['point', 'V'],
                'V': ['L', 'point', 'bent-V'],
                'C': ['open', 'claw', 'flat-O'],
                'pinch': ['C', 'fist', 'flat-O'],
                'claw': ['C', 'open', 'bent-V'],
                'thumbsUp': ['fist', 'L'],
                'Y-hand': ['horns', 'ILY'],
                'ILY': ['Y-hand', 'horns', 'open'],
                'horns': ['Y-hand', 'ILY'],
                'three': ['V', 'open'],
                'bent-V': ['V', 'claw'],
                'bent-flat': ['flat', 'claw'],
                'flat-O': ['pinch', 'C'],
                'S-hand': ['fist'],
            };
            for (const req of acceptableDom) {
                const similar = similarMap[req] || [];
                if (similar.includes(domObs.shape)) { domScore = 0.25; break; }
            }
        }

        // Non-dominant hand shape
        if (shapeReq.non) {
            const nonObs = classifyHandshape(snapshot.non);
            const acceptableNon = Array.isArray(shapeReq.non) ? shapeReq.non : [shapeReq.non];
            let nonScore = 0;
            if (acceptableNon.includes(nonObs.shape)) {
                nonScore = nonObs.confidence;
            } else {
                const similarMap = {
                    'open': ['flat', 'C'], 'flat': ['open'], 'fist': ['thumbsUp'],
                    'point': ['L'], 'L': ['point'], 'V': ['L'],
                };
                for (const req of acceptableNon) {
                    const similar = similarMap[req] || [];
                    if (similar.includes(nonObs.shape)) { nonScore = 0.35; break; }
                }
            }
            return (domScore * 0.6 + nonScore * 0.4);
        }

        return domScore;
    }

    /**
     * Score an orientation requirement.
     * Uses proximity groups for partial credit instead of binary matching.
     */
    function scoreOrientation(oriReq, snapshot) {
        if (!oriReq) return 1.0;
        const observed = classifyOrientation(snapshot.dom);
        const acceptable = Array.isArray(oriReq) ? oriReq : [oriReq];
        if (acceptable.includes(observed)) return 1.0;

        // Close orientations — axes that are adjacent in 3D space
        const closeMap = {
            'palm-forward': ['palm-up', 'palm-side'],
            'palm-back': ['palm-down', 'palm-side'],
            'palm-up': ['palm-forward', 'palm-side'],
            'palm-down': ['palm-back', 'palm-side'],
            'palm-side': ['palm-forward', 'palm-back'],
        };
        // Opposite orientations — clearly wrong
        const oppositeMap = {
            'palm-forward': ['palm-back'],
            'palm-back': ['palm-forward'],
            'palm-up': ['palm-down'],
            'palm-down': ['palm-up'],
        };

        for (const req of acceptable) {
            const close = closeMap[req] || [];
            if (close.includes(observed)) return 0.45;
        }
        for (const req of acceptable) {
            const opp = oppositeMap[req] || [];
            if (opp.includes(observed)) return 0.08;
        }
        return 0.15;
    }

    /**
     * Score two-handed configuration requirement.
     */
    function scoreTwoHanded(twoReq, snapshot, domTraj, nonTraj) {
        if (!twoReq) return 1.0; // not required = always matches

        const domPresent = snapshot.dom.present;
        const nonPresent = snapshot.non.present;
        const handDist = snapshot.distances.handDist;

        if (twoReq === 'one-handed') {
            if (!domPresent) return 0.3;
            // Penalise if non-dom hand is also moving (suggests two-handed sign)
            if (nonPresent && nonTraj && nonTraj.type !== 'static' && nonTraj.speed > 0.5) {
                return 0.5;
            }
            return 0.9;
        }

        if (!domPresent || !nonPresent) return 0.15; // two hands required but not detected

        let score = 0.3; // base for having both hands

        if (twoReq === 'symmetric') {
            // Both hands doing same movement
            if (domTraj && nonTraj && domTraj.type === nonTraj.type) score += 0.5;
            else score += 0.15;
            // Check if hands are roughly mirrored in x
            const xDiff = Math.abs(snapshot.dom.pos.x + snapshot.non.pos.x);
            if (xDiff < 0.5) score += 0.2;
        } else if (twoReq === 'alternating') {
            // Hands moving in opposite phases
            if (domTraj && nonTraj) {
                const domDisp = domTraj.displacement || { x: 0, y: 0 };
                const nonDisp = nonTraj.displacement || { x: 0, y: 0 };
                const antiCorr = -(domDisp.y * nonDisp.y);
                if (antiCorr > 0) score += 0.5;
                else score += 0.1;
            }
            score += 0.2;
        } else if (twoReq === 'together') {
            // Hands close together
            if (handDist < 0.4) score += 0.5;
            else if (handDist < 0.7) score += 0.25;
            score += 0.2;
        } else if (twoReq === 'apart-to-together') {
            // Hands start apart and come together (check buffer)
            if (smoothedBuffer.length >= 10) {
                const earlyDist = smoothedBuffer[smoothedBuffer.length - 10].distances.handDist;
                if (earlyDist > handDist + 0.15) score += 0.5;
                else score += 0.1;
            }
            score += 0.2;
        } else if (twoReq === 'together-to-apart') {
            if (smoothedBuffer.length >= 10) {
                const earlyDist = smoothedBuffer[smoothedBuffer.length - 10].distances.handDist;
                if (handDist > earlyDist + 0.15) score += 0.5;
                else score += 0.1;
            }
            score += 0.2;
        } else if (twoReq === 'non-static') {
            // Non-dominant stays still, dominant moves
            if (nonTraj && nonTraj.type === 'static') score += 0.4;
            else score += 0.1;
            if (domTraj && domTraj.type !== 'static') score += 0.3;
        }

        return clamp01(score);
    }

    /**
     * Score a contact requirement (hand touching face/body/other hand).
     */
    function scoreContact(contactReq, snapshot) {
        if (!contactReq) return 0.3; // no contact requirement = low neutral (prevents attractors)

        const d = snapshot.distances;
        const contactThreshold = 0.35;

        if (contactReq === 'face' || contactReq === 'chin') {
            return d.domToChin < contactThreshold ? 1.0 : d.domToChin < 0.55 ? 0.5 : 0.1;
        }
        if (contactReq === 'forehead') {
            return d.domToForehead < contactThreshold ? 1.0 : d.domToForehead < 0.55 ? 0.5 : 0.1;
        }
        if (contactReq === 'nose') {
            return d.domToNose < contactThreshold ? 1.0 : d.domToNose < 0.5 ? 0.5 : 0.1;
        }
        if (contactReq === 'chest') {
            return d.domToChest < contactThreshold ? 1.0 : d.domToChest < 0.55 ? 0.5 : 0.1;
        }
        if (contactReq === 'stomach' || contactReq === 'upper-chest') {
            // Use average of chest and waist distance as proxy for stomach
            const dist = (d.domToChest + d.domToWaist) / 2;
            return dist < contactThreshold ? 1.0 : dist < 0.55 ? 0.5 : 0.1;
        }
        if (contactReq === 'shoulder') {
            return d.domToChest < 0.4 ? 0.7 : 0.2; // approximate
        }
        if (contactReq === 'ear') {
            // ear is near chin/face
            return d.domToChin < 0.4 ? 0.8 : d.domToChin < 0.6 ? 0.4 : 0.1;
        }
        if (contactReq === 'other-hand') {
            return d.handDist < 0.3 ? 1.0 : d.handDist < 0.5 ? 0.5 : 0.1;
        }
        if (contactReq === 'none') {
            const minDist = Math.min(d.domToNose, d.domToChin, d.domToChest);
            return minDist > 0.5 ? 1.0 : minDist > 0.3 ? 0.6 : 0.2;
        }

        return 0.5;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Sign dictionary — data-driven BSL sign definitions
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Each sign definition uses BSL phonological parameters:
     *   sign:        display name (uppercase)
     *   location:    where dom hand is (string or array of acceptable locations)
     *   handshape:   { dom, non? } shape name(s)
     *   movement:    { dom: { type, speed?, direction? }, non? }
     *   orientation: palm direction(s)
     *   twoHanded:   'one-handed' | 'symmetric' | 'alternating' | 'together' | etc.
     *   contact:     body part the hand touches, or 'none'
     *   category:    for dictionary grouping
     *   nmfAlternative: (optional) non-manual feature alternative recognition path
     */
    // Use external dictionary if available, otherwise fall back to inline definitions
    const SIGN_DEFINITIONS = (typeof BSL_SIGN_DICTIONARY !== 'undefined' && BSL_SIGN_DICTIONARY.length > 0)
        ? BSL_SIGN_DICTIONARY
        : [

        // ── GREETINGS (FALLBACK — external dictionary preferred) ──────

        {
            sign: 'HELLO',
            category: 'greetings',
            location: ['face', 'forehead', 'nose'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['oscillate-h', 'any-movement'], speed: 'moderate' } },
            orientation: ['palm-forward'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'GOODBYE',
            category: 'greetings',
            location: ['face', 'chin', 'upper-chest'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['oscillate-h', 'linear-down', 'any-movement'], speed: 'moderate' } },
            orientation: ['palm-forward'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'GOOD-MORNING',
            category: 'greetings',
            location: ['chin', 'chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-up', 'arc-cw'], direction: 'up' } },
            orientation: ['palm-forward', 'palm-up'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'GOOD-AFTERNOON',
            category: 'greetings',
            location: ['chin', 'face'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'arc-cw'], direction: 'forward-down' } },
            orientation: ['palm-down', 'palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'GOOD-EVENING',
            category: 'greetings',
            location: ['chin', 'chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down'], direction: 'down' } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },

        // ── RESPONSES ──────────────────────────────────────────────────

        {
            sign: 'YES',
            category: 'responses',
            location: ['chin', 'face', 'nose'],
            handshape: { dom: ['fist'] },
            movement: { dom: { type: ['oscillate-v', 'tap'], speed: 'moderate' } },
            orientation: ['palm-forward', 'palm-side'],
            twoHanded: 'one-handed',
            contact: null,
            // Alternative: non-manual only (head nod)
            nmfAlternative: { headNod: true, minNods: 2 },
        },
        {
            sign: 'NO',
            category: 'responses',
            location: ['face', 'chin', 'neutral'],
            handshape: { dom: ['point', 'L'] },
            movement: { dom: { type: ['oscillate-h'], speed: 'moderate' } },
            orientation: ['palm-forward', 'palm-side'],
            twoHanded: 'one-handed',
            contact: null,
            // Alternative: non-manual only (head shake)
            nmfAlternative: { headShake: true, minShakes: 2 },
        },
        {
            sign: 'MAYBE',
            category: 'responses',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['oscillate-v', 'oscillate-h'] }, non: { type: ['oscillate-v', 'oscillate-h'] } },
            orientation: ['palm-up'],
            twoHanded: 'alternating',
            contact: null,
        },
        {
            sign: 'PLEASE',
            category: 'responses',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['circular-cw', 'circular-ccw', 'oscillate-circular'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'THANK-YOU',
            category: 'responses',
            location: ['chin', 'face'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'linear'], direction: 'forward-down' } },
            orientation: ['palm-back', 'palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'SORRY',
            category: 'responses',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['fist'] },
            movement: { dom: { type: ['circular-cw', 'circular-ccw', 'oscillate-circular'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'EXCUSE-ME',
            category: 'responses',
            location: ['chin', 'face'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear', 'arc-cw'], direction: 'forward-down' } },
            orientation: ['palm-up'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },

        // ── QUESTIONS ──────────────────────────────────────────────────

        {
            sign: 'WHAT',
            category: 'questions',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['point', 'L'] },
            movement: { dom: { type: ['oscillate-h', 'linear-right', 'linear-left'] } },
            orientation: ['palm-up', 'palm-forward'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'WHERE',
            category: 'questions',
            location: ['neutral', 'upper-chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['oscillate-h', 'any-movement'] } },
            orientation: ['palm-down', 'palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'WHEN',
            category: 'questions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point'], non: ['point'] },
            movement: { dom: { type: ['circular-cw', 'arc-cw'] } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: null,
        },
        {
            sign: 'WHO',
            category: 'questions',
            location: ['chin', 'face'],
            handshape: { dom: ['point', 'L'] },
            movement: { dom: { type: ['circular-cw', 'oscillate-circular', 'any-movement'] } },
            orientation: ['palm-side', 'palm-forward'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'WHY',
            category: 'questions',
            location: ['forehead', 'face'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'HOW',
            category: 'questions',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['fist', 'C'], non: ['fist', 'C'] },
            movement: { dom: { type: ['circular-cw', 'arc-cw', 'any-movement'] } },
            orientation: ['palm-down', 'palm-back'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'HOW-MUCH',
            category: 'questions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['claw', 'C', 'open'] },
            movement: { dom: { type: ['linear-up', 'any-movement'], direction: 'up' } },
            orientation: ['palm-up'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'HOW-MANY',
            category: 'questions',
            location: ['neutral', 'upper-chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['linear-up', 'any-movement'], direction: 'up' }, non: { type: ['linear-up', 'any-movement'], direction: 'up' } },
            orientation: ['palm-up'],
            twoHanded: 'symmetric',
            contact: null,
        },

        // ── PEOPLE ─────────────────────────────────────────────────────

        {
            sign: 'MAN',
            category: 'people',
            location: ['chin', 'face'],
            handshape: { dom: ['flat', 'C', 'open'] },
            movement: { dom: { type: ['linear-down', 'linear'], direction: 'down' } },
            orientation: ['palm-side', 'palm-back'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'WOMAN',
            category: 'people',
            location: ['chin', 'face', 'nose'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'linear'], direction: 'down' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'BOY',
            category: 'people',
            location: ['chin', 'face'],
            handshape: { dom: ['pinch', 'C'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'GIRL',
            category: 'people',
            location: ['chin', 'nose', 'face'],
            handshape: { dom: ['point', 'L'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'BABY',
            category: 'people',
            location: ['neutral', 'chest', 'stomach'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['oscillate-h', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'CHILD',
            category: 'people',
            location: ['neutral', 'waist', 'stomach'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['oscillate-v', 'tap', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'FRIEND',
            category: 'people',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['open', 'flat', 'claw'], non: ['open', 'flat', 'claw'] },
            movement: { dom: { type: ['oscillate-h', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'FAMILY',
            category: 'people',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['circular-cw', 'circular-ccw', 'oscillate-circular'] }, non: { type: ['circular-cw', 'circular-ccw', 'oscillate-circular'] } },
            orientation: ['palm-forward'],
            twoHanded: 'symmetric',
            contact: null,
        },
        {
            sign: 'MOTHER',
            category: 'people',
            location: ['chin', 'face'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'FATHER',
            category: 'people',
            location: ['forehead', 'face'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'BROTHER',
            category: 'people',
            location: ['nose', 'face'],
            handshape: { dom: ['V', 'point'] },
            movement: { dom: { type: ['tap', 'any-movement'] } },
            orientation: ['palm-forward'],
            twoHanded: 'one-handed',
            contact: 'nose',
        },
        {
            sign: 'SISTER',
            category: 'people',
            location: ['nose', 'chin', 'face'],
            handshape: { dom: ['point', 'V'] },
            movement: { dom: { type: ['linear-down', 'tap', 'any-movement'], direction: 'down' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'nose',
        },
        {
            sign: 'HUSBAND',
            category: 'people',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist', 'C'], non: ['fist', 'C'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'WIFE',
            category: 'people',
            location: ['neutral', 'chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'DOCTOR',
            category: 'people',
            location: ['neutral', 'chest'],
            handshape: { dom: ['V', 'point'], non: ['flat', 'open'] },
            movement: { dom: { type: ['tap', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'TEACHER',
            category: 'people',
            location: ['neutral', 'upper-chest', 'face'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['oscillate-v', 'linear-down', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'POLICE',
            category: 'people',
            location: ['upper-chest', 'chest'],
            handshape: { dom: ['claw', 'C'] },
            movement: { dom: { type: ['tap', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },

        // ── ACTIONS ────────────────────────────────────────────────────

        {
            sign: 'GO',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['linear-right', 'linear-left', 'linear'], direction: 'right' } },
            orientation: ['palm-down', 'palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'COME',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['linear-left', 'linear-right', 'linear'], direction: 'left' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'STOP',
            category: 'actions',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['static', 'linear-down'], speed: 'slow' } },
            orientation: ['palm-forward'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'WAIT',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['static', 'oscillate-v'] }, non: { type: ['static', 'oscillate-v'] } },
            orientation: ['palm-up'],
            twoHanded: 'symmetric',
            contact: null,
        },
        {
            sign: 'SIT',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['V'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'tap'], direction: 'down' } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'STAND',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['V'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-up', 'any-movement'], direction: 'up' } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'WALK',
            category: 'actions',
            location: ['neutral', 'chest', 'stomach'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['alternating', 'oscillate-v', 'any-movement'] }, non: { type: ['alternating', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'alternating',
            contact: null,
        },
        {
            sign: 'RUN',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist', 'L'], non: ['fist', 'L'] },
            movement: { dom: { type: ['oscillate-v', 'alternating', 'any-movement'], speed: 'fast' }, non: { type: ['oscillate-v', 'alternating', 'any-movement'], speed: 'fast' } },
            orientation: ['palm-side'],
            twoHanded: 'alternating',
            contact: null,
        },
        {
            sign: 'EAT',
            category: 'actions',
            location: ['chin', 'face'],
            handshape: { dom: ['pinch', 'C', 'flat'] },
            movement: { dom: { type: ['tap', 'oscillate-v'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'DRINK',
            category: 'actions',
            location: ['chin', 'face'],
            handshape: { dom: ['C'] },
            movement: { dom: { type: ['linear-up', 'arc-cw', 'tap'], direction: 'forward-up' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'SLEEP',
            category: 'actions',
            location: ['face', 'chin', 'forehead'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'static'], speed: 'slow' } },
            orientation: ['palm-side', 'palm-back'],
            twoHanded: 'one-handed',
            contact: 'face',
        },
        {
            sign: 'WAKE',
            category: 'actions',
            location: ['face', 'forehead'],
            handshape: { dom: ['fist', 'pinch'] },
            movement: { dom: { type: ['any-movement'], speed: 'moderate' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'WORK',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist'], non: ['fist'] },
            movement: { dom: { type: ['oscillate-v', 'tap', 'any-movement'] }, non: { type: ['oscillate-v', 'tap', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'alternating',
            contact: null,
        },
        {
            sign: 'HELP',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist', 'thumbsUp'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-up', 'any-movement'], direction: 'up' } },
            orientation: ['palm-up'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'WANT',
            category: 'actions',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['claw', 'C'] },
            movement: { dom: { type: ['linear-left', 'linear', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'NEED',
            category: 'actions',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'LIKE',
            category: 'actions',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['linear-down', 'linear', 'any-movement'], direction: 'forward-down' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'LOVE',
            category: 'actions',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['fist', 'claw'], non: ['fist', 'claw'] },
            movement: { dom: { type: ['static', 'any-movement'], speed: 'slow' } },
            orientation: ['palm-back'],
            twoHanded: 'together',
            contact: 'chest',
        },
        {
            sign: 'KNOW',
            category: 'actions',
            location: ['forehead', 'face'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['tap', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'UNDERSTAND',
            category: 'actions',
            location: ['forehead', 'face'],
            handshape: { dom: ['point', 'flat'] },
            movement: { dom: { type: ['any-movement', 'linear-up'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'THINK',
            category: 'actions',
            location: ['forehead', 'face'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['tap', 'static', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'REMEMBER',
            category: 'actions',
            location: ['forehead', 'face'],
            handshape: { dom: ['thumbsUp', 'fist'] },
            movement: { dom: { type: ['tap', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'FORGET',
            category: 'actions',
            location: ['forehead', 'face'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'LEARN',
            category: 'actions',
            location: ['forehead', 'face', 'neutral'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['linear-up', 'any-movement'], direction: 'up' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'TEACH',
            category: 'actions',
            location: ['neutral', 'upper-chest', 'face'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'oscillate-v', 'any-movement'] }, non: { type: ['linear-down', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'symmetric',
            contact: null,
        },
        {
            sign: 'GIVE',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'TAKE',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['claw', 'C', 'fist'] },
            movement: { dom: { type: ['linear-left', 'linear', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'BUY',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'tap', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'SELL',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'MAKE',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist'], non: ['fist'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'BREAK',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist'], non: ['fist'] },
            movement: { dom: { type: ['any-movement'], speed: 'fast' } },
            orientation: ['palm-down'],
            twoHanded: 'together-to-apart',
            contact: null,
        },
        {
            sign: 'OPEN',
            category: 'actions',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['arc-cw', 'linear', 'any-movement'] }, non: { type: ['arc-ccw', 'linear', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'together-to-apart',
            contact: null,
        },
        {
            sign: 'CLOSE',
            category: 'actions',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['arc-ccw', 'linear', 'any-movement'] }, non: { type: ['arc-cw', 'linear', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'apart-to-together',
            contact: null,
        },
        {
            sign: 'START',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'FINISH',
            category: 'actions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' }, non: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-down'],
            twoHanded: 'symmetric',
            contact: null,
        },

        // ── DESCRIPTIONS ───────────────────────────────────────────────

        {
            sign: 'GOOD',
            category: 'descriptions',
            location: ['chin', 'face'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'linear', 'any-movement'], direction: 'forward-down' } },
            orientation: ['palm-back', 'palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'BAD',
            category: 'descriptions',
            location: ['chin', 'face'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down', speed: 'fast' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'BIG',
            category: 'descriptions',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['open', 'flat', 'C'], non: ['open', 'flat', 'C'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'] }, non: { type: ['linear-left', 'linear', 'any-movement'] } },
            orientation: ['palm-forward'],
            twoHanded: 'together-to-apart',
            contact: null,
        },
        {
            sign: 'SMALL',
            category: 'descriptions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-left', 'linear', 'any-movement'] }, non: { type: ['linear-right', 'linear', 'any-movement'] } },
            orientation: ['palm-forward'],
            twoHanded: 'apart-to-together',
            contact: null,
        },
        {
            sign: 'HOT',
            category: 'descriptions',
            location: ['chin', 'face'],
            handshape: { dom: ['claw', 'C', 'open'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'forward-down' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'COLD',
            category: 'descriptions',
            location: ['neutral', 'upper-chest', 'chest'],
            handshape: { dom: ['fist'], non: ['fist'] },
            movement: { dom: { type: ['oscillate-h', 'oscillate-v', 'any-movement'] }, non: { type: ['oscillate-h', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'symmetric',
            contact: null,
        },
        {
            sign: 'HAPPY',
            category: 'descriptions',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['circular-cw', 'oscillate-circular', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'SAD',
            category: 'descriptions',
            location: ['face', 'chin'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'ANGRY',
            category: 'descriptions',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['claw', 'fist'] },
            movement: { dom: { type: ['linear-up', 'any-movement'], direction: 'up', speed: 'fast' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'TIRED',
            category: 'descriptions',
            location: ['chest', 'upper-chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' }, non: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-back'],
            twoHanded: 'symmetric',
            contact: 'chest',
        },
        {
            sign: 'HUNGRY',
            category: 'descriptions',
            location: ['chest', 'stomach', 'upper-chest'],
            handshape: { dom: ['claw', 'C', 'flat'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'THIRSTY',
            category: 'descriptions',
            location: ['chin', 'face', 'upper-chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'BEAUTIFUL',
            category: 'descriptions',
            location: ['face', 'chin'],
            handshape: { dom: ['open', 'flat'] },
            movement: { dom: { type: ['circular-cw', 'arc-cw', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'UGLY',
            category: 'descriptions',
            location: ['face', 'nose'],
            handshape: { dom: ['claw', 'C'] },
            movement: { dom: { type: ['linear-down', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'EASY',
            category: 'descriptions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['oscillate-v', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'HARD',
            category: 'descriptions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'tap', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'FAST',
            category: 'descriptions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist', 'point'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'], speed: 'fast' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'SLOW',
            category: 'descriptions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['open', 'flat'], non: ['open', 'flat'] },
            movement: { dom: { type: ['linear-up', 'linear', 'any-movement'], speed: 'slow' } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'NEW',
            category: 'descriptions',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['arc-cw', 'linear', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'OLD',
            category: 'descriptions',
            location: ['chin', 'face'],
            handshape: { dom: ['C', 'claw', 'fist'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },

        // ── TIME ───────────────────────────────────────────────────────

        {
            sign: 'TODAY',
            category: 'time',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['linear-down', 'tap', 'any-movement'], direction: 'down' } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'TOMORROW',
            category: 'time',
            location: ['face', 'chin', 'forehead'],
            handshape: { dom: ['flat', 'open', 'L'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'YESTERDAY',
            category: 'time',
            location: ['face', 'chin'],
            handshape: { dom: ['thumbsUp', 'flat', 'L'] },
            movement: { dom: { type: ['linear-left', 'arc-ccw', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'MORNING',
            category: 'time',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-up', 'arc-cw', 'any-movement'], direction: 'up' } },
            orientation: ['palm-up'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'AFTERNOON',
            category: 'time',
            location: ['face', 'forehead', 'upper-chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'arc-cw', 'any-movement'], direction: 'forward-down' } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'EVENING',
            category: 'time',
            location: ['chest', 'neutral', 'upper-chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' }, non: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-down'],
            twoHanded: 'symmetric',
            contact: null,
        },
        {
            sign: 'NIGHT',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['arc-cw', 'linear-down', 'any-movement'] }, non: { type: ['arc-ccw', 'linear-down', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'apart-to-together',
            contact: null,
        },
        {
            sign: 'NOW',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point', 'flat'] },
            movement: { dom: { type: ['linear-down', 'tap', 'any-movement'], direction: 'down' } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'LATER',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['L', 'point'] },
            movement: { dom: { type: ['linear-right', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'BEFORE',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-left', 'any-movement'] } },
            orientation: ['palm-back'],
            twoHanded: 'non-static',
            contact: null,
        },
        {
            sign: 'AFTER',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-right', 'any-movement'] } },
            orientation: ['palm-forward'],
            twoHanded: 'non-static',
            contact: null,
        },
        {
            sign: 'ALWAYS',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['circular-cw', 'circular-ccw', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'NEVER',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'oscillate-h', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'SOMETIMES',
            category: 'time',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point'] },
            movement: { dom: { type: ['oscillate-v', 'tap', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },

        // ── PLACES ─────────────────────────────────────────────────────

        {
            sign: 'HOME',
            category: 'places',
            location: ['chin', 'face'],
            handshape: { dom: ['flat', 'pinch'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'SCHOOL',
            category: 'places',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['oscillate-v', 'tap', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'HOSPITAL',
            category: 'places',
            location: ['upper-chest', 'chest'],
            handshape: { dom: ['V', 'point'] },
            movement: { dom: { type: ['linear-right', 'linear', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chest',
        },
        {
            sign: 'SHOP',
            category: 'places',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'tap', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'CHURCH',
            category: 'places',
            location: ['neutral', 'chest', 'upper-chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-up', 'any-movement'] } },
            orientation: ['palm-forward'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'OFFICE',
            category: 'places',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['arc-cw', 'any-movement'] }, non: { type: ['arc-ccw', 'any-movement'] } },
            orientation: ['palm-forward'],
            twoHanded: 'symmetric',
            contact: null,
        },

        // ── OTHER ──────────────────────────────────────────────────────

        {
            sign: 'NAME',
            category: 'other',
            location: ['forehead', 'face'],
            handshape: { dom: ['V', 'point'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'forehead',
        },
        {
            sign: 'AGE',
            category: 'other',
            location: ['chin', 'face', 'nose'],
            handshape: { dom: ['point', 'L'] },
            movement: { dom: { type: ['linear-down', 'any-movement'], direction: 'down' } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'NUMBER',
            category: 'other',
            location: ['neutral', 'chest'],
            handshape: { dom: ['pinch', 'C'], non: ['pinch', 'C'] },
            movement: { dom: { type: ['oscillate-v', 'tap', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'MONEY',
            category: 'other',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
        {
            sign: 'FOOD',
            category: 'other',
            location: ['chin', 'face'],
            handshape: { dom: ['pinch', 'C', 'flat'] },
            movement: { dom: { type: ['tap', 'oscillate-v'] } },
            orientation: ['palm-back'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'WATER',
            category: 'other',
            location: ['chin', 'face'],
            handshape: { dom: ['point', 'V'] },
            movement: { dom: { type: ['tap', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: 'chin',
        },
        {
            sign: 'CAR',
            category: 'other',
            location: ['neutral', 'chest'],
            handshape: { dom: ['fist'], non: ['fist'] },
            movement: { dom: { type: ['oscillate-h', 'oscillate-circular', 'any-movement'] }, non: { type: ['oscillate-h', 'oscillate-circular', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'symmetric',
            contact: null,
        },
        {
            sign: 'HOUSE',
            category: 'other',
            location: ['neutral', 'upper-chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['linear-down', 'arc-cw', 'any-movement'] }, non: { type: ['linear-down', 'arc-ccw', 'any-movement'] } },
            orientation: ['palm-forward'],
            twoHanded: 'symmetric',
            contact: null,
        },
        {
            sign: 'BOOK',
            category: 'other',
            location: ['neutral', 'chest'],
            handshape: { dom: ['flat', 'open'], non: ['flat', 'open'] },
            movement: { dom: { type: ['arc-cw', 'any-movement'] } },
            orientation: ['palm-up'],
            twoHanded: 'together',
            contact: 'other-hand',
        },
        {
            sign: 'PHONE',
            category: 'other',
            location: ['face', 'chin'],
            handshape: { dom: ['L', 'point', 'fist'] },
            movement: { dom: { type: ['static', 'any-movement'] } },
            orientation: ['palm-side'],
            twoHanded: 'one-handed',
            contact: null,
        },
        {
            sign: 'COMPUTER',
            category: 'other',
            location: ['neutral', 'chest'],
            handshape: { dom: ['point', 'L'], non: ['flat', 'open'] },
            movement: { dom: { type: ['tap', 'oscillate-v', 'any-movement'] } },
            orientation: ['palm-down'],
            twoHanded: 'non-static',
            contact: 'other-hand',
        },
    ];

    console.log(`SignRecogniser: using ${SIGN_DEFINITIONS.length} sign definitions` +
        (typeof BSL_SIGN_DICTIONARY !== 'undefined' ? ' (external dictionary)' : ' (inline fallback)'));

    // ═══════════════════════════════════════════════════════════════════════
    // Sign scoring — aggregate all phonological parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Score a single sign definition against the current observations.
     * Returns a confidence value in [0, 1].
     * If detailed=true, returns { total, components: { mov, loc, shape, ori, two, contact } }
     */
    function scoreSign(def, snapshot, domTraj, nonTraj, detailed) {
        // CRITICAL: if dominant hand is not detected, don't recognise manual signs
        // This prevents false positives when hands are out of frame or at rest
        if (!snapshot.dom.present) return detailed ? { total: 0, components: null } : 0;

        const movScore = scoreMovement(def.movement, domTraj, nonTraj);
        const locScore = scoreLocation(def.location, snapshot);
        const shapeScore = scoreHandshape(def.handshape, snapshot);
        const oriScore = scoreOrientation(def.orientation, snapshot);
        const twoScore = scoreTwoHanded(def.twoHanded, snapshot, domTraj, nonTraj);
        const contactScore = scoreContact(def.contact, snapshot);

        // Multiplicative scoring: each specified parameter must match well.
        // Uses weighted geometric mean — a low score on ANY parameter pulls
        // the total down much more than additive scoring, preventing
        // generic signs from being false-positive attractors.
        // Unspecified parameters (null) use a neutral 0.5 baseline.
        const params = [
            { score: Math.max(movScore, 0.01), weight: W_MOVEMENT },
            { score: Math.max(locScore, 0.01), weight: W_LOCATION },
            { score: Math.max(shapeScore, 0.01), weight: W_HANDSHAPE },
            { score: Math.max(oriScore, 0.01), weight: W_ORIENTATION },
            { score: Math.max(twoScore, 0.01), weight: W_TWO_HANDED },
            { score: Math.max(contactScore, 0.01), weight: W_CONTACT },
        ];

        // Weighted geometric mean: product of (score ^ weight)
        let total = 1.0;
        for (const p of params) {
            total *= Math.pow(p.score, p.weight);
        }

        // If two-handed sign but non-dom hand not present, penalise
        if (def.twoHanded && def.twoHanded !== 'one-handed' && !snapshot.non.present) {
            total *= 0.5;
        }

        total = clamp01(total);

        if (detailed) {
            return {
                total,
                components: {
                    mov: +movScore.toFixed(3),
                    loc: +locScore.toFixed(3),
                    shape: +shapeScore.toFixed(3),
                    ori: +oriScore.toFixed(3),
                    two: +twoScore.toFixed(3),
                    contact: +contactScore.toFixed(3),
                },
            };
        }
        return total;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Non-Manual Feature (NMF) Detection
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Extract face landmark data for NMF detection.
     * Uses MediaPipe face landmarks (468 points) to track head nods, shakes,
     * eyebrow raises, and mouth patterns.
     *
     * Key landmarks:
     *   1  = nose tip
     *   13 = upper lip (top)
     *   14 = lower lip (bottom)
     *   70, 63 = left eyebrow inner/outer
     *   300, 293 = right eyebrow inner/outer
     *   159, 145 = left eye upper/lower lid
     *   386, 374 = right eye upper/lower lid
     *   234 = right ear tragion
     *   454 = left ear tragion
     */
    function extractFaceSnapshot(faceLandmarks) {
        if (!faceLandmarks || faceLandmarks.length < 468) return null;

        const noseTip = faceLandmarks[1];
        const upperLip = faceLandmarks[13];
        const lowerLip = faceLandmarks[14];

        // Eyebrow landmarks
        const leftBrowInner = faceLandmarks[70];
        const leftBrowOuter = faceLandmarks[63];
        const rightBrowInner = faceLandmarks[300];
        const rightBrowOuter = faceLandmarks[293];

        // Eye landmarks (for brow-to-eye distance)
        const leftEyeUpper = faceLandmarks[159];
        const leftEyeLower = faceLandmarks[145];
        const rightEyeUpper = faceLandmarks[386];
        const rightEyeLower = faceLandmarks[374];

        // Ear landmarks for head tilt
        const rightEar = faceLandmarks[234];
        const leftEar = faceLandmarks[454];

        // Compute derived features
        const mouthDist = Math.abs(upperLip.y - lowerLip.y);

        // Brow-to-eye distances (normalised)
        const leftBrowEyeDist = Math.abs(
            ((leftBrowInner.y + leftBrowOuter.y) / 2) -
            ((leftEyeUpper.y + leftEyeLower.y) / 2)
        );
        const rightBrowEyeDist = Math.abs(
            ((rightBrowInner.y + rightBrowOuter.y) / 2) -
            ((rightEyeUpper.y + rightEyeLower.y) / 2)
        );
        const avgBrowEyeDist = (leftBrowEyeDist + rightBrowEyeDist) / 2;

        // Head tilt: difference in ear Y positions
        const earYDiff = rightEar.y - leftEar.y; // positive = tilted right

        return {
            noseTip: { x: noseTip.x, y: noseTip.y },
            mouthDist,
            avgBrowEyeDist,
            earYDiff,
            timestamp: performance.now(),
        };
    }

    /**
     * Detect non-manual features from the face landmark buffer.
     * Updates the currentNmf state object.
     */
    function detectNonManualFeatures() {
        if (nmfBuffer.length < 6) {
            currentNmf = {
                headNod: false, headShake: false, headTilt: 0,
                eyebrowRaise: false, mouthOpen: false,
                nodCount: 0, shakeCount: 0,
            };
            return;
        }

        // ── Head nod detection (vertical nose tip oscillation) ──
        const yPositions = nmfBuffer.map(f => f.noseTip.y);
        let verticalReversals = 0;
        let verticalAmplitude = 0;
        for (let i = 2; i < yPositions.length; i++) {
            const dy1 = yPositions[i - 1] - yPositions[i - 2];
            const dy2 = yPositions[i] - yPositions[i - 1];
            if (dy1 * dy2 < 0 && Math.abs(dy1) > NMF_NOD_THRESHOLD * 0.3) {
                verticalReversals++;
                verticalAmplitude = Math.max(verticalAmplitude, Math.abs(dy1));
            }
        }
        const headNod = verticalReversals >= NMF_NOD_MIN_REVERSALS &&
                         verticalAmplitude > NMF_NOD_THRESHOLD;

        // ── Head shake detection (horizontal nose tip oscillation) ──
        const xPositions = nmfBuffer.map(f => f.noseTip.x);
        let horizontalReversals = 0;
        let horizontalAmplitude = 0;
        for (let i = 2; i < xPositions.length; i++) {
            const dx1 = xPositions[i - 1] - xPositions[i - 2];
            const dx2 = xPositions[i] - xPositions[i - 1];
            if (dx1 * dx2 < 0 && Math.abs(dx1) > NMF_SHAKE_THRESHOLD * 0.3) {
                horizontalReversals++;
                horizontalAmplitude = Math.max(horizontalAmplitude, Math.abs(dx1));
            }
        }
        const headShake = horizontalReversals >= NMF_SHAKE_MIN_REVERSALS &&
                           horizontalAmplitude > NMF_SHAKE_THRESHOLD;

        // ── Head tilt ──
        const latestTilt = nmfBuffer[nmfBuffer.length - 1].earYDiff;
        const headTilt = Math.abs(latestTilt) > NMF_TILT_THRESHOLD ? latestTilt : 0;

        // ── Eyebrow raise detection ──
        // Compare recent brow-eye distance to baseline (first few frames in buffer)
        const baselineFrames = Math.min(6, Math.floor(nmfBuffer.length / 2));
        let baselineBrowDist = 0;
        for (let i = 0; i < baselineFrames; i++) {
            baselineBrowDist += nmfBuffer[i].avgBrowEyeDist;
        }
        baselineBrowDist /= baselineFrames;

        const recentFrames = 4;
        let recentBrowDist = 0;
        for (let i = nmfBuffer.length - recentFrames; i < nmfBuffer.length; i++) {
            recentBrowDist += nmfBuffer[i].avgBrowEyeDist;
        }
        recentBrowDist /= recentFrames;

        const eyebrowRaise = (recentBrowDist - baselineBrowDist) > NMF_BROW_RAISE_THRESHOLD;

        // ── Mouth open detection ──
        const latestMouth = nmfBuffer[nmfBuffer.length - 1].mouthDist;
        const mouthOpen = latestMouth > NMF_MOUTH_OPEN_THRESHOLD;

        currentNmf = {
            headNod,
            headShake,
            headTilt,
            eyebrowRaise,
            mouthOpen,
            nodCount: verticalReversals,
            shakeCount: horizontalReversals,
        };
    }

    /**
     * Check if the current NMF state matches an nmfAlternative definition.
     * Returns a confidence score 0-1, or 0 if no match.
     */
    function scoreNmfAlternative(nmfAlt) {
        if (!nmfAlt) return 0;

        if (nmfAlt.headNod) {
            if (!currentNmf.headNod) return 0;
            const minNods = nmfAlt.minNods || 2;
            if (currentNmf.nodCount < minNods) return 0;
            // Scale confidence by nod count (more nods = more confident)
            return clamp01(0.6 + 0.1 * Math.min(currentNmf.nodCount - minNods, 4));
        }

        if (nmfAlt.headShake) {
            if (!currentNmf.headShake) return 0;
            const minShakes = nmfAlt.minShakes || 2;
            if (currentNmf.shakeCount < minShakes) return 0;
            return clamp01(0.6 + 0.1 * Math.min(currentNmf.shakeCount - minShakes, 4));
        }

        return 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Sign Segmentation — velocity-based stroke detection
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Compute the instantaneous velocity magnitude of the dominant hand
     * between two consecutive smoothed snapshots.
     */
    let lastValidVelocity = 0;
    let handDropoutFrames = 0;

    function computeInstantVelocity(curr, prev) {
        if (!curr || !prev) return 0;
        // When hand landmarks aren't detected, allow 2 frames of dropout
        // before zeroing velocity — prevents single-frame flicker from
        // killing segments
        if (!curr.dom.present || !prev.dom.present) {
            handDropoutFrames++;
            if (handDropoutFrames <= 2) return lastValidVelocity * 0.5;
            return 0;
        }
        handDropoutFrames = 0;
        const dt = (curr.timestamp - prev.timestamp) / 1000;
        if (dt < 0.001) return 0;
        const dx = curr.dom.pos.x - prev.dom.pos.x;
        const dy = curr.dom.pos.y - prev.dom.pos.y;
        lastValidVelocity = Math.sqrt(dx * dx + dy * dy) / dt;
        return lastValidVelocity;
    }

    /**
     * Update the segmentation state machine.
     * Called each frame with the latest smoothed snapshot and velocity.
     *
     * State transitions:
     *   IDLE → ONSET: velocity rises above threshold
     *   ONSET → STROKE: velocity continues above threshold for several frames
     *   ONSET → IDLE: velocity drops back below threshold quickly (false start)
     *   STROKE → HOLD: velocity drops below threshold after sustained movement
     *   STROKE → RETRACTION: velocity drops while hand moves away from sign space
     *   HOLD → RETRACTION: hold phase ends (too many frames or velocity rises slightly)
     *   HOLD → IDLE: hold was the end of the sign
     *   RETRACTION → IDLE: velocity returns to idle level
     *
     * Returns: { event: 'segment-complete' | 'segment-active' | null, segment: [...] | null }
     */
    function updateSegmentation(smoothed, velocity) {
        const frameIndex = smoothedBuffer.length - 1;
        const isAboveThreshold = velocity > SEG_VELOCITY_THRESHOLD;
        const isWellAboveThreshold = velocity > SEG_VELOCITY_THRESHOLD * 1.8;

        let result = { event: null, segment: null };

        switch (segState) {
            case 'IDLE':
                if (isAboveThreshold) {
                    segState = 'ONSET';
                    segOnsetFrame = frameIndex;
                    segIdleCounter = 0;
                    segPeakVelocity = velocity;
                    segSegmentBuffer = [smoothed];
                } else {
                    segIdleCounter++;
                }
                break;

            case 'ONSET':
                segSegmentBuffer.push(smoothed);
                segPeakVelocity = Math.max(segPeakVelocity, velocity);

                if (!isAboveThreshold) {
                    // False start — velocity dropped before establishing a stroke
                    if (segSegmentBuffer.length < 4) {
                        segState = 'IDLE';
                        segSegmentBuffer = [];
                        segIdleCounter = 0;
                    } else {
                        // Short burst — might be a tap, treat as a complete segment
                        segState = 'IDLE';
                        result = { event: 'segment-complete', segment: segSegmentBuffer.slice() };
                        segSegmentBuffer = [];
                        segIdleCounter = 0;
                    }
                } else if (frameIndex - segOnsetFrame >= 3) {
                    // Sustained above threshold — transition to STROKE
                    segState = 'STROKE';
                    segStrokeStart = segOnsetFrame;
                }

                // Safety: don't let onset run too long
                if (frameIndex - segOnsetFrame > 15) {
                    segState = 'STROKE';
                    segStrokeStart = segOnsetFrame;
                }
                break;

            case 'STROKE':
                segSegmentBuffer.push(smoothed);
                segPeakVelocity = Math.max(segPeakVelocity, velocity);

                if (!isAboveThreshold) {
                    // Velocity dropped — enter HOLD phase
                    segState = 'HOLD';
                    segHoldCounter = 0;
                } else if (segSegmentBuffer.length > SEG_MAX_SEGMENT_FRAMES) {
                    // Segment too long — force complete and start new
                    segState = 'IDLE';
                    result = { event: 'segment-complete', segment: segSegmentBuffer.slice() };
                    segSegmentBuffer = [];
                    segIdleCounter = 0;
                }
                break;

            case 'HOLD':
                segSegmentBuffer.push(smoothed);
                segHoldCounter++;

                if (isWellAboveThreshold) {
                    // Movement resumed — back to stroke (might be multi-stroke sign)
                    segState = 'STROKE';
                } else if (isAboveThreshold && segHoldCounter < 3) {
                    // Brief dip — back to stroke
                    segState = 'STROKE';
                } else if (segHoldCounter >= SEG_HOLD_MAX_FRAMES) {
                    // Hold too long — this is the end of the sign
                    segState = 'RETRACTION';
                    segRetractionCounter = 0;
                }
                break;

            case 'RETRACTION':
                segSegmentBuffer.push(smoothed);
                segRetractionCounter++;

                if (!isAboveThreshold) {
                    segIdleCounter++;
                } else {
                    segIdleCounter = 0;
                }

                // Complete when idle for enough frames or retraction runs too long
                if (segIdleCounter >= SEG_IDLE_FRAMES || segRetractionCounter > 20) {
                    segState = 'IDLE';
                    const seg = segSegmentBuffer.slice();
                    segSegmentBuffer = [];
                    segIdleCounter = 0;
                    if (seg.length >= SEG_MIN_STROKE_FRAMES) {
                        result = { event: 'segment-complete', segment: seg };
                    }
                }
                break;
        }

        // Mark active segment
        if (segState !== 'IDLE' && result.event === null) {
            result.event = 'segment-active';
        }

        return result;
    }

    /**
     * Flush any in-progress segment, forcing recognition on whatever has
     * accumulated so far. Useful when video playback ends mid-sign.
     */
    function flushSegment() {
        let seg = null;

        if (segState !== 'IDLE' && segSegmentBuffer.length >= SEG_MIN_STROKE_FRAMES) {
            seg = segSegmentBuffer.slice();
        } else if (segState === 'IDLE' && smoothedBuffer.length >= SEG_MIN_STROKE_FRAMES) {
            // Fallback: segmenter never fired. Find the frames with most hand movement
            // and use those as a synthetic segment.
            const framesWithHands = smoothedBuffer.filter(f => f.dom.present);
            if (framesWithHands.length >= SEG_MIN_STROKE_FRAMES) {
                // Use the middle 60% of hand-present frames (likely the stroke)
                const start = Math.floor(framesWithHands.length * 0.2);
                const end = Math.ceil(framesWithHands.length * 0.8);
                seg = framesWithHands.slice(start, end);
                console.log(`[SignRecogniser] Segmenter fallback: using ${seg.length} frames from ${framesWithHands.length} hand-present frames`);
            }
        }

        resetSegmentation();

        if (seg && seg.length >= SEG_MIN_STROKE_FRAMES) {
            lastCompletedSegment = seg;
            const recognition = recogniseSegment(seg);
            if (recognition && recognition.best.confidence >= MATCH_THRESHOLD) {
                const best = recognition.best;
                const now = Date.now();

                if (best.sign !== lastRecognition.sign || now - lastRecognition.time > COOLDOWN_MS) {
                    lastRecognition = { sign: best.sign, time: now };
                    appendToSentence(best);

                    if (onRecognition) {
                        onRecognition({
                            sign: best.sign,
                            confidence: best.confidence,
                            source: best.source,
                            segmentDuration: recognition.segmentDuration,
                            alternatives: recognition.alternatives,
                            sentence: getSentence(),
                            nmf: { ...currentNmf },
                        });
                    }
                }
            }
        }
    }

    /**
     * Reset segmentation state machine.
     */
    function resetSegmentation() {
        segState = 'IDLE';
        segOnsetFrame = 0;
        segStrokeStart = 0;
        segIdleCounter = 0;
        segHoldCounter = 0;
        segRetractionCounter = 0;
        segPeakVelocity = 0;
        segSegmentBuffer = [];
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Segment-based recognition
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Run full sign recognition on a completed segment.
     * This analyses the segment's trajectory, handshape, location etc.
     * and scores against all sign definitions.
     *
     * Returns the top N matches with confidence scores.
     */
    function recogniseSegment(segment) {
        if (!segment || segment.length < SEG_MIN_STROKE_FRAMES) return null;

        // Require at least 40% of frames to have actual hand landmarks detected.
        // Without this, segments triggered by residual pose wrist movement (when
        // hands are barely in frame) would produce unreliable recognition.
        const framesWithHands = segment.filter(s => s.dom.present).length;
        if (framesWithHands / segment.length < 0.4) return null;

        // Classify trajectory over the stroke portion (middle 60%) of the segment,
        // excluding onset and retraction which can dominate the trajectory.
        const trimStart = Math.floor(segment.length * 0.2);
        const trimEnd = Math.ceil(segment.length * 0.8);
        const strokeSegment = segment.length > 10
            ? segment.slice(trimStart, trimEnd)
            : segment;
        const domTraj = classifyTrajectory(strokeSegment, 'dom');
        const nonTraj = classifyTrajectory(strokeSegment, 'non');

        // MULTI-FRAME SCORING: Sample multiple frames from the stroke portion
        // and score each, then take the median. This dramatically reduces noise
        // from single-frame MediaPipe jitter in location/handshape/orientation.
        const sampleFrames = [];
        const strokeFrames = strokeSegment.filter(s => s.dom.present);
        if (strokeFrames.length >= 3) {
            // Sample 5 evenly-spaced frames from the stroke
            const step = Math.max(1, Math.floor(strokeFrames.length / 5));
            for (let i = 0; i < strokeFrames.length; i += step) {
                sampleFrames.push(strokeFrames[i]);
                if (sampleFrames.length >= 5) break;
            }
        } else {
            // Too few frames — use what we have
            sampleFrames.push(...strokeFrames);
        }
        // Always include the midpoint as a sample
        const midIndex = Math.floor(segment.length / 2);
        const strokeSnapshot = segment[midIndex];
        if (strokeSnapshot.dom.present && !sampleFrames.includes(strokeSnapshot)) {
            sampleFrames.push(strokeSnapshot);
        }

        // Classify observed features for diagnostics (from midpoint)
        const obsLocation = classifyLocation(strokeSnapshot, 'dom');
        const obsHandshape = classifyHandshape(strokeSnapshot.dom);
        const obsOrientation = classifyOrientation(strokeSnapshot.dom);
        const obsNonPresent = strokeSnapshot.non.present;
        const obsNonHandshape = obsNonPresent ? classifyHandshape(strokeSnapshot.non) : null;

        const diagnostics = {
            frames: segment.length,
            framesWithHands,
            sampleCount: sampleFrames.length,
            location: obsLocation,
            handshape: obsHandshape,
            orientation: obsOrientation,
            nonPresent: obsNonPresent,
            nonHandshape: obsNonHandshape,
            domTrajectory: domTraj,
            nonTrajectory: nonTraj,
            nmf: { ...currentNmf },
        };

        console.log('[SignRecogniser] Segment diagnostics:', JSON.stringify(diagnostics, null, 2));

        // Score all sign definitions using multi-frame median scoring
        const scores = SIGN_DEFINITIONS.map(def => {
            // Score on each sample frame
            const frameScores = sampleFrames.map(frame =>
                scoreSign(def, frame, domTraj, nonTraj)
            );
            // Use median score — robust to outlier frames
            frameScores.sort((a, b) => a - b);
            const medianIdx = Math.floor(frameScores.length / 2);
            let manualScore = frameScores.length % 2 === 1
                ? frameScores[medianIdx]
                : (frameScores[medianIdx - 1] + frameScores[medianIdx]) / 2;

            // Check NMF alternative path
            let nmfScore = 0;
            if (def.nmfAlternative) {
                nmfScore = scoreNmfAlternative(def.nmfAlternative);
            }

            // Use the better of manual or NMF score
            const bestScore = Math.max(manualScore, nmfScore);
            const source = nmfScore > manualScore ? 'nmf' : 'segment';

            return {
                sign: def.sign,
                confidence: bestScore,
                source,
                manualScore,
                nmfScore,
            };
        });

        // NMF priority: if any sign has a strong NMF match (≥0.65), it should
        // beat manual scores from other signs. NMF (head nod/shake) is very
        // specific and reliable — boost NMF-matched signs to ensure they win.
        const bestNmfScore = Math.max(...scores.map(s => s.nmfScore));
        if (bestNmfScore >= 0.65) {
            for (const s of scores) {
                if (s.nmfScore >= 0.65) {
                    // Boost NMF score to ensure it wins over other manual scores
                    s.confidence = Math.max(s.confidence, s.nmfScore + 0.10);
                    s.source = 'nmf';
                }
            }
        }

        scores.sort((a, b) => b.confidence - a.confidence);

        // Get detailed component breakdown for top 5 signs (using midpoint snapshot)
        const top5 = scores.slice(0, 5);
        for (const s of top5) {
            const def = SIGN_DEFINITIONS.find(d => d.sign === s.sign);
            if (def) {
                const detail = scoreSign(def, strokeSnapshot, domTraj, nonTraj, true);
                s.components = detail.components;
            }
        }
        console.log('[SignRecogniser] Top 5 matches:', top5.map(s => {
            const c = s.components;
            const breakdown = c ? ` [m=${c.mov} l=${c.loc} s=${c.shape} o=${c.ori} t=${c.two} c=${c.contact}]` : '';
            return `${s.sign}: ${(s.confidence * 100).toFixed(1)}%${breakdown}`;
        }).join(' | '));

        // ── DTW template matching (blended scoring) ──
        // When DTW templates are loaded, blend categorical and DTW scores.
        // Templates are recorded from segmented strokes (same representation as queries).
        if (dtwEnabled && dtwTemplates.size > 0) {
            const dtwScores = scoreDtwAll(segment);
            if (dtwScores.length > 0) {
                const dtwMap = new Map(dtwScores.map(d => [d.sign, d]));

                for (const s of scores) {
                    const dtw = dtwMap.get(s.sign);
                    s.categoricalScore = s.confidence;
                    s.dtwScore = dtw ? dtw.dtwScore : 0;
                    s.dtwDist = dtw ? dtw.dtwDist : Infinity;

                    if (dtw) {
                        // Blend: categorical + DTW weighted
                        s.confidence = s.confidence * DTW_BLEND_CAT + s.dtwScore * DTW_BLEND_DTW;
                    }
                    // Signs without DTW templates keep pure categorical score
                }

                scores.sort((a, b) => b.confidence - a.confidence);

                console.log('[SignRecogniser] DTW blend:', scores.slice(0, 5).map(s =>
                    `${s.sign}: ${(s.confidence * 100).toFixed(1)}% (cat=${(s.categoricalScore * 100).toFixed(1)}% dtw=${(s.dtwScore * 100).toFixed(1)}%)`
                ).join(' | '));
            }
        }

        // No ambiguity penalty — multi-frame median scoring already handles noise,
        // and with 270 signs there's always a close second place.
        // Log the gap for diagnostics only.
        let best = scores[0];
        const second = scores[1];
        if (best && second && best.confidence > 0 && second.confidence > 0) {
            const gap = best.confidence - second.confidence;
            if (gap < 0.03) {
                console.log(`[SignRecogniser] Close match: ${best.sign}=${(best.confidence*100).toFixed(1)}% vs ${second.sign}=${(second.confidence*100).toFixed(1)}%`);
            }
        }

        const result = {
            best,
            alternatives: scores.slice(1, TOP_N).filter(s => s.confidence > MATCH_THRESHOLD * 0.5),
            segmentDuration: segment[segment.length - 1].timestamp - segment[0].timestamp,
            diagnostics,
        };

        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Sentence accumulation
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Append a recognised sign to the sentence buffer.
     * Handles NMF modifiers (head nod = emphasis, head shake = negation,
     * eyebrow raise = question marker).
     */
    function appendToSentence(signData) {
        const now = Date.now();

        // If too much time has passed since last sign, start a new sentence
        if (sentenceBuffer.length > 0 && (now - lastSignTime) > SENTENCE_TIMEOUT_MS) {
            sentenceBuffer = [];
        }

        // Require high confidence for sentence accumulation
        if (signData.confidence < 0.65) return;

        // Only segment-based or NMF recognition contributes to sentences
        // (continuous/fallback is too noisy for sentence building)
        if (signData.source !== 'segment' && signData.source !== 'nmf') return;

        // Never add the same sign consecutively (wait for a different sign first)
        if (sentenceBuffer.length > 0) {
            const lastEntry = sentenceBuffer[sentenceBuffer.length - 1];
            if (lastEntry.sign === signData.sign) {
                return; // suppress consecutive duplicate
            }
        }

        const entry = {
            sign: signData.sign,
            confidence: signData.confidence,
            timestamp: now,
            source: signData.source,
            nmf: { ...currentNmf },
        };

        sentenceBuffer.push(entry);
        lastSignTime = now;

        // Trim sentence if too long
        if (sentenceBuffer.length > SENTENCE_MAX_SIGNS) {
            sentenceBuffer = sentenceBuffer.slice(-SENTENCE_MAX_SIGNS);
        }

        // Fire sentence update callback
        if (onSentenceUpdate) {
            onSentenceUpdate({
                sentence: sentenceBuffer.map(e => e.sign),
                entries: sentenceBuffer.slice(),
                nmf: { ...currentNmf },
            });
        }
    }

    /**
     * Get the current accumulated sentence as an array of sign names.
     */
    function getSentence() {
        // Check for timeout
        const now = Date.now();
        if (sentenceBuffer.length > 0 && (now - lastSignTime) > SENTENCE_TIMEOUT_MS) {
            // Sentence has timed out but we still return it until cleared
        }
        return sentenceBuffer.map(e => e.sign);
    }

    /**
     * Get full sentence data including confidence and timing.
     */
    function getSentenceData() {
        return sentenceBuffer.slice();
    }

    /**
     * Clear the sentence buffer.
     */
    function clearSentence() {
        sentenceBuffer = [];
        lastSignTime = 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Main recognition pipeline
    // ═══════════════════════════════════════════════════════════════════════

    function processLandmarks(results, timestampMs) {
        const snapshot = extractSnapshot(results, timestampMs);
        if (!snapshot) return;

        // Add to raw buffer
        rawBuffer.push(snapshot);
        if (rawBuffer.length > BUFFER_SIZE) rawBuffer.shift();

        // Apply exponential smoothing
        const prevSmoothed = smoothedBuffer.length > 0 ? smoothedBuffer[smoothedBuffer.length - 1] : null;
        const smoothed = smoothSnapshot(snapshot, prevSmoothed);
        smoothedBuffer.push(smoothed);
        if (smoothedBuffer.length > BUFFER_SIZE) smoothedBuffer.shift();

        // ── NMF: Extract face landmarks and detect non-manual features ──
        const faceLm = results.faceLandmarks;
        const faceSnap = extractFaceSnapshot(faceLm);
        if (faceSnap) {
            nmfBuffer.push(faceSnap);
            if (nmfBuffer.length > NMF_BUFFER_SIZE) nmfBuffer.shift();
        }
        detectNonManualFeatures();

        // ── Check for NMF-only signs (head nod = YES, head shake = NO) ──
        // These fire independently of hand segmentation
        checkNmfOnlySigns();

        // Need enough frames for trajectory analysis
        if (smoothedBuffer.length < MIN_FRAMES) return;

        // ── Segmentation: compute velocity and update state machine ──
        const velocity = computeInstantVelocity(
            smoothedBuffer[smoothedBuffer.length - 1],
            smoothedBuffer.length > 1 ? smoothedBuffer[smoothedBuffer.length - 2] : null
        );

        const segResult = updateSegmentation(smoothed, velocity);

        // ── Segment-based recognition ──
        if (segResult.event === 'segment-complete' && segResult.segment) {
            lastCompletedSegment = segResult.segment;
            const recognition = recogniseSegment(segResult.segment);
            if (recognition && recognition.best.confidence >= MATCH_THRESHOLD) {
                const best = recognition.best;
                const now = Date.now();

                // Cooldown check
                if (best.sign !== lastRecognition.sign || now - lastRecognition.time > COOLDOWN_MS) {
                    lastRecognition = { sign: best.sign, time: now };

                    // Append to sentence
                    appendToSentence(best);

                    // Fire callback with enriched data
                    if (onRecognition) {
                        onRecognition({
                            sign: best.sign,
                            confidence: best.confidence,
                            source: best.source,
                            segmentDuration: recognition.segmentDuration,
                            alternatives: recognition.alternatives,
                            sentence: getSentence(),
                            nmf: { ...currentNmf },
                        });
                    }
                }
            }
        }

        // ── Continuous (legacy) recognition as fallback ──
        // Also run continuous per-frame scoring for real-time feedback
        // when no segment has completed. This ensures responsiveness.
        const domTraj = classifyTrajectory(smoothedBuffer, 'dom');
        const nonTraj = classifyTrajectory(smoothedBuffer, 'non');
        trajectoryCache = { dom: domTraj, non: nonTraj };

        const scores = SIGN_DEFINITIONS.map(def => {
            let manualScore = scoreSign(def, smoothed, domTraj, nonTraj);

            // Check NMF alternative
            let nmfScore = 0;
            if (def.nmfAlternative) {
                nmfScore = scoreNmfAlternative(def.nmfAlternative);
            }

            const bestScore = Math.max(manualScore, nmfScore);
            const source = nmfScore > manualScore ? 'nmf' : 'landmarks';

            return { sign: def.sign, confidence: bestScore, source };
        });

        scores.sort((a, b) => b.confidence - a.confidence);

        // Temporal consistency for continuous mode
        recentTopSigns.push(scores[0].sign);
        if (recentTopSigns.length > TEMPORAL_CONSISTENCY * 2) {
            recentTopSigns = recentTopSigns.slice(-TEMPORAL_CONSISTENCY * 2);
        }

        const best = scores[0];
        const now = Date.now();

        // Only fire continuous recognition if no segment just completed
        // and the sign meets threshold + temporal consistency
        if (segResult.event !== 'segment-complete' && best.confidence >= MATCH_THRESHOLD) {
            const recentWindow = recentTopSigns.slice(-TEMPORAL_CONSISTENCY);
            const topCount = recentWindow.filter(s => s === best.sign).length;

            if (topCount >= Math.ceil(TEMPORAL_CONSISTENCY * 0.7)) {
                if (best.sign !== lastRecognition.sign || now - lastRecognition.time > COOLDOWN_MS) {
                    lastRecognition = { sign: best.sign, time: now };

                    appendToSentence(best);

                    if (onRecognition) {
                        onRecognition({
                            sign: best.sign,
                            confidence: best.confidence,
                            source: best.source,
                            segmentDuration: 0,
                            alternatives: scores.slice(1, TOP_N).filter(s => s.confidence > MATCH_THRESHOLD * 0.5),
                            sentence: getSentence(),
                            nmf: { ...currentNmf },
                        });
                    }
                }
            }
        }

        // ── Send idle/low-confidence callback when hands are not detected ──
        // This lets the UI clear the caption to "Watching..." instead of showing
        // stale recognition results indefinitely.
        if (!smoothed.dom.present && best.confidence < MATCH_THRESHOLD && onRecognition) {
            onRecognition({
                sign: '',
                confidence: 0,
                source: 'idle',
                segmentDuration: 0,
                alternatives: [],
                sentence: getSentence(),
                nmf: { ...currentNmf },
            });
        }

        return scores.slice(0, TOP_N);
    }

    /**
     * Check for signs that can be recognised purely from non-manual features
     * (e.g. head nod = YES, head shake = NO) without any hand movement.
     * This runs every frame but has its own cooldown to avoid spam.
     */
    function checkNmfOnlySigns() {
        const now = Date.now();

        for (const def of SIGN_DEFINITIONS) {
            if (!def.nmfAlternative) continue;

            const nmfScore = scoreNmfAlternative(def.nmfAlternative);
            if (nmfScore < 0.7) continue;

            // During STROKE/ONSET, only allow NMF if the score is very high (≥0.75)
            // This prevents NMF from firing during hand signing, but allows strong
            // NMF signals (clear head nod/shake) to override the segmenter
            if ((segState === 'STROKE' || segState === 'ONSET') && nmfScore < 0.75) continue;

            // Cooldown check
            if (def.sign === lastRecognition.sign && now - lastRecognition.time < COOLDOWN_MS * 1.5) continue;

            // Fire recognition
            lastRecognition = { sign: def.sign, time: now };
            appendToSentence({ sign: def.sign, confidence: nmfScore, source: 'nmf' });

            if (onRecognition) {
                onRecognition({
                    sign: def.sign,
                    confidence: nmfScore,
                    source: 'nmf',
                    segmentDuration: 0,
                    alternatives: [],
                    sentence: getSentence(),
                    nmf: { ...currentNmf },
                });
            }

            // Only fire one NMF sign per frame
            break;
        }
    }

    function setCallback(cb) {
        onRecognition = cb;
    }

    function setSentenceCallback(cb) {
        onSentenceUpdate = cb;
    }

    function reset() {
        rawBuffer = [];
        smoothedBuffer = [];
        trajectoryCache = null;
        recentTopSigns = [];
        lastRecognition = { sign: null, time: 0 };
        resetSegmentation();
        nmfBuffer = [];
        currentNmf = {
            headNod: false, headShake: false, headTilt: 0,
            eyebrowRaise: false, mouthOpen: false,
            nodCount: 0, shakeCount: 0,
        };
        sentenceBuffer = [];
        lastSignTime = 0;
    }

    function getSignDictionary() {
        return SIGN_DEFINITIONS.map(d => d.sign);
    }

    function getSegmentationState() {
        return {
            state: segState,
            segmentLength: segSegmentBuffer.length,
            peakVelocity: segPeakVelocity,
            idleCounter: segIdleCounter,
        };
    }

    function getNmfState() {
        return { ...currentNmf };
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DTW Template Matching
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Extract a compact feature vector from a smoothed snapshot for DTW.
     * Returns an array of DTW_FEATURE_DIM numbers.
     */
    function extractDtwFrame(snapshot) {
        // Position relative to nose (signer-invariant location encoding)
        const noseX = snapshot.body ? snapshot.body.nose.x : 0;
        const noseY = snapshot.body ? snapshot.body.nose.y : 0;
        return [
            snapshot.dom.pos.x - noseX,                                     // dom hand x relative to nose
            snapshot.dom.pos.y - noseY,                                     // dom hand y relative to nose
            snapshot.non.present ? (snapshot.non.pos.x - noseX) : 0,        // non-dom hand x relative to nose
            snapshot.non.present ? (snapshot.non.pos.y - noseY) : 0,        // non-dom hand y relative to nose
            ...snapshot.dom.fingerExtension,                                 // 5 binary finger values
            snapshot.dom.openness,                                           // continuous hand openness
            snapshot.dom.spread,                                             // continuous finger spread
            snapshot.dom.palmCross,                                          // continuous palm orientation
            snapshot.dom.fingerDir.x,                                        // finger pointing direction x
            snapshot.dom.fingerDir.y,                                        // finger pointing direction y
            snapshot.dom.pinchDist,                                           // thumb-index distance
            snapshot.non.present ? 1 : 0,                                    // non-dom hand present flag
            snapshot.distances ? snapshot.distances.handDist : 0,            // inter-hand distance
        ];
    }

    /**
     * Weighted Euclidean distance between two DTW feature vectors.
     */
    function dtwFrameDistance(a, b) {
        let sum = 0;
        for (let i = 0; i < DTW_FEATURE_DIM; i++) {
            const d = (a[i] - b[i]) * DTW_WEIGHTS[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /**
     * Dynamic Time Warping distance between two sequences of feature vectors.
     * Uses O(m) memory with two-row approach.
     * Returns normalised distance (total cost / sqrt(n*m)).
     */
    function dtwDistance(seq1, seq2) {
        const n = seq1.length, m = seq2.length;
        if (n === 0 || m === 0) return Infinity;

        // Two-row DP for memory efficiency
        let prev = new Float64Array(m + 1);
        let curr = new Float64Array(m + 1);
        for (let j = 0; j <= m; j++) prev[j] = Infinity;
        prev[0] = 0;

        for (let i = 1; i <= n; i++) {
            curr[0] = Infinity;
            for (let j = 1; j <= m; j++) {
                const cost = dtwFrameDistance(seq1[i - 1], seq2[j - 1]);
                curr[j] = cost + Math.min(prev[j], curr[j - 1], prev[j - 1]);
            }
            [prev, curr] = [curr, prev];
        }

        // Normalise by geometric mean of lengths
        return prev[m] / Math.sqrt(n * m);
    }

    /**
     * Register a DTW template from smoothed snapshots.
     * Extracts features from frames where dom hand is present,
     * subsamples to max 60 frames.
     */
    /**
     * Register a DTW template from an already-segmented stroke.
     * This ensures templates match the same representation used during recognition.
     */
    function registerSegmentTemplate(sign, segment) {
        const frames = segment
            .filter(s => s.dom && s.dom.present && s.body)
            .map(s => extractDtwFrame(s));

        if (frames.length < 4) return false;

        // Subsample long sequences to 60 frames
        let template = frames;
        if (frames.length > 60) {
            const step = frames.length / 60;
            template = [];
            for (let i = 0; i < 60; i++) {
                template.push(frames[Math.floor(i * step)]);
            }
        }

        if (!dtwTemplates.has(sign)) {
            dtwTemplates.set(sign, []);
        }
        dtwTemplates.get(sign).push(template);
        return true;
    }

    /**
     * Get the last completed segment (for template recording).
     */
    function getLastSegment() {
        return lastCompletedSegment;
    }

    function registerTemplate(sign, snapshots) {
        const frames = snapshots
            .filter(s => s.dom && s.dom.present && s.body)
            .map(s => extractDtwFrame(s));

        if (frames.length < 4) return false;

        // Trim idle frames from start and end using velocity-based detection.
        // Compute inter-frame position deltas and trim frames below threshold.
        const velocities = [];
        for (let i = 1; i < frames.length; i++) {
            const dx = frames[i][0] - frames[i - 1][0]; // dom x
            const dy = frames[i][1] - frames[i - 1][1]; // dom y
            velocities.push(Math.sqrt(dx * dx + dy * dy));
        }
        // Find the peak velocity to set an adaptive threshold
        const maxVel = Math.max(...velocities);
        const velThreshold = maxVel * 0.08; // 8% of peak velocity

        // Trim from start: find first frame above threshold
        let trimStart = 0;
        for (let i = 0; i < velocities.length; i++) {
            if (velocities[i] >= velThreshold) {
                trimStart = Math.max(0, i - 2); // keep 2 frames before onset
                break;
            }
        }
        // Trim from end: find last frame above threshold
        let trimEnd = frames.length;
        for (let i = velocities.length - 1; i >= 0; i--) {
            if (velocities[i] >= velThreshold) {
                trimEnd = Math.min(frames.length, i + 3); // keep 2 frames after
                break;
            }
        }

        let trimmed = frames.slice(trimStart, trimEnd);
        if (trimmed.length < 4) trimmed = frames; // fallback to untrimmed

        // Subsample long sequences to 60 frames
        let template = trimmed;
        if (trimmed.length > 60) {
            const step = trimmed.length / 60;
            template = [];
            for (let i = 0; i < 60; i++) {
                template.push(trimmed[Math.floor(i * step)]);
            }
        }

        if (!dtwTemplates.has(sign)) {
            dtwTemplates.set(sign, []);
        }
        dtwTemplates.get(sign).push(template);
        return true;
    }

    /**
     * Score a segment against all DTW templates.
     * Returns array of { sign, dtwScore, dtwDist } sorted by score desc.
     */
    function scoreDtwAll(segment) {
        const inputFrames = segment
            .filter(s => s.dom && s.dom.present && s.body)
            .map(s => extractDtwFrame(s));

        if (inputFrames.length < 4) return [];

        const results = [];
        for (const [sign, templates] of dtwTemplates) {
            // Leave-one-out: skip the sign being tested
            if (dtwExcludeSign && sign === dtwExcludeSign) continue;

            // Find minimum distance across all templates for this sign
            let minDist = Infinity;
            for (const template of templates) {
                const dist = dtwDistance(inputFrames, template);
                if (dist < minDist) minDist = dist;
            }

            // Convert distance to score using exponential decay
            const score = Math.exp(-minDist / DTW_SIGMA);
            results.push({ sign, dtwScore: score, dtwDist: minDist });
        }

        results.sort((a, b) => b.dtwScore - a.dtwScore);
        return results;
    }

    /**
     * Export all templates as a JSON-serialisable object.
     */
    function exportDtwTemplates() {
        const data = {};
        for (const [sign, templates] of dtwTemplates) {
            data[sign] = templates;
        }
        return {
            version: 2,
            templateCount: dtwTemplates.size,
            featureDim: DTW_FEATURE_DIM,
            weights: [...DTW_WEIGHTS],
            sigma: DTW_SIGMA,
            templates: data,
        };
    }

    /**
     * Import templates from a JSON object (produced by exportDtwTemplates).
     */
    function importDtwTemplates(data) {
        dtwTemplates = new Map();
        const templates = data.templates || data; // support both wrapped and raw format
        for (const sign of Object.keys(templates)) {
            dtwTemplates.set(sign, templates[sign]);
        }
        dtwEnabled = dtwTemplates.size > 0;
        console.log(`[SignRecogniser] Loaded ${dtwTemplates.size} DTW templates (${
            Array.from(dtwTemplates.values()).reduce((s, t) => s + t.length, 0)} total sequences)`);
    }

    /**
     * Get a copy of the current smoothed buffer (for template recording).
     */
    function getSmoothedBuffer() {
        return smoothedBuffer.slice();
    }

    /**
     * Set the ML classifier module (MLClassifier).
     */
    function setMlClassifier(classifier) {
        mlClassifier = classifier;
        mlEnabled = classifier && classifier.isReady();
        console.log(`[SignRecogniser] ML classifier ${mlEnabled ? 'enabled' : 'disabled'}`);
    }

    function setMlEnabled(enabled) {
        mlEnabled = enabled && mlClassifier && mlClassifier.isReady();
        console.log(`[SignRecogniser] ML scoring ${mlEnabled ? 'enabled' : 'disabled'}`);
    }

    function setMlMode(mode) {
        mlMode = mode; // 'blend' | 'override' | 'only'
    }

    function setMlBlendWeight(weight) {
        mlBlendWeight = Math.max(0, Math.min(1, weight));
    }

    /**
     * Classify a segment using the ML model. Returns a promise.
     * Used by batch test harness for ML-only or ML-blend scoring.
     */
    async function classifySegmentMl(segment) {
        if (!mlEnabled || !mlClassifier || !mlClassifier.isReady()) return null;
        return mlClassifier.classify(segment);
    }

    /**
     * Enable/disable DTW scoring.
     */
    function setDtwEnabled(enabled) {
        dtwEnabled = enabled && dtwTemplates.size > 0;
        console.log(`[SignRecogniser] DTW scoring ${dtwEnabled ? 'enabled' : 'disabled'} (${dtwTemplates.size} templates)`);
    }

    /**
     * Set the sign to exclude during DTW matching (for leave-one-out testing).
     */
    function setDtwExclude(sign) {
        dtwExcludeSign = sign ? sign.toUpperCase() : null;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Public API
    // ═══════════════════════════════════════════════════════════════════════

    return {
        processLandmarks,
        flushSegment,
        setCallback,
        setSentenceCallback,
        reset,
        getSignDictionary,
        getSentence,
        getSentenceData,
        clearSentence,
        getSegmentationState,
        getNmfState,
        // ML classifier
        setMlClassifier,
        setMlEnabled,
        setMlMode,
        setMlBlendWeight,
        classifySegmentMl,
        // DTW template matching
        registerTemplate,
        registerSegmentTemplate,
        exportDtwTemplates,
        importDtwTemplates,
        getSmoothedBuffer,
        getLastSegment,
        setDtwEnabled,
        setDtwExclude,
    };
})();
