/**
 * MediaPipe Holistic overlay — draws pose, hand, and face landmarks
 * on a canvas overlay. Also provides segmentation mask and pose data
 * for frame preprocessing (background removal + crop for BSL-1K).
 */
const MediaPipeOverlay = (() => {
    let holistic = null;
    let mpCamera = null;
    let canvasCtx = null;
    let canvasElement = null;
    let isRunning = false;
    let videoEl = null;

    // Latest results for preprocessing
    let latestResults = null;
    let latestSegMask = null;

    const HAND_COLOR = '#00FF00';
    const POSE_COLOR = '#00D4FF';
    const LINE_WIDTH = 2;
    const POINT_RADIUS = 3;

    async function start(videoElement, canvasEl) {
        videoEl = videoElement;
        canvasElement = canvasEl;
        canvasCtx = canvasEl.getContext('2d');

        canvasEl.width = videoElement.videoWidth || 640;
        canvasEl.height = videoElement.videoHeight || 480;

        try {
            holistic = new Holistic({
                locateFile: (file) =>
                    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
            });

            holistic.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                enableSegmentation: true,
                smoothSegmentation: true,
                refineFaceLandmarks: false,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5,
            });

            holistic.onResults(onResults);

            mpCamera = new Camera_MP(videoElement, {
                onFrame: async () => {
                    if (holistic && isRunning) {
                        await holistic.send({ image: videoElement });
                    }
                },
                width: 640,
                height: 480,
            });

            isRunning = true;
            mpCamera.start();
        } catch (err) {
            console.warn('MediaPipe Holistic init failed:', err);
            isRunning = false;
        }
    }

    function onResults(results) {
        latestResults = results;
        if (results.segmentationMask) {
            latestSegMask = results.segmentationMask;
        }

        // Feed landmarks to sign recogniser
        if (typeof SignRecogniser !== 'undefined') {
            SignRecogniser.processLandmarks(results);
        }

        if (!canvasCtx || !canvasElement) return;

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Draw pose landmarks (upper body)
        if (results.poseLandmarks) {
            drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
                color: POSE_COLOR, lineWidth: LINE_WIDTH,
            });
            results.poseLandmarks.slice(0, 25).forEach((lm) => {
                drawPoint(canvasCtx, lm, POSE_COLOR, POINT_RADIUS);
            });
        }

        // Draw hands
        if (results.leftHandLandmarks) {
            drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
                color: HAND_COLOR, lineWidth: LINE_WIDTH + 1,
            });
            results.leftHandLandmarks.forEach((lm) => drawPoint(canvasCtx, lm, HAND_COLOR, POINT_RADIUS + 1));
        }
        if (results.rightHandLandmarks) {
            drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
                color: HAND_COLOR, lineWidth: LINE_WIDTH + 1,
            });
            results.rightHandLandmarks.forEach((lm) => drawPoint(canvasCtx, lm, HAND_COLOR, POINT_RADIUS + 1));
        }

        canvasCtx.restore();
    }

    /**
     * Get the latest pose landmarks for frame preprocessing.
     * Returns null if no pose detected.
     */
    function getLatestPose() {
        if (!latestResults || !latestResults.poseLandmarks) return null;
        return latestResults.poseLandmarks;
    }

    /**
     * Get the latest segmentation mask (ImageData-like).
     */
    function getSegmentationMask() {
        return latestSegMask;
    }

    /**
     * Get the video element being tracked.
     */
    function getVideoElement() {
        return videoEl;
    }

    /**
     * Render a preprocessed frame: person segmented onto dark background,
     * cropped to upper body signing area. Returns a canvas or null.
     */
    function renderPreprocessedFrame(targetCanvas) {
        const pose = latestResults && latestResults.poseLandmarks;
        const seg = latestSegMask;
        if (!pose || !videoEl) return false;

        const vw = videoEl.videoWidth || 640;
        const vh = videoEl.videoHeight || 480;
        const tw = targetCanvas.width;
        const th = targetCanvas.height;
        const ctx = targetCanvas.getContext('2d');

        // Calculate signing area bounding box from pose landmarks
        const lShoulder = pose[11];
        const rShoulder = pose[12];
        const nose = pose[0];
        const lWrist = pose[15];
        const rWrist = pose[16];
        const lElbow = pose[13];
        const rElbow = pose[14];
        const lHip = pose[23];
        const rHip = pose[24];

        const shoulderW = Math.abs(rShoulder.x - lShoulder.x);
        if (shoulderW < 0.02) return false; // no valid pose

        // Bounding box: head to hips, arm span + padding
        const pad = shoulderW * 0.6;
        const allX = [lShoulder.x, rShoulder.x, lWrist.x, rWrist.x, lElbow.x, rElbow.x];
        const allY = [nose.y, lShoulder.y, rShoulder.y, lWrist.y, rWrist.y];

        let minX = Math.min(...allX) - pad;
        let maxX = Math.max(...allX) + pad;
        let minY = Math.min(nose.y - shoulderW * 0.5, ...allY) - pad * 0.5;
        let maxY = Math.max(lHip.y, rHip.y, ...allY) + pad * 0.3;

        // Clamp to [0, 1]
        minX = Math.max(0, minX);
        maxX = Math.min(1, maxX);
        minY = Math.max(0, minY);
        maxY = Math.min(1, maxY);

        // Convert to pixel coords
        const sx = Math.floor(minX * vw);
        const sy = Math.floor(minY * vh);
        const sw = Math.floor((maxX - minX) * vw);
        const sh = Math.floor((maxY - minY) * vh);

        if (sw < 50 || sh < 50) return false;

        // Dark background
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, tw, th);

        if (seg) {
            // Use segmentation mask: draw video through mask onto dark background
            // Create temporary canvas for masking
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = vw;
            tmpCanvas.height = vh;
            const tmpCtx = tmpCanvas.getContext('2d');

            // Draw video
            tmpCtx.drawImage(videoEl, 0, 0, vw, vh);

            // Apply segmentation mask
            const imgData = tmpCtx.getImageData(0, 0, vw, vh);
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = seg.width || vw;
            maskCanvas.height = seg.height || vh;
            const maskCtx = maskCanvas.getContext('2d');
            maskCtx.drawImage(seg, 0, 0);
            const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

            // Apply mask: set alpha based on segmentation
            const scaleX = maskCanvas.width / vw;
            const scaleY = maskCanvas.height / vh;
            for (let y = 0; y < vh; y++) {
                for (let x = 0; x < vw; x++) {
                    const mi = (Math.floor(y * scaleY) * maskCanvas.width + Math.floor(x * scaleX)) * 4;
                    const ii = (y * vw + x) * 4;
                    // Mask red channel indicates person probability
                    const maskVal = maskData.data[mi] / 255;
                    if (maskVal < 0.5) {
                        // Background: replace with dark color
                        imgData.data[ii] = 26;     // R
                        imgData.data[ii + 1] = 26;  // G
                        imgData.data[ii + 2] = 46;  // B
                    }
                }
            }
            tmpCtx.putImageData(imgData, 0, 0);

            // Draw cropped region scaled to target
            ctx.drawImage(tmpCanvas, sx, sy, sw, sh, 0, 0, tw, th);
        } else {
            // No segmentation: just crop and place on dark background
            ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, tw, th);
        }

        return true;
    }

    function drawPoint(ctx, landmark, color, radius) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(
            landmark.x * canvasElement.width,
            landmark.y * canvasElement.height,
            radius, 0, 2 * Math.PI
        );
        ctx.fill();
    }

    function drawConnectors(ctx, landmarks, connections, style) {
        if (!landmarks || !connections) return;
        ctx.strokeStyle = style.color;
        ctx.lineWidth = style.lineWidth;
        connections.forEach(([start, end]) => {
            if (landmarks[start] && landmarks[end]) {
                ctx.beginPath();
                ctx.moveTo(landmarks[start].x * canvasElement.width, landmarks[start].y * canvasElement.height);
                ctx.lineTo(landmarks[end].x * canvasElement.width, landmarks[end].y * canvasElement.height);
                ctx.stroke();
            }
        });
    }

    function stop() {
        isRunning = false;
        if (mpCamera) { mpCamera.stop(); mpCamera = null; }
        if (holistic) { holistic.close(); holistic = null; }
        latestResults = null;
        latestSegMask = null;
    }

    const POSE_CONNECTIONS = [
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
        [11, 23], [12, 24], [23, 24],
        [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
    ];

    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
    ];

    const Camera_MP = typeof window !== 'undefined' && window.Camera ? window.Camera : class {
        constructor() {}
        start() {}
        stop() {}
    };

    return { start, stop, getLatestPose, getSegmentationMask, getVideoElement, renderPreprocessedFrame };
})();
