/**
 * Camera module — WebRTC capture.
 * When onBatch callback is provided: accumulates 32 frames (1.28s) at 25fps,
 * sends preprocessed batches every 8 new frames for server-side BSL-1K inference.
 * When onBatch is null: just starts the video stream for MediaPipe landmark detection.
 */
const Camera = (() => {
    const CAPTURE_FPS = 25;
    const BUFFER_SIZE = 32;
    const STRIDE = 8;
    const JPEG_QUALITY = 0.85;

    let stream = null;
    let videoElement = null;
    let captureCanvas = null;
    let captureCtx = null;
    let preprocessCanvas = null;
    let frameBuffer = [];
    let frameCount = 0;
    let captureInterval = null;
    let onBatchReady = null;
    let isCapturing = false;

    async function start(videoEl, onBatch) {
        videoElement = videoEl;
        onBatchReady = onBatch;

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: CAPTURE_FPS },
                    facingMode: 'user',
                },
                audio: false,
            });

            videoElement.srcObject = stream;
            await videoElement.play();

            isCapturing = true;

            // Only start frame capture loop if server-side recognition is enabled
            if (onBatchReady) {
                captureCanvas = document.createElement('canvas');
                captureCanvas.width = 640;
                captureCanvas.height = 480;
                captureCtx = captureCanvas.getContext('2d');

                preprocessCanvas = document.createElement('canvas');
                preprocessCanvas.width = 640;
                preprocessCanvas.height = 480;

                startCapture();
            }

            return true;
        } catch (err) {
            console.error('Camera access failed:', err);
            return false;
        }
    }

    function startCapture() {
        const intervalMs = 1000 / CAPTURE_FPS;

        captureInterval = setInterval(() => {
            if (!isCapturing || !videoElement || videoElement.readyState < 2) return;

            let sourceCanvas;

            // Try to use preprocessed frame (pose-cropped + dark background)
            if (typeof MediaPipeOverlay !== 'undefined' &&
                MediaPipeOverlay.renderPreprocessedFrame &&
                MediaPipeOverlay.renderPreprocessedFrame(preprocessCanvas)) {
                sourceCanvas = preprocessCanvas;
            } else {
                captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
                sourceCanvas = captureCanvas;
            }

            const dataUrl = sourceCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
            const base64 = dataUrl.split(',')[1];

            frameBuffer.push(base64);
            if (frameBuffer.length > BUFFER_SIZE) {
                frameBuffer.shift();
            }

            frameCount++;

            if (frameCount % STRIDE === 0 && frameBuffer.length >= BUFFER_SIZE && onBatchReady) {
                const batch = frameBuffer.slice();
                onBatchReady(batch);
            }
        }, intervalMs);
    }

    function stop() {
        isCapturing = false;
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        frameBuffer = [];
        frameCount = 0;
    }

    function getStream() { return stream; }
    function isActive() { return isCapturing; }
    function getVideoElement() { return videoElement; }

    return { start, stop, getStream, isActive, getVideoElement };
})();
