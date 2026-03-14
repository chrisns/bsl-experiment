/**
 * BSL video player — handles PiP overlay playback for sign video output.
 */
const Player = (() => {
    let pipOverlay = null;
    let pipVideo = null;

    function init() {
        pipOverlay = document.getElementById('pipOverlay');
        pipVideo = document.getElementById('pipVideo');
    }

    function playVideo(url) {
        if (!pipVideo || !pipOverlay) init();
        if (!url) return;

        pipVideo.src = url;
        pipOverlay.style.display = 'block';

        pipVideo.play().catch(err => {
            console.warn('Video playback failed:', err);
        });

        pipVideo.onended = () => {
            // Keep showing last frame for a moment
            setTimeout(() => {
                pipOverlay.style.display = 'none';
            }, 2000);
        };
    }

    function stop() {
        if (pipVideo) {
            pipVideo.pause();
            pipVideo.src = '';
        }
        if (pipOverlay) {
            pipOverlay.style.display = 'none';
        }
    }

    function isPlaying() {
        return pipVideo && !pipVideo.paused && !pipVideo.ended;
    }

    return { init, playVideo, stop, isPlaying };
})();
