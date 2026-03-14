/**
 * Practice mode — learn BSL signs with recognition-based scoring.
 */
const Practice = (() => {
    // Sign sets — curated for BSL-1K model recognition accuracy.
    // Uses BSL-1K vocabulary words (lowercase in model, displayed uppercase).
    const SIGN_SETS = {
        greetings: ['HELLO', 'GOODBYE', 'GOOD-MORNING', 'GOOD-AFTERNOON', 'GOOD-EVENING', 'YES', 'NO', 'PLEASE', 'SORRY', 'THANK-YOU'],
        questions: ['WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW', 'HOW-MUCH', 'HOW-MANY', 'NAME', 'AGE'],
        actions: ['GO', 'COME', 'STOP', 'WAIT', 'EAT', 'DRINK', 'SLEEP', 'WORK', 'HELP', 'WANT'],
        descriptions: ['GOOD', 'BAD', 'BIG', 'SMALL', 'HAPPY', 'SAD', 'EASY', 'HARD', 'FAST', 'SLOW'],
    };

    let currentSet = 'greetings';
    let currentIndex = 0;
    let learned = {};
    let practiceCamera = null;
    let isActive = false;

    function init() {
        // Load progress from localStorage
        try {
            const saved = localStorage.getItem('bsl_practice_progress');
            if (saved) learned = JSON.parse(saved);
        } catch (e) { /* ignore */ }

        // Set up set buttons
        document.querySelectorAll('.practice-set-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.practice-set-btn').forEach(b => {
                    b.classList.remove('active');
                    b.setAttribute('aria-checked', 'false');
                });
                btn.classList.add('active');
                btn.setAttribute('aria-checked', 'true');
                currentSet = btn.dataset.set;
                currentIndex = 0;
                showCurrentSign();
            });
        });

        document.getElementById('nextSignBtn').addEventListener('click', nextSign);
    }

    function activate() {
        isActive = true;
        showCurrentSign();
        updateProgress();
        startPracticeCamera();

        // Hook landmark recogniser for practice scoring
        if (typeof SignRecogniser !== 'undefined') {
            SignRecogniser.setCallback(handleLandmarkResult);
        }
    }

    function deactivate() {
        if (isActive) {
            Camera.stop();
        }
        isActive = false;
    }

    function showCurrentSign() {
        const signs = SIGN_SETS[currentSet] || [];
        if (signs.length === 0) return;

        const word = signs[currentIndex % signs.length];
        document.getElementById('practiceWord').textContent = word;

        // Reset stars
        const stars = document.querySelectorAll('#practiceScore .star');
        stars.forEach(s => s.classList.remove('filled'));

        // Load reference video if available
        const practiceVideo = document.getElementById('practiceVideo');
        const config = ApiClient.getConfig();
        if (config.dataBucket) {
            // Try to load from S3 via CloudFront
            practiceVideo.src = `https://${config.cloudfrontDomain}/lexicon/signbank/${word.toLowerCase()}.mp4`;
            practiceVideo.onerror = () => { practiceVideo.src = ''; };
        }
    }

    function nextSign() {
        currentIndex++;
        showCurrentSign();
    }

    async function startPracticeCamera() {
        // Stop any existing camera session before starting a new one
        if (Camera.isActive()) {
            Camera.stop();
        }

        const videoEl = document.getElementById('practiceCameraVideo');
        const canvasEl = document.getElementById('practiceCanvas');

        const started = await Camera.start(videoEl, handlePracticeBatch);
        if (started) {
            videoEl.style.display = 'block';
            canvasEl.style.display = 'block';
            MediaPipeOverlay.start(videoEl, canvasEl);
        }
    }

    async function handlePracticeBatch(frames) {
        if (!isActive) return;

        try {
            const batchId = `practice-${Date.now()}`;
            const result = await ApiClient.recogniseFrames(frames, batchId);
            const signs = result.signs || [];

            if (signs.length > 0) {
                const currentWord = (SIGN_SETS[currentSet] || [])[currentIndex % (SIGN_SETS[currentSet] || []).length];

                // Check all returned predictions (top-10), not just top-1
                const match = signs.find(s => s.sign.toUpperCase() === currentWord.toUpperCase());

                if (match) {
                    // Score based on confidence and ranking
                    const rank = signs.indexOf(match);
                    const rankBonus = rank === 0 ? 1.0 : rank <= 2 ? 0.8 : 0.6;
                    const effectiveConf = match.confidence * rankBonus;
                    const starCount = Math.min(5, Math.max(1, Math.ceil(effectiveConf * 5)));
                    updateStars(starCount);

                    if (starCount >= 2) {
                        learned[currentWord] = Math.max(learned[currentWord] || 0, starCount);
                        saveProgress();
                        updateProgress();
                    }
                }

                // Show what model sees (feedback for the user)
                const feedback = document.getElementById('practiceWord');
                const top = signs[0];
                if (top.sign.toUpperCase() !== currentWord.toUpperCase() && top.confidence > 0.15) {
                    feedback.title = `Model sees: ${top.sign} (${(top.confidence * 100).toFixed(0)}%)`;
                } else {
                    feedback.title = '';
                }
            }
        } catch (err) {
            console.warn('Practice recognition failed:', err);
        }
    }

    function handleLandmarkResult(result) {
        if (!isActive) return;

        const signs = SIGN_SETS[currentSet] || [];
        const currentWord = signs[currentIndex % signs.length];
        if (!currentWord) return;

        // Check if the recognised sign matches the target
        if (result.sign === currentWord) {
            const starCount = Math.min(5, Math.max(1, Math.ceil(result.confidence * 5)));
            updateStars(starCount);

            if (starCount >= 2) {
                learned[currentWord] = Math.max(learned[currentWord] || 0, starCount);
                saveProgress();
                updateProgress();
            }

            document.getElementById('practiceWord').title = '';
        } else if (result.confidence > 0.4) {
            // Show what recogniser sees as feedback
            document.getElementById('practiceWord').title =
                `Seeing: ${result.sign} (${(result.confidence * 100).toFixed(0)}%)`;
        }
    }

    function updateStars(count) {
        const stars = document.querySelectorAll('#practiceScore .star');
        stars.forEach((s, i) => {
            s.classList.toggle('filled', i < count);
        });
    }

    function updateProgress() {
        const signs = SIGN_SETS[currentSet] || [];
        const learnedCount = signs.filter(s => learned[s] >= 3).length;
        document.getElementById('progressText').textContent =
            `${learnedCount} of ${signs.length} signs learned`;
        const pct = signs.length > 0 ? (learnedCount / signs.length * 100) : 0;
        document.getElementById('progressFill').style.width = `${pct}%`;
    }

    function saveProgress() {
        try {
            localStorage.setItem('bsl_practice_progress', JSON.stringify(learned));
        } catch (e) { /* ignore */ }
    }

    return { init, activate, deactivate };
})();
