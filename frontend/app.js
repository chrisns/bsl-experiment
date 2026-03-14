/**
 * Main application — mode switching, camera lifecycle, text-to-sign flow.
 * Primary recognition is landmark-based (client-side via SignRecogniser).
 * Displays sign-by-sign recognition + accumulated sentence transcript.
 * Shows NMF indicators (head nod, shake, eyebrow raise).
 */
(function () {
    'use strict';

    let currentMode = 'live';
    let cameraStarted = false;

    // Server-side recognition (BSL-1K via SageMaker) — disabled by default
    // as CPU inference is too slow for real-time. Enable via console:
    //   window.__BSL_CONFIG__.enableServerRecognition = true
    let serverRecognitionEnabled = false;
    let isProcessingServer = false;

    // ============ INIT ============
    document.addEventListener('DOMContentLoaded', () => {
        Player.init();
        Practice.init();
        Kiosk.init();

        setupModeNav();
        setupLiveMode();
        setupTextPanel();
        setupSpeech();
        setupAuth();
        setupDominantHand();
        setupSentenceBar();

        // Set up landmark-based sign recogniser callback (primary recognition)
        if (typeof SignRecogniser !== 'undefined') {
            SignRecogniser.setCallback(displayLandmarkPrediction);
            SignRecogniser.setSentenceCallback(updateSentenceDisplay);
        }

        // Initialize ML classifier for browser-side ONNX inference
        if (typeof MLClassifier !== 'undefined') {
            MLClassifier.init('bsl_classifier.onnx', 'model_metadata.json').then(ok => {
                if (ok && typeof SignRecogniser !== 'undefined') {
                    SignRecogniser.setMlClassifier(MLClassifier);
                    SignRecogniser.setMlEnabled(true);
                    SignRecogniser.setMlMode('only');
                }
            });
        }

        // Check if server recognition is enabled via config
        const config = window.__BSL_CONFIG__ || {};
        serverRecognitionEnabled = config.enableServerRecognition === true;

        updateStatus('disconnected');

        // Auto-start camera on page load
        startCamera();
    });

    // ============ MODE NAVIGATION ============
    function setupModeNav() {
        document.querySelectorAll('.nav-btn[data-mode]').forEach(btn => {
            btn.addEventListener('click', () => switchMode(btn.dataset.mode));
        });

        const tabs = document.querySelectorAll('.nav-btn[data-mode]');
        tabs.forEach((tab, i) => {
            tab.addEventListener('keydown', (e) => {
                let newIndex;
                if (e.key === 'ArrowRight') newIndex = (i + 1) % tabs.length;
                else if (e.key === 'ArrowLeft') newIndex = (i - 1 + tabs.length) % tabs.length;
                else return;
                e.preventDefault();
                tabs[newIndex].click();
                tabs[newIndex].focus();
            });
        });
    }

    function switchMode(mode) {
        currentMode = mode;

        document.querySelectorAll('.nav-btn[data-mode]').forEach(btn => {
            const isActive = btn.dataset.mode === mode;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-selected', isActive);
            btn.setAttribute('tabindex', isActive ? '0' : '-1');
        });

        const panels = {
            'live': document.getElementById('live-panel'),
            'practice': document.getElementById('practice-panel'),
            'kiosk': document.getElementById('kiosk-panel'),
        };

        Object.entries(panels).forEach(([key, panel]) => {
            if (key === 'live') {
                panel.style.display = key === mode ? 'flex' : 'none';
            } else {
                panel.hidden = key !== mode;
                panel.classList.toggle('active', key === mode);
            }
        });

        if (mode === 'practice') Practice.activate();
        else Practice.deactivate();

        if (mode === 'kiosk') Kiosk.activate();
        else Kiosk.deactivate();

        // Re-attach landmark recogniser callback for live mode
        if (mode === 'live' && typeof SignRecogniser !== 'undefined') {
            SignRecogniser.setCallback(displayLandmarkPrediction);
            SignRecogniser.setSentenceCallback(updateSentenceDisplay);
        }
    }

    // ============ LIVE MODE ============
    function setupLiveMode() {
        document.getElementById('startCameraBtn').addEventListener('click', startCamera);
    }

    async function startCamera() {
        const videoEl = document.getElementById('cameraVideo');
        const canvasEl = document.getElementById('overlayCanvas');
        const placeholder = document.getElementById('cameraPlaceholder');

        updateStatus('processing');
        showLoading(true);

        const onBatch = serverRecognitionEnabled ? handleFrameBatch : null;
        const started = await Camera.start(videoEl, onBatch);

        showLoading(false);

        if (started) {
            cameraStarted = true;
            placeholder.style.display = 'none';
            videoEl.style.display = 'block';
            canvasEl.style.display = 'block';
            updateStatus('connected');

            MediaPipeOverlay.start(videoEl, canvasEl);
        } else {
            updateStatus('disconnected');
            alert('Camera access was denied. Please allow camera access and try again.');
        }
    }

    async function handleFrameBatch(frames) {
        if (!serverRecognitionEnabled || isProcessingServer) return;
        isProcessingServer = true;
        try {
            const batchId = `live-${Date.now()}`;
            const result = await ApiClient.recogniseFrames(frames, batchId);
            displayServerPrediction(result);
        } catch (err) {
            console.warn('Server recognition error:', err.message);
        } finally {
            isProcessingServer = false;
        }
    }

    // ============ LANDMARK-BASED RECOGNITION (PRIMARY) ============
    function displayLandmarkPrediction(result) {
        const captionText = document.getElementById('captionText');
        const captionConfidence = document.getElementById('captionConfidence');
        const nmfIndicators = document.getElementById('nmfIndicators');

        if (result.confidence >= 0.55) {
            captionText.textContent = result.sign;

            // Source indicator
            let sourceTag = '';
            if (result.source === 'segment') sourceTag = ' [segment]';
            else if (result.source === 'nmf') sourceTag = ' [face]';

            // Duration if segment-based
            let durationTag = '';
            if (result.segmentDuration > 0) {
                durationTag = ` ${result.segmentDuration}ms`;
            }

            let confText = `${(result.confidence * 100).toFixed(0)}%${sourceTag}${durationTag}`;
            if (result.alternatives && result.alternatives.length > 0) {
                const alts = result.alternatives
                    .filter(a => a.confidence >= 0.25)
                    .slice(0, 3)
                    .map(a => `${a.sign} ${(a.confidence * 100).toFixed(0)}%`)
                    .join(' | ');
                if (alts) confText += `  also: ${alts}`;
            }
            captionConfidence.textContent = confText;
            updateStatus('connected');
        } else {
            // Below threshold — show waiting state
            captionText.textContent = 'Watching...';
            captionConfidence.textContent = '';
        }

        // Update NMF indicators
        if (result.nmf && nmfIndicators) {
            updateNmfDisplay(result.nmf);
        }

        // Update sentence display from result
        if (result.sentence && result.sentence.length > 0) {
            renderSentence(result.sentence);
        }
    }

    // ============ NMF DISPLAY ============
    function updateNmfDisplay(nmf) {
        const el = document.getElementById('nmfIndicators');
        if (!el) return;

        const tags = [];
        if (nmf.headNod) tags.push('Nod');
        if (nmf.headShake) tags.push('Shake');
        if (nmf.eyebrowRaise) tags.push('Brow↑');
        if (nmf.mouthOpen) tags.push('Mouth');

        el.innerHTML = tags.map(t => `<span class="nmf-tag">${t}</span>`).join('');
    }

    // ============ SENTENCE DISPLAY ============
    function setupSentenceBar() {
        const clearBtn = document.getElementById('clearSentenceBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                if (typeof SignRecogniser !== 'undefined') {
                    SignRecogniser.clearSentence();
                }
                renderSentence([]);
            });
        }
    }

    function updateSentenceDisplay(data) {
        // data is { sentence: string[], entries: [{sign, confidence, timestamp, source}...], nmf }
        const signs = data.sentence || [];
        const entries = data.entries || [];
        renderSentence(signs, entries);
    }

    function renderSentence(signs, fullData) {
        const sentenceText = document.getElementById('sentenceText');
        const clearBtn = document.getElementById('clearSentenceBtn');
        if (!sentenceText) return;

        if (!signs || signs.length === 0) {
            sentenceText.innerHTML = '';
            if (clearBtn) clearBtn.style.display = 'none';
            return;
        }

        sentenceText.innerHTML = signs.map((s, i) => {
            const source = fullData && fullData[i] ? fullData[i].source : 'landmarks';
            const cls = source === 'nmf' ? 'sign-word nmf-sign' : 'sign-word';
            return `<span class="${cls}">${s}</span>`;
        }).join('');

        if (clearBtn) clearBtn.style.display = 'inline-block';
    }

    // ============ SERVER-SIDE RECOGNITION (SUPPLEMENTARY) ============
    function displayServerPrediction(result) {
        const signs = result.signs || [];
        if (signs.length === 0) return;

        const top = signs[0];
        if (top.confidence < 0.50) return;

        const captionText = document.getElementById('captionText');
        const captionConfidence = document.getElementById('captionConfidence');

        captionText.textContent = top.sign.toUpperCase();
        const alternatives = signs.slice(1, 4)
            .filter(s => s.confidence > 0.10)
            .map(s => `${s.sign} ${(s.confidence * 100).toFixed(0)}%`)
            .join(' | ');

        let confText = `${(top.confidence * 100).toFixed(0)}% (model)`;
        if (alternatives) confText += `  also: ${alternatives}`;
        captionConfidence.textContent = confText;
    }

    // ============ TEXT PANEL ============
    function setupTextPanel() {
        const toggleBtn = document.getElementById('togglePanelBtn');
        const textPanel = document.getElementById('textPanel');

        toggleBtn.addEventListener('click', () => {
            textPanel.classList.toggle('collapsed');
            toggleBtn.textContent = textPanel.classList.contains('collapsed') ? '▶' : '◀';
        });

        document.getElementById('translateBtn').addEventListener('click', handleTranslate);
        document.getElementById('textInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleTranslate();
            }
        });
    }

    async function handleTranslate() {
        const text = document.getElementById('textInput').value.trim();
        if (!text) return;

        const glossOutput = document.getElementById('glossOutput');
        const speakBtn = document.getElementById('speakBtn');
        const config = window.__BSL_CONFIG__ || {};

        // In standalone mode, text-to-BSL requires server-side APIs (Bedrock Claude)
        if (config.standalone) {
            glossOutput.textContent = '';
            const infoSpan = document.createElement('span');
            infoSpan.style.color = 'var(--text-secondary, #666)';
            infoSpan.textContent = 'Text-to-BSL translation requires the full AWS deployment (Bedrock Claude for glossing). This standalone demo supports sign recognition only — try signing at the camera!';
            glossOutput.appendChild(infoSpan);
            return;
        }

        glossOutput.innerHTML = '<div class="spinner" style="width:24px;height:24px;margin:0 auto"></div>';
        showLoading(true);

        try {
            const result = await ApiClient.translateToSign(text);
            showLoading(false);

            if (result.executionArn) {
                glossOutput.textContent = '';
                const em = document.createElement('em');
                em.textContent = 'Translating... (pipeline running)';
                glossOutput.appendChild(em);
                return;
            }

            glossOutput.textContent = '';
            if (result.glossBreakdown) {
                result.glossBreakdown.forEach(item => {
                    const span = document.createElement('span');
                    span.className = item.type === 'fingerspell' ? 'gloss-word fingerspell' : 'gloss-word';
                    span.title = item.source || '';
                    span.textContent = item.word;
                    glossOutput.appendChild(span);
                });
            } else if (result.gloss) {
                result.gloss.split(' ').forEach(w => {
                    const span = document.createElement('span');
                    span.className = 'gloss-word';
                    span.textContent = w;
                    glossOutput.appendChild(span);
                });
            }

            if (result.videoUrl) Player.playVideo(result.videoUrl);

            speakBtn.style.display = 'block';
            speakBtn.onclick = () => speakText(text);

        } catch (err) {
            showLoading(false);
            glossOutput.textContent = '';
            const errSpan = document.createElement('span');
            errSpan.style.color = 'var(--error)';
            errSpan.textContent = 'Error: ' + err.message;
            glossOutput.appendChild(errSpan);
        }
    }

    // ============ SPEECH ============
    function setupSpeech() {
        const micBtn = document.getElementById('micBtn');
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            micBtn.addEventListener('click', startDictation);
        } else {
            micBtn.style.display = 'none';
        }
    }

    function startDictation() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-GB';
        recognition.continuous = false;
        recognition.interimResults = false;

        const micBtn = document.getElementById('micBtn');
        micBtn.textContent = '🔴 Listening...';
        micBtn.disabled = true;

        recognition.onresult = (event) => {
            document.getElementById('textInput').value = event.results[0][0].transcript;
            micBtn.textContent = '🎤 Dictate';
            micBtn.disabled = false;
        };

        recognition.onerror = () => { micBtn.textContent = '🎤 Dictate'; micBtn.disabled = false; };
        recognition.onend = () => { micBtn.textContent = '🎤 Dictate'; micBtn.disabled = false; };
        recognition.start();
    }

    function speakText(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-GB';
            const voices = speechSynthesis.getVoices();
            const britishVoice = voices.find(v => v.lang === 'en-GB');
            if (britishVoice) utterance.voice = britishVoice;
            speechSynthesis.speak(utterance);
        }
    }

    // ============ DOMINANT HAND TOGGLE ============
    function setupDominantHand() {
        const btn = document.getElementById('domHandBtn');
        const label = document.getElementById('domHandLabel');
        if (!btn || !label) return;

        // Default: right-handed (selfie camera already mirrors)
        let isRightHanded = true;

        btn.addEventListener('click', () => {
            isRightHanded = !isRightHanded;
            label.textContent = isRightHanded ? 'Right' : 'Left';
            btn.setAttribute('aria-label', `Dominant hand: ${isRightHanded ? 'Right' : 'Left'}`);
            btn.classList.toggle('active', isRightHanded);
        });
    }

    // ============ AUTH ============
    function setupAuth() {
        const authBtn = document.getElementById('authBtn');
        if (authBtn) {
            authBtn.addEventListener('click', () => alert('Authentication not yet configured'));
        }
    }

    // ============ UTILS ============
    function updateStatus(status) {
        const dot = document.getElementById('statusDot');
        dot.className = 'status-dot';
        if (status === 'connected') dot.classList.add('connected');
        else if (status === 'processing') dot.classList.add('processing');
        dot.setAttribute('aria-label', `Connection status: ${status}`);
    }

    function showLoading(show) {
        document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
    }
})();
