/**
 * Kiosk mode — tablet-optimised reception desk interface.
 * Auto-resets after 2 minutes of inactivity.
 */
const Kiosk = (() => {
    const TIMEOUT_SECONDS = 120;
    let inactivityTimer = null;
    let countdownTimer = null;
    let countdownValue = TIMEOUT_SECONDS;
    let isActive = false;
    let subMode = null; // 'sign' or 'type'

    function init() {
        document.getElementById('kioskSignBtn').addEventListener('click', () => enterSubMode('sign'));
        document.getElementById('kioskTypeBtn').addEventListener('click', () => enterSubMode('type'));

        // Reset on any interaction
        document.addEventListener('click', resetTimer);
        document.addEventListener('touchstart', resetTimer);
        document.addEventListener('keydown', resetTimer);
    }

    function activate() {
        isActive = true;
        subMode = null;
        showMainButtons();
        resetTimer();
    }

    function deactivate() {
        isActive = false;
        clearTimers();
        document.getElementById('kioskTimeout').style.display = 'none';
    }

    function enterSubMode(mode) {
        subMode = mode;

        if (mode === 'sign') {
            // Switch to live translation mode from kiosk context
            document.querySelector('[data-mode="live"]').click();
            // Auto-start camera
            setTimeout(() => {
                document.getElementById('startCameraBtn')?.click();
            }, 500);
        } else if (mode === 'type') {
            document.querySelector('[data-mode="live"]').click();
            // Focus text input
            setTimeout(() => {
                document.getElementById('textInput')?.focus();
            }, 500);
        }

        resetTimer();
    }

    function showMainButtons() {
        // Reset to main kiosk screen
        const kioskPanel = document.getElementById('kiosk-panel');
        // Buttons are already in the HTML
    }

    function resetTimer() {
        if (!isActive) return;

        clearTimers();
        countdownValue = TIMEOUT_SECONDS;
        document.getElementById('kioskTimeout').style.display = 'none';

        inactivityTimer = setTimeout(() => {
            showCountdown();
        }, (TIMEOUT_SECONDS - 10) * 1000); // Show countdown 10s before reset
    }

    function showCountdown() {
        const timeoutEl = document.getElementById('kioskTimeout');
        const countdownEl = document.getElementById('kioskCountdown');
        timeoutEl.style.display = 'block';
        countdownValue = 10;
        countdownEl.textContent = countdownValue;

        countdownTimer = setInterval(() => {
            countdownValue--;
            countdownEl.textContent = countdownValue;

            if (countdownValue <= 0) {
                resetSession();
            }
        }, 1000);
    }

    function resetSession() {
        clearTimers();
        Camera.stop();
        Player.stop();

        // Return to kiosk main screen
        document.querySelector('[data-mode="kiosk"]').click();

        // Clear text input
        const textInput = document.getElementById('textInput');
        if (textInput) textInput.value = '';

        const glossOutput = document.getElementById('glossOutput');
        if (glossOutput) glossOutput.innerHTML = '';

        document.getElementById('kioskTimeout').style.display = 'none';
        subMode = null;
        resetTimer();
    }

    function clearTimers() {
        if (inactivityTimer) { clearTimeout(inactivityTimer); inactivityTimer = null; }
        if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null; }
    }

    return { init, activate, deactivate };
})();
