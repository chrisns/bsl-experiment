/**
 * API client for Lambda Function URL endpoints.
 */
const ApiClient = (() => {
    function getConfig() {
        return window.__BSL_CONFIG__ || {};
    }

    async function recogniseFrames(frames, batchId) {
        const config = getConfig();
        const url = config.recogniseUrl;
        if (!url) throw new Error('Recognise URL not configured');

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames, batchId }),
        });

        if (!response.ok) {
            throw new Error(`Recognition failed: ${response.status}`);
        }

        return response.json();
    }

    async function translateToSign(text, type = 'text') {
        const config = getConfig();
        const url = config.textToSignUrl;
        if (!url) throw new Error('Text-to-sign URL not configured');

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, type }),
        });

        if (!response.ok) {
            throw new Error(`Translation failed: ${response.status}`);
        }

        return response.json();
    }

    async function pollExecution(executionArn) {
        // For Step Functions async execution polling
        // In practice, the Lambda Function URL would handle this
        return { status: 'RUNNING' };
    }

    return { recogniseFrames, translateToSign, pollExecution, getConfig };
})();
