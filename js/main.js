/**
 * Main Application
 *
 * Wires together: Camera → MediaPipe FaceMesh → PainEngine → PainGauge
 */

(function () {
    'use strict';

    // ── DOM refs ──────────────────────────────────────────────────

    const videoEl = document.getElementById('webcam');
    const overlayEl = document.getElementById('overlay');
    const overlayCtx = overlayEl.getContext('2d');
    const noFaceWarning = document.getElementById('no-face-warning');
    const calibrateBtn = document.getElementById('calibrate-btn');
    const resetBtn = document.getElementById('reset-btn');
    const statusEl = document.getElementById('status');
    const fpsEl = document.getElementById('fps');
    const calibStatusEl = document.getElementById('calibration-status');

    // AU bar elements
    const auBars = {
        au4: { bar: document.getElementById('au4-bar'), value: document.getElementById('au4-value') },
        au6_7: { bar: document.getElementById('au67-bar'), value: document.getElementById('au67-value') },
        au9_10: { bar: document.getElementById('au910-bar'), value: document.getElementById('au910-value') },
        au43: { bar: document.getElementById('au43-bar'), value: document.getElementById('au43-value') },
    };

    // ── Initialize engine & gauge ─────────────────────────────────

    const engine = new PainEngine({ smoothingSize: 8 });
    const gauge = new PainGauge('gauge');

    // ── FPS tracking ──────────────────────────────────────────────

    let frameCount = 0;
    let lastFpsTime = performance.now();

    function updateFps() {
        const now = performance.now();
        const elapsed = now - lastFpsTime;
        if (elapsed >= 1000) {
            fpsEl.textContent = `FPS: ${Math.round((frameCount * 1000) / elapsed)}`;
            frameCount = 0;
            lastFpsTime = now;
        }
        frameCount++;
    }

    // ── Calibration ───────────────────────────────────────────────

    let calibrationFrames = 0;
    let isCalibrating = false;
    const CALIBRATION_FRAME_COUNT = 30; // ~1 second of frames

    calibrateBtn.addEventListener('click', () => {
        isCalibrating = true;
        calibrationFrames = 0;
        engine.resetCalibration();
        calibrateBtn.disabled = true;
        calibrateBtn.textContent = 'Hold still...';
        statusEl.textContent = 'Calibrating — keep a neutral face...';
    });

    resetBtn.addEventListener('click', () => {
        engine.resetCalibration();
        calibStatusEl.textContent = 'Uncalibrated (using defaults)';
        calibStatusEl.classList.remove('calibrated');
        gauge.setScore(0);
    });

    // ── MediaPipe FaceMesh setup ──────────────────────────────────

    const faceMesh = new FaceMesh({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        },
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });

    faceMesh.onResults(onResults);

    // ── Process each frame ────────────────────────────────────────

    function onResults(results) {
        updateFps();

        // Resize overlay to match video
        overlayEl.width = results.image.width;
        overlayEl.height = results.image.height;
        overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            noFaceWarning.classList.remove('hidden');
            statusEl.textContent = 'No face detected';
            return;
        }

        noFaceWarning.classList.add('hidden');
        const landmarks = results.multiFaceLandmarks[0];

        // Draw subtle face mesh on overlay
        drawFaceMesh(landmarks);

        // Calibration mode
        if (isCalibrating) {
            engine.calibrate(landmarks);
            calibrationFrames++;

            if (calibrationFrames >= CALIBRATION_FRAME_COUNT) {
                isCalibrating = false;
                calibrateBtn.disabled = false;
                calibrateBtn.textContent = 'Calibrate Neutral Face';
                calibStatusEl.textContent = `Calibrated (${engine.baseline.samples} samples)`;
                calibStatusEl.classList.add('calibrated');
                statusEl.textContent = 'Tracking face';
            }
            return;
        }

        // Score pain
        const result = engine.process(landmarks);

        // Update gauge
        gauge.setScore(result.score);

        // Update AU bars
        updateAUBars(result.aus);

        statusEl.textContent = 'Tracking face';
    }

    // ── Draw face mesh overlay ────────────────────────────────────

    function drawFaceMesh(landmarks) {
        // Draw the tesselation (subtle wireframe)
        drawConnectors(overlayCtx, landmarks, FACEMESH_TESSELATION, {
            color: 'rgba(100, 180, 255, 0.15)',
            lineWidth: 0.5,
        });

        // Highlight pain-relevant landmarks
        const painLandmarkIndices = [
            // Brows
            55, 285, 65, 295,
            // Eyes
            33, 160, 158, 133, 153, 144,
            263, 387, 385, 362, 380, 373,
            // Nose / lip
            4, 13, 168,
        ];

        overlayCtx.fillStyle = 'rgba(255, 100, 100, 0.6)';
        for (const idx of painLandmarkIndices) {
            const lm = landmarks[idx];
            overlayCtx.beginPath();
            overlayCtx.arc(
                lm.x * overlayEl.width,
                lm.y * overlayEl.height,
                2.5,
                0,
                2 * Math.PI
            );
            overlayCtx.fill();
        }
    }

    // ── Update AU breakdown bars ──────────────────────────────────

    function updateAUBars(aus) {
        for (const [key, elements] of Object.entries(auBars)) {
            const value = aus[key] || 0;
            const pct = (value / 5) * 100;
            elements.bar.style.width = `${pct}%`;
            elements.bar.style.backgroundColor = PainGauge.painColor(value * 2);
            elements.value.textContent = value.toFixed(1);
        }
    }

    // ── Start camera ──────────────────────────────────────────────

    async function startCamera() {
        statusEl.textContent = 'Starting camera...';

        try {
            const camera = new Camera(videoEl, {
                onFrame: async () => {
                    await faceMesh.send({ image: videoEl });
                },
                width: 640,
                height: 480,
            });

            await camera.start();
            statusEl.textContent = 'Loading face mesh model...';
            calibrateBtn.disabled = false;
        } catch (err) {
            statusEl.textContent = `Camera error: ${err.message}`;
            console.error('Camera start failed:', err);
        }
    }

    startCamera();
})();
