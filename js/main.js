/**
 * Main Application
 *
 * Flow: Calibration Screen → Main Screen (Start/End session)
 *
 * - Calibration: full-screen camera, capture neutral face baseline
 * - Main: gauge + camera/history toggle, session start/end controls
 * - Start: begins measuring and recording pain scores
 * - End: stops recording, graph remains visible
 * - Next Start: clears previous graph, begins new session
 */

(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────────

    let sessionActive = false;

    // ── Screens ───────────────────────────────────────────────────

    const screenCalib = document.getElementById('screen-calibration');
    const screenMain = document.getElementById('screen-main');

    function showScreen(screen) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        screen.classList.add('active');
    }

    // ── DOM refs: Calibration ─────────────────────────────────────

    const videoEl = document.getElementById('webcam');
    const overlayEl = document.getElementById('overlay');
    const overlayCtx = overlayEl.getContext('2d');
    const noFaceWarning = document.getElementById('no-face-warning');
    const calibrateBtn = document.getElementById('calibrate-btn');
    const calibInstructions = document.getElementById('calib-instructions');
    const calibProgress = document.getElementById('calib-progress');
    const calibProgressBar = document.getElementById('calib-progress-bar');
    const calibStatus = document.getElementById('calib-status');

    // ── DOM refs: Main ────────────────────────────────────────────

    const recalibrateBtn = document.getElementById('recalibrate-btn');
    const startBtn = document.getElementById('start-btn');
    const endBtn = document.getElementById('end-btn');
    const sessionStatusEl = document.getElementById('session-status');
    const sessionTimerEl = document.getElementById('session-timer');
    const fpsEl = document.getElementById('fps');
    const mainVideoContainer = document.getElementById('main-video-container');

    // AU bar elements
    const auBars = {
        au4: { bar: document.getElementById('au4-bar'), value: document.getElementById('au4-value') },
        au6_7: { bar: document.getElementById('au67-bar'), value: document.getElementById('au67-value') },
        au9_10: { bar: document.getElementById('au910-bar'), value: document.getElementById('au910-value') },
        au43: { bar: document.getElementById('au43-bar'), value: document.getElementById('au43-value') },
    };

    // ── Initialize engine, gauge & chart ──────────────────────────

    const engine = new PainEngine({ smoothingSize: 8 });
    const gauge = new PainGauge('gauge');
    const chart = new SessionChart('session-chart');

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

    // ── Session timer ─────────────────────────────────────────────

    let sessionStartTime = null;
    let timerInterval = null;

    function startTimer() {
        sessionStartTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
            const m = Math.floor(elapsed / 60);
            const s = elapsed % 60;
            sessionTimerEl.textContent = `${m}:${s.toString().padStart(2, '0')}`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
        timerInterval = null;
    }

    // ── Calibration ───────────────────────────────────────────────

    let calibrationFrames = 0;
    let isCalibrating = false;
    const CALIBRATION_FRAME_COUNT = 45; // ~1.5 seconds

    calibrateBtn.addEventListener('click', () => {
        isCalibrating = true;
        calibrationFrames = 0;
        engine.resetCalibration();
        calibrateBtn.disabled = true;
        calibrateBtn.textContent = 'Hold still...';
        calibInstructions.innerHTML = 'Keep your face <strong>still and relaxed</strong>...';
        calibProgress.classList.remove('hidden');
        calibProgressBar.style.width = '0%';
    });

    function onCalibrationComplete() {
        isCalibrating = false;
        calibrateBtn.textContent = 'Calibrated!';
        calibInstructions.innerHTML = 'Calibration complete. Entering session view...';
        calibProgress.classList.add('hidden');

        // Move video + overlay to main screen after short delay
        setTimeout(() => {
            moveVideoToMain();
            showScreen(screenMain);
        }, 800);
    }

    // ── Move video element between screens ────────────────────────

    function moveVideoToMain() {
        mainVideoContainer.appendChild(videoEl);
        mainVideoContainer.appendChild(overlayEl);
        // Re-create the no-face warning in main
        const warning = document.createElement('div');
        warning.id = 'no-face-warning-main';
        warning.className = 'hidden';
        warning.textContent = 'No face detected';
        mainVideoContainer.appendChild(warning);
    }

    function moveVideoToCalib() {
        const calibArea = document.querySelector('.calib-video-area');
        calibArea.insertBefore(videoEl, calibArea.firstChild);
        calibArea.insertBefore(overlayEl, videoEl.nextSibling);
    }

    // ── Recalibrate ───────────────────────────────────────────────

    recalibrateBtn.addEventListener('click', () => {
        // Stop any active session
        if (sessionActive) {
            endSession();
        }
        // Move video back to calibration screen
        moveVideoToCalib();
        // Reset calibration UI
        calibrateBtn.disabled = false;
        calibrateBtn.textContent = 'Calibrate';
        calibInstructions.innerHTML = 'Position your face in the frame and keep a <strong>neutral, relaxed expression</strong>.';
        calibStatus.textContent = 'Ready to calibrate';
        engine.resetCalibration();
        showScreen(screenCalib);
    });

    // ── Session Start / End ───────────────────────────────────────

    startBtn.addEventListener('click', () => {
        startSession();
    });

    endBtn.addEventListener('click', () => {
        endSession();
    });

    function startSession() {
        // Clear previous session data
        chart.resetRecording();
        chart.startRecording();
        engine.smoothingWindow = [];

        sessionActive = true;
        startBtn.classList.add('hidden');
        endBtn.classList.remove('hidden');
        sessionStatusEl.textContent = 'Recording';
        sessionStatusEl.style.color = '#c0392b';
        startTimer();

        // Switch to camera view
        switchView(0);
    }

    function endSession() {
        sessionActive = false;
        chart.stopRecording();
        stopTimer();

        endBtn.classList.add('hidden');
        startBtn.classList.remove('hidden');
        startBtn.textContent = 'New Session';
        sessionStatusEl.textContent = 'Session ended';
        sessionStatusEl.style.color = '#888';

        // Switch to history view to show results
        switchView(1);
    }

    // ── Session chart sampling ────────────────────────────────────

    let lastSampleTime = 0;
    const SAMPLE_INTERVAL_MS = 500;

    function maybeRecordSample(score) {
        if (!sessionActive) return;
        const now = performance.now();
        if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
            chart.addSample(score);
            lastSampleTime = now;
        }
    }

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

        const hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;

        // Update face warning
        noFaceWarning.classList.toggle('hidden', hasFace);
        const mainWarning = document.getElementById('no-face-warning-main');
        if (mainWarning) mainWarning.classList.toggle('hidden', hasFace);

        if (!hasFace) return;

        const landmarks = results.multiFaceLandmarks[0];

        // Draw face mesh on overlay
        drawFaceMesh(landmarks);

        // Calibration mode
        if (isCalibrating) {
            engine.calibrate(landmarks);
            calibrationFrames++;
            const pct = Math.round((calibrationFrames / CALIBRATION_FRAME_COUNT) * 100);
            calibProgressBar.style.width = pct + '%';

            if (calibrationFrames >= CALIBRATION_FRAME_COUNT) {
                onCalibrationComplete();
            }
            return;
        }

        // Score pain (always process for live gauge, but only record in session)
        const result = engine.process(landmarks);
        gauge.setScore(result.score);
        updateAUBars(result.aus);
        maybeRecordSample(result.score);
    }

    // ── Draw face mesh overlay ────────────────────────────────────

    function drawFaceMesh(landmarks) {
        drawConnectors(overlayCtx, landmarks, FACEMESH_TESSELATION, {
            color: 'rgba(100, 180, 255, 0.15)',
            lineWidth: 0.5,
        });

        const painLandmarkIndices = [
            55, 285, 65, 295,
            33, 160, 158, 133, 153, 144,
            263, 387, 385, 362, 380, 373,
            4, 13, 168,
        ];

        overlayCtx.fillStyle = 'rgba(255, 100, 100, 0.6)';
        for (const idx of painLandmarkIndices) {
            const lm = landmarks[idx];
            overlayCtx.beginPath();
            overlayCtx.arc(lm.x * overlayEl.width, lm.y * overlayEl.height, 2.5, 0, 2 * Math.PI);
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
        calibStatus.textContent = 'Starting camera...';

        try {
            const camera = new Camera(videoEl, {
                onFrame: async () => {
                    await faceMesh.send({ image: videoEl });
                },
                width: 640,
                height: 480,
            });

            await camera.start();
            calibStatus.textContent = 'Camera ready. Click Calibrate when ready.';
            calibrateBtn.disabled = false;
        } catch (err) {
            calibStatus.textContent = `Camera error: ${err.message}`;
            console.error('Camera start failed:', err);
        }
    }

    startCamera();
})();
