/**
 * Main Application
 *
 * Wires together: Camera → MediaPipe FaceMesh → PainEngine → PainGauge
 * Plus: Session chart, view switching (tabs + swipe)
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

    // Swipe / tabs
    const swipeTrack = document.querySelector('.swipe-track');
    const viewTabs = document.querySelectorAll('.view-tab');

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

    // ── View switching (tabs + swipe) ─────────────────────────────

    let currentView = 0; // 0 = camera, 1 = history

    function switchView(index) {
        currentView = index;
        swipeTrack.style.transform = `translateX(-${index * 100}%)`;
        viewTabs.forEach((tab, i) => {
            tab.classList.toggle('active', i === index);
        });
    }

    // Tab clicks
    viewTabs.forEach((tab, i) => {
        tab.addEventListener('click', () => switchView(i));
    });

    // Touch swipe
    let touchStartX = 0;
    let touchDeltaX = 0;
    let isSwiping = false;
    const swipeContainer = document.querySelector('.swipe-container');

    swipeContainer.addEventListener('touchstart', (e) => {
        touchStartX = e.touches[0].clientX;
        touchDeltaX = 0;
        isSwiping = true;
        swipeTrack.classList.add('swiping');
    }, { passive: true });

    swipeContainer.addEventListener('touchmove', (e) => {
        if (!isSwiping) return;
        touchDeltaX = e.touches[0].clientX - touchStartX;
        const baseOffset = -currentView * 100;
        const dragPercent = (touchDeltaX / swipeContainer.offsetWidth) * 100;
        swipeTrack.style.transform = `translateX(${baseOffset + dragPercent}%)`;
    }, { passive: true });

    swipeContainer.addEventListener('touchend', () => {
        if (!isSwiping) return;
        isSwiping = false;
        swipeTrack.classList.remove('swiping');

        const threshold = swipeContainer.offsetWidth * 0.25;
        if (touchDeltaX < -threshold && currentView < 1) {
            switchView(1);
        } else if (touchDeltaX > threshold && currentView > 0) {
            switchView(0);
        } else {
            switchView(currentView); // snap back
        }
    });

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
        chart.resetRecording();
        calibStatusEl.textContent = 'Uncalibrated (using defaults)';
        calibStatusEl.classList.remove('calibrated');
        gauge.setScore(0);
    });

    // ── Session chart sampling ────────────────────────────────────
    // Sample at ~2 Hz to keep the chart readable (not every video frame)

    let lastSampleTime = 0;
    const SAMPLE_INTERVAL_MS = 500;

    function maybeRecordSample(score) {
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

                // Start session recording after calibration
                chart.startRecording();
            }
            return;
        }

        // Score pain
        const result = engine.process(landmarks);

        // Update gauge
        gauge.setScore(result.score);

        // Update AU bars
        updateAUBars(result.aus);

        // Record to session chart
        maybeRecordSample(result.score);

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
