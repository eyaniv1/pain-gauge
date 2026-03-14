/**
 * Main Application
 *
 * Flow: Main Screen (live gauge, start/end session) → optional Calibration
 *
 * - Main: gauge + camera/history toggle, session start/end controls
 * - Calibration: accessed via Calibrate button, captures face baseline
 * - Settings: tunable sensitivity parameters, persisted to localStorage
 */

(function () {
    'use strict';

    // ── Settings persistence ─────────────────────────────────────

    const SETTINGS_KEY = 'painGaugeSettings';
    const DEFAULTS = {
        au4Sensitivity: 12,
        au67Sensitivity: 8,
        au910Sensitivity: 12,
        au43Threshold: 0.55,
        au43Sensitivity: 11,
        talkingSuppression: 0.5,
        smoothingSize: 8,
        sampleIntervalMs: 500,
    };

    function loadSettings() {
        try {
            const saved = localStorage.getItem(SETTINGS_KEY);
            if (saved) return { ...DEFAULTS, ...JSON.parse(saved) };
        } catch (e) { /* ignore */ }
        return { ...DEFAULTS };
    }

    function saveSettings(s) {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    }

    let settings = loadSettings();

    // ── State ─────────────────────────────────────────────────────

    let sessionActive = false;
    let SAMPLE_INTERVAL_MS = settings.sampleIntervalMs;
    let inputMode = 'camera'; // 'camera' or 'video'
    let cameraInstance = null;
    let videoLoopId = null;

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
    const baselinePainInput = document.getElementById('baseline-pain');

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

    // ── DOM refs: Settings ────────────────────────────────────────

    const settingsBtn = document.getElementById('settings-btn');
    const settingsPanel = document.getElementById('settings-panel');
    const resetSettingsBtn = document.getElementById('reset-settings-btn');

    const settingInputs = {
        au4Sensitivity:     { input: 'set-au4-sens',    display: 'val-au4-sens' },
        au67Sensitivity:    { input: 'set-au67-sens',   display: 'val-au67-sens' },
        au910Sensitivity:   { input: 'set-au910-sens',  display: 'val-au910-sens' },
        au43Threshold:      { input: 'set-au43-thresh', display: 'val-au43-thresh' },
        au43Sensitivity:    { input: 'set-au43-sens',   display: 'val-au43-sens' },
        talkingSuppression: { input: 'set-talk-supp',   display: 'val-talk-supp' },
        smoothingSize:      { input: 'set-smooth-win',  display: 'val-smooth-win' },
        sampleIntervalMs:   { input: 'set-sample-int',  display: 'val-sample-int' },
    };

    // ── DOM refs: Video input ────────────────────────────────────

    const loadVideoBtn = document.getElementById('load-video-btn');
    const backToCameraBtn = document.getElementById('back-to-camera-btn');
    const videoFileInput = document.getElementById('video-file-input');

    // ── Initialize engine, gauge & chart ──────────────────────────

    const engine = new PainEngine(settings);
    const gauge = new PainGauge('gauge');
    const chart = new SessionChart('session-chart');

    // ── Settings panel logic ─────────────────────────────────────

    settingsBtn.addEventListener('click', () => {
        settingsPanel.classList.toggle('hidden');
        settingsBtn.classList.toggle('active');
    });

    function applySettingsToUI(s) {
        for (const [key, { input, display }] of Object.entries(settingInputs)) {
            document.getElementById(input).value = s[key];
            document.getElementById(display).textContent = s[key];
        }
    }

    function bindSettingInputs() {
        for (const [key, { input, display }] of Object.entries(settingInputs)) {
            const el = document.getElementById(input);
            const valEl = document.getElementById(display);
            el.addEventListener('input', () => {
                const val = parseFloat(el.value);
                valEl.textContent = val;
                settings[key] = val;

                // Apply to engine
                if (key === 'smoothingSize') {
                    engine.smoothingSize = val;
                } else if (key === 'sampleIntervalMs') {
                    SAMPLE_INTERVAL_MS = val;
                } else {
                    engine[key] = val;
                }

                saveSettings(settings);
            });
        }
    }

    applySettingsToUI(settings);
    bindSettingInputs();

    resetSettingsBtn.addEventListener('click', () => {
        settings = { ...DEFAULTS };
        applySettingsToUI(settings);
        // Apply all to engine
        for (const key of Object.keys(DEFAULTS)) {
            if (key === 'sampleIntervalMs') {
                SAMPLE_INTERVAL_MS = DEFAULTS[key];
            } else {
                engine[key] = DEFAULTS[key];
            }
        }
        saveSettings(settings);
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
        // Set the baseline pain level from hidden input
        engine.baselinePainLevel = parseInt(baselinePainInput.value);
        calibrateBtn.disabled = true;
        calibrateBtn.textContent = 'Hold still...';
        calibInstructions.innerHTML = 'Capturing face at <strong>current pain level</strong>...';
        calibProgress.classList.remove('hidden');
        calibProgressBar.style.width = '0%';
        // Start video playback if in video mode
        if (inputMode === 'video') {
            videoEl.currentTime = 0;
            videoEl.play();
        }
    });

    function onCalibrationComplete() {
        isCalibrating = false;
        calibrateBtn.textContent = 'Calibrated!';
        const level = engine.baselinePainLevel;
        calibInstructions.innerHTML = `Calibrated at pain level <strong>${level}</strong>. Returning...`;
        calibProgress.classList.add('hidden');
        // Pause video after calibration capture
        if (inputMode === 'video') {
            videoEl.pause();
        }

        // Move video back to main screen after short delay
        setTimeout(() => {
            moveVideoToMain();
            showScreen(screenMain);
        }, 800);
    }

    // ── Move video element between screens ────────────────────────

    function moveVideoToMain() {
        mainVideoContainer.appendChild(videoEl);
        mainVideoContainer.appendChild(overlayEl);
        // Ensure no-face warning exists in main
        if (!document.getElementById('no-face-warning-main')) {
            const warning = document.createElement('div');
            warning.id = 'no-face-warning-main';
            warning.className = 'hidden';
            warning.textContent = 'No face detected';
            mainVideoContainer.appendChild(warning);
        }
    }

    function moveVideoToCalib() {
        const calibArea = document.querySelector('.calib-video-area');
        calibArea.insertBefore(videoEl, calibArea.firstChild);
        calibArea.insertBefore(overlayEl, videoEl.nextSibling);
    }

    // ── Calibrate (go to calibration screen) ─────────────────────

    recalibrateBtn.addEventListener('click', () => {
        // Stop any active session
        if (sessionActive) {
            endSession();
        }
        // Pause video so it waits for Calibrate press
        if (inputMode === 'video') {
            videoEl.pause();
            videoEl.currentTime = 0;
        }
        // Move video to calibration screen
        moveVideoToCalib();
        // Reset calibration UI
        calibrateBtn.disabled = false;
        calibrateBtn.textContent = 'Calibrate';
        calibInstructions.innerHTML = 'What is the patient\'s <strong>current pain level</strong>?';
        calibStatus.textContent = inputMode === 'video' ? 'Video paused. Click Calibrate to begin.' : 'Camera ready. Click Calibrate when ready.';
        selectPainLevel(0);
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
        // Restart video from beginning
        if (inputMode === 'video') {
            videoEl.currentTime = 0;
            videoEl.play();
        }

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
        // Pause video on session end
        if (inputMode === 'video') {
            videoEl.pause();
        }

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
        stopVideoLoop();
        try {
            cameraInstance = new Camera(videoEl, {
                onFrame: async () => {
                    await faceMesh.send({ image: videoEl });
                },
                width: 640,
                height: 480,
            });

            await cameraInstance.start();
            inputMode = 'camera';
            loadVideoBtn.classList.remove('hidden');
            backToCameraBtn.classList.add('hidden');
            videoEl.style.transform = '';
            overlayEl.style.transform = '';
        } catch (err) {
            console.error('Camera start failed:', err);
        }
    }

    // ── Video file input ─────────────────────────────────────────

    loadVideoBtn.addEventListener('click', () => {
        videoFileInput.click();
    });

    backToCameraBtn.addEventListener('click', () => {
        startCamera();
    });

    videoFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        loadVideoFile(file);
        // Reset so the same file can be re-selected
        videoFileInput.value = '';
    });

    function loadVideoFile(file) {
        // Stop camera if running
        if (cameraInstance) {
            cameraInstance.stop();
            cameraInstance = null;
        }
        stopVideoLoop();

        const url = URL.createObjectURL(file);
        videoEl.srcObject = null;
        videoEl.src = url;
        videoEl.muted = true;
        videoEl.loop = true;
        // Don't mirror file videos (they're not selfie-cam)
        videoEl.style.transform = 'none';
        overlayEl.style.transform = 'none';

        videoEl.addEventListener('loadeddata', function onLoaded() {
            videoEl.removeEventListener('loadeddata', onLoaded);
            inputMode = 'video';
            loadVideoBtn.classList.add('hidden');
            backToCameraBtn.classList.remove('hidden');
            // Don't autoplay — wait for Start Session or Calibrate
            videoEl.pause();
            videoEl.currentTime = 0;
            startVideoLoop();
        });
    }

    function startVideoLoop() {
        async function loop() {
            if (inputMode !== 'video') return;
            if (!videoEl.paused && !videoEl.ended && videoEl.readyState >= 2) {
                await faceMesh.send({ image: videoEl });
            }
            videoLoopId = requestAnimationFrame(loop);
        }
        loop();
    }

    function stopVideoLoop() {
        if (videoLoopId) {
            cancelAnimationFrame(videoLoopId);
            videoLoopId = null;
        }
    }

    startCamera();
})();
