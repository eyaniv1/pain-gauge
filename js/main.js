/**
 * Main Application
 *
 * Flow: Main Screen (live gauge, start/end session) → optional Calibration
 *
 * - Main: gauge + camera/history toggle, session start/end controls
 * - Calibration: accessed via Calibrate button, captures face baseline
 * - Settings: tunable sensitivity parameters, persisted to localStorage
 * - Backend: patient management, session persistence, frame capture
 */

(function () {
    'use strict';

    // ── Settings persistence ─────────────────────────────────────

    const SETTINGS_KEY = 'painGaugeSettings';
    const DEFAULTS = {
        au4Sensitivity: 5,
        au67Sensitivity: 4,
        au910Sensitivity: 5,
        au43Threshold: 0.55,
        au43Sensitivity: 8,
        talkingSuppression: 0.5,
        smoothingSize: 8,
        sampleIntervalMs: 500,
        backendUrl: '',
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
    let autoCalibFrames = 0;
    const AUTO_CALIB_COUNT = 30;
    let cameraInstance = null;
    let videoLoopId = null;

    // Backend state
    let currentPatientId = localStorage.getItem('painGaugePatientId') || '';
    let currentSessionId = null;
    let sampleBuffer = []; // accumulated samples to send in batches
    let sampleFlushInterval = null;

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
    const backendUrlInput = document.getElementById('set-backend-url');
    const connectionStatusEl = document.getElementById('connection-status');

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

    const cameraBtn = document.getElementById('camera-btn');
    const loadVideoBtn = document.getElementById('load-video-btn');
    const videoFileInput = document.getElementById('video-file-input');

    // ── DOM refs: Patient ────────────────────────────────────────

    const patientSelect = document.getElementById('patient-select');
    const addPatientBtn = document.getElementById('add-patient-btn');
    const patientModal = document.getElementById('patient-modal');
    const patientModalTitle = document.getElementById('patient-modal-title');
    const patientNameInput = document.getElementById('patient-name');
    const patientDobInput = document.getElementById('patient-dob');
    const patientExtIdInput = document.getElementById('patient-ext-id');
    const patientNotesInput = document.getElementById('patient-notes');
    const patientSaveBtn = document.getElementById('patient-save-btn');
    const patientCancelBtn = document.getElementById('patient-cancel-btn');

    // ── Initialize engine, gauge & chart ──────────────────────────

    const engine = new PainEngine(settings);
    const gauge = new PainGauge('gauge');
    const chart = new SessionChart('session-chart');

    // ── Chart zoom controls ──────────────────────────────────────
    document.getElementById('chart-zoom-in').addEventListener('click', () => chart.zoomIn());
    document.getElementById('chart-zoom-out').addEventListener('click', () => chart.zoomOut());

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
        backendUrlInput.value = s.backendUrl || '';
    }

    function bindSettingInputs() {
        for (const [key, { input, display }] of Object.entries(settingInputs)) {
            const el = document.getElementById(input);
            const valEl = document.getElementById(display);
            el.addEventListener('input', () => {
                const val = parseFloat(el.value);
                valEl.textContent = val;
                settings[key] = val;

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

    // Backend URL input
    let backendUrlDebounce = null;
    backendUrlInput.addEventListener('input', () => {
        clearTimeout(backendUrlDebounce);
        backendUrlDebounce = setTimeout(() => {
            settings.backendUrl = backendUrlInput.value.trim();
            saveSettings(settings);
            initBackend();
        }, 800);
    });

    resetSettingsBtn.addEventListener('click', () => {
        const keepBackendUrl = settings.backendUrl;
        settings = { ...DEFAULTS, backendUrl: keepBackendUrl };
        applySettingsToUI(settings);
        for (const key of Object.keys(DEFAULTS)) {
            if (key === 'sampleIntervalMs') {
                SAMPLE_INTERVAL_MS = DEFAULTS[key];
            } else if (key === 'backendUrl') {
                // keep current backend URL
            } else {
                engine[key] = DEFAULTS[key];
            }
        }
        saveSettings(settings);
    });

    // ── Backend initialization ───────────────────────────────────

    function updateConnectionStatus() {
        if (PainGaugeAPI.isConnected()) {
            connectionStatusEl.className = 'connection-status connected';
            connectionStatusEl.title = 'Connected';
        } else {
            connectionStatusEl.className = 'connection-status disconnected';
            connectionStatusEl.title = 'Not connected';
        }
    }

    async function initBackend() {
        const url = settings.backendUrl;
        PainGaugeAPI.setBaseUrl(url);
        if (!url) {
            updateConnectionStatus();
            return;
        }
        await PainGaugeAPI.checkConnection();
        updateConnectionStatus();
        if (PainGaugeAPI.isConnected()) {
            await loadPatients();
            loadHistorySessions();
        }
    }

    // ── Patient management ──────────────────────────────────────

    async function loadPatients() {
        const patients = await PainGaugeAPI.listPatients();
        patientSelect.innerHTML = '<option value="">-- No Patient --</option>';
        for (const p of patients) {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name + (p.patient_id ? ` (${p.patient_id})` : '');
            patientSelect.appendChild(opt);
        }
        // Restore last selected patient
        if (currentPatientId) {
            patientSelect.value = currentPatientId;
        }
    }

    patientSelect.addEventListener('change', () => {
        currentPatientId = patientSelect.value;
        localStorage.setItem('painGaugePatientId', currentPatientId);
        // Clear chart and reload history for new patient
        chart.resetRecording();
        loadHistorySessions();
    });

    addPatientBtn.addEventListener('click', () => {
        if (!PainGaugeAPI.isConnected()) {
            alert('Backend not connected. Set the Backend URL in Settings first.');
            return;
        }
        patientModalTitle.textContent = 'New Patient';
        patientNameInput.value = '';
        patientDobInput.value = '';
        patientExtIdInput.value = '';
        patientNotesInput.value = '';
        patientModal.classList.remove('hidden');
        patientNameInput.focus();
    });

    patientCancelBtn.addEventListener('click', () => {
        patientModal.classList.add('hidden');
    });

    patientSaveBtn.addEventListener('click', async () => {
        const name = patientNameInput.value.trim();
        if (!name) {
            patientNameInput.focus();
            return;
        }
        const patient = await PainGaugeAPI.createPatient({
            name,
            dob: patientDobInput.value || null,
            patient_id: patientExtIdInput.value.trim() || null,
            notes: patientNotesInput.value.trim() || null,
        });
        if (patient) {
            await loadPatients();
            patientSelect.value = patient.id;
            currentPatientId = patient.id;
            localStorage.setItem('painGaugePatientId', currentPatientId);
        }
        patientModal.classList.add('hidden');
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
    const CALIBRATION_FRAME_COUNT = 45;

    calibrateBtn.addEventListener('click', () => {
        isCalibrating = true;
        calibrationFrames = 0;
        engine.resetCalibration();
        autoCalibFrames = AUTO_CALIB_COUNT;
        engine.baselinePainLevel = parseInt(baselinePainInput.value);
        calibrateBtn.disabled = true;
        calibrateBtn.textContent = 'Hold still...';
        calibInstructions.innerHTML = 'Capturing face at <strong>current pain level</strong>...';
        calibProgress.classList.remove('hidden');
        calibProgressBar.style.width = '0%';
        if (inputMode === 'video') {
            videoEl.currentTime = 0;
            videoEl.play();
        }
    });

    async function onCalibrationComplete() {
        isCalibrating = false;
        calibrateBtn.textContent = 'Calibrated!';
        const level = engine.baselinePainLevel;
        calibInstructions.innerHTML = `Calibrated at pain level <strong>${level}</strong>. Returning...`;
        calibProgress.classList.add('hidden');
        if (inputMode === 'video') {
            videoEl.pause();
        }

        // Send calibration frame to AI inference service
        if (PainGaugeAPI.isConnected()) {
            const frame = captureFrame();
            const aiResult = await PainGaugeAPI.calibrateAI(frame, level);
            if (aiResult && aiResult.ok) {
                console.log(`AI calibrated at pain level ${level}, face_detected=${aiResult.face_detected}`);
            }
        }

        setTimeout(() => {
            moveVideoToMain();
            showScreen(screenMain);
        }, 800);
    }

    // ── Move video element between screens ────────────────────────

    function moveVideoToMain() {
        mainVideoContainer.appendChild(videoEl);
        mainVideoContainer.appendChild(overlayEl);
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

    recalibrateBtn.addEventListener('click', async () => {
        if (sessionActive) {
            endSession();
        }
        if (inputMode === 'video') {
            videoEl.pause();
            videoEl.currentTime = 0;
        }
        moveVideoToCalib();
        calibrateBtn.disabled = false;
        calibrateBtn.textContent = 'Calibrate';
        calibInstructions.innerHTML = 'What is the patient\'s <strong>current pain level</strong>?';
        calibStatus.textContent = inputMode === 'video' ? 'Video paused. Click Calibrate to begin.' : 'Camera ready. Click Calibrate when ready.';
        selectPainLevel(0);
        engine.resetCalibration();

        // Clear AI calibration too
        if (PainGaugeAPI.isConnected()) {
            await PainGaugeAPI.clearAICalibration();
        }

        showScreen(screenCalib);
    });

    // ── Frame capture ────────────────────────────────────────────

    const frameCanvas = document.createElement('canvas');
    const frameCtx = frameCanvas.getContext('2d');

    function captureFrame() {
        const w = 160;
        const h = 120;
        frameCanvas.width = w;
        frameCanvas.height = h;
        frameCtx.drawImage(videoEl, 0, 0, w, h);
        return frameCanvas.toDataURL('image/jpeg', 0.6);
    }

    // ── Session Start / End ───────────────────────────────────────

    startBtn.addEventListener('click', () => {
        startSession();
    });

    endBtn.addEventListener('click', () => {
        endSession();
    });

    async function startSession() {
        chart.resetRecording();
        chart.startRecording();
        engine.smoothingWindow = [];
        sampleBuffer = [];

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
        switchView(0);

        // Create session on backend
        currentSessionId = null;
        if (PainGaugeAPI.isConnected() && currentPatientId) {
            const session = await PainGaugeAPI.createSession(currentPatientId, {
                baseline_pain_level: engine.baselinePainLevel,
                calibration_data: engine.baseline,
                settings_snapshot: settings,
            });
            if (session) {
                currentSessionId = session.id;
            }
        }

        // Flush samples to backend every 5 seconds
        sampleFlushInterval = setInterval(flushSamples, 5000);
    }

    async function endSession() {
        sessionActive = false;
        chart.stopRecording();
        stopTimer();
        if (inputMode === 'video') {
            videoEl.pause();
        }

        endBtn.classList.add('hidden');
        startBtn.classList.remove('hidden');
        startBtn.textContent = 'New Session';
        sessionStatusEl.textContent = 'Session ended';
        sessionStatusEl.style.color = '#888';

        // Flush remaining samples and end session on backend
        clearInterval(sampleFlushInterval);
        await flushSamples();
        if (currentSessionId) {
            await PainGaugeAPI.endSession(currentSessionId, {});
            currentSessionId = null;
        }

        switchView(1);
        loadHistorySessions();
    }

    async function flushSamples() {
        if (!currentSessionId || sampleBuffer.length === 0) return;
        const batch = sampleBuffer.splice(0);
        const result = await PainGaugeAPI.sendSamples(currentSessionId, batch);

        // Feed AI scores back to the chart for real-time plotting
        if (result && result.ai_results) {
            for (const ai of result.ai_results) {
                chart.addAiSample(ai.timestamp_ms / 1000, ai.ai_score);
            }
        }
    }

    // ── Session chart sampling ────────────────────────────────────

    let lastSampleTime = 0;
    let lastResult = null;

    function maybeRecordSample(result) {
        if (!sessionActive) return;
        const now = performance.now();
        if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
            chart.addSample(result.score);
            lastSampleTime = now;

            // Buffer sample for backend
            if (currentSessionId) {
                const elapsed = chart.getElapsedTime() * 1000;
                const sample = {
                    timestamp_ms: Math.round(elapsed),
                    score: result.score,
                    raw_score: result.rawScore,
                    pspi: result.pspi,
                    sensor_data: {
                        face: result.aus,
                    },
                    frame: captureFrame(),
                };
                sampleBuffer.push(sample);
            }
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

        overlayEl.width = results.image.width;
        overlayEl.height = results.image.height;
        overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);

        const hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;

        noFaceWarning.classList.toggle('hidden', hasFace);
        const mainWarning = document.getElementById('no-face-warning-main');
        if (mainWarning) mainWarning.classList.toggle('hidden', hasFace);

        if (!hasFace) return;

        const landmarks = results.multiFaceLandmarks[0];
        drawFaceMesh(landmarks);

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

        if (!engine.isCalibrated() && autoCalibFrames < AUTO_CALIB_COUNT) {
            engine.calibrate(landmarks);
            autoCalibFrames++;
            return;
        }

        const result = engine.process(landmarks);
        gauge.setScore(result.score);
        updateAUBars(result.aus);
        lastResult = result;
        maybeRecordSample(result);
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

    let useFrontCamera = true;
    const flipCameraBtn = document.getElementById('flip-camera-btn');

    async function startCamera() {
        stopVideoLoop();

        // Stop existing camera if any
        if (cameraInstance) {
            try { cameraInstance.stop(); } catch (e) { /* ignore */ }
            cameraInstance = null;
        }

        try {
            cameraInstance = new Camera(videoEl, {
                onFrame: async () => {
                    await faceMesh.send({ image: videoEl });
                },
                facingMode: useFrontCamera ? 'user' : 'environment',
                width: 640,
                height: 480,
            });

            await cameraInstance.start();
            inputMode = 'camera';
            cameraBtn.classList.add('active');
            loadVideoBtn.classList.remove('active');
            // Mirror front camera, don't mirror rear
            videoEl.style.transform = useFrontCamera ? '' : 'scaleX(1)';
            overlayEl.style.transform = useFrontCamera ? '' : 'scaleX(1)';
        } catch (err) {
            console.error('Camera start failed:', err);
            // If rear camera failed, fall back to front
            if (!useFrontCamera) {
                useFrontCamera = true;
                startCamera();
            }
        }
    }

    // ── Video file input ─────────────────────────────────────────

    cameraBtn.addEventListener('click', () => {
        if (inputMode === 'camera') return;
        startCamera();
    });

    flipCameraBtn.addEventListener('click', () => {
        useFrontCamera = !useFrontCamera;
        if (inputMode === 'camera') {
            startCamera();
        }
    });

    loadVideoBtn.addEventListener('click', () => {
        videoFileInput.click();
    });

    videoFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        loadVideoFile(file);
        videoFileInput.value = '';
    });

    function loadVideoFile(file) {
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
        videoEl.style.transform = 'none';
        overlayEl.style.transform = 'none';

        videoEl.addEventListener('loadeddata', function onLoaded() {
            videoEl.removeEventListener('loadeddata', onLoaded);
            inputMode = 'video';
            loadVideoBtn.classList.add('active');
            cameraBtn.classList.remove('active');
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

    // ── History Panel ──────────────────────────────────────────

    const historySessionsView = document.getElementById('history-sessions-view');
    const historySamplesView = document.getElementById('history-samples-view');
    const historySessionsTbody = document.getElementById('history-sessions-tbody');
    const historySamplesTbody = document.getElementById('history-samples-tbody');
    const historyRefreshBtn = document.getElementById('history-refresh-btn');
    const historyDeleteBtn = document.getElementById('history-delete-btn');
    const historySelectAll = document.getElementById('history-select-all');
    const historyBackBtn = document.getElementById('history-back-btn');
    const historySessionTitle = document.getElementById('history-session-title');
    const historyViewChartBtn = document.getElementById('history-view-chart-btn');
    const historyEmpty = document.getElementById('history-empty');

    // Frame viewer modal
    const frameModal = document.getElementById('frame-modal');
    const frameModalImg = document.getElementById('frame-modal-img');
    const frameModalTitle = document.getElementById('frame-modal-title');
    const frameModalClose = document.getElementById('frame-modal-close');

    let historySessions = [];
    let historySelectedIds = new Set();
    let currentHistorySamples = [];

    historyRefreshBtn.addEventListener('click', () => loadHistorySessions());

    historySelectAll.addEventListener('change', () => {
        const checked = historySelectAll.checked;
        historySelectedIds.clear();
        if (checked) {
            historySessions.forEach(s => historySelectedIds.add(s.id));
        }
        historySessionsTbody.querySelectorAll('.session-check').forEach(cb => {
            cb.checked = checked;
        });
        updateDeleteBtn();
    });

    historyDeleteBtn.addEventListener('click', async () => {
        if (historySelectedIds.size === 0) return;
        const count = historySelectedIds.size;
        if (!confirm(`Delete ${count} session${count > 1 ? 's' : ''}? This cannot be undone.`)) return;
        const ids = Array.from(historySelectedIds);
        await PainGaugeAPI.deleteSessions(ids);
        historySelectedIds.clear();
        historySelectAll.checked = false;
        await loadHistorySessions();
    });

    historyBackBtn.addEventListener('click', () => {
        historySamplesView.classList.add('hidden');
        historySessionsView.classList.remove('hidden');
    });

    historyViewChartBtn.addEventListener('click', () => {
        if (currentHistorySamples.length > 0) {
            chart.loadSamples(currentHistorySamples, historySessionTitle.textContent);
            switchView(1);
        }
    });

    frameModalClose.addEventListener('click', () => {
        frameModal.classList.add('hidden');
    });
    frameModal.addEventListener('click', (e) => {
        if (e.target === frameModal) frameModal.classList.add('hidden');
    });

    function updateDeleteBtn() {
        historyDeleteBtn.disabled = historySelectedIds.size === 0;
    }

    async function loadHistorySessions() {
        historySessionsTbody.innerHTML = '';
        historySelectedIds.clear();
        historySelectAll.checked = false;
        updateDeleteBtn();

        if (!PainGaugeAPI.isConnected() || !currentPatientId) {
            historyEmpty.classList.remove('hidden');
            historyEmpty.textContent = !currentPatientId ? 'Select a patient to view history.' : 'Backend not connected.';
            return;
        }

        historySessions = await PainGaugeAPI.listSessions(currentPatientId);
        if (historySessions.length === 0) {
            historyEmpty.classList.remove('hidden');
            historyEmpty.textContent = 'No sessions found for this patient.';
            return;
        }
        historyEmpty.classList.add('hidden');

        for (const s of historySessions) {
            const tr = document.createElement('tr');

            const startDate = new Date(s.start_time);
            const endDate = s.end_time ? new Date(s.end_time) : null;
            const duration = endDate ? formatDuration(endDate - startDate) : 'In progress';
            const shortId = s.id.substring(0, 8);

            tr.innerHTML = `
                <td class="col-check"><input type="checkbox" class="session-check" data-id="${s.id}"></td>
                <td><span class="session-link" data-id="${s.id}">${shortId}</span></td>
                <td>${formatDateTime(startDate)}</td>
                <td>${duration}</td>
                <td class="sample-count" data-id="${s.id}">...</td>
                <td><span class="session-link" data-id="${s.id}" style="font-size:11px">View</span></td>
            `;

            // Checkbox handler
            tr.querySelector('.session-check').addEventListener('change', (e) => {
                if (e.target.checked) {
                    historySelectedIds.add(s.id);
                } else {
                    historySelectedIds.delete(s.id);
                }
                historySelectAll.checked = historySelectedIds.size === historySessions.length;
                updateDeleteBtn();
            });

            // Session link handlers
            tr.querySelectorAll('.session-link').forEach(link => {
                link.addEventListener('click', () => loadHistorySamples(s));
            });

            historySessionsTbody.appendChild(tr);

            // Load sample count async
            PainGaugeAPI.getSamples(s.id).then(samples => {
                const cell = historySessionsTbody.querySelector(`.sample-count[data-id="${s.id}"]`);
                if (cell) cell.textContent = samples.length;
            });
        }
    }

    async function loadHistorySamples(session) {
        historySessionsView.classList.add('hidden');
        historySamplesView.classList.remove('hidden');

        const startDate = new Date(session.start_time);
        historySessionTitle.textContent = `Session ${session.id.substring(0, 8)} — ${formatDateTime(startDate)}`;

        historySamplesTbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:#999">Loading...</td></tr>';

        const samples = await PainGaugeAPI.getSamples(session.id);
        currentHistorySamples = samples;
        historySamplesTbody.innerHTML = '';

        if (samples.length === 0) {
            historySamplesTbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:#999">No samples</td></tr>';
            return;
        }

        // Also load into chart
        chart.loadSamples(samples, `Session ${session.id.substring(0, 8)}`);

        for (const s of samples) {
            const tr = document.createElement('tr');
            const face = s.sensor_data && s.sensor_data.face ? s.sensor_data.face : {};
            const timeStr = formatMs(s.timestamp_ms);

            // AI score badge with confidence coloring
            let aiScoreHtml = '-';
            if (s.ai_score != null) {
                const conf = s.ai_confidence || 0;
                const confClass = conf > 0.7 ? 'high-confidence' : conf > 0.5 ? 'med-confidence' : 'low-confidence';
                aiScoreHtml = `<span class="ai-score-badge ${confClass}">${s.ai_score.toFixed(1)}</span>`;
            }

            // Correction cell with edit button
            let correctionHtml;
            if (s.corrected_score != null) {
                correctionHtml = `<span class="corrected-value">${s.corrected_score.toFixed(1)}</span>`;
            } else {
                correctionHtml = `<button class="correct-btn" data-sample-id="${s.id}" title="Correct this score">Edit</button>`;
            }

            if (s.corrected_score != null) tr.classList.add('sample-corrected');

            tr.innerHTML = `
                <td>${timeStr}</td>
                <td>${s.score != null ? s.score.toFixed(1) : '-'}</td>
                <td class="col-hide-mobile">${s.pspi != null ? s.pspi.toFixed(2) : '-'}</td>
                <td>${aiScoreHtml}</td>
                <td class="correction-cell col-hide-mobile">${correctionHtml}</td>
                <td class="col-hide-mobile">${face.au4 != null ? face.au4.toFixed(1) : '-'}</td>
                <td class="col-hide-mobile">${face.au6_7 != null ? face.au6_7.toFixed(1) : '-'}</td>
                <td class="col-hide-mobile">${face.au9_10 != null ? face.au9_10.toFixed(1) : '-'}</td>
                <td class="col-hide-mobile">${face.au43 != null ? face.au43.toFixed(1) : '-'}</td>
                <td class="col-hide-mobile">${s.frame_filename ? '<span class="frame-link" data-frame="' + s.frame_filename + '" data-time="' + timeStr + '">View</span>' : '-'}</td>
            `;

            // Frame click handler
            const frameLink = tr.querySelector('.frame-link');
            if (frameLink) {
                frameLink.addEventListener('click', () => {
                    frameModalImg.src = PainGaugeAPI.frameUrl(frameLink.dataset.frame);
                    frameModalTitle.textContent = `Frame at ${frameLink.dataset.time}`;
                    frameModal.classList.remove('hidden');
                });
            }

            // Correction button handler
            const correctBtn = tr.querySelector('.correct-btn');
            if (correctBtn) {
                correctBtn.addEventListener('click', async () => {
                    const sampleId = correctBtn.dataset.sampleId;
                    const input = document.createElement('input');
                    input.type = 'number';
                    input.min = '0';
                    input.max = '10';
                    input.step = '0.1';
                    input.value = s.ai_score != null ? s.ai_score.toFixed(1) : s.score.toFixed(1);
                    input.className = 'correction-input';

                    const cell = correctBtn.parentElement;
                    cell.innerHTML = '';
                    cell.appendChild(input);
                    input.focus();
                    input.select();

                    async function submitCorrection() {
                        const val = parseFloat(input.value);
                        if (isNaN(val) || val < 0 || val > 10) {
                            cell.innerHTML = correctionHtml;
                            return;
                        }
                        const result = await PainGaugeAPI.correctSample(sampleId, val);
                        if (result) {
                            cell.innerHTML = `<span class="corrected-value">${val.toFixed(1)}</span>`;
                            tr.classList.add('sample-corrected');
                        } else {
                            cell.innerHTML = correctionHtml;
                        }
                    }

                    input.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter') submitCorrection();
                        if (e.key === 'Escape') cell.innerHTML = correctionHtml;
                    });
                    input.addEventListener('blur', submitCorrection);
                });
            }

            historySamplesTbody.appendChild(tr);
        }
    }

    // ── History helpers ────────────────────────────────────────

    function formatDateTime(date) {
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function formatDuration(ms) {
        const totalSec = Math.floor(ms / 1000);
        const m = Math.floor(totalSec / 60);
        const s = totalSec % 60;
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    function formatMs(ms) {
        const totalSec = Math.round(ms / 1000);
        const m = Math.floor(totalSec / 60);
        const s = totalSec % 60;
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    // ── AI Status Polling ─────────────────────────────────────────

    const aiStatusDot = document.getElementById('ai-status-dot');
    let aiStatusInterval = null;

    async function pollAIStatus() {
        if (!PainGaugeAPI.isConnected()) {
            if (aiStatusDot) {
                aiStatusDot.className = 'ai-status-dot offline';
                aiStatusDot.title = 'AI: Backend not connected';
            }
            return;
        }
        const status = await PainGaugeAPI.getInferenceStatus();
        if (aiStatusDot) {
            if (status && status.online) {
                const detail = status.detail || {};
                const ver = detail.model_version || '';
                const cal = detail.calibrated ? ' [calibrated]' : '';
                aiStatusDot.className = 'ai-status-dot online';
                aiStatusDot.title = `AI: Online${ver ? ` (${ver})` : ''}${cal}`;
            } else {
                aiStatusDot.className = 'ai-status-dot offline';
                aiStatusDot.title = 'AI: Offline';
            }
        }
    }

    function startAIStatusPolling() {
        pollAIStatus();
        aiStatusInterval = setInterval(pollAIStatus, 30000);
    }

    // ── Initialize ───────────────────────────────────────────────

    initBackend();
    startCamera();
    startAIStatusPolling();
})();
