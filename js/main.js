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
        cnnWeight: 0.4,
        hrElevationSensitivity: 1.0,
        hrvSuppressionSensitivity: 1.0,
        faceWeight: 0.5,
        hrWeight: 0.3,
        bodyWeight: 0.2,
        guardingSensitivity: 1.0,
        bracingSensitivity: 1.0,
        restlessnessSensitivity: 1.0,
        freezingSensitivity: 1.0,
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
    let lastCnnScore = null; // latest CNN engine score from backend

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
    const cancelCalibrateBtn = document.getElementById('cancel-calibrate-btn');
    const backToMainBtn = document.getElementById('back-to-main-btn');

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
    const presetSelect = document.getElementById('preset-select');
    const presetSaveBtn = document.getElementById('preset-save-btn');
    const presetDeleteBtn = document.getElementById('preset-delete-btn');
    const PRESETS_KEY = 'paingauge_presets';
    const ACTIVE_PRESET_KEY = 'paingauge_active_preset';
    const backendUrlInput = document.getElementById('set-backend-url');
    const connectionStatusEl = document.getElementById('connection-status');

    const settingInputs = {
        au4Sensitivity:     { input: 'set-au4-sens',    display: 'val-au4-sens' },
        au67Sensitivity:    { input: 'set-au67-sens',   display: 'val-au67-sens' },
        au910Sensitivity:   { input: 'set-au910-sens',  display: 'val-au910-sens' },
        au43Threshold:      { input: 'set-au43-thresh', display: 'val-au43-thresh' },
        au43Sensitivity:    { input: 'set-au43-sens',   display: 'val-au43-sens' },
        talkingSuppression: { input: 'set-talk-supp',   display: 'val-talk-supp' },
        cnnWeight:          { input: 'set-cnn-weight',  display: 'val-cnn-weight' },
        hrElevationSensitivity:   { input: 'set-hr-elev-sens',  display: 'val-hr-elev-sens' },
        hrvSuppressionSensitivity: { input: 'set-hrv-supp-sens', display: 'val-hrv-supp-sens' },
        faceWeight:         { input: 'set-face-weight',  display: 'val-face-weight' },
        hrWeight:           { input: 'set-hr-weight',   display: 'val-hr-weight' },
        bodyWeight:         { input: 'set-body-weight', display: 'val-body-weight' },
        guardingSensitivity:      { input: 'set-guard-sens',   display: 'val-guard-sens' },
        bracingSensitivity:       { input: 'set-brace-sens',   display: 'val-brace-sens' },
        restlessnessSensitivity:  { input: 'set-restless-sens', display: 'val-restless-sens' },
        freezingSensitivity:      { input: 'set-freeze-sens',  display: 'val-freeze-sens' },
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

    // Wire up chart point click → frame viewer
    chart.onPointSelected = (info) => {
        const frameModal = document.getElementById('frame-modal');
        const frameModalImg = document.getElementById('frame-modal-img');
        const frameModalTitle = document.getElementById('frame-modal-title');
        if (info && info.frame) {
            frameModalImg.src = info.isFilename ? PainGaugeAPI.frameUrl(info.frame) : info.frame;
            frameModalTitle.textContent = `${info.label}: ${info.score.toFixed(1)} @ ${Math.floor(info.time / 60)}:${Math.floor(info.time % 60).toString().padStart(2, '0')}`;
            frameModal.classList.remove('hidden');
        }
    };

    const bleHR = new BleHeartRate();
    const bodyMotion = new BodyMotion(settings);

    // ── Heart Rate Sensor UI ──────────────────────────────────

    const hrConnectBtn = document.getElementById('hr-connect-btn');
    const hrSimBtn = document.getElementById('hr-sim-btn');
    const hrDisconnectBtn = document.getElementById('hr-disconnect-btn');
    const hrBpmEl = document.getElementById('hr-bpm');
    const hrHrvEl = document.getElementById('hr-hrv');
    const hrRrEl = document.getElementById('hr-rr');
    const hrDeviceName = document.getElementById('hr-device-name');
    const hrStatusBadge = document.getElementById('hr-status-badge');

    hrConnectBtn.addEventListener('click', async () => {
        try {
            await bleHR.connect();
        } catch (err) {
            console.error('BLE connect failed:', err);
        }
    });

    hrSimBtn.addEventListener('click', () => {
        bleHR.startSimulation();
    });

    hrDisconnectBtn.addEventListener('click', () => {
        bleHR.disconnect();
    });

    bleHR.onUpdate = ({ hr, hrv, rr }) => {
        hrBpmEl.textContent = Math.round(hr);
        hrHrvEl.textContent = hrv > 0 ? hrv.toFixed(1) : '--';
        hrRrEl.textContent = rr > 0 ? Math.round(rr) : '--';

        // Add HR to chart during recording
        if (sessionActive) {
            chart.addHrSample(hr);
        }
    };

    bleHR.onStatusChange = (status) => {
        hrStatusBadge.textContent = status === 'simulating' ? 'Simulating' :
            status === 'connected' ? 'Connected' : 'Disconnected';
        hrStatusBadge.className = 'hr-status-badge ' + status;
        hrDeviceName.textContent = bleHR.getDeviceName();

        const isActive = status === 'connected' || status === 'simulating';
        hrConnectBtn.classList.toggle('hidden', isActive);
        hrSimBtn.classList.toggle('hidden', isActive);
        hrDisconnectBtn.classList.toggle('hidden', !isActive);
    };

    // ── Body Motion Panel DOM refs ────────────────────────────
    const bodyMotionPanel = document.getElementById('body-motion-panel');
    const poseStatusBadge = document.getElementById('pose-status-badge');
    const bodyPainScoreEl = document.getElementById('body-pain-score');
    const bodyBars = {
        guarding:     { bar: document.getElementById('guard-bar'),    value: document.getElementById('guard-value') },
        bracing:      { bar: document.getElementById('brace-bar'),    value: document.getElementById('brace-value') },
        restlessness: { bar: document.getElementById('restless-bar'), value: document.getElementById('restless-value') },
        freezing:     { bar: document.getElementById('freeze-bar'),   value: document.getElementById('freeze-value') },
    };

    // Track latest pose results
    let latestPoseLandmarks = null;
    let poseActive = false;

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

    const fusionWeightKeys = ['faceWeight', 'hrWeight', 'bodyWeight'];
    const lockedWeights = new Set();

    // Wire up lock buttons
    document.querySelectorAll('.weight-lock-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.weight;
            if (lockedWeights.has(key)) {
                lockedWeights.delete(key);
                btn.classList.remove('locked');
                btn.title = 'Lock this weight';
            } else {
                lockedWeights.add(key);
                btn.classList.add('locked');
                btn.title = 'Unlock this weight';
            }
        });
    });

    function redistributeFusionWeights(changedKey, newVal) {
        // Only redistribute to unlocked, non-changed sliders
        const otherKeys = fusionWeightKeys.filter(k => k !== changedKey && !lockedWeights.has(k));
        const lockedSum = fusionWeightKeys
            .filter(k => k !== changedKey && lockedWeights.has(k))
            .reduce((s, k) => s + settings[k], 0);
        const remaining = Math.max(0, 1 - newVal - lockedSum);

        if (otherKeys.length === 0) return; // all others locked, can't redistribute

        const otherSum = otherKeys.reduce((s, k) => s + settings[k], 0);

        for (const k of otherKeys) {
            // Distribute remaining proportionally; if others were all zero, split evenly
            const proportion = otherSum > 0 ? settings[k] / otherSum : 1 / otherKeys.length;
            const adjusted = Math.round(remaining * proportion * 100) / 100;
            settings[k] = adjusted;
            engine[k] = adjusted;

            // Update slider and display
            const { input, display } = settingInputs[k];
            document.getElementById(input).value = adjusted;
            document.getElementById(display).textContent = adjusted;
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

                const bodyMotionKeys = ['guardingSensitivity', 'bracingSensitivity',
                    'restlessnessSensitivity', 'freezingSensitivity'];

                if (fusionWeightKeys.includes(key)) {
                    if (lockedWeights.has(key)) {
                        // Revert — can't change a locked slider
                        el.value = settings[key];
                        valEl.textContent = settings[key];
                        return;
                    }
                    engine[key] = val;
                    redistributeFusionWeights(key, val);
                } else if (key === 'smoothingSize') {
                    engine.smoothingSize = val;
                } else if (key === 'sampleIntervalMs') {
                    SAMPLE_INTERVAL_MS = val;
                } else if (bodyMotionKeys.includes(key)) {
                    bodyMotion[key] = val;
                } else if (key === 'cnnWeight') {
                    // Used directly from settings in updateEngineScores
                } else {
                    engine[key] = val;
                }

                saveSettings(settings);
                markPresetModified();
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

    // ── Presets ────────────────────────────────────────────────

    function markPresetModified() {
        const active = presetSelect.value;
        const currentOption = presetSelect.options[presetSelect.selectedIndex];
        if (!currentOption) return;
        const baseName = active === '__defaults__' ? 'Default' : active;
        if (!currentOption.textContent.endsWith(' *')) {
            currentOption.textContent = baseName + ' *';
        }
    }

    function loadPresets() {
        try {
            const saved = localStorage.getItem(PRESETS_KEY);
            if (saved) return JSON.parse(saved);
        } catch (e) { /* ignore */ }
        return {};
    }

    function savePresets(presets) {
        localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
    }

    function getSettingsSnapshot() {
        // Capture all tunable settings (exclude backendUrl — it's per-machine)
        const snap = {};
        for (const key of Object.keys(DEFAULTS)) {
            if (key === 'backendUrl') continue;
            snap[key] = settings[key];
        }
        return snap;
    }

    function applyPresetSettings(preset) {
        const keepBackendUrl = settings.backendUrl;
        settings = { ...DEFAULTS, ...preset, backendUrl: keepBackendUrl };
        applySettingsToUI(settings);
        const bodyMotionKeys = ['guardingSensitivity', 'bracingSensitivity',
            'restlessnessSensitivity', 'freezingSensitivity'];
        for (const key of Object.keys(DEFAULTS)) {
            if (key === 'sampleIntervalMs') {
                SAMPLE_INTERVAL_MS = settings[key];
            } else if (key === 'backendUrl') {
                // keep current backend URL
            } else if (bodyMotionKeys.includes(key)) {
                bodyMotion[key] = settings[key];
            } else {
                engine[key] = settings[key];
            }
        }
        saveSettings(settings);
    }

    function populatePresetDropdown() {
        const presets = loadPresets();
        const activePreset = localStorage.getItem(ACTIVE_PRESET_KEY) || '__defaults__';
        presetSelect.innerHTML = '<option value="__defaults__">Default</option>';
        for (const name of Object.keys(presets).sort()) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            presetSelect.appendChild(option);
        }
        presetSelect.value = activePreset;
        // If saved preset no longer exists, fall back to defaults
        if (presetSelect.value !== activePreset) {
            presetSelect.value = '__defaults__';
            localStorage.setItem(ACTIVE_PRESET_KEY, '__defaults__');
        }
    }

    presetSelect.addEventListener('change', () => {
        const name = presetSelect.value;
        localStorage.setItem(ACTIVE_PRESET_KEY, name);
        if (name === '__defaults__') {
            applyPresetSettings(DEFAULTS);
        } else {
            const presets = loadPresets();
            if (presets[name]) {
                applyPresetSettings(presets[name]);
            }
        }
    });

    presetSaveBtn.addEventListener('click', () => {
        const activePreset = presetSelect.value;
        let name;
        if (activePreset !== '__defaults__') {
            // Offer to overwrite current or save new
            name = prompt('Save preset as:', activePreset);
        } else {
            name = prompt('Preset name:');
        }
        if (!name || !name.trim()) return;
        name = name.trim();
        if (name === '__defaults__' || name.toLowerCase() === 'default') {
            alert('Cannot overwrite the Default preset.');
            return;
        }
        const presets = loadPresets();
        presets[name] = getSettingsSnapshot();
        savePresets(presets);
        populatePresetDropdown();
        presetSelect.value = name;
        localStorage.setItem(ACTIVE_PRESET_KEY, name);
    });

    presetDeleteBtn.addEventListener('click', () => {
        const name = presetSelect.value;
        if (name === '__defaults__') {
            alert('Cannot delete the Default preset.');
            return;
        }
        if (!confirm(`Delete preset "${name}"?`)) return;
        const presets = loadPresets();
        delete presets[name];
        savePresets(presets);
        localStorage.setItem(ACTIVE_PRESET_KEY, '__defaults__');
        applyPresetSettings(DEFAULTS);
        populatePresetDropdown();
    });

    populatePresetDropdown();

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

    // Mark that calibration is needed
    function markNeedsCalibration() {
        recalibrateBtn.classList.add('needs-calibration');
    }
    function clearNeedsCalibration() {
        recalibrateBtn.classList.remove('needs-calibration');
    }

    // On startup, calibration is always needed
    markNeedsCalibration();

    patientSelect.addEventListener('change', () => {
        currentPatientId = patientSelect.value;
        localStorage.setItem('painGaugePatientId', currentPatientId);
        // Clear chart and reload history for new patient
        chart.resetRecording();
        loadHistorySessions();
        markNeedsCalibration();
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
        bodyMotion.resetCalibration();
        autoCalibFrames = AUTO_CALIB_COUNT;
        engine.baselinePainLevel = parseInt(baselinePainInput.value);
        calibrateBtn.disabled = true;
        calibrateBtn.textContent = 'Hold still...';
        cancelCalibrateBtn.classList.remove('hidden');
        calibInstructions.innerHTML = 'Capturing face at <strong>current pain level</strong>...';
        calibProgress.classList.remove('hidden');
        calibProgressBar.style.width = '0%';
        if (inputMode === 'video') {
            videoEl.currentTime = 0;
            videoEl.play();
        }
    });

    cancelCalibrateBtn.addEventListener('click', () => {
        isCalibrating = false;
        calibrationFrames = 0;
        engine.resetCalibration();
        bodyMotion.resetCalibration();
        cancelCalibrateBtn.classList.add('hidden');
        calibrateBtn.disabled = false;
        calibrateBtn.textContent = 'Calibrate';
        calibProgress.classList.add('hidden');
        calibInstructions.innerHTML = 'What is the patient\'s <strong>current pain level</strong>?';
        calibStatus.textContent = 'Calibration cancelled.';
        markNeedsCalibration();
        if (inputMode === 'video') {
            videoEl.pause();
        }
    });

    backToMainBtn.addEventListener('click', () => {
        // Abort calibration if in progress
        if (isCalibrating) {
            isCalibrating = false;
            calibrationFrames = 0;
            engine.resetCalibration();
            bodyMotion.resetCalibration();
            cancelCalibrateBtn.classList.add('hidden');
            calibProgress.classList.add('hidden');
            if (inputMode === 'video') videoEl.pause();
        }
        markNeedsCalibration();
        moveVideoToMain();
        showScreen(screenMain);
    });

    async function onCalibrationComplete() {
        isCalibrating = false;
        cancelCalibrateBtn.classList.add('hidden');
        calibrateBtn.textContent = 'Calibrated!';
        clearNeedsCalibration();
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
            // Start loading Pose model after calibration (non-blocking)
            initPose();
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
        bodyMotion.resetCalibration();

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
        const w = 640;
        const h = 480;
        frameCanvas.width = w;
        frameCanvas.height = h;
        frameCtx.drawImage(videoEl, 0, 0, w, h);
        return frameCanvas.toDataURL('image/jpeg', 0.85);
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
        lastCnnScore = null;

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
                lastCnnScore = ai.ai_score;
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
            // Compute combined face score (AU + CNN blend)
            let faceScore = result.facePain;
            if (lastCnnScore != null && result.facePain != null) {
                const cw = settings.cnnWeight;
                faceScore = result.facePain * (1 - cw) + lastCnnScore * cw;
            }

            // Recompute total using CNN-blended face score instead of AU-only
            // This ensures the CNN contribution flows into the aggregate score
            // Only include modalities that have data AND nonzero weight
            const activeWeights = {};
            if (result.facePain != null && engine.faceWeight > 0) activeWeights.face = engine.faceWeight;
            if (result.hrPain !== null && engine.hrWeight > 0) activeWeights.hr = engine.hrWeight;
            if (result.bodyPain !== null && engine.bodyWeight > 0) activeWeights.body = engine.bodyWeight;
            const totalW = (activeWeights.face || 0) + (activeWeights.hr || 0) + (activeWeights.body || 0);
            let totalScore = 0;
            if (totalW > 0) {
                totalScore = (faceScore || 0) * ((activeWeights.face || 0) / totalW)
                    + (result.hrPain || 0) * ((activeWeights.hr || 0) / totalW)
                    + (result.bodyPain || 0) * ((activeWeights.body || 0) / totalW);
            }
            totalScore = Math.max(0, Math.min(10, Math.round(totalScore * 10) / 10));

            const frame = captureFrame();
            chart.addSample(
                totalScore,       // total aggregate (includes CNN via face blend)
                faceScore,        // combined face (AU + CNN)
                result.facePain,  // AU engine (rules-based)
                result.bodyPain,  // body motion
                result.hrPain,    // HR pain
                frame             // captured frame
            );
            lastSampleTime = now;

            // Buffer sample for backend
            if (currentSessionId) {
                const elapsed = chart.getElapsedTime() * 1000;
                const sample = {
                    timestamp_ms: Math.round(elapsed),
                    score: totalScore,
                    raw_score: result.rawScore,
                    pspi: result.pspi,
                    face_score: faceScore != null ? Math.round(faceScore * 10) / 10 : null,
                    au_score: result.facePain != null ? result.facePain : null,
                    cnn_score: lastCnnScore != null ? Math.round(lastCnnScore * 10) / 10 : null,
                    body_score: result.bodyPain,
                    hr_score: result.hrPain,
                    sensor_data: {
                        face: result.aus,
                        hr: bleHR.isActive() ? {
                            bpm: bleHR.heartRate,
                            hrv: bleHR.hrv,
                            rr: bleHR.rrIntervals.length > 0 ? bleHR.rrIntervals[bleHR.rrIntervals.length - 1] : null,
                        } : undefined,
                        body: (bodyMotion.lastResult && poseReady) ? {
                            guarding: bodyMotion.lastResult.guarding || 0,
                            bracing: bodyMotion.lastResult.bracing || 0,
                            restlessness: bodyMotion.lastResult.restlessness || 0,
                            freezing: bodyMotion.lastResult.freezing || 0,
                        } : undefined,
                    },
                    frame,
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

    // ── MediaPipe Pose setup (lazy — loaded after calibration) ──

    let pose = null;
    let poseReady = false;
    let poseInitializing = false;
    let poseSendPending = false;

    async function initPose() {
        if (poseInitializing || poseReady) return;
        poseInitializing = true;
        try {
            pose = new Pose({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
                },
            });

            pose.setOptions({
                modelComplexity: 0,       // 0=lite for speed
                smoothLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5,
            });

            pose.onResults(onPoseResults);

            // First send triggers WASM download + compile
            await pose.send({ image: videoEl });
            poseReady = true;
            console.log('Pose model ready');
        } catch (e) {
            console.warn('Pose init failed, will retry:', e);
            pose = null;
        }
        poseInitializing = false;
    }

    function sendToPose() {
        if (!poseReady || poseSendPending) return;
        poseSendPending = true;
        pose.send({ image: videoEl }).then(() => {
            poseSendPending = false;
        }).catch(() => {
            poseSendPending = false;
        });
    }

    function onPoseResults(results) {
        if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
            latestPoseLandmarks = results.poseLandmarks;

            if (!poseActive) {
                poseActive = true;
                poseStatusBadge.textContent = 'Active';
                poseStatusBadge.classList.add('active');
            }

            // Auto-calibrate body motion from first ~30 frames
            if (!bodyMotion.isCalibrated() || bodyMotion._calibSamples.length < 30) {
                bodyMotion.calibrate(latestPoseLandmarks);
            }

            const bodyResult = bodyMotion.process(latestPoseLandmarks);
            updateBodyMotionUI(bodyResult);

            // Draw pose skeleton on overlay
            drawPoseSkeleton(results.poseLandmarks);
        }
    }

    function updateBodyMotionUI(result) {
        bodyPainScoreEl.textContent = result.score.toFixed(1);
        for (const [key, elements] of Object.entries(bodyBars)) {
            const value = result[key] || 0;
            const pct = (value / 5) * 100;
            elements.bar.style.width = `${pct}%`;
            elements.bar.style.backgroundColor = PainGauge.painColor(value * 2);
            elements.value.textContent = value.toFixed(1);
        }
    }

    function drawPoseSkeleton(landmarks) {
        const w = overlayEl.width;
        const h = overlayEl.height;
        if (!w || !h) return;

        // Skeleton connections
        const connections = [
            [11, 12], // shoulders
            [11, 13], [13, 15], // left arm
            [12, 14], [14, 16], // right arm
            [11, 23], [12, 24], // torso sides
            [23, 24], // hips
            [23, 25], [25, 27], // left leg
            [24, 26], [26, 28], // right leg
        ];

        overlayCtx.strokeStyle = 'rgba(0, 220, 130, 0.5)';
        overlayCtx.lineWidth = 2;

        for (const [i, j] of connections) {
            const a = landmarks[i];
            const b = landmarks[j];
            if ((a.visibility || 0) > 0.4 && (b.visibility || 0) > 0.4) {
                overlayCtx.beginPath();
                overlayCtx.moveTo(a.x * w, a.y * h);
                overlayCtx.lineTo(b.x * w, b.y * h);
                overlayCtx.stroke();
            }
        }

        // Draw key joint dots
        overlayCtx.fillStyle = 'rgba(0, 220, 130, 0.7)';
        for (let i = 11; i <= 28; i++) {
            const lm = landmarks[i];
            if ((lm.visibility || 0) > 0.4) {
                overlayCtx.beginPath();
                overlayCtx.arc(lm.x * w, lm.y * h, 3, 0, 2 * Math.PI);
                overlayCtx.fill();
            }
        }
    }

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

        // When no face detected, still compute score from other modalities
        if (!hasFace) {
            if (!engine.isCalibrated() || isCalibrating) return;

            const physio = bleHR.isActive() ? { hr: bleHR.heartRate, hrv: bleHR.hrv } : null;
            const bodyScore = (bodyMotion.lastResult && poseReady) ? bodyMotion.lastResult.score : null;

            // Only proceed if we have at least one non-face modality with data
            const hasHR = physio && engine.hrWeight > 0 && engine.hrBaseline;
            const hasBody = bodyScore !== null && engine.bodyWeight > 0;
            if (!hasHR && !hasBody) return;

            // Compute HR pain if available
            let hrPainScore = null;
            if (hasHR) {
                hrPainScore = engine.computeHRPainScore(physio.hr, physio.hrv);
            }

            // Build weighted fusion from available non-face modalities
            const gw = {};
            if (hrPainScore !== null && engine.hrWeight > 0) gw.hr = engine.hrWeight;
            if (hasBody) gw.body = engine.bodyWeight;
            const gwTotal = (gw.hr || 0) + (gw.body || 0);
            let gaugeScore = 0;
            if (gwTotal > 0) {
                gaugeScore = (hrPainScore || 0) * ((gw.hr || 0) / gwTotal)
                    + (bodyScore || 0) * ((gw.body || 0) / gwTotal);
            }
            gaugeScore = Math.max(0, Math.min(10, Math.round(gaugeScore * 10) / 10));
            gauge.setScore(gaugeScore);

            // Build a result-like object for sampling and display
            const noFaceResult = {
                score: gaugeScore,
                rawScore: gaugeScore,
                facePain: null,
                hrPain: hrPainScore,
                bodyPain: bodyScore,
                aus: { au4: 0, au6_7: 0, au9_10: 0, au43: 0 },
                pspi: 0,
            };
            updateAUBars(noFaceResult.aus);
            updateEngineScores(noFaceResult);
            lastResult = noFaceResult;

            const hrPainEl = document.getElementById('hr-pain');
            if (hrPainEl) {
                hrPainEl.textContent = hrPainScore !== null ? hrPainScore.toFixed(1) : '--';
            }
            maybeRecordSample(noFaceResult);
            return;
        }

        const landmarks = results.multiFaceLandmarks[0];
        drawFaceMesh(landmarks);

        if (isCalibrating) {
            engine.calibrate(landmarks);
            if (bleHR.isActive()) engine.calibrateHR(bleHR.heartRate, bleHR.hrv);
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
            if (bleHR.isActive()) engine.calibrateHR(bleHR.heartRate, bleHR.hrv);
            autoCalibFrames++;
            return;
        }

        // Late HR baseline capture: if sensor became active after calibration,
        // collect samples until we have enough for a stable baseline (~10 samples)
        if (bleHR.isActive() && bleHR.heartRate > 0 &&
            (!engine.hrBaseline || engine._hrCalibSamples.length < 10)) {
            engine.calibrateHR(bleHR.heartRate, bleHR.hrv);
        }

        // Build multimodal data for fusion
        const physio = bleHR.isActive() ? { hr: bleHR.heartRate, hrv: bleHR.hrv } : null;
        const bodyScore = (bodyMotion.lastResult && poseReady) ? bodyMotion.lastResult.score : null;
        const result = engine.process(landmarks, physio, bodyScore);
        // Recompute gauge score with same weighted fusion as chart total
        // (result.score from pain-engine always includes face; we need to respect weight=0)
        let faceForGauge = result.score; // smoothed AU face score
        if (lastCnnScore != null && result.facePain != null) {
            const cw = settings.cnnWeight;
            const cnnDelta = (result.facePain * (1 - cw) + lastCnnScore * cw) - result.facePain;
            faceForGauge = Math.max(0, Math.min(10, result.score + cnnDelta));
        }
        const gw = {};
        if (engine.faceWeight > 0) gw.face = engine.faceWeight;
        if (result.hrPain !== null && engine.hrWeight > 0) gw.hr = engine.hrWeight;
        if (result.bodyPain !== null && engine.bodyWeight > 0) gw.body = engine.bodyWeight;
        const gwTotal = (gw.face || 0) + (gw.hr || 0) + (gw.body || 0);
        let gaugeScore = 0;
        if (gwTotal > 0) {
            gaugeScore = (faceForGauge || 0) * ((gw.face || 0) / gwTotal)
                + (result.hrPain || 0) * ((gw.hr || 0) / gwTotal)
                + (result.bodyPain || 0) * ((gw.body || 0) / gwTotal);
        }
        gaugeScore = Math.max(0, Math.min(10, Math.round(gaugeScore * 10) / 10));
        gauge.setScore(gaugeScore);
        updateAUBars(result.aus);
        updateEngineScores(result);
        lastResult = result;
        if (bleHR.simulating) bleHR.setSimPainLevel(result.facePain);
        // Update HR pain display
        const hrPainEl = document.getElementById('hr-pain');
        if (hrPainEl) {
            hrPainEl.textContent = result.hrPain !== null ? result.hrPain.toFixed(1) : '--';
        }
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

    // ── Update engine score summary ────────────────────────────────

    const auEngineScoreEl = document.getElementById('au-engine-score');
    const cnnEngineScoreEl = document.getElementById('cnn-engine-score');
    const faceCombinedScoreEl = document.getElementById('face-combined-score');

    function updateEngineScores(result) {
        // AU engine = the rules-based face pain score
        auEngineScoreEl.textContent = result.facePain != null ? result.facePain.toFixed(1) : '--';
        // CNN engine = latest score from backend (async)
        cnnEngineScoreEl.textContent = lastCnnScore != null ? lastCnnScore.toFixed(1) : '--';
        // Combined face score: weighted blend of AU and CNN
        if (lastCnnScore != null && result.facePain != null) {
            const cw = settings.cnnWeight;
            const combined = result.facePain * (1 - cw) + lastCnnScore * cw;
            faceCombinedScoreEl.textContent = combined.toFixed(1);
        } else {
            faceCombinedScoreEl.textContent = result.facePain != null ? result.facePain.toFixed(1) : '--';
        }
    }

    // ── Camera device selection ──────────────────────────────────

    const cameraSelect = document.getElementById('camera-select');
    let cameraLoopId = null;

    async function enumeratecameras() {
        try {
            // Request permission first so labels are available
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(t => t.stop());

            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');

            cameraSelect.innerHTML = '';
            videoDevices.forEach((device, i) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Camera ${i + 1}`;
                cameraSelect.appendChild(option);
            });

            // If we had a previously selected device, restore it
            const saved = localStorage.getItem('selectedCameraId');
            if (saved && videoDevices.some(d => d.deviceId === saved)) {
                cameraSelect.value = saved;
            }
        } catch (err) {
            console.error('Failed to enumerate cameras:', err);
            cameraSelect.innerHTML = '<option value="">No cameras found</option>';
        }
    }

    function stopCameraLoop() {
        if (cameraLoopId) {
            cancelAnimationFrame(cameraLoopId);
            cameraLoopId = null;
        }
    }

    async function startCamera() {
        stopVideoLoop();
        stopCameraLoop();

        // Stop existing camera stream
        if (cameraInstance) {
            try {
                if (cameraInstance instanceof MediaStream) {
                    cameraInstance.getTracks().forEach(t => t.stop());
                } else {
                    cameraInstance.stop();
                }
            } catch (e) { /* ignore */ }
            cameraInstance = null;
        }

        const deviceId = cameraSelect.value;
        if (!deviceId) return;

        localStorage.setItem('selectedCameraId', deviceId);

        try {
            const constraints = {
                video: {
                    deviceId: { exact: deviceId },
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                },
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraInstance = stream;
            videoEl.srcObject = stream;
            videoEl.muted = true;
            await videoEl.play();

            inputMode = 'camera';
            cameraBtn.classList.add('active');
            loadVideoBtn.classList.remove('active');
            videoEl.style.transform = '';
            overlayEl.style.transform = '';

            // Start frame processing loop
            async function cameraLoop() {
                if (inputMode !== 'camera') return;
                if (videoEl.readyState >= 2) {
                    await faceMesh.send({ image: videoEl });
                    sendToPose();
                }
                cameraLoopId = requestAnimationFrame(cameraLoop);
            }
            cameraLoop();
        } catch (err) {
            console.error('Camera start failed:', err);
        }
    }

    cameraSelect.addEventListener('change', () => {
        if (inputMode === 'camera') {
            startCamera();
        }
    });

    // ── Video file input ─────────────────────────────────────────

    cameraBtn.addEventListener('click', () => {
        if (inputMode === 'camera') return;
        startCamera();
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
        stopCameraLoop();
        if (cameraInstance) {
            if (cameraInstance instanceof MediaStream) {
                cameraInstance.getTracks().forEach(t => t.stop());
            } else {
                try { cameraInstance.stop(); } catch (e) { /* ignore */ }
            }
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
                sendToPose();
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
    let lastCheckedIndex = -1; // for shift-click range selection

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
        lastCheckedIndex = -1;
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

        for (let idx = 0; idx < historySessions.length; idx++) {
            const s = historySessions[idx];
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

            // Checkbox handler with shift-click range selection
            const cb = tr.querySelector('.session-check');
            cb.addEventListener('click', (e) => {
                const checked = cb.checked;
                if (e.shiftKey && lastCheckedIndex >= 0 && lastCheckedIndex !== idx) {
                    const start = Math.min(lastCheckedIndex, idx);
                    const end = Math.max(lastCheckedIndex, idx);
                    const allCbs = historySessionsTbody.querySelectorAll('.session-check');
                    for (let i = start; i <= end; i++) {
                        allCbs[i].checked = checked;
                        if (checked) historySelectedIds.add(historySessions[i].id);
                        else historySelectedIds.delete(historySessions[i].id);
                    }
                } else {
                    if (checked) historySelectedIds.add(s.id);
                    else historySelectedIds.delete(s.id);
                }
                lastCheckedIndex = idx;
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

        historySamplesTbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#999">Loading...</td></tr>';

        const samples = await PainGaugeAPI.getSamples(session.id);
        currentHistorySamples = samples;
        historySamplesTbody.innerHTML = '';

        if (samples.length === 0) {
            historySamplesTbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#999">No samples</td></tr>';
            return;
        }

        // Also load into chart
        chart.loadSamples(samples, `Session ${session.id.substring(0, 8)}`);

        for (const s of samples) {
            const tr = document.createElement('tr');
            const hr = s.sensor_data && s.sensor_data.hr ? s.sensor_data.hr : {};
            const body = s.sensor_data && s.sensor_data.body ? s.sensor_data.body : {};
            const timeStr = formatMs(s.timestamp_ms);

            // Correction cell with edit button
            let correctionHtml;
            if (s.corrected_score != null) {
                correctionHtml = `<span class="corrected-value">${s.corrected_score.toFixed(1)}</span>`;
            } else {
                correctionHtml = `<button class="correct-btn" data-sample-id="${s.id}" title="Correct this score">Edit</button>`;
            }

            if (s.corrected_score != null) tr.classList.add('sample-corrected');

            const frameThumb = s.frame_filename
                ? `<img class="frame-thumb" src="${PainGaugeAPI.frameUrl(s.frame_filename)}" data-frame="${s.frame_filename}" data-time="${timeStr}" alt="Frame at ${timeStr}" loading="lazy">`
                : '-';

            // Face score with tooltip showing AU and CNN breakdown
            const faceVal = s.face_score != null ? s.face_score : (s.score != null ? s.score : null);
            const auVal = s.au_score != null ? s.au_score.toFixed(1) : (s.pspi != null ? s.pspi.toFixed(1) : '-');
            const cnnVal = s.cnn_score != null ? s.cnn_score.toFixed(1) : (s.ai_score != null ? s.ai_score.toFixed(1) : '-');
            const faceTooltip = `AU=${auVal}  CNN=${cnnVal}`;
            const faceHtml = faceVal != null
                ? `<span class="score-with-tip" title="${faceTooltip}">${faceVal.toFixed(1)}</span>`
                : '-';

            // Body score with tooltip showing component breakdown
            const bodyVal = s.body_score != null ? s.body_score : null;
            const bodyTooltip = `Guarding=${(body.guarding || 0).toFixed(1)}  Bracing=${(body.bracing || 0).toFixed(1)}  Restlessness=${(body.restlessness || 0).toFixed(1)}  Freezing=${(body.freezing || 0).toFixed(1)}`;
            const bodyHtml = bodyVal != null
                ? `<span class="score-with-tip" title="${bodyTooltip}">${bodyVal.toFixed(1)}</span>`
                : '-';

            // HR score with tooltip showing BPM and HRV
            const hrVal = s.hr_score != null ? s.hr_score : null;
            const hrTooltip = `BPM=${hr.bpm != null ? hr.bpm : '-'}  HRV=${hr.hrv != null ? hr.hrv.toFixed(0) + 'ms' : '-'}`;
            const hrHtml = hrVal != null
                ? `<span class="score-with-tip" title="${hrTooltip}">${hrVal.toFixed(1)}</span>`
                : '-';

            tr.innerHTML = `
                <td>${timeStr}</td>
                <td>${s.score != null ? s.score.toFixed(1) : '-'}</td>
                <td class="correction-cell">${correctionHtml}</td>
                <td>${faceHtml}</td>
                <td>${bodyHtml}</td>
                <td>${hrHtml}</td>
                <td class="frame-cell">${frameThumb}</td>
            `;

            // Frame click handler
            const frameThumbEl = tr.querySelector('.frame-thumb');
            if (frameThumbEl) {
                frameThumbEl.addEventListener('click', () => {
                    frameModalImg.src = PainGaugeAPI.frameUrl(frameThumbEl.dataset.frame);
                    frameModalTitle.textContent = `Frame at ${frameThumbEl.dataset.time}`;
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
    enumeratecameras().then(() => startCamera());
    startAIStatusPolling();
})();
