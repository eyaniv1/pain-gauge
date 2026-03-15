/**
 * Pain Gauge Backend Server
 *
 * Express + SQLite backend for persisting patient data, sessions, and samples.
 * Designed to run on a local machine and be exposed via Cloudflare Tunnel.
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const Database = require('better-sqlite3');

const app = express();
const PORT = process.env.PORT || 3000;
const INFERENCE_URL = process.env.INFERENCE_URL || 'http://127.0.0.1:5000';

// ── Inference Service State ───────────────────────────────

let inferenceStatus = { online: false, lastCheck: 0, detail: null };
const INFERENCE_CACHE_MS = 30000; // Cache health status for 30 seconds

// ── Middleware ──────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Serve frame images
const FRAMES_DIR = path.join(__dirname, 'frames');
if (!fs.existsSync(FRAMES_DIR)) fs.mkdirSync(FRAMES_DIR, { recursive: true });
app.use('/api/frames', express.static(FRAMES_DIR));

// ── Database setup ─────────────────────────────────────────

const DB_PATH = path.join(__dirname, 'paingauge.db');
const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        dob TEXT,
        patient_id TEXT,
        notes TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        start_time TEXT NOT NULL,
        end_time TEXT,
        baseline_pain_level INTEGER DEFAULT 0,
        calibration_data TEXT,
        settings_snapshot TEXT,
        notes TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        timestamp_ms REAL NOT NULL,
        score REAL NOT NULL,
        raw_score REAL,
        pspi REAL,
        sensor_data TEXT,
        frame_filename TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
    CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id);
`);

// ── AI Column Migration ──────────────────────────────────
// Add AI-related columns to samples table (idempotent)

const aiColumns = [
    { name: 'ai_score', type: 'REAL' },
    { name: 'ai_confidence', type: 'REAL' },
    { name: 'model_version', type: 'TEXT' },
    { name: 'corrected_score', type: 'REAL' },
    { name: 'corrected_by', type: 'TEXT' },
    { name: 'corrected_at', type: 'TEXT' },
];

const existingColumns = db.pragma('table_info(samples)').map(c => c.name);
for (const col of aiColumns) {
    if (!existingColumns.includes(col.name)) {
        db.exec(`ALTER TABLE samples ADD COLUMN ${col.name} ${col.type}`);
    }
}

db.exec('CREATE INDEX IF NOT EXISTS idx_samples_corrected ON samples(corrected_score)');

// ── Prepared statements ────────────────────────────────────

const stmts = {
    // Patients
    listPatients: db.prepare('SELECT * FROM patients ORDER BY name'),
    getPatient: db.prepare('SELECT * FROM patients WHERE id = ?'),
    insertPatient: db.prepare('INSERT INTO patients (id, name, dob, patient_id, notes) VALUES (?, ?, ?, ?, ?)'),
    updatePatient: db.prepare('UPDATE patients SET name = ?, dob = ?, patient_id = ?, notes = ? WHERE id = ?'),
    deletePatient: db.prepare('DELETE FROM patients WHERE id = ?'),

    // Sessions
    listSessions: db.prepare('SELECT * FROM sessions WHERE patient_id = ? ORDER BY start_time DESC'),
    getSession: db.prepare('SELECT * FROM sessions WHERE id = ?'),
    insertSession: db.prepare('INSERT INTO sessions (id, patient_id, start_time, baseline_pain_level, calibration_data, settings_snapshot) VALUES (?, ?, ?, ?, ?, ?)'),
    endSession: db.prepare('UPDATE sessions SET end_time = ?, notes = ? WHERE id = ?'),
    deleteSession: db.prepare('DELETE FROM sessions WHERE id = ?'),

    // Samples
    listSamples: db.prepare('SELECT * FROM samples WHERE session_id = ? ORDER BY timestamp_ms'),
    insertSample: db.prepare('INSERT INTO samples (session_id, timestamp_ms, score, raw_score, pspi, sensor_data, frame_filename, ai_score, ai_confidence, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'),
    correctSample: db.prepare('UPDATE samples SET corrected_score = ?, corrected_by = ?, corrected_at = ? WHERE id = ?'),
    getSample: db.prepare('SELECT * FROM samples WHERE id = ?'),
};

// Batch insert samples in a transaction
const insertSamplesBatch = db.transaction((samples) => {
    for (const s of samples) {
        stmts.insertSample.run(
            s.session_id, s.timestamp_ms, s.score, s.raw_score, s.pspi,
            s.sensor_data, s.frame_filename,
            s.ai_score || null, s.ai_confidence || null, s.model_version || null
        );
    }
});

// ── Inference Service Helpers ──────────────────────────────

async function checkInferenceHealth() {
    const now = Date.now();
    if (now - inferenceStatus.lastCheck < INFERENCE_CACHE_MS) {
        return inferenceStatus;
    }

    try {
        const resp = await fetch(`${INFERENCE_URL}/health`, {
            signal: AbortSignal.timeout(2000),
        });
        const data = await resp.json();
        inferenceStatus = { online: true, lastCheck: now, detail: data };
    } catch {
        inferenceStatus = { online: false, lastCheck: now, detail: null };
    }
    return inferenceStatus;
}

async function requestInference(imageBase64, sensorData) {
    try {
        const body = { image: imageBase64 };
        if (sensorData) {
            const parsed = typeof sensorData === 'string' ? JSON.parse(sensorData) : sensorData;
            if (parsed.face) body.au_data = parsed.face;
        }

        const resp = await fetch(`${INFERENCE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(2000),
        });

        if (!resp.ok) return null;
        return await resp.json();
    } catch (err) {
        console.warn(`Inference request failed: ${err.message}`);
        return null;
    }
}

async function requestCalibration(imageBase64, painLevel) {
    try {
        const resp = await fetch(`${INFERENCE_URL}/calibrate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageBase64, pain_level: painLevel }),
            signal: AbortSignal.timeout(3000),
        });

        if (!resp.ok) return null;
        return await resp.json();
    } catch (err) {
        console.warn(`Calibration request failed: ${err.message}`);
        return null;
    }
}

async function requestClearCalibration() {
    try {
        const resp = await fetch(`${INFERENCE_URL}/clear-calibration`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(2000),
        });
        if (!resp.ok) return null;
        return await resp.json();
    } catch (err) {
        console.warn(`Clear calibration failed: ${err.message}`);
        return null;
    }
}

// ── API Routes: Patients ───────────────────────────────────

app.get('/api/patients', (req, res) => {
    res.json(stmts.listPatients.all());
});

app.get('/api/patients/:id', (req, res) => {
    const patient = stmts.getPatient.get(req.params.id);
    if (!patient) return res.status(404).json({ error: 'Patient not found' });
    res.json(patient);
});

app.post('/api/patients', (req, res) => {
    const { name, dob, patient_id, notes } = req.body;
    if (!name) return res.status(400).json({ error: 'Name is required' });
    const id = uuidv4();
    stmts.insertPatient.run(id, name, dob || null, patient_id || null, notes || null);
    res.status(201).json(stmts.getPatient.get(id));
});

app.put('/api/patients/:id', (req, res) => {
    const existing = stmts.getPatient.get(req.params.id);
    if (!existing) return res.status(404).json({ error: 'Patient not found' });
    const { name, dob, patient_id, notes } = req.body;
    stmts.updatePatient.run(
        name || existing.name,
        dob !== undefined ? dob : existing.dob,
        patient_id !== undefined ? patient_id : existing.patient_id,
        notes !== undefined ? notes : existing.notes,
        req.params.id
    );
    res.json(stmts.getPatient.get(req.params.id));
});

app.delete('/api/patients/:id', (req, res) => {
    stmts.deletePatient.run(req.params.id);
    res.json({ ok: true });
});

// ── API Routes: Sessions ───────────────────────────────────

app.get('/api/patients/:pid/sessions', (req, res) => {
    res.json(stmts.listSessions.all(req.params.pid));
});

app.post('/api/patients/:pid/sessions', (req, res) => {
    const patient = stmts.getPatient.get(req.params.pid);
    if (!patient) return res.status(404).json({ error: 'Patient not found' });

    const { baseline_pain_level, calibration_data, settings_snapshot } = req.body;
    const id = uuidv4();
    const startTime = new Date().toISOString();

    stmts.insertSession.run(
        id, req.params.pid, startTime,
        baseline_pain_level || 0,
        calibration_data ? JSON.stringify(calibration_data) : null,
        settings_snapshot ? JSON.stringify(settings_snapshot) : null
    );

    res.status(201).json(stmts.getSession.get(id));
});

app.get('/api/sessions/:id', (req, res) => {
    const session = stmts.getSession.get(req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    res.json(session);
});

app.put('/api/sessions/:id', (req, res) => {
    const session = stmts.getSession.get(req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    const { notes } = req.body;
    const endTime = new Date().toISOString();
    stmts.endSession.run(endTime, notes || null, req.params.id);
    res.json(stmts.getSession.get(req.params.id));
});

app.delete('/api/sessions/:id', (req, res) => {
    // Also delete associated frame files
    const samples = stmts.listSamples.all(req.params.id);
    for (const s of samples) {
        if (s.frame_filename) {
            const framePath = path.join(FRAMES_DIR, s.frame_filename);
            if (fs.existsSync(framePath)) fs.unlinkSync(framePath);
        }
    }
    stmts.deleteSession.run(req.params.id);
    res.json({ ok: true });
});

// Batch delete sessions
app.post('/api/sessions/batch-delete', (req, res) => {
    const { session_ids } = req.body;
    if (!Array.isArray(session_ids)) return res.status(400).json({ error: 'session_ids array required' });

    const deleteMany = db.transaction((ids) => {
        for (const id of ids) {
            const samples = stmts.listSamples.all(id);
            for (const s of samples) {
                if (s.frame_filename) {
                    const framePath = path.join(FRAMES_DIR, s.frame_filename);
                    if (fs.existsSync(framePath)) fs.unlinkSync(framePath);
                }
            }
            stmts.deleteSession.run(id);
        }
    });

    deleteMany(session_ids);
    res.json({ ok: true, deleted: session_ids.length });
});

// ── API Routes: Samples ────────────────────────────────────

app.get('/api/sessions/:id/samples', (req, res) => {
    const samples = stmts.listSamples.all(req.params.id);
    res.json(samples.map(s => ({
        ...s,
        sensor_data: s.sensor_data ? JSON.parse(s.sensor_data) : null,
    })));
});

app.post('/api/sessions/:id/samples', async (req, res) => {
    const session = stmts.getSession.get(req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });

    const { samples } = req.body;
    if (!Array.isArray(samples)) return res.status(400).json({ error: 'samples array required' });

    const rows = [];
    for (const sample of samples) {
        let frameFilename = null;
        let aiResult = null;

        // Save frame image if provided (base64 JPEG)
        if (sample.frame) {
            frameFilename = `${req.params.id}_${sample.timestamp_ms}.jpg`;
            const buffer = Buffer.from(sample.frame.replace(/^data:image\/\w+;base64,/, ''), 'base64');
            fs.writeFileSync(path.join(FRAMES_DIR, frameFilename), buffer);

            // Request AI inference (non-blocking for the batch, but sequential per sample)
            aiResult = await requestInference(sample.frame, sample.sensor_data);
        }

        rows.push({
            session_id: req.params.id,
            timestamp_ms: sample.timestamp_ms,
            score: sample.score,
            raw_score: sample.raw_score || null,
            pspi: sample.pspi || null,
            sensor_data: sample.sensor_data ? JSON.stringify(sample.sensor_data) : null,
            frame_filename: frameFilename,
            ai_score: aiResult ? aiResult.ai_score : null,
            ai_confidence: aiResult ? aiResult.confidence : null,
            model_version: aiResult ? aiResult.model_version : null,
        });
    }

    insertSamplesBatch(rows);
    res.status(201).json({ inserted: rows.length });
});

// ── API Routes: Sample Correction ─────────────────────────

app.put('/api/samples/:id/correct', (req, res) => {
    const sample = stmts.getSample.get(req.params.id);
    if (!sample) return res.status(404).json({ error: 'Sample not found' });

    const { corrected_score, corrected_by } = req.body;
    if (corrected_score === undefined || corrected_score === null) {
        return res.status(400).json({ error: 'corrected_score is required' });
    }

    const corrected_at = new Date().toISOString();
    stmts.correctSample.run(corrected_score, corrected_by || null, corrected_at, req.params.id);
    res.json(stmts.getSample.get(req.params.id));
});

// ── Health check ───────────────────────────────────────────

app.get('/api/health', async (req, res) => {
    const inference = await checkInferenceHealth();
    res.json({
        status: 'ok',
        patients: stmts.listPatients.all().length,
        inference_online: inference.online,
    });
});

app.get('/api/inference/status', async (req, res) => {
    const inference = await checkInferenceHealth();
    res.json({
        online: inference.online,
        url: INFERENCE_URL,
        detail: inference.detail,
        last_check: inference.lastCheck ? new Date(inference.lastCheck).toISOString() : null,
    });
});

// ── API Routes: AI Calibration ────────────────────────────

app.post('/api/inference/calibrate', async (req, res) => {
    const { image, pain_level } = req.body;
    if (!image) return res.status(400).json({ error: 'image is required' });
    if (pain_level === undefined) return res.status(400).json({ error: 'pain_level is required' });

    const result = await requestCalibration(image, pain_level);
    if (!result) return res.status(502).json({ error: 'Inference service unavailable' });
    res.json(result);
});

app.post('/api/inference/clear-calibration', async (req, res) => {
    const result = await requestClearCalibration();
    if (!result) return res.status(502).json({ error: 'Inference service unavailable' });
    res.json(result);
});

// ── Start server ───────────────────────────────────────────

app.listen(PORT, () => {
    console.log(`Pain Gauge backend running on http://localhost:${PORT}`);
    console.log(`Database: ${DB_PATH}`);
    console.log(`Frames:   ${FRAMES_DIR}`);
});
