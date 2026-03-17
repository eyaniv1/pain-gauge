/**
 * API Client for Pain Gauge Backend
 *
 * Handles all communication with the backend server.
 * Gracefully degrades when backend is unreachable (offline mode).
 */

const PainGaugeAPI = (function () {
    'use strict';

    let baseUrl = '';
    let connected = false;

    function setBaseUrl(url) {
        baseUrl = url.replace(/\/+$/, '');
    }

    function getBaseUrl() {
        return baseUrl;
    }

    function isConnected() {
        return connected;
    }

    const MAX_RETRIES = 3;
    const BASE_DELAY_MS = 1000;

    async function request(method, path, body, attempt = 0) {
        if (!baseUrl) {
            connected = false;
            return null;
        }
        try {
            const opts = {
                method,
                headers: { 'Content-Type': 'application/json' },
            };
            if (body) opts.body = JSON.stringify(body);
            const res = await fetch(baseUrl + path, opts);
            if (!res.ok) {
                // Retry on 429 (rate limit) or 529 (overloaded)
                if ((res.status === 429 || res.status === 529) && attempt < MAX_RETRIES) {
                    const delay = BASE_DELAY_MS * Math.pow(2, attempt) + Math.random() * 500;
                    console.warn(`API ${method} ${path}: ${res.status}, retrying in ${Math.round(delay)}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
                    await new Promise(r => setTimeout(r, delay));
                    return request(method, path, body, attempt + 1);
                }
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || `HTTP ${res.status}`);
            }
            connected = true;
            return await res.json();
        } catch (e) {
            // Retry on network errors (fetch failures)
            if (!e.message.startsWith('HTTP') && attempt < MAX_RETRIES) {
                const delay = BASE_DELAY_MS * Math.pow(2, attempt) + Math.random() * 500;
                console.warn(`API ${method} ${path}: ${e.message}, retrying in ${Math.round(delay)}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
                await new Promise(r => setTimeout(r, delay));
                return request(method, path, body, attempt + 1);
            }
            if (e.message && !e.message.startsWith('HTTP')) {
                connected = false;
            }
            console.error(`API ${method} ${path}:`, e.message);
            return null;
        }
    }

    // ── Health check ───────────────────────────────────────

    async function checkConnection() {
        if (!baseUrl) { connected = false; return false; }
        const result = await request('GET', '/api/health');
        return result !== null;
    }

    // ── Patients ───────────────────────────────────────────

    async function listPatients() {
        return await request('GET', '/api/patients') || [];
    }

    async function getPatient(id) {
        return await request('GET', '/api/patients/' + id);
    }

    async function createPatient(data) {
        return await request('POST', '/api/patients', data);
    }

    async function updatePatient(id, data) {
        return await request('PUT', '/api/patients/' + id, data);
    }

    async function deletePatient(id) {
        return await request('DELETE', '/api/patients/' + id);
    }

    // ── Sessions ───────────────────────────────────────────

    async function listSessions(patientId) {
        return await request('GET', '/api/patients/' + patientId + '/sessions') || [];
    }

    async function createSession(patientId, data) {
        return await request('POST', '/api/patients/' + patientId + '/sessions', data);
    }

    async function getSession(sessionId) {
        return await request('GET', '/api/sessions/' + sessionId);
    }

    async function endSession(sessionId, data) {
        return await request('PUT', '/api/sessions/' + sessionId, data);
    }

    async function deleteSession(sessionId) {
        return await request('DELETE', '/api/sessions/' + sessionId);
    }

    async function deleteSessions(sessionIds) {
        return await request('POST', '/api/sessions/batch-delete', { session_ids: sessionIds });
    }

    // ── Samples ────────────────────────────────────────────

    async function getSamples(sessionId) {
        return await request('GET', '/api/sessions/' + sessionId + '/samples') || [];
    }

    async function sendSamples(sessionId, samples) {
        return await request('POST', '/api/sessions/' + sessionId + '/samples', { samples });
    }

    // ── Sample Correction ──────────────────────────────────

    async function correctSample(sampleId, correctedScore, correctedBy) {
        return await request('PUT', '/api/samples/' + sampleId + '/correct', {
            corrected_score: correctedScore,
            corrected_by: correctedBy || 'clinician',
        });
    }

    // ── Inference Status ──────────────────────────────────

    async function getInferenceStatus() {
        return await request('GET', '/api/inference/status');
    }

    // ── AI Calibration ──────────────────────────────────

    async function calibrateAI(imageBase64, painLevel) {
        return await request('POST', '/api/inference/calibrate', {
            image: imageBase64,
            pain_level: painLevel,
        });
    }

    async function clearAICalibration() {
        return await request('POST', '/api/inference/clear-calibration');
    }

    // ── Frame URL helper ───────────────────────────────────

    function frameUrl(filename) {
        return baseUrl + '/api/frames/' + filename;
    }

    return {
        setBaseUrl,
        getBaseUrl,
        isConnected,
        checkConnection,
        listPatients,
        getPatient,
        createPatient,
        updatePatient,
        deletePatient,
        listSessions,
        createSession,
        getSession,
        endSession,
        deleteSession,
        deleteSessions,
        getSamples,
        sendSamples,
        correctSample,
        getInferenceStatus,
        calibrateAI,
        clearAICalibration,
        frameUrl,
    };
})();
