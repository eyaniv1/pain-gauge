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

    async function request(method, path, body) {
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
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || `HTTP ${res.status}`);
            }
            connected = true;
            return await res.json();
        } catch (e) {
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

    // ── Samples ────────────────────────────────────────────

    async function getSamples(sessionId) {
        return await request('GET', '/api/sessions/' + sessionId + '/samples') || [];
    }

    async function sendSamples(sessionId, samples) {
        return await request('POST', '/api/sessions/' + sessionId + '/samples', { samples });
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
        getSamples,
        sendSamples,
        frameUrl,
    };
})();
