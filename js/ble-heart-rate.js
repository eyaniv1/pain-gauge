/**
 * BLE Heart Rate Sensor Module
 *
 * Connects to any BLE device exposing the standard Heart Rate Service (0x180d).
 * Parses heart rate and RR intervals, computes real-time HRV (RMSSD).
 * Includes simulation mode for development without hardware.
 */

class BleHeartRate {
    constructor() {
        // State
        this.device = null;
        this.server = null;
        this.characteristic = null;
        this.connected = false;
        this.simulating = false;

        // Data
        this.heartRate = 0;
        this.rrIntervals = [];     // last N RR intervals in ms
        this.hrv = 0;              // RMSSD in ms
        this.maxRRHistory = 60;    // keep last 60 RR intervals for HRV

        // Callbacks
        this.onUpdate = null;      // ({ hr, hrv, rr }) => {}
        this.onStatusChange = null; // (status: 'connected'|'disconnected'|'simulating') => {}

        // Simulation
        this._simInterval = null;
        this._simBaseHR = 72;
        this._simPainLevel = 0;
    }

    /** Check if Web Bluetooth is available */
    static isSupported() {
        return !!navigator.bluetooth;
    }

    /** Connect to a real BLE Heart Rate sensor */
    async connect() {
        if (!BleHeartRate.isSupported()) {
            throw new Error('Web Bluetooth not supported in this browser');
        }

        try {
            this.device = await navigator.bluetooth.requestDevice({
                filters: [{ services: ['heart_rate'] }],
                optionalServices: ['battery_service']
            });

            this.device.addEventListener('gattserverdisconnected', () => {
                this.connected = false;
                this._notify('disconnected');
            });

            this.server = await this.device.gatt.connect();
            const service = await this.server.getPrimaryService('heart_rate');
            this.characteristic = await service.getCharacteristic('heart_rate_measurement');

            await this.characteristic.startNotifications();
            this.characteristic.addEventListener('characteristicvaluechanged', (e) => {
                this._parseHeartRate(e.target.value);
            });

            this.connected = true;
            this.simulating = false;
            this._notify('connected');

        } catch (err) {
            if (err.name === 'NotFoundError') {
                // User cancelled the picker
                return;
            }
            throw err;
        }
    }

    /** Disconnect from real device */
    disconnect() {
        if (this.device && this.device.gatt.connected) {
            this.device.gatt.disconnect();
        }
        this.stopSimulation();
        this.connected = false;
        this.simulating = false;
        this.heartRate = 0;
        this.hrv = 0;
        this.rrIntervals = [];
        this._notify('disconnected');
    }

    /** Start simulated heart rate data */
    startSimulation() {
        this.stopSimulation();
        this.simulating = true;
        this.connected = false;
        this._simBaseHR = 68 + Math.random() * 10; // 68-78 resting
        this._notify('simulating');

        this._simInterval = setInterval(() => {
            this._generateSimData();
        }, 1000);
        // Immediate first reading
        this._generateSimData();
    }

    /** Stop simulation */
    stopSimulation() {
        if (this._simInterval) {
            clearInterval(this._simInterval);
            this._simInterval = null;
        }
        this.simulating = false;
    }

    /** Update simulation with current pain level (0-10) for realistic correlation */
    setSimPainLevel(painLevel) {
        this._simPainLevel = painLevel;
    }

    /** Reset data (call on session start) */
    reset() {
        this.rrIntervals = [];
        this.hrv = 0;
        this.heartRate = 0;
    }

    // ── Parsing ──────────────────────────────────────────────

    /** Parse BLE Heart Rate Measurement characteristic (standard 0x2A37) */
    _parseHeartRate(dataView) {
        const flags = dataView.getUint8(0);
        const is16Bit = flags & 0x01;
        let offset = 1;

        // Heart rate value
        if (is16Bit) {
            this.heartRate = dataView.getUint16(offset, true);
            offset += 2;
        } else {
            this.heartRate = dataView.getUint8(offset);
            offset += 1;
        }

        // RR intervals (if present, bit 4 of flags)
        const hasRR = flags & 0x10;
        if (hasRR) {
            while (offset + 1 < dataView.byteLength) {
                const rr = dataView.getUint16(offset, true);
                // RR is in 1/1024 seconds, convert to ms
                const rrMs = (rr / 1024) * 1000;
                this._addRR(rrMs);
                offset += 2;
            }
        }

        this._emitUpdate();
    }

    // ── HRV Calculation ──────────────────────────────────────

    /** Add an RR interval and recalculate HRV */
    _addRR(rrMs) {
        // Filter out physiologically impossible values
        if (rrMs < 300 || rrMs > 2000) return;

        this.rrIntervals.push(rrMs);
        if (this.rrIntervals.length > this.maxRRHistory) {
            this.rrIntervals.shift();
        }

        this._calculateHRV();
    }

    /** Calculate RMSSD (Root Mean Square of Successive Differences) */
    _calculateHRV() {
        if (this.rrIntervals.length < 3) {
            this.hrv = 0;
            return;
        }

        let sumSquaredDiffs = 0;
        let count = 0;

        for (let i = 1; i < this.rrIntervals.length; i++) {
            const diff = this.rrIntervals[i] - this.rrIntervals[i - 1];
            sumSquaredDiffs += diff * diff;
            count++;
        }

        this.hrv = Math.sqrt(sumSquaredDiffs / count);
    }

    // ── Simulation ───────────────────────────────────────────

    _generateSimData() {
        const pain = this._simPainLevel;

        // Pain increases HR: baseline + up to 30 bpm at pain 10
        const painHRBoost = pain * 3;
        // Add some natural variability
        const noise = (Math.random() - 0.5) * 4;
        const targetHR = this._simBaseHR + painHRBoost + noise;

        // Smooth transition
        this.heartRate = this.heartRate === 0
            ? targetHR
            : this.heartRate * 0.7 + targetHR * 0.3;
        this.heartRate = Math.round(Math.max(50, Math.min(180, this.heartRate)));

        // Generate RR interval from HR (with realistic variability)
        const meanRR = 60000 / this.heartRate;
        // HRV decreases with pain: more pain = less variability
        const hrvFactor = Math.max(0.1, 1 - pain * 0.08);
        const rrVariability = 30 * hrvFactor; // ms of variability
        const rr = meanRR + (Math.random() - 0.5) * 2 * rrVariability;

        this._addRR(rr);
        this._emitUpdate();
    }

    // ── Events ───────────────────────────────────────────────

    _emitUpdate() {
        if (this.onUpdate) {
            this.onUpdate({
                hr: this.heartRate,
                hrv: Math.round(this.hrv * 10) / 10,
                rr: this.rrIntervals.length > 0
                    ? this.rrIntervals[this.rrIntervals.length - 1]
                    : 0
            });
        }
    }

    _notify(status) {
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
    }

    /** Get device name */
    getDeviceName() {
        if (this.simulating) return 'Simulated Sensor';
        if (this.device) return this.device.name || 'BLE HR Sensor';
        return '';
    }

    /** Check if active (connected or simulating) */
    isActive() {
        return this.connected || this.simulating;
    }
}
