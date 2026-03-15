/**
 * Session History Chart
 *
 * Canvas-based line chart that records and displays pain scores over time.
 * Recording starts after calibration and runs until stopped.
 */

class SessionChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // Data
        this.samples = [];       // { time: seconds, score: 0-10 }
        this.isRecording = false;
        this.startTime = null;
        this.maxVisibleSeconds = 120; // 2 minutes visible window, scrolls

        // Layout
        this.padding = { top: 30, right: 20, bottom: 40, left: 50 };

        // Animation
        this._animFrame = null;
        this._startRender();
    }

    startRecording() {
        this.samples = [];
        this.startTime = performance.now();
        this.isRecording = true;
    }

    stopRecording() {
        this.isRecording = false;
        this.stoppedElapsed = this.getElapsedTime();
    }

    resetRecording() {
        this.samples = [];
        this.startTime = null;
        this.stoppedElapsed = null;
        this.isRecording = false;
        this.chartTitle = null;
    }

    addSample(score) {
        if (!this.isRecording || this.startTime === null) return;
        const elapsed = (performance.now() - this.startTime) / 1000;
        this.samples.push({ time: elapsed, score });
    }

    /** Load historical samples from backend (array of { timestamp_ms, score }) */
    loadSamples(samples, title) {
        this.isRecording = false;
        this.startTime = null;
        this.stoppedElapsed = null;
        this.chartTitle = title || 'Session Pain History';
        this.samples = samples.map(s => ({
            time: s.timestamp_ms / 1000,
            score: s.score,
        }));
        if (this.samples.length > 0) {
            this.stoppedElapsed = this.samples[this.samples.length - 1].time;
        }
    }

    getElapsedTime() {
        if (!this.startTime) return 0;
        if (!this.isRecording && this.stoppedElapsed != null) return this.stoppedElapsed;
        return (performance.now() - this.startTime) / 1000;
    }

    // ── Rendering ─────────────────────────────────────────────────

    _resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this._displayWidth = rect.width;
        this._displayHeight = rect.height;
    }

    _startRender() {
        const render = () => {
            this._resizeCanvas();
            this._draw();
            this._animFrame = requestAnimationFrame(render);
        };
        render();
    }

    _draw() {
        const ctx = this.ctx;
        const w = this._displayWidth || this.canvas.width;
        const h = this._displayHeight || this.canvas.height;
        const p = this.padding;

        ctx.clearRect(0, 0, w, h);

        // Chart area
        const chartX = p.left;
        const chartY = p.top;
        const chartW = w - p.left - p.right;
        const chartH = h - p.top - p.bottom;

        // Background
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, w, h);

        // Pain zone bands (horizontal)
        const zones = [
            { from: 0, to: 2, color: 'rgba(255,255,255,0.8)', label: '' },
            { from: 2, to: 4, color: 'rgba(255,220,220,0.5)', label: '' },
            { from: 4, to: 6, color: 'rgba(255,180,180,0.5)', label: '' },
            { from: 6, to: 8, color: 'rgba(255,120,120,0.4)', label: '' },
            { from: 8, to: 10, color: 'rgba(255,60,60,0.3)', label: '' },
        ];

        for (const zone of zones) {
            const y1 = chartY + chartH - (zone.to / 10) * chartH;
            const y2 = chartY + chartH - (zone.from / 10) * chartH;
            ctx.fillStyle = zone.color;
            ctx.fillRect(chartX, y1, chartW, y2 - y1);
        }

        // Time range
        const elapsed = this.getElapsedTime();
        const maxTime = Math.max(this.maxVisibleSeconds, elapsed);
        const timeStart = elapsed > this.maxVisibleSeconds ? elapsed - this.maxVisibleSeconds : 0;
        const timeEnd = Math.max(this.maxVisibleSeconds, elapsed);

        // Grid lines & Y-axis labels
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#888';
        ctx.font = '12px "Segoe UI", Arial, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        for (let i = 0; i <= 10; i += 2) {
            const y = chartY + chartH - (i / 10) * chartH;
            ctx.beginPath();
            ctx.moveTo(chartX, y);
            ctx.lineTo(chartX + chartW, y);
            ctx.stroke();
            ctx.fillText(i.toString(), chartX - 8, y);
        }

        // X-axis time labels
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const timeSpan = timeEnd - timeStart;
        const timeStep = timeSpan <= 60 ? 10 : timeSpan <= 120 ? 15 : 30;

        for (let t = Math.ceil(timeStart / timeStep) * timeStep; t <= timeEnd; t += timeStep) {
            const x = chartX + ((t - timeStart) / timeSpan) * chartW;
            ctx.beginPath();
            ctx.moveTo(x, chartY);
            ctx.lineTo(x, chartY + chartH);
            ctx.strokeStyle = '#e0e0e0';
            ctx.stroke();
            ctx.fillStyle = '#888';
            ctx.fillText(this._formatTime(t), x, chartY + chartH + 6);
        }

        // Plot data
        const visibleSamples = this.samples.filter(s => s.time >= timeStart && s.time <= timeEnd);

        if (visibleSamples.length > 1) {
            // Filled area under curve
            ctx.beginPath();
            const firstX = chartX + ((visibleSamples[0].time - timeStart) / timeSpan) * chartW;
            ctx.moveTo(firstX, chartY + chartH);

            for (const s of visibleSamples) {
                const x = chartX + ((s.time - timeStart) / timeSpan) * chartW;
                const y = chartY + chartH - (s.score / 10) * chartH;
                ctx.lineTo(x, y);
            }

            const lastX = chartX + ((visibleSamples[visibleSamples.length - 1].time - timeStart) / timeSpan) * chartW;
            ctx.lineTo(lastX, chartY + chartH);
            ctx.closePath();

            const gradient = ctx.createLinearGradient(0, chartY, 0, chartY + chartH);
            gradient.addColorStop(0, 'rgba(220, 50, 50, 0.3)');
            gradient.addColorStop(1, 'rgba(220, 50, 50, 0.02)');
            ctx.fillStyle = gradient;
            ctx.fill();

            // Line
            ctx.beginPath();
            for (let i = 0; i < visibleSamples.length; i++) {
                const s = visibleSamples[i];
                const x = chartX + ((s.time - timeStart) / timeSpan) * chartW;
                const y = chartY + chartH - (s.score / 10) * chartH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = '#c0392b';
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.stroke();

            // Current value dot
            const last = visibleSamples[visibleSamples.length - 1];
            const dotX = chartX + ((last.time - timeStart) / timeSpan) * chartW;
            const dotY = chartY + chartH - (last.score / 10) * chartH;
            ctx.beginPath();
            ctx.arc(dotX, dotY, 5, 0, 2 * Math.PI);
            ctx.fillStyle = '#c0392b';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Chart border
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.strokeRect(chartX, chartY, chartW, chartH);

        // Title
        ctx.fillStyle = '#333';
        ctx.font = '14px "Segoe UI", Arial, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(this.chartTitle || 'Session Pain History', chartX, 8);

        // Stats (if recording)
        if (this.samples.length > 0) {
            const scores = this.samples.map(s => s.score);
            const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
            const max = Math.max(...scores);
            const statsText = `Avg: ${avg.toFixed(1)}  |  Peak: ${max.toFixed(1)}  |  Duration: ${this._formatTime(elapsed)}`;
            ctx.textAlign = 'right';
            ctx.fillStyle = '#888';
            ctx.font = '12px "Segoe UI", Arial, sans-serif';
            ctx.fillText(statsText, chartX + chartW, 10);
        }

        // Y-axis label
        ctx.save();
        ctx.translate(14, chartY + chartH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#888';
        ctx.font = '12px "Segoe UI", Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Pain Level', 0, 0);
        ctx.restore();

        // No data message
        if (this.samples.length === 0) {
            ctx.fillStyle = '#aaa';
            ctx.font = '16px "Segoe UI", Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(
                this.isRecording ? 'Waiting for data...' : 'Start a session or select one from History',
                chartX + chartW / 2,
                chartY + chartH / 2
            );
        }
    }

    _formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
}
