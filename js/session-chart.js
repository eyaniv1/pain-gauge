/**
 * Session History Chart
 *
 * Canvas-based line chart that records and displays pain scores over time.
 * Supports dual lines: PSPI (blue) and AI score (red) with legend.
 * Recording starts after calibration and runs until stopped.
 */

class SessionChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // Data
        this.samples = [];       // { time: seconds, score: 0-10 }
        this.aiSamples = [];     // { time: seconds, score: 0-10 } — AI scores
        this.isRecording = false;
        this.startTime = null;
        this.maxVisibleSeconds = 120; // 2 minutes visible window, scrolls

        // Layout
        this.padding = { top: 30, right: 20, bottom: 40, left: 50 };

        // Colors
        this.pspiColor = '#2980b9';   // blue for PSPI
        this.aiColor = '#c0392b';     // red for AI

        // Animation
        this._animFrame = null;
        this._startRender();
    }

    startRecording() {
        this.samples = [];
        this.aiSamples = [];
        this.startTime = performance.now();
        this.isRecording = true;
    }

    stopRecording() {
        this.isRecording = false;
        this.stoppedElapsed = this.getElapsedTime();
    }

    resetRecording() {
        this.samples = [];
        this.aiSamples = [];
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

    /** Load historical samples from backend (array of { timestamp_ms, score, ai_score }) */
    loadSamples(samples, title) {
        this.isRecording = false;
        this.startTime = null;
        this.stoppedElapsed = null;
        this.chartTitle = title || 'Session Pain History';
        this.samples = samples.map(s => ({
            time: s.timestamp_ms / 1000,
            score: s.score,
        }));
        // Extract AI samples (only where ai_score exists)
        this.aiSamples = samples
            .filter(s => s.ai_score != null)
            .map(s => ({
                time: s.timestamp_ms / 1000,
                score: s.ai_score,
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
            { from: 0, to: 2, color: 'rgba(255,255,255,0.8)' },
            { from: 2, to: 4, color: 'rgba(255,220,220,0.5)' },
            { from: 4, to: 6, color: 'rgba(255,180,180,0.5)' },
            { from: 6, to: 8, color: 'rgba(255,120,120,0.4)' },
            { from: 8, to: 10, color: 'rgba(255,60,60,0.3)' },
        ];

        for (const zone of zones) {
            const y1 = chartY + chartH - (zone.to / 10) * chartH;
            const y2 = chartY + chartH - (zone.from / 10) * chartH;
            ctx.fillStyle = zone.color;
            ctx.fillRect(chartX, y1, chartW, y2 - y1);
        }

        // Time range
        const elapsed = this.getElapsedTime();
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

        // Plot PSPI data (blue)
        const visibleSamples = this.samples.filter(s => s.time >= timeStart && s.time <= timeEnd);
        const hasAI = this.aiSamples.length > 0;
        const visibleAI = hasAI ? this.aiSamples.filter(s => s.time >= timeStart && s.time <= timeEnd) : [];

        if (visibleSamples.length > 1) {
            // Filled area under PSPI curve
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
            gradient.addColorStop(0, 'rgba(41, 128, 185, 0.2)');
            gradient.addColorStop(1, 'rgba(41, 128, 185, 0.02)');
            ctx.fillStyle = gradient;
            ctx.fill();

            // PSPI Line
            this._drawLine(ctx, visibleSamples, chartX, chartY, chartH, timeStart, timeSpan, this.pspiColor);
        }

        // Plot AI data (red) — only when we have AI samples
        if (visibleAI.length > 1) {
            // Filled area under AI curve
            ctx.beginPath();
            const firstAIX = chartX + ((visibleAI[0].time - timeStart) / timeSpan) * chartW;
            ctx.moveTo(firstAIX, chartY + chartH);

            for (const s of visibleAI) {
                const x = chartX + ((s.time - timeStart) / timeSpan) * chartW;
                const y = chartY + chartH - (s.score / 10) * chartH;
                ctx.lineTo(x, y);
            }

            const lastAIX = chartX + ((visibleAI[visibleAI.length - 1].time - timeStart) / timeSpan) * chartW;
            ctx.lineTo(lastAIX, chartY + chartH);
            ctx.closePath();

            const aiGradient = ctx.createLinearGradient(0, chartY, 0, chartY + chartH);
            aiGradient.addColorStop(0, 'rgba(192, 57, 43, 0.15)');
            aiGradient.addColorStop(1, 'rgba(192, 57, 43, 0.02)');
            ctx.fillStyle = aiGradient;
            ctx.fill();

            // AI Line
            this._drawLine(ctx, visibleAI, chartX, chartY, chartH, timeStart, timeSpan, this.aiColor);
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

        // Legend (when AI data present)
        if (hasAI) {
            const legendX = chartX + chartW - 150;
            const legendY = 6;
            ctx.font = '11px "Segoe UI", Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';

            // PSPI legend
            ctx.strokeStyle = this.pspiColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(legendX, legendY + 6);
            ctx.lineTo(legendX + 20, legendY + 6);
            ctx.stroke();
            ctx.fillStyle = '#555';
            ctx.fillText('PSPI', legendX + 24, legendY);

            // AI legend
            ctx.strokeStyle = this.aiColor;
            ctx.beginPath();
            ctx.moveTo(legendX + 65, legendY + 6);
            ctx.lineTo(legendX + 85, legendY + 6);
            ctx.stroke();
            ctx.fillStyle = '#555';
            ctx.fillText('AI', legendX + 89, legendY);
        }

        // Stats
        if (this.samples.length > 0) {
            const scores = this.samples.map(s => s.score);
            const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
            const max = Math.max(...scores);
            let statsText = `PSPI Avg: ${avg.toFixed(1)}  Peak: ${max.toFixed(1)}`;

            if (this.aiSamples.length > 0) {
                const aiScores = this.aiSamples.map(s => s.score);
                const aiAvg = aiScores.reduce((a, b) => a + b, 0) / aiScores.length;
                const aiMax = Math.max(...aiScores);
                statsText += `  |  AI Avg: ${aiAvg.toFixed(1)}  Peak: ${aiMax.toFixed(1)}`;
            }

            statsText += `  |  ${this._formatTime(elapsed)}`;

            ctx.textAlign = 'right';
            ctx.fillStyle = '#888';
            ctx.font = '11px "Segoe UI", Arial, sans-serif';
            ctx.fillText(statsText, chartX + chartW, chartY + chartH + 24);
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

    _drawLine(ctx, data, chartX, chartY, chartH, timeStart, timeSpan, color) {
        const chartW = this._displayWidth - this.padding.left - this.padding.right;
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const s = data[i];
            const x = chartX + ((s.time - timeStart) / timeSpan) * chartW;
            const y = chartY + chartH - (s.score / 10) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.stroke();

        // Current value dot
        const last = data[data.length - 1];
        const dotX = chartX + ((last.time - timeStart) / timeSpan) * chartW;
        const dotY = chartY + chartH - (last.score / 10) * chartH;
        ctx.beginPath();
        ctx.arc(dotX, dotY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    _formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
}
