/**
 * Session History Chart
 *
 * Canvas-based line chart that records and displays pain scores over time.
 * Supports multiple lines with clickable legend toggling and point inspection.
 */

class SessionChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // Data series
        this.series = {
            total:  { data: [], label: 'Total',     color: '#1a1a2e', dash: [],     lineWidth: 3,   visible: true },
            face:   { data: [], label: 'Face',       color: '#8e44ad', dash: [],     lineWidth: 2,   visible: false },
            au:     { data: [], label: 'AU Engine',  color: '#2980b9', dash: [5, 4], lineWidth: 1.5, visible: false },
            cnn:    { data: [], label: 'CNN',        color: '#c0392b', dash: [5, 4], lineWidth: 1.5, visible: false },
            body:   { data: [], label: 'Body',       color: '#27ae60', dash: [],     lineWidth: 2,   visible: false },
            hr:     { data: [], label: 'HR Pain',    color: '#e67e22', dash: [],     lineWidth: 2,   visible: false },
        };
        // HR BPM uses secondary Y-axis, tracked separately
        this.hrBpmSamples = [];

        // Frames for point inspection
        this.frameSamples = []; // { time, frame (dataURL) }

        this.isRecording = false;
        this.startTime = null;
        this.maxVisibleSeconds = 120;

        // Layout
        this.padding = { top: 30, right: 50, bottom: 40, left: 50 };

        // Zoom
        this.zoomLevels = [30, 60, 120, 300, 600];
        this.zoomIndex = 2;
        this.maxVisibleSeconds = this.zoomLevels[this.zoomIndex];

        // Legend interaction
        this._legendHitAreas = []; // { key, x, y, w, h }

        // Point inspection
        this.selectedPoint = null; // { seriesKey, index, x, y, time, score }
        this.onPointSelected = null; // callback({ time, score, frame })

        // Click handler
        this.canvas.addEventListener('click', (e) => this._handleClick(e));

        // Cursor handler — pointer over legend checkboxes, crosshair elsewhere
        this.canvas.addEventListener('mousemove', (e) => this._handleMouseMove(e));

        // Animation
        this._animFrame = null;
        this._startRender();
    }

    // ── Recording lifecycle ─────────────────────────────────────

    startRecording() {
        for (const s of Object.values(this.series)) s.data = [];
        this.hrBpmSamples = [];
        this.frameSamples = [];
        this.startTime = performance.now();
        this.isRecording = true;
        this.selectedPoint = null;
        this._resetVisibility();
    }

    stopRecording() {
        this.isRecording = false;
        this.stoppedElapsed = this.getElapsedTime();
    }

    resetRecording() {
        for (const s of Object.values(this.series)) s.data = [];
        this.hrBpmSamples = [];
        this.frameSamples = [];
        this.startTime = null;
        this.stoppedElapsed = null;
        this.isRecording = false;
        this.chartTitle = null;
        this.selectedPoint = null;
        this._resetVisibility();
    }

    _resetVisibility() {
        for (const [key, s] of Object.entries(this.series)) {
            s.visible = (key === 'total');
        }
    }

    // ── Data input ──────────────────────────────────────────────

    addSample(totalScore, faceScore, auScore, bodyScore, hrPainScore, frame) {
        if (!this.isRecording || this.startTime === null) return;
        const time = (performance.now() - this.startTime) / 1000;

        this.series.total.data.push({ time, score: totalScore });
        if (faceScore != null) this.series.face.data.push({ time, score: faceScore });
        if (auScore != null) this.series.au.data.push({ time, score: auScore });
        if (bodyScore != null) this.series.body.data.push({ time, score: bodyScore });
        if (hrPainScore != null) this.series.hr.data.push({ time, score: hrPainScore });

        if (frame) this.frameSamples.push({ time, frame });
    }

    addCnnSample(time, score) {
        this.series.cnn.data.push({ time, score });
    }

    addHrBpmSample(bpm) {
        if (!this.isRecording || this.startTime === null) return;
        const time = (performance.now() - this.startTime) / 1000;
        this.hrBpmSamples.push({ time, bpm });
    }

    // Legacy compatibility
    addAiSample(time, score) { this.addCnnSample(time, score); }
    addHrSample(bpm) { this.addHrBpmSample(bpm); }
    addBodySample(_score) { /* now handled via addSample */ }

    /** Load historical samples from backend */
    loadSamples(samples, title) {
        this.isRecording = false;
        this.startTime = null;
        this.stoppedElapsed = null;
        this.chartTitle = title || 'Session Pain History';
        this.selectedPoint = null;

        // Reset all data
        for (const s of Object.values(this.series)) s.data = [];
        this.hrBpmSamples = [];
        this.frameSamples = [];

        for (const s of samples) {
            const time = s.timestamp_ms / 1000;
            if (s.score != null) this.series.total.data.push({ time, score: s.score });
            if (s.ai_score != null) this.series.cnn.data.push({ time, score: s.ai_score });
            if (s.frame) this.frameSamples.push({ time, frame: s.frame });
        }

        if (this.series.total.data.length > 0) {
            this.stoppedElapsed = this.series.total.data[this.series.total.data.length - 1].time;
        }

        // Show total by default for loaded sessions
        this._resetVisibility();
    }

    getElapsedTime() {
        if (!this.startTime) return 0;
        if (!this.isRecording && this.stoppedElapsed != null) return this.stoppedElapsed;
        return (performance.now() - this.startTime) / 1000;
    }

    // ── Click handling ──────────────────────────────────────────

    _handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Check legend hits first
        for (const hit of this._legendHitAreas) {
            if (mx >= hit.x && mx <= hit.x + hit.w && my >= hit.y && my <= hit.y + hit.h) {
                this.series[hit.key].visible = !this.series[hit.key].visible;
                return;
            }
        }

        // Check data point proximity
        const w = this._displayWidth || this.canvas.width;
        const h = this._displayHeight || this.canvas.height;
        const p = this.padding;
        const chartX = p.left;
        const chartY = p.top;
        const chartW = w - p.left - p.right;
        const chartH = h - p.top - p.bottom;

        // Only check within chart area
        if (mx < chartX || mx > chartX + chartW || my < chartY || my > chartY + chartH) {
            this.selectedPoint = null;
            return;
        }

        const elapsed = this.getElapsedTime();
        const timeStart = elapsed > this.maxVisibleSeconds ? elapsed - this.maxVisibleSeconds : 0;
        const timeEnd = Math.max(this.maxVisibleSeconds, elapsed);
        const timeSpan = timeEnd - timeStart;

        let closest = null;
        let closestDist = 15; // max pixel distance to snap

        for (const [key, series] of Object.entries(this.series)) {
            if (!series.visible) continue;
            for (let i = 0; i < series.data.length; i++) {
                const s = series.data[i];
                if (s.time < timeStart || s.time > timeEnd) continue;
                const px = chartX + ((s.time - timeStart) / timeSpan) * chartW;
                const py = chartY + chartH - (s.score / 10) * chartH;
                const dist = Math.sqrt((mx - px) ** 2 + (my - py) ** 2);
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = { seriesKey: key, index: i, x: px, y: py, time: s.time, score: s.score };
                }
            }
        }

        this.selectedPoint = closest;

        if (closest && this.onPointSelected) {
            // Find nearest frame
            const frame = this._findNearestFrame(closest.time);
            this.onPointSelected({
                time: closest.time,
                score: closest.score,
                seriesKey: closest.seriesKey,
                label: this.series[closest.seriesKey].label,
                frame,
            });
        } else if (!closest && this.onPointSelected) {
            this.onPointSelected(null);
        }
    }

    _findNearestFrame(time) {
        if (this.frameSamples.length === 0) return null;
        let best = this.frameSamples[0];
        let bestDist = Math.abs(best.time - time);
        for (const f of this.frameSamples) {
            const d = Math.abs(f.time - time);
            if (d < bestDist) { best = f; bestDist = d; }
        }
        // Only return if within 2 seconds
        return bestDist < 2 ? best.frame : null;
    }

    // ── Rendering ───────────────────────────────────────────────

    _resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) return false;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this._displayWidth = rect.width;
        this._displayHeight = rect.height;
        return true;
    }

    _startRender() {
        const render = () => {
            if (this._resizeCanvas()) {
                this._draw();
            }
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

        const chartX = p.left;
        const chartY = p.top;
        const chartW = w - p.left - p.right;
        const chartH = h - p.top - p.bottom;

        // Background
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, w, h);

        // Pain zone bands
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
        const timeSpan = timeEnd - timeStart;

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

        // Draw filled area under total score
        if (this.series.total.visible) {
            const vis = this.series.total.data.filter(s => s.time >= timeStart && s.time <= timeEnd);
            if (vis.length > 1) {
                ctx.beginPath();
                ctx.moveTo(chartX + ((vis[0].time - timeStart) / timeSpan) * chartW, chartY + chartH);
                for (const s of vis) {
                    ctx.lineTo(
                        chartX + ((s.time - timeStart) / timeSpan) * chartW,
                        chartY + chartH - (s.score / 10) * chartH
                    );
                }
                ctx.lineTo(chartX + ((vis[vis.length - 1].time - timeStart) / timeSpan) * chartW, chartY + chartH);
                ctx.closePath();
                const gradient = ctx.createLinearGradient(0, chartY, 0, chartY + chartH);
                gradient.addColorStop(0, 'rgba(26, 26, 46, 0.15)');
                gradient.addColorStop(1, 'rgba(26, 26, 46, 0.02)');
                ctx.fillStyle = gradient;
                ctx.fill();
            }
        }

        // Draw each visible series
        const drawOrder = ['au', 'cnn', 'face', 'body', 'hr', 'total'];
        for (const key of drawOrder) {
            const series = this.series[key];
            if (!series.visible || series.data.length < 2) continue;
            const vis = series.data.filter(s => s.time >= timeStart && s.time <= timeEnd);
            if (vis.length < 2) continue;
            this._drawLine(ctx, vis, chartX, chartY, chartW, chartH, timeStart, timeSpan, series);
        }

        // HR BPM on secondary axis (only if hr pain line is visible)
        const hasHRBpm = this.hrBpmSamples.length > 0;
        if (hasHRBpm && this.series.hr.visible) {
            const hrMin = 40, hrMax = 180;
            const visHR = this.hrBpmSamples.filter(s => s.time >= timeStart && s.time <= timeEnd);
            if (visHR.length > 1) {
                ctx.beginPath();
                for (let i = 0; i < visHR.length; i++) {
                    const s = visHR[i];
                    const x = chartX + ((s.time - timeStart) / timeSpan) * chartW;
                    const y = chartY + chartH - ((s.bpm - hrMin) / (hrMax - hrMin)) * chartH;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.strokeStyle = 'rgba(230, 126, 34, 0.3)';
                ctx.lineWidth = 1;
                ctx.lineJoin = 'round';
                ctx.setLineDash([3, 3]);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Right Y-axis labels for BPM
            ctx.fillStyle = 'rgba(230, 126, 34, 0.5)';
            ctx.font = '10px "Segoe UI", Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            for (let bpm = 40; bpm <= 180; bpm += 20) {
                const y = chartY + chartH - ((bpm - hrMin) / (hrMax - hrMin)) * chartH;
                ctx.fillText(bpm.toString(), chartX + chartW + 6, y);
            }
            ctx.save();
            ctx.translate(w - 4, chartY + chartH / 2);
            ctx.rotate(Math.PI / 2);
            ctx.fillStyle = 'rgba(230, 126, 34, 0.5)';
            ctx.font = '10px "Segoe UI", Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('BPM', 0, 0);
            ctx.restore();
        }

        // Selected point highlight
        if (this.selectedPoint) {
            const sp = this.selectedPoint;
            const series = this.series[sp.seriesKey];
            if (series && sp.index < series.data.length) {
                const s = series.data[sp.index];
                const px = chartX + ((s.time - timeStart) / timeSpan) * chartW;
                const py = chartY + chartH - (s.score / 10) * chartH;

                // Vertical crosshair
                ctx.strokeStyle = 'rgba(0,0,0,0.2)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(px, chartY);
                ctx.lineTo(px, chartY + chartH);
                ctx.stroke();
                ctx.setLineDash([]);

                // Larger highlighted dot
                ctx.beginPath();
                ctx.arc(px, py, 7, 0, 2 * Math.PI);
                ctx.fillStyle = series.color;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2.5;
                ctx.stroke();

                // Value tooltip
                const tooltipText = `${series.label}: ${s.score.toFixed(1)} @ ${this._formatTime(s.time)}`;
                ctx.font = 'bold 11px "Segoe UI", Arial, sans-serif';
                const tw = ctx.measureText(tooltipText).width + 12;
                const tx = Math.min(px - tw / 2, chartX + chartW - tw);
                const ty = py - 28;
                ctx.fillStyle = 'rgba(0,0,0,0.8)';
                this._roundRect(ctx, Math.max(chartX, tx), ty, tw, 20, 4);
                ctx.fill();
                ctx.fillStyle = '#fff';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(tooltipText, Math.max(chartX + tw / 2, tx + tw / 2), ty + 10);
            }
        }

        // Chart border
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        ctx.strokeRect(chartX, chartY, chartW, chartH);

        // Title
        ctx.fillStyle = '#333';
        ctx.font = '14px "Segoe UI", Arial, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(this.chartTitle || 'Session Pain History', chartX, 8);

        // Interactive legend
        this._drawLegend(ctx, chartX, chartW, w);

        // Stats bar
        this._drawStats(ctx, chartX, chartY, chartW, chartH, elapsed);

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
        if (this.series.total.data.length === 0) {
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

    _drawLine(ctx, data, chartX, chartY, chartW, chartH, timeStart, timeSpan, series) {
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const s = data[i];
            const x = chartX + ((s.time - timeStart) / timeSpan) * chartW;
            const y = chartY + chartH - (s.score / 10) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = series.color;
        ctx.lineWidth = series.lineWidth;
        ctx.lineJoin = 'round';
        ctx.setLineDash(series.dash);
        ctx.stroke();
        ctx.setLineDash([]);

        // Current value dot
        const last = data[data.length - 1];
        const dotX = chartX + ((last.time - timeStart) / timeSpan) * chartW;
        const dotY = chartY + chartH - (last.score / 10) * chartH;
        ctx.beginPath();
        ctx.arc(dotX, dotY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = series.color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    _handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        for (const hit of this._legendHitAreas) {
            if (mx >= hit.x && mx <= hit.x + hit.w && my >= hit.y && my <= hit.y + hit.h) {
                this.canvas.style.cursor = 'pointer';
                return;
            }
        }
        this.canvas.style.cursor = 'crosshair';
    }

    _drawLegend(ctx, chartX, chartW, w) {
        this._legendHitAreas = [];
        const legendY = 6;
        const itemH = 16;
        const checkSize = 10;
        ctx.font = '11px "Segoe UI", Arial, sans-serif';

        // Measure total legend width: checkbox(10) + gap(5) + lineSwath(18) + gap(4) + label
        const items = Object.entries(this.series);
        const itemWidths = items.map(([, s]) => checkSize + 5 + 18 + 4 + ctx.measureText(s.label).width);
        const totalWidth = itemWidths.reduce((a, b) => a + b + 10, -10);
        let legendX = Math.max(chartX + 120, chartX + chartW - totalWidth);

        for (let i = 0; i < items.length; i++) {
            const [key, series] = items[i];
            const labelW = ctx.measureText(series.label).width;
            const itemW = checkSize + 5 + 18 + 4 + labelW;

            // Hit area
            this._legendHitAreas.push({ key, x: legendX - 2, y: legendY - 2, w: itemW + 4, h: itemH + 4 });

            const alpha = series.visible ? 1 : 0.35;
            ctx.globalAlpha = alpha;

            // Checkbox
            const cbX = legendX;
            const cbY = legendY + 2;
            ctx.strokeStyle = series.visible ? series.color : '#aaa';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
            ctx.strokeRect(cbX, cbY, checkSize, checkSize);

            if (series.visible) {
                // Checkmark
                ctx.strokeStyle = series.color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(cbX + 2, cbY + 5);
                ctx.lineTo(cbX + 4, cbY + 8);
                ctx.lineTo(cbX + 8, cbY + 2);
                ctx.stroke();
            }

            // Line sample swatch
            const swatchStartX = cbX + checkSize + 5;
            ctx.strokeStyle = series.color;
            ctx.lineWidth = series.lineWidth;
            ctx.setLineDash(series.dash);
            ctx.beginPath();
            ctx.moveTo(swatchStartX, legendY + 7);
            ctx.lineTo(swatchStartX + 18, legendY + 7);
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = series.visible ? '#333' : '#aaa';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(series.label, swatchStartX + 22, legendY);

            ctx.globalAlpha = 1;
            legendX += itemW + 10;
        }
    }

    _drawStats(ctx, chartX, chartY, chartW, chartH, elapsed) {
        const parts = [];

        if (this.series.total.data.length > 0) {
            const scores = this.series.total.data.map(s => s.score);
            const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
            const max = Math.max(...scores);
            parts.push(`Avg: ${avg.toFixed(1)}  Peak: ${max.toFixed(1)}`);
        }

        parts.push(this._formatTime(elapsed));

        ctx.textAlign = 'right';
        ctx.fillStyle = '#888';
        ctx.font = '11px "Segoe UI", Arial, sans-serif';
        ctx.fillText(parts.join('  |  '), chartX + chartW, chartY + chartH + 24);
    }

    _roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    // ── Zoom ────────────────────────────────────────────────────

    zoomIn() {
        if (this.zoomIndex > 0) {
            this.zoomIndex--;
            this.maxVisibleSeconds = this.zoomLevels[this.zoomIndex];
        }
    }

    zoomOut() {
        if (this.zoomIndex < this.zoomLevels.length - 1) {
            this.zoomIndex++;
            this.maxVisibleSeconds = this.zoomLevels[this.zoomIndex];
        }
    }

    getZoomLabel() {
        const s = this.maxVisibleSeconds;
        return s >= 60 ? `${s / 60}m` : `${s}s`;
    }

    _formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
}
