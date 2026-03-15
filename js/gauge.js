/**
 * Pain Gauge Visualization
 *
 * Draws a semicircular gauge on a canvas element.
 * Color gradient: white (0) → deep red (10).
 * Smooth needle animation with digital readout.
 */

class PainGauge {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.startAngle = Math.PI;
        this.endAngle = 2 * Math.PI;

        // Animation
        this.displayScore = 0;
        this.targetScore = 0;
        this.animationSpeed = 0.12;

        // Responsive sizing
        this._resize();
        this._resizeObserver = new ResizeObserver(() => this._resize());
        this._resizeObserver.observe(this.canvas.parentElement);

        // Start render loop
        this._animate();
    }

    _resize() {
        const parent = this.canvas.parentElement;
        const w = Math.min(400, parent ? parent.clientWidth : 400);
        const h = Math.round(w * 0.7);
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = w * dpr;
        this.canvas.height = h * dpr;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Scale factor relative to reference width of 400
        this.s = w / 400;

        // Recompute geometry based on size
        this.cx = w / 2;
        this.cy = h - (h * 0.14);
        this.radius = w * 0.375;
    }

    /**
     * Set the target pain score (0–10). The needle animates to it.
     */
    setScore(score) {
        this.targetScore = Math.max(0, Math.min(10, score));
    }

    // ── Color mapping ─────────────────────────────────────────────

    /** Map a value 0–10 to a white→red color */
    static painColor(value) {
        const t = Math.max(0, Math.min(1, value / 10));
        const r = 255;
        const g = Math.round(255 * (1 - t));
        const b = Math.round(255 * (1 - t));
        return `rgb(${r}, ${g}, ${b})`;
    }

    /** Darker variant for the arc border */
    static painColorDark(value) {
        const t = Math.max(0, Math.min(1, value / 10));
        const r = Math.round(200 + 55 * t);
        const g = Math.round(200 * (1 - t));
        const b = Math.round(200 * (1 - t));
        return `rgb(${r}, ${g}, ${b})`;
    }

    // ── Drawing ───────────────────────────────────────────────────

    _drawArc() {
        const ctx = this.ctx;
        const s = this.s;
        const segments = 100;
        const arcWidth = 30 * s;
        const half = arcWidth / 2;

        for (let i = 0; i < segments; i++) {
            const t1 = i / segments;
            const t2 = (i + 1) / segments;
            const a1 = this.startAngle + (this.endAngle - this.startAngle) * t1;
            const a2 = this.startAngle + (this.endAngle - this.startAngle) * t2;
            const value = t1 * 10;

            ctx.beginPath();
            ctx.arc(this.cx, this.cy, this.radius, a1, a2);
            ctx.strokeStyle = PainGauge.painColor(value);
            ctx.lineWidth = arcWidth;
            ctx.lineCap = 'butt';
            ctx.stroke();
        }

        // Outer border
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, this.radius + half, this.startAngle, this.endAngle, false);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Inner border
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, this.radius - half, this.startAngle, this.endAngle, false);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    _drawTicks() {
        const ctx = this.ctx;
        const s = this.s;

        for (let i = 0; i <= 10; i++) {
            const t = i / 10;
            const angle = this.startAngle + (this.endAngle - this.startAngle) * t;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);

            // Major tick
            const innerR = this.radius - 20 * s;
            const outerR = this.radius + 20 * s;
            ctx.beginPath();
            ctx.moveTo(this.cx + innerR * cos, this.cy + innerR * sin);
            ctx.lineTo(this.cx + outerR * cos, this.cy + outerR * sin);
            ctx.strokeStyle = '#444';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Label
            const labelR = this.radius + 32 * s;
            ctx.fillStyle = '#333';
            ctx.font = `${Math.round(14 * s)}px "Segoe UI", Arial, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(i.toString(), this.cx + labelR * cos, this.cy + labelR * sin);

            // Minor ticks (half-points)
            if (i < 10) {
                const halfT = (i + 0.5) / 10;
                const halfAngle = this.startAngle + (this.endAngle - this.startAngle) * halfT;
                const hCos = Math.cos(halfAngle);
                const hSin = Math.sin(halfAngle);
                ctx.beginPath();
                ctx.moveTo(this.cx + (this.radius - 17 * s) * hCos, this.cy + (this.radius - 17 * s) * hSin);
                ctx.lineTo(this.cx + (this.radius + 17 * s) * hCos, this.cy + (this.radius + 17 * s) * hSin);
                ctx.strokeStyle = '#999';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }

    _drawNeedle(score) {
        const ctx = this.ctx;
        const s = this.s;
        const t = score / 10;
        const angle = this.startAngle + (this.endAngle - this.startAngle) * t;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        // Needle shadow
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.3)';
        ctx.shadowBlur = 6 * s;
        ctx.shadowOffsetX = 2 * s;
        ctx.shadowOffsetY = 2 * s;

        // Needle body
        ctx.beginPath();
        ctx.moveTo(this.cx + (this.radius - 25 * s) * cos, this.cy + (this.radius - 25 * s) * sin);
        const perpCos = Math.cos(angle + Math.PI / 2);
        const perpSin = Math.sin(angle + Math.PI / 2);
        const needleHalf = 6 * s;
        ctx.lineTo(this.cx + needleHalf * perpCos, this.cy + needleHalf * perpSin);
        ctx.lineTo(this.cx - needleHalf * perpCos, this.cy - needleHalf * perpSin);
        ctx.closePath();
        ctx.fillStyle = '#222';
        ctx.fill();

        ctx.restore();

        // Center cap
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, 10 * s, 0, 2 * Math.PI);
        ctx.fillStyle = '#444';
        ctx.fill();
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    _drawDigitalReadout(score) {
        const ctx = this.ctx;
        const s = this.s;

        // Score number
        ctx.fillStyle = PainGauge.painColor(score);
        ctx.font = `bold ${Math.round(48 * s)}px "Segoe UI", Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(score.toFixed(1), this.cx, this.cy - 50 * s);

        // Label
        ctx.fillStyle = '#666';
        ctx.font = `${Math.round(14 * s)}px "Segoe UI", Arial, sans-serif`;
        ctx.fillText('PAIN LEVEL', this.cx, this.cy - 80 * s);

        // Severity text
        const severity = PainGauge.getSeverityLabel(score);
        ctx.fillStyle = PainGauge.painColor(score);
        ctx.font = `${Math.round(16 * s)}px "Segoe UI", Arial, sans-serif`;
        ctx.fillText(severity, this.cx, this.cy - 20 * s);
    }

    static getSeverityLabel(score) {
        if (score < 0.5) return 'No Pain';
        if (score < 2) return 'Minimal';
        if (score < 4) return 'Mild';
        if (score < 6) return 'Moderate';
        if (score < 8) return 'Severe';
        return 'Very Severe';
    }

    // ── Animation loop ────────────────────────────────────────────

    _animate() {
        // Smooth interpolation toward target
        this.displayScore += (this.targetScore - this.displayScore) * this.animationSpeed;

        // Clear and redraw
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this._drawArc();
        this._drawTicks();
        this._drawNeedle(this.displayScore);
        this._drawDigitalReadout(this.displayScore);

        requestAnimationFrame(() => this._animate());
    }
}
