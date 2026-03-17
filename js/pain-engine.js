/**
 * Pain Scoring Engine (Rules-Based)
 *
 * Computes a pain score (0-10) from MediaPipe Face Mesh landmarks
 * using the PSPI (Prkachin and Solomon Pain Intensity) formula:
 *
 *   PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
 *
 * Each Action Unit is scored 0-5 based on deviation from a baseline
 * (calibrated neutral face or population defaults).
 *
 * This engine sits behind a simple interface so it can be swapped
 * for an ML-based model later without changing the rest of the app.
 */

class PainEngine {
    constructor(options = {}) {
        this.baseline = null;
        this.baselinePainLevel = 0; // patient's reported pain at calibration (0-10)
        this.currentScore = 0;
        this.smoothingWindow = [];
        this.smoothingSize = options.smoothingSize || 10;
        this.lastResult = null;

        // Tunable sensitivity parameters
        this.au4Sensitivity = options.au4Sensitivity ?? 5;
        this.au67Sensitivity = options.au67Sensitivity ?? 4;
        this.au910Sensitivity = options.au910Sensitivity ?? 5;
        this.au43Threshold = options.au43Threshold ?? 0.55;
        this.au43Sensitivity = options.au43Sensitivity ?? 8;
        this.talkingSuppression = options.talkingSuppression ?? 0.5;

        // Multimodal fusion parameters
        this.hrWeight = options.hrWeight ?? 0.3;    // HR contribution
        this.bodyWeight = options.bodyWeight ?? 0.2; // Body motion contribution
        this.hrBaseline = null;  // { hr, hrv } captured during calibration
        this._hrCalibSamples = [];
    }

    // ── MediaPipe Face Mesh landmark indices ──────────────────────

    static L = {
        // Face scale reference
        FOREHEAD: 10,
        CHIN: 152,

        // AU4 — Brow Lowerer
        LEFT_INNER_BROW: 55,
        RIGHT_INNER_BROW: 285,
        NOSE_BRIDGE_TOP: 168,

        // AU6/7 — Cheek Raiser / Lid Tightener (Eye Aspect Ratio)
        // Left eye
        L_EYE_OUTER: 33,
        L_EYE_P2: 160,
        L_EYE_P3: 158,
        L_EYE_INNER: 133,
        L_EYE_P5: 153,
        L_EYE_P6: 144,
        // Right eye
        R_EYE_OUTER: 263,
        R_EYE_P2: 387,
        R_EYE_P3: 385,
        R_EYE_INNER: 362,
        R_EYE_P5: 380,
        R_EYE_P6: 373,

        // AU9/10 — Nose Wrinkler / Upper Lip Raiser
        NOSE_TIP: 4,
        UPPER_LIP_TOP: 13,

        // Mouth (for filtering out talking)
        LOWER_LIP_BOTTOM: 14,
    };

    // ── Geometry helpers ──────────────────────────────────────────

    static dist(a, b) {
        return Math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        );
    }

    // ── Raw measurement functions ─────────────────────────────────

    /** Face height for normalization */
    faceScale(lm) {
        return PainEngine.dist(lm[PainEngine.L.FOREHEAD], lm[PainEngine.L.CHIN]);
    }

    /** AU4: distance from inner brows to nose bridge, normalized */
    measureAU4(lm, scale) {
        const L = PainEngine.L;
        const left = PainEngine.dist(lm[L.LEFT_INNER_BROW], lm[L.NOSE_BRIDGE_TOP]);
        const right = PainEngine.dist(lm[L.RIGHT_INNER_BROW], lm[L.NOSE_BRIDGE_TOP]);
        return ((left + right) / 2) / scale;
    }

    /** AU6/7: Eye Aspect Ratio (average of both eyes) */
    measureEAR(lm) {
        const ear = (side) => {
            const L = PainEngine.L;
            const prefix = side === 'left' ? 'L' : 'R';
            const p1 = lm[L[`${prefix}_EYE_OUTER`]];
            const p2 = lm[L[`${prefix}_EYE_P2`]];
            const p3 = lm[L[`${prefix}_EYE_P3`]];
            const p4 = lm[L[`${prefix}_EYE_INNER`]];
            const p5 = lm[L[`${prefix}_EYE_P5`]];
            const p6 = lm[L[`${prefix}_EYE_P6`]];
            const v1 = PainEngine.dist(p2, p6);
            const v2 = PainEngine.dist(p3, p5);
            const h = PainEngine.dist(p1, p4);
            return (v1 + v2) / (2 * h);
        };
        return (ear('left') + ear('right')) / 2;
    }

    /** AU9/10: distance from nose tip to upper lip, normalized */
    measureAU910(lm, scale) {
        const L = PainEngine.L;
        return PainEngine.dist(lm[L.NOSE_TIP], lm[L.UPPER_LIP_TOP]) / scale;
    }

    /** Mouth openness — used to suppress false positives from talking */
    measureMouthOpen(lm, scale) {
        const L = PainEngine.L;
        return PainEngine.dist(lm[L.UPPER_LIP_TOP], lm[L.LOWER_LIP_BOTTOM]) / scale;
    }

    // ── Calibration ───────────────────────────────────────────────

    /**
     * Capture the subject's neutral face as baseline.
     * Call this when the person is relaxed and not in pain.
     * Averages over multiple frames if called repeatedly.
     */
    calibrate(landmarks) {
        const scale = this.faceScale(landmarks);
        const measurement = {
            au4: this.measureAU4(landmarks, scale),
            ear: this.measureEAR(landmarks),
            au910: this.measureAU910(landmarks, scale),
            mouthOpen: this.measureMouthOpen(landmarks, scale),
        };

        if (!this.baseline) {
            this.baseline = { ...measurement, samples: 1 };
        } else {
            // Running average
            const n = this.baseline.samples;
            this.baseline.au4 = (this.baseline.au4 * n + measurement.au4) / (n + 1);
            this.baseline.ear = (this.baseline.ear * n + measurement.ear) / (n + 1);
            this.baseline.au910 = (this.baseline.au910 * n + measurement.au910) / (n + 1);
            this.baseline.mouthOpen = (this.baseline.mouthOpen * n + measurement.mouthOpen) / (n + 1);
            this.baseline.samples = n + 1;
        }
    }

    /** Capture HR baseline sample during calibration */
    calibrateHR(hr, hrv) {
        if (hr > 0) {
            this._hrCalibSamples.push({ hr, hrv: hrv || 0 });
            // Compute running average
            const n = this._hrCalibSamples.length;
            this.hrBaseline = {
                hr: this._hrCalibSamples.reduce((s, v) => s + v.hr, 0) / n,
                hrv: this._hrCalibSamples.reduce((s, v) => s + v.hrv, 0) / n,
            };
        }
    }

    /**
     * Compute physiological pain contribution (0-10) from HR/HRV.
     *
     * Pain correlates:
     *   - HR elevation above baseline (sympathetic activation)
     *   - HRV suppression below baseline (reduced parasympathetic tone)
     *
     * The baseline pain level is accounted for: if calibrated at pain 5,
     * the captured HR already reflects that pain. We estimate the resting
     * HR by subtracting the expected pain-related elevation, then measure
     * deviation from that estimated rest state.
     *
     * Based on literature: ~3 bpm per pain level, ~8% HRV drop per pain level.
     */
    computeHRPainScore(hr, hrv) {
        if (!this.hrBaseline || this.hrBaseline.hr === 0) return null;

        const base = this.hrBaseline;
        const bpl = this.baselinePainLevel;

        // Estimate true resting values by removing pain-related component
        // ~3 bpm elevation per pain level (literature: 10-30 bpm at pain 10)
        const estimatedRestHR = base.hr - (bpl * 3);
        // ~8% HRV suppression per pain level (literature: ~80% at pain 10)
        const estimatedRestHRV = base.hrv > 0 ? base.hrv / Math.max(0.2, 1 - bpl * 0.08) : 0;

        // HR component: elevation from estimated rest → 0-5 score
        // +30 bpm above rest = score 5 (max contribution)
        const hrElevation = Math.max(0, hr - estimatedRestHR);
        const hrScore = Math.min(5, (hrElevation / 30) * 5);

        // HRV component: suppression from estimated rest → 0-5 score
        // 80% reduction = score 5 (max contribution)
        let hrvScore = 0;
        if (estimatedRestHRV > 5) {
            const hrvDrop = Math.max(0, estimatedRestHRV - hrv) / estimatedRestHRV;
            hrvScore = Math.min(5, (hrvDrop / 0.8) * 5);
        }

        // Combined: 0-10 scale (equal weight HR and HRV)
        return Math.min(10, hrScore + hrvScore);
    }

    resetCalibration() {
        this.baseline = null;
        this.baselinePainLevel = 0;
        this.smoothingWindow = [];
        this.currentScore = 0;
        this.hrBaseline = null;
        this._hrCalibSamples = [];
    }

    isCalibrated() {
        return this.baseline !== null;
    }

    // ── Main scoring ──────────────────────────────────────────────

    /**
     * Process a frame of face landmarks and return a pain result.
     *
     * @param {Array} landmarks - MediaPipe Face Mesh landmarks (468 points)
     * @param {{ hr: number, hrv: number }|null} physio - optional physiological data
     * @param {number|null} bodyPainScore - optional body motion pain score (0-10)
     * @returns {{ score: number, rawScore: number, aus: object, pspi: number, hrPain: number|null, bodyPain: number|null }}
     */
    process(landmarks, physio, bodyPainScore) {
        if (!landmarks || landmarks.length < 468) {
            return this.lastResult || { score: 0, rawScore: 0, aus: { au4: 0, au6_7: 0, au9_10: 0, au43: 0 }, pspi: 0 };
        }

        const scale = this.faceScale(landmarks);

        // Raw measurements
        const rawAU4 = this.measureAU4(landmarks, scale);
        const rawEAR = this.measureEAR(landmarks);
        const rawAU910 = this.measureAU910(landmarks, scale);
        const mouthOpen = this.measureMouthOpen(landmarks, scale);

        // Baseline (calibrated or population defaults)
        const base = this.baseline || {
            au4: 0.18,
            ear: 0.27,
            au910: 0.06,
            mouthOpen: 0.01,
        };

        // ── Score each AU (–5 to +5, signed) ──
        // Positive = more pain than baseline, negative = more relaxed

        // AU4: brow lowers → distance decreases
        const au4Dev = (base.au4 - rawAU4) / base.au4;
        const au4Score = Math.max(-5, Math.min(5, au4Dev * this.au4Sensitivity));

        // AU6/7: eyes tighten → EAR decreases
        const earDev = (base.ear - rawEAR) / base.ear;
        const au6_7Score = Math.max(-5, Math.min(5, earDev * this.au67Sensitivity));

        // AU9/10: nose wrinkle / lip raise → nose-lip distance decreases
        const au910Dev = (base.au910 - rawAU910) / base.au910;
        const au9_10Score = Math.max(-5, Math.min(5, au910Dev * this.au910Sensitivity));

        // AU43: eye closure → EAR near zero (only positive, can't have negative closure)
        let au43Score = 0;
        if (earDev > this.au43Threshold) {
            au43Score = Math.min(5, (earDev - this.au43Threshold) * this.au43Sensitivity);
        }

        // ── Suppress talking false positives ──
        // When mouth is wide open (talking), reduce AU9/10 contribution
        const mouthDev = (mouthOpen - (base.mouthOpen || 0.01)) / 0.05;
        const talkSuppression = Math.max(0, Math.min(1, 1 - mouthDev * this.talkingSuppression));
        const adjustedAU910 = au9_10Score * talkSuppression;

        // ── PSPI formula (signed — can be negative if face relaxes) ──
        const pspi = au4Score + au6_7Score + adjustedAU910 + au43Score;

        // Map PSPI to a deviation from baseline: ±15 maps to ±10
        const deviationPain = Math.max(-10, Math.min(10, (pspi / 15) * 10));

        // Apply deviation bidirectionally from baseline pain level
        // Positive deviation scales into remaining range above baseline
        // Negative deviation scales into range below baseline
        let facePain;
        if (deviationPain >= 0) {
            const remainingRange = 10 - this.baselinePainLevel;
            facePain = this.baselinePainLevel + (deviationPain / 10) * remainingRange;
        } else {
            facePain = this.baselinePainLevel + (deviationPain / 10) * this.baselinePainLevel;
        }
        facePain = Math.max(0, Math.min(10, facePain));

        // ── Multimodal fusion ──
        // Blend face score with HR and body motion scores when available
        let hrPainScore = null;
        let rawPain = facePain;

        if (physio && this.hrBaseline) {
            hrPainScore = this.computeHRPainScore(physio.hr, physio.hrv);
        }

        // Compute weighted fusion: normalize weights for active modalities
        const activeWeights = { face: 1 };
        if (hrPainScore !== null && this.hrWeight > 0) activeWeights.hr = this.hrWeight;
        if (bodyPainScore !== null && this.bodyWeight > 0) activeWeights.body = this.bodyWeight;

        // Face gets remaining weight after HR and body
        const totalExtra = (activeWeights.hr || 0) + (activeWeights.body || 0);
        if (totalExtra > 0) {
            // Cap total extra at 0.8 so face always contributes at least 20%
            const scale = totalExtra > 0.8 ? 0.8 / totalExtra : 1;
            const hrW = (activeWeights.hr || 0) * scale;
            const bodyW = (activeWeights.body || 0) * scale;
            const faceW = 1 - hrW - bodyW;

            rawPain = facePain * faceW
                + (hrPainScore || 0) * hrW
                + (bodyPainScore || 0) * bodyW;
            rawPain = Math.max(0, Math.min(10, rawPain));
        }

        // Temporal smoothing (exponential moving average)
        this.smoothingWindow.push(rawPain);
        if (this.smoothingWindow.length > this.smoothingSize) {
            this.smoothingWindow.shift();
        }
        this.currentScore = this.smoothingWindow.reduce((a, b) => a + b, 0) / this.smoothingWindow.length;

        this.lastResult = {
            score: Math.round(this.currentScore * 10) / 10,
            rawScore: Math.round(rawPain * 10) / 10,
            facePain: Math.round(facePain * 10) / 10,
            hrPain: hrPainScore !== null ? Math.round(hrPainScore * 10) / 10 : null,
            bodyPain: bodyPainScore !== null ? Math.round(bodyPainScore * 10) / 10 : null,
            aus: {
                au4: Math.round(Math.max(0, au4Score) * 10) / 10,
                au6_7: Math.round(Math.max(0, au6_7Score) * 10) / 10,
                au9_10: Math.round(Math.max(0, adjustedAU910) * 10) / 10,
                au43: Math.round(au43Score * 10) / 10,
            },
            pspi: Math.round(pspi * 10) / 10,
        };

        return this.lastResult;
    }
}
