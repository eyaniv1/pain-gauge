/**
 * Body Motion Analysis Engine
 *
 * Uses MediaPipe Pose (33 landmarks) to detect pain-related body behaviors:
 *   - Guarding: protecting a body area with hands/arms
 *   - Bracing: rigid posture, stiffness
 *   - Restlessness: excessive shifting/fidgeting
 *   - Freezing: abnormally low movement (pain avoidance)
 *
 * Outputs a body motion pain score (0-10) that can be fused with
 * the facial AU score and HR score for multimodal pain assessment.
 */

class BodyMotion {
    constructor(options = {}) {
        // Pose landmark history for motion analysis
        this.history = [];          // recent pose snapshots
        this.historySize = options.historySize || 30;  // ~1 second at 30fps

        // Baseline (captured during calibration)
        this.baseline = null;       // { avgMovement, posture }
        this._calibSamples = [];

        // Scores
        this.currentScore = 0;
        this.lastResult = null;
        this.smoothingWindow = [];
        this.smoothingSize = options.smoothingSize || 15;

        // Sensitivity tuning
        this.guardingSensitivity = options.guardingSensitivity ?? 1.0;
        this.restlessnessSensitivity = options.restlessnessSensitivity ?? 1.0;
        this.freezingSensitivity = options.freezingSensitivity ?? 1.0;
        this.bracingSensitivity = options.bracingSensitivity ?? 1.0;
    }

    // ── MediaPipe Pose landmark indices ────────────────────────

    static L = {
        NOSE: 0,
        LEFT_SHOULDER: 11,
        RIGHT_SHOULDER: 12,
        LEFT_ELBOW: 13,
        RIGHT_ELBOW: 14,
        LEFT_WRIST: 15,
        RIGHT_WRIST: 16,
        LEFT_HIP: 23,
        RIGHT_HIP: 24,
        LEFT_KNEE: 25,
        RIGHT_KNEE: 26,
        LEFT_ANKLE: 27,
        RIGHT_ANKLE: 28,
    };

    // Key body regions for guarding detection
    static BODY_REGIONS = {
        head:  { center: 0,  radius: 0.08 },
        chest: { landmarks: [11, 12], radius: 0.12 },
        abdomen: { landmarks: [11, 12, 23, 24], radius: 0.10 },
        leftArm:  { landmarks: [11, 13, 15], radius: 0.06 },
        rightArm: { landmarks: [12, 14, 16], radius: 0.06 },
    };

    // ── Geometry helpers ──────────────────────────────────────

    static dist(a, b) {
        return Math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            ((a.z || 0) - (b.z || 0)) ** 2
        );
    }

    static midpoint(a, b) {
        return {
            x: (a.x + b.x) / 2,
            y: (a.y + b.y) / 2,
            z: ((a.z || 0) + (b.z || 0)) / 2,
        };
    }

    // ── Measurement functions ─────────────────────────────────

    /**
     * Compute total body movement between two pose snapshots.
     * Returns average displacement of key joints (normalized).
     */
    computeMovement(prev, curr) {
        const joints = [
            BodyMotion.L.NOSE,
            BodyMotion.L.LEFT_SHOULDER, BodyMotion.L.RIGHT_SHOULDER,
            BodyMotion.L.LEFT_ELBOW, BodyMotion.L.RIGHT_ELBOW,
            BodyMotion.L.LEFT_WRIST, BodyMotion.L.RIGHT_WRIST,
            BodyMotion.L.LEFT_HIP, BodyMotion.L.RIGHT_HIP,
        ];

        let totalDist = 0;
        let count = 0;

        for (const idx of joints) {
            if (prev[idx] && curr[idx] &&
                (prev[idx].visibility || 0) > 0.5 &&
                (curr[idx].visibility || 0) > 0.5) {
                totalDist += BodyMotion.dist(prev[idx], curr[idx]);
                count++;
            }
        }

        return count > 0 ? totalDist / count : 0;
    }

    /**
     * Detect guarding: hands positioned protectively over a body area.
     * Returns 0-5 score.
     */
    measureGuarding(landmarks) {
        const L = BodyMotion.L;
        const lw = landmarks[L.LEFT_WRIST];
        const rw = landmarks[L.RIGHT_WRIST];

        if (!lw || !rw ||
            (lw.visibility || 0) < 0.4 ||
            (rw.visibility || 0) < 0.4) {
            return 0;
        }

        // Check if hands are near torso (chest/abdomen area)
        const chest = BodyMotion.midpoint(
            landmarks[L.LEFT_SHOULDER],
            landmarks[L.RIGHT_SHOULDER]
        );
        const abdomen = BodyMotion.midpoint(
            BodyMotion.midpoint(landmarks[L.LEFT_SHOULDER], landmarks[L.LEFT_HIP]),
            BodyMotion.midpoint(landmarks[L.RIGHT_SHOULDER], landmarks[L.RIGHT_HIP])
        );

        // Shoulder width for normalization
        const shoulderWidth = BodyMotion.dist(
            landmarks[L.LEFT_SHOULDER],
            landmarks[L.RIGHT_SHOULDER]
        );
        if (shoulderWidth < 0.01) return 0;

        // Distance of each wrist to torso center
        const lwToChest = BodyMotion.dist(lw, chest) / shoulderWidth;
        const rwToChest = BodyMotion.dist(rw, chest) / shoulderWidth;
        const lwToAbdomen = BodyMotion.dist(lw, abdomen) / shoulderWidth;
        const rwToAbdomen = BodyMotion.dist(rw, abdomen) / shoulderWidth;

        // Closer to torso = more guarding
        const minDistL = Math.min(lwToChest, lwToAbdomen);
        const minDistR = Math.min(rwToChest, rwToAbdomen);

        // Score: hands very close to torso (< 0.5 shoulder-widths) = guarding
        let score = 0;
        if (minDistL < 0.6) score += Math.max(0, (0.6 - minDistL) / 0.6) * 2.5;
        if (minDistR < 0.6) score += Math.max(0, (0.6 - minDistR) / 0.6) * 2.5;

        return Math.min(5, score * this.guardingSensitivity);
    }

    /**
     * Detect bracing: rigid, stiff posture (shoulder elevation, arm tension).
     * Returns 0-5 score.
     */
    measureBracing(landmarks) {
        const L = BodyMotion.L;

        // Shoulder elevation relative to normal posture
        // When bracing, shoulders tend to rise (shrug)
        const ls = landmarks[L.LEFT_SHOULDER];
        const rs = landmarks[L.RIGHT_SHOULDER];
        const lh = landmarks[L.LEFT_HIP];
        const rh = landmarks[L.RIGHT_HIP];
        const nose = landmarks[L.NOSE];

        if (!ls || !rs || !lh || !rh || !nose) return 0;
        if ((ls.visibility || 0) < 0.4 || (rs.visibility || 0) < 0.4) return 0;

        // Torso length
        const torsoLength = (BodyMotion.dist(ls, lh) + BodyMotion.dist(rs, rh)) / 2;
        if (torsoLength < 0.01) return 0;

        // Arm-to-body angle: elbows tight to body = bracing
        const le = landmarks[L.LEFT_ELBOW];
        const re = landmarks[L.RIGHT_ELBOW];
        let armTightness = 0;

        if (le && (le.visibility || 0) > 0.4) {
            const elbowDist = BodyMotion.dist(le, BodyMotion.midpoint(ls, lh)) / torsoLength;
            armTightness += Math.max(0, 0.4 - elbowDist) / 0.4;
        }
        if (re && (re.visibility || 0) > 0.4) {
            const elbowDist = BodyMotion.dist(re, BodyMotion.midpoint(rs, rh)) / torsoLength;
            armTightness += Math.max(0, 0.4 - elbowDist) / 0.4;
        }

        return Math.min(5, armTightness * 2.5 * this.bracingSensitivity);
    }

    /**
     * Compute restlessness from movement variance over history window.
     * High, erratic movement = restlessness.
     * Returns 0-5 score.
     */
    measureRestlessness() {
        if (this.history.length < 5) return 0;

        // Compute per-frame movements
        const movements = [];
        for (let i = 1; i < this.history.length; i++) {
            movements.push(this.computeMovement(this.history[i - 1], this.history[i]));
        }

        const avgMovement = movements.reduce((a, b) => a + b, 0) / movements.length;

        // Variance of movement (erratic = high variance)
        const variance = movements.reduce((sum, m) => sum + (m - avgMovement) ** 2, 0) / movements.length;
        const stdDev = Math.sqrt(variance);

        // Baseline comparison
        const baselineMovement = this.baseline ? this.baseline.avgMovement : 0.005;

        // Score based on excess movement above baseline
        const excessMovement = Math.max(0, avgMovement - baselineMovement * 2);
        const movementScore = Math.min(2.5, (excessMovement / 0.02) * 2.5);

        // Score based on movement variability (jittery/erratic)
        const erraticScore = Math.min(2.5, (stdDev / 0.01) * 2.5);

        return Math.min(5, (movementScore + erraticScore) * this.restlessnessSensitivity);
    }

    /**
     * Detect freezing: abnormally low movement (pain avoidance).
     * Returns 0-5 score. Only scores when there IS a baseline to compare to.
     */
    measureFreezing() {
        if (!this.baseline || this.history.length < 10) return 0;

        const movements = [];
        for (let i = 1; i < this.history.length; i++) {
            movements.push(this.computeMovement(this.history[i - 1], this.history[i]));
        }

        const avgMovement = movements.reduce((a, b) => a + b, 0) / movements.length;

        // If current movement is much less than baseline normal movement
        if (this.baseline.avgMovement < 0.001) return 0;
        const movementRatio = avgMovement / this.baseline.avgMovement;

        // Very low movement ratio = freezing
        if (movementRatio < 0.3) {
            return Math.min(5, ((0.3 - movementRatio) / 0.3) * 5 * this.freezingSensitivity);
        }
        return 0;
    }

    // ── Calibration ───────────────────────────────────────────

    calibrate(landmarks) {
        this._calibSamples.push([...landmarks]);

        if (this._calibSamples.length >= 2) {
            // Compute average movement during calibration
            const movements = [];
            for (let i = 1; i < this._calibSamples.length; i++) {
                movements.push(this.computeMovement(this._calibSamples[i - 1], this._calibSamples[i]));
            }
            const avgMovement = movements.reduce((a, b) => a + b, 0) / movements.length;

            this.baseline = {
                avgMovement,
                samples: this._calibSamples.length,
            };
        }
    }

    resetCalibration() {
        this.baseline = null;
        this._calibSamples = [];
        this.history = [];
        this.smoothingWindow = [];
        this.currentScore = 0;
        this.lastResult = null;
    }

    isCalibrated() {
        return this.baseline !== null;
    }

    // ── Main scoring ──────────────────────────────────────────

    /**
     * Process a frame of pose landmarks and return body motion pain indicators.
     *
     * @param {Array} landmarks - MediaPipe Pose landmarks (33 points)
     * @returns {{ score: number, guarding: number, bracing: number, restlessness: number, freezing: number }}
     */
    process(landmarks) {
        if (!landmarks || landmarks.length < 33) {
            return this.lastResult || {
                score: 0,
                guarding: 0,
                bracing: 0,
                restlessness: 0,
                freezing: 0,
            };
        }

        // Add to history
        this.history.push([...landmarks]);
        if (this.history.length > this.historySize) {
            this.history.shift();
        }

        // Score each behavior
        const guarding = this.measureGuarding(landmarks);
        const bracing = this.measureBracing(landmarks);
        const restlessness = this.measureRestlessness();
        const freezing = this.measureFreezing();

        // Combined score: take the max pain signal approach
        // Guarding and bracing are direct indicators, restlessness/freezing are secondary
        const directPain = Math.max(guarding, bracing);
        const indirectPain = Math.max(restlessness, freezing);

        // Weighted combination: direct signals dominate
        const rawScore = Math.min(10, directPain * 1.5 + indirectPain * 0.5);

        // Temporal smoothing
        this.smoothingWindow.push(rawScore);
        if (this.smoothingWindow.length > this.smoothingSize) {
            this.smoothingWindow.shift();
        }
        this.currentScore = this.smoothingWindow.reduce((a, b) => a + b, 0) / this.smoothingWindow.length;

        this.lastResult = {
            score: Math.round(this.currentScore * 10) / 10,
            rawScore: Math.round(rawScore * 10) / 10,
            guarding: Math.round(guarding * 10) / 10,
            bracing: Math.round(bracing * 10) / 10,
            restlessness: Math.round(restlessness * 10) / 10,
            freezing: Math.round(freezing * 10) / 10,
        };

        return this.lastResult;
    }
}
