# Product Requirements Document: Pain Gauge AI Engine

**Date:** 2026-03-15
**Author:** Eran
**Version:** 1.0
**Project Type:** AI/ML Integration — Pain Detection
**Project Level:** 2
**Status:** Draft

---

## Document Overview

This Product Requirements Document (PRD) defines the functional and non-functional requirements for the Pain Gauge AI Engine — Phase 1 (Facial Pain Detection). It serves as the source of truth for what will be built and provides traceability from requirements through implementation.

**Related Documents:**
- Research: MDPI Weighted AU-Zone Pain Estimation (2025)
- Research: Exploration Medicine — AI for Pain Detection via Facial Expression (2025)
- Research: PMC — Automated Assessment of Pain Workshop (AAP 2025)
- Research: ACM — AI4Pain Grand Challenge 2025

---

## Executive Summary

Pain Gauge currently uses a fixed PSPI (Prkachin and Solomon Pain Intensity) formula to estimate pain from facial action units. While functional, this formula is rigid and does not adapt to individual patients or leverage advances in deep learning for facial expression analysis.

This PRD defines requirements for integrating an AI/ML-based pain estimation engine that replaces the fixed formula with a pre-trained model (fine-tuned on established pain datasets like UNBC-McMaster). The engine will run as a Python microservice alongside the existing Node.js backend, providing real-time inference with confidence scores, clinician correction capabilities, and a foundation for future multimodal pain detection (e.g., Apple Watch physiological signals).

---

## Product Goals

### Business Objectives

1. **Replace the fixed PSPI formula** with an AI/ML-based pain estimation model that delivers more accurate, clinically relevant scores
2. **Integrate a pre-trained facial expression model** (trained on established pain datasets like UNBC-McMaster) that can be fine-tuned over time with clinician feedback
3. **Build a foundation for multimodal pain detection** — architecture that supports future addition of physiological sensors (Apple Watch, EDA, respiration, BVP)

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Pain classification accuracy | >= 90% on UNBC-McMaster benchmark | Validation set evaluation |
| Inference latency | < 500ms per frame (CPU) | 95th percentile over 100 frames |
| Clinician alignment | >= 80% agreement with clinical assessment | Clinician review of session scores |
| System availability | No session interruption when AI service fails | Graceful fallback to PSPI |

---

## Functional Requirements

Functional Requirements (FRs) define **what** the system does - specific features and behaviors.

Each requirement includes:
- **ID**: Unique identifier (FR-001, FR-002, etc.)
- **Priority**: Must Have / Should Have / Could Have (MoSCoW)
- **Description**: What the system should do
- **Acceptance Criteria**: How to verify it's complete

---

### FR-001: Pre-trained Model Integration

**Priority:** Must Have

**Description:**
System shall integrate a pre-trained facial pain detection model (e.g., VGG16 or ResNet fine-tuned on UNBC-McMaster) that loads on service startup and is ready to process frames.

**Acceptance Criteria:**
- [ ] Model loads successfully on Python service startup
- [ ] Model accepts a face image (JPEG/PNG) as input
- [ ] Model returns a pain score in the 0-10 range
- [ ] Startup time with model loading < 30 seconds

**Dependencies:** None

---

### FR-002: Frame-to-Score Inference

**Priority:** Must Have

**Description:**
System shall accept a webcam frame (JPEG/PNG) via HTTP API and return an AI-predicted pain score alongside the existing PSPI score.

**Acceptance Criteria:**
- [ ] API endpoint `POST /predict` accepts image payload
- [ ] Response includes `{ ai_score, pspi_score, confidence, model_version }`
- [ ] Endpoint handles malformed input gracefully (400 error)
- [ ] Supports JPEG and PNG formats

**Dependencies:** FR-001

---

### FR-003: AU-Weighted Pain Estimation

**Priority:** Must Have

**Description:**
System shall implement a weighted AU-zone approach (per MDPI research paper) that assigns configurable weights to different facial regions based on their pain relevance, improving estimation accuracy from ~83% to ~93%.

**Acceptance Criteria:**
- [ ] Weights are defined in a configuration file (JSON/YAML)
- [ ] AU zones (AU4, AU6-7, AU9-10, AU43) have individual weight multipliers
- [ ] Output score reflects weighted AU contributions
- [ ] Default weights match MDPI paper recommendations
- [ ] Weights can be updated without code changes

**Dependencies:** FR-001

---

### FR-004: Confidence Score

**Priority:** Should Have

**Description:**
Each AI prediction shall include a confidence value (0.0 to 1.0) indicating model certainty about the predicted pain score.

**Acceptance Criteria:**
- [ ] Confidence score returned with every prediction
- [ ] Low-confidence predictions (< 0.5) are flagged in API response
- [ ] UI displays confidence indicator (color-coded or numeric)
- [ ] Confidence is stored in sample records

**Dependencies:** FR-002

---

### FR-005: Model Versioning

**Priority:** Should Have

**Description:**
System shall support loading different model versions, tracking which model produced each score, and switching active models without downtime.

**Acceptance Criteria:**
- [ ] Model files follow naming convention: `model-{version}.{format}`
- [ ] Session/sample records include `model_version` field
- [ ] Admin API endpoint to list available models and switch active model
- [ ] Model switch takes effect for next prediction (no restart required)

**Dependencies:** FR-001, FR-011

---

### FR-006: UNBC-McMaster Dataset Integration

**Priority:** Must Have

**Description:**
System shall include tooling to load and preprocess the UNBC-McMaster Shoulder Pain Expression dataset for model training, validation, and benchmarking.

**Acceptance Criteria:**
- [ ] Script loads UNBC-McMaster dataset from local directory
- [ ] Extracts frames, AU labels, and observer pain scores
- [ ] Splits data into train/validation/test sets (70/15/15)
- [ ] Outputs preprocessed data in standard format for training pipeline
- [ ] Documents dataset structure and licensing requirements

**Dependencies:** None

---

### FR-007: Fine-tuning Pipeline

**Priority:** Should Have

**Description:**
System shall support fine-tuning the pre-trained model on new labeled data, including clinician-corrected scores from production sessions.

**Acceptance Criteria:**
- [ ] Training script accepts labeled samples (image + score pairs)
- [ ] Supports incremental fine-tuning (doesn't require full retraining)
- [ ] Produces updated model weights with new version number
- [ ] Logs training metrics (loss, accuracy, validation score)
- [ ] Outputs comparison report: new model vs. previous version

**Dependencies:** FR-001, FR-006

---

### FR-008: Clinician Score Correction

**Priority:** Should Have

**Description:**
Clinicians shall be able to review completed sessions and correct individual AI pain scores, creating high-quality labeled training data.

**Acceptance Criteria:**
- [ ] UI shows per-sample AI score with edit capability in session detail view
- [ ] Corrected scores stored with `corrected_score`, `corrected_by`, `corrected_at` fields
- [ ] Original AI score is preserved (not overwritten)
- [ ] Corrections are exportable as training data
- [ ] Bulk correction mode for efficient review

**Dependencies:** FR-011

---

### FR-009: Python Inference Service

**Priority:** Must Have

**Description:**
A Python-based inference microservice shall run alongside the Node.js backend, handling model loading, preprocessing, and prediction via HTTP API.

**Acceptance Criteria:**
- [ ] Service starts independently (separate process from Node.js)
- [ ] Health check endpoint `GET /health` confirms model is loaded and ready
- [ ] Service is configurable via environment variables (port, model path, device)
- [ ] Graceful shutdown on SIGTERM
- [ ] Startup script (`start.bat`) updated to launch both services

**Dependencies:** None

---

### FR-010: Score Comparison Dashboard

**Priority:** Could Have

**Description:**
UI shall display both PSPI and AI scores side-by-side during a live session, allowing clinicians to compare approaches in real time.

**Acceptance Criteria:**
- [ ] Session chart shows dual lines (PSPI = blue, AI = red)
- [ ] Legend distinguishes the two score sources
- [ ] Stats bar shows averages and peaks for both scores
- [ ] Can toggle individual lines on/off

**Dependencies:** FR-002, FR-011

---

### FR-011: AI Scores in Sample Records

**Priority:** Must Have

**Description:**
Each sample record in the database shall store `ai_score`, `confidence`, and `model_version` alongside existing PSPI and AU fields.

**Acceptance Criteria:**
- [ ] Database schema includes `ai_score` (REAL), `confidence` (REAL), `model_version` (TEXT) columns
- [ ] Migration script adds columns to existing database without data loss
- [ ] API responses for samples include AI fields
- [ ] History/detail views display AI score when available
- [ ] Null AI fields handled gracefully (pre-AI sessions)

**Dependencies:** None

---

### FR-012: Training Dataset Management

**Priority:** Could Have

**Description:**
System shall maintain a curated dataset of labeled pain expressions for ongoing model improvement, combining clinician corrections with external benchmark data.

**Acceptance Criteria:**
- [ ] Admin can export all clinician-corrected samples as a labeled dataset
- [ ] Export format compatible with training pipeline input
- [ ] Can import external datasets (UNBC-McMaster format)
- [ ] Dataset statistics dashboard (sample count, score distribution, source breakdown)

**Dependencies:** FR-007, FR-008

---

## Non-Functional Requirements

Non-Functional Requirements (NFRs) define **how** the system performs - quality attributes and constraints.

---

### NFR-001: Performance — Inference Latency

**Priority:** Must Have

**Description:**
AI inference shall complete in < 500ms per frame on a standard CPU machine. GPU-equipped machines should achieve < 100ms.

**Acceptance Criteria:**
- [ ] 95th percentile latency < 500ms measured over 100 consecutive frames (CPU)
- [ ] 95th percentile latency < 100ms on GPU (when available)
- [ ] Latency does not degrade over extended sessions (> 30 minutes)

**Rationale:**
Real-time pain monitoring requires frame-rate-compatible inference. At 2 samples/second, 500ms is the maximum acceptable latency.

---

### NFR-002: Performance — Concurrent Sessions

**Priority:** Should Have

**Description:**
System shall support at least 3 concurrent sessions performing real-time AI inference without significant degradation.

**Acceptance Criteria:**
- [ ] Load test with 3 simultaneous streams maintains < 1s inference time per frame
- [ ] Memory usage stays below 4GB for 3 concurrent sessions
- [ ] No request timeouts or dropped frames under load

**Rationale:**
Clinical environments may have multiple patients being monitored simultaneously.

---

### NFR-003: Security — Patient Data Privacy

**Priority:** Must Have

**Description:**
All patient images, pain data, and model inference shall remain on the local server. No patient data shall be transmitted to external cloud ML services.

**Acceptance Criteria:**
- [ ] Network audit confirms zero outbound calls to third-party AI/ML APIs during inference
- [ ] Model runs entirely locally (no cloud inference fallback)
- [ ] Patient images are never logged or cached outside the designated frames directory

**Rationale:**
Medical data privacy (potential HIPAA/GDPR implications). Patient facial images are highly sensitive.

---

### NFR-004: Security — Model Access Control

**Priority:** Should Have

**Description:**
Model files and training data shall be access-controlled. Only authorized users can modify models or trigger retraining.

**Acceptance Criteria:**
- [ ] Model directory has restricted file permissions (owner-only write)
- [ ] Model management API endpoints require authentication
- [ ] Training pipeline cannot be triggered from unauthenticated requests

**Rationale:**
Prevents unauthorized model tampering that could produce incorrect pain assessments.

---

### NFR-005: Reliability — Graceful Degradation

**Priority:** Must Have

**Description:**
If the Python inference service is unavailable (crashed, not started, overloaded), the system shall automatically fall back to the existing PSPI formula without any interruption to active sessions.

**Acceptance Criteria:**
- [ ] Killing the Python service mid-session does not crash the app
- [ ] PSPI scores continue flowing within 1 second of AI service failure
- [ ] UI indicates when running in fallback mode
- [ ] System automatically reconnects when AI service recovers

**Rationale:**
Clinical tool reliability is paramount. No patient monitoring session should ever be interrupted by an AI service failure.

---

### NFR-006: Compatibility — Model Formats

**Priority:** Should Have

**Description:**
System shall support both ONNX and PyTorch model formats, enabling flexibility in sourcing pre-trained models from different research groups.

**Acceptance Criteria:**
- [ ] Both `.onnx` and `.pt`/`.pth` models load successfully
- [ ] Equivalent scores produced from same input across formats (< 0.1 difference)
- [ ] Model format auto-detected from file extension

**Rationale:**
Research models are published in various formats. ONNX provides optimized inference; PyTorch provides training flexibility.

---

### NFR-007: Usability — Transparent Scoring

**Priority:** Must Have

**Description:**
The UI shall clearly indicate the source of each pain score (AI model vs. PSPI formula) and display confidence when available.

**Acceptance Criteria:**
- [ ] Score source labeled in real-time display and session history
- [ ] Confidence indicator visible (color-coded badge or numeric value)
- [ ] Fallback mode clearly indicated when AI is unavailable

**Rationale:**
Clinician trust requires transparency about how scores are generated.

---

### NFR-008: Maintainability — Modular Architecture

**Priority:** Must Have

**Description:**
The Python inference service shall be fully decoupled from the Node.js backend, communicating only via HTTP API. Either service can be updated, restarted, or replaced independently.

**Acceptance Criteria:**
- [ ] Python service can be restarted without restarting Node.js server
- [ ] Node.js server can be restarted without restarting Python service
- [ ] API contract is documented (OpenAPI/Swagger or equivalent)
- [ ] No shared state between services (stateless inference)

**Rationale:**
Enables independent development cycles, easier debugging, and future replacement of either component.

---

## Epics

Epics are logical groupings of related functionality that will be broken down into user stories during sprint planning (Phase 4).

Each epic maps to multiple functional requirements and will generate 2-10 stories.

---

### EPIC-001: AI Inference Service

**Description:**
Stand up the Python-based inference microservice with pre-trained model loading, weighted AU-zone scoring, health check, and frame-to-score HTTP API.

**Functional Requirements:**
- FR-001: Pre-trained Model Integration
- FR-002: Frame-to-Score Inference
- FR-003: AU-Weighted Pain Estimation
- FR-004: Confidence Score
- FR-009: Python Inference Service

**Story Count Estimate:** 5-6 stories

**Priority:** Must Have

**Business Value:**
Core capability — replaces the fixed PSPI formula with an ML-based pain detection engine that can adapt and improve over time.

---

### EPIC-002: Data Pipeline & Training

**Description:**
Build tooling to load and preprocess benchmark datasets (UNBC-McMaster), fine-tune models on new labeled data, and manage training datasets.

**Functional Requirements:**
- FR-006: UNBC-McMaster Dataset Integration
- FR-007: Fine-tuning Pipeline
- FR-012: Training Dataset Management

**Story Count Estimate:** 3-5 stories

**Priority:** Should Have

**Business Value:**
Enables continuous model improvement from clinician feedback and established research datasets. Without this, the model is static.

---

### EPIC-003: Backend & Storage Integration

**Description:**
Extend the Node.js backend and SQLite database to route captured frames to the Python inference service, store AI scores in sample records, and support model versioning.

**Functional Requirements:**
- FR-005: Model Versioning
- FR-011: AI Scores in Sample Records

**Story Count Estimate:** 2-3 stories

**Priority:** Must Have

**Business Value:**
Connects the AI engine to the existing app. Without this, AI predictions have no way to reach the user or be persisted.

---

### EPIC-004: Clinical UI Enhancements

**Description:**
Add UI features for side-by-side score comparison (PSPI vs AI), clinician score correction, confidence display, and dual-line session charts.

**Functional Requirements:**
- FR-008: Clinician Score Correction
- FR-010: Score Comparison Dashboard

**Story Count Estimate:** 3-4 stories

**Priority:** Should Have

**Business Value:**
Enables clinician oversight, builds trust in the AI system, and generates labeled training data through score corrections — creating a virtuous improvement cycle.

---

## User Stories (High-Level)

User stories follow the format: "As a [user type], I want [goal] so that [benefit]."

These are preliminary stories. Detailed stories will be refined during sprint planning (Phase 4).

---

### EPIC-001: AI Inference Service

**US-001:** As a developer, I want a Python inference service that loads a pre-trained pain detection model on startup, so that the system can process frames through an ML pipeline.

**US-002:** As a developer, I want an HTTP API endpoint that accepts a face image and returns `{ ai_score, confidence, model_version }`, so that the Node.js backend can request predictions.

**US-003:** As a developer, I want the inference service to implement weighted AU-zone scoring (per the MDPI research), so that pain estimation accounts for the relative importance of different facial regions.

**US-004:** As a clinician, I want to see a confidence indicator alongside each AI pain score, so that I know when the model is uncertain about its prediction.

**US-005:** As a developer, I want the inference service to expose a health check endpoint, so that the backend can verify the AI engine is available before routing frames.

### EPIC-002: Data Pipeline & Training

**US-006:** As a developer, I want a script that downloads and preprocesses the UNBC-McMaster dataset into a standard format (frames + AU labels + pain scores), so that I have benchmark data for training and validation.

**US-007:** As a developer, I want a fine-tuning script that accepts labeled samples and produces updated model weights, so that the model can improve over time from new data.

**US-008:** As an admin, I want to export clinician-corrected samples as a labeled dataset, so that corrections feed back into model training.

### EPIC-003: Backend & Storage Integration

**US-009:** As a developer, I want the Node.js backend to forward captured frames to the Python inference service and store the returned AI score, confidence, and model version in each sample record, so that AI predictions are persisted alongside existing data.

**US-010:** As a developer, I want the database schema extended with `ai_score`, `confidence`, and `model_version` columns on the samples table, so that AI results are queryable and traceable.

**US-011:** As a developer, I want the system to fall back to PSPI-only scoring if the Python service is unavailable, so that sessions are never interrupted by AI service downtime.

### EPIC-004: Clinical UI Enhancements

**US-012:** As a clinician, I want to see both the PSPI score and AI score displayed side-by-side during a live session, so that I can compare the two approaches in real time.

**US-013:** As a clinician, I want to review a completed session and correct individual AI pain scores, so that my expert judgment creates labeled training data for model improvement.

**US-014:** As a clinician, I want the session chart to show dual lines (PSPI vs AI) with stats for both, so that I can evaluate model performance over the session duration.

---

## User Personas

### Clinician (Primary User)
- Medical professional (doctor, nurse, pain specialist) monitoring patient pain
- Needs reliable, transparent pain scores during sessions
- May not have technical ML knowledge — needs clear UI indicators
- Values clinical accuracy over technical sophistication

### Administrator / Developer
- Sets up and maintains the Pain Gauge system
- Manages model versions, training pipelines, and datasets
- Needs command-line tooling and clear documentation

### Patient (Indirect User)
- Subject being monitored — does not interact with the system directly
- Benefits from more accurate pain assessment

---

## User Flows

### Flow 1: Real-time AI Pain Monitoring
1. Clinician selects patient and starts session
2. System begins capturing webcam frames
3. Each frame is sent to Python inference service
4. AI score + confidence returned and displayed alongside PSPI score
5. Both scores recorded in sample records
6. Session ends; history shows AI and PSPI scores

### Flow 2: Clinician Score Correction
1. Clinician opens session from History tab
2. Reviews sample-by-sample AI scores
3. Corrects scores that don't match clinical assessment
4. Corrections saved with clinician attribution
5. Corrections available for model fine-tuning

### Flow 3: Model Improvement Cycle
1. Developer exports clinician-corrected data
2. Runs fine-tuning pipeline with new labeled data
3. Evaluates new model against benchmark
4. Deploys new model version
5. System uses updated model for future sessions

---

## Dependencies

### Internal Dependencies

- **Existing Pain Gauge frontend** — UI enhancements build on current Camera/Chart/History tabs
- **Existing Node.js backend** — API extensions and database schema changes
- **Existing PSPI calculation** — Continues as fallback and comparison baseline
- **Cloudflare tunnel** — Remote access to both Node.js and Python services

### External Dependencies

- **UNBC-McMaster Dataset** — Requires license/access for training and benchmarking
- **Pre-trained model** — VGG16, ResNet, or equivalent fine-tuned on pain detection
- **Python ML libraries** — PyTorch, ONNX Runtime, OpenCV, Flask/FastAPI
- **Hardware** — CPU minimum; GPU recommended for < 100ms inference

---

## Assumptions

1. The UNBC-McMaster dataset can be obtained for research/clinical use
2. A pre-trained facial expression model (VGG16/ResNet) fine-tuned for pain is available or can be trained from UNBC-McMaster data
3. The deployment machine has sufficient resources to run both Node.js and Python services (minimum 8GB RAM)
4. Clinicians are willing to spend time correcting AI scores to build training data
5. The existing AU detection (face-api.js) provides sufficiently accurate AU values for the weighted approach
6. CPU inference at < 500ms is achievable with model optimization (quantization, ONNX)

---

## Out of Scope

- **Multimodal signals** — Apple Watch, EDA, respiration, BVP integration (Phase 2)
- **Cloud deployment** — Inference remains local only; no cloud ML services
- **Mobile app** — AI engine is server-side only; mobile is browser-based
- **Real-time video streaming** — Inference operates on individual frames, not video streams
- **Multi-language support** — English-only UI
- **Regulatory certification** — FDA/CE marking not in scope for this phase
- **Patient-facing features** — All AI features are clinician-facing

---

## Open Questions

1. **Dataset access:** What is the licensing process for UNBC-McMaster? Are there usage restrictions for clinical tools?
2. **Model selection:** Should we start with a publicly available pre-trained model, or train from scratch on UNBC-McMaster?
3. **AU source:** Should the Python service compute its own AUs (via OpenFace/MediaPipe), or consume AUs from the existing face-api.js frontend pipeline?
4. **Inference architecture:** Should frames be sent from the browser directly to the Python service, or proxied through Node.js?
5. **GPU availability:** Will deployment machines have GPU access, or should we optimize exclusively for CPU?

---

## Approval & Sign-off

### Stakeholders

| Role | Name | Status |
|------|------|--------|
| Product Owner | Eran | Pending |
| Developer | Eran | Pending |

### Approval Status

- [ ] Product Owner
- [ ] Engineering Lead

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-15 | Eran | Initial PRD |

---

## Next Steps

### Phase 3: Architecture

Run `/architecture` to create system architecture based on these requirements.

The architecture will address:
- All functional requirements (FRs)
- All non-functional requirements (NFRs)
- Technical stack decisions (PyTorch vs ONNX, FastAPI vs Flask)
- Data models and APIs
- System component diagram
- Inference pipeline design

### Phase 4: Sprint Planning

After architecture is complete, run `/sprint-planning` to:
- Break epics into detailed user stories
- Estimate story complexity
- Plan sprint iterations
- Begin implementation

---

**This document was created using BMAD Method v6 - Phase 2 (Planning)**

*To continue: Run `/workflow-status` to see your progress and next recommended workflow.*

---

## Appendix A: Requirements Traceability Matrix

| Epic ID | Epic Name | Functional Requirements | Story Count (Est.) |
|---------|-----------|-------------------------|-------------------|
| EPIC-001 | AI Inference Service | FR-001, FR-002, FR-003, FR-004, FR-009 | 5-6 |
| EPIC-002 | Data Pipeline & Training | FR-006, FR-007, FR-012 | 3-5 |
| EPIC-003 | Backend & Storage Integration | FR-005, FR-011 | 2-3 |
| EPIC-004 | Clinical UI Enhancements | FR-008, FR-010 | 3-4 |

---

## Appendix B: Prioritization Details

### Functional Requirements

| Priority | Count | Requirements |
|----------|-------|-------------|
| Must Have | 5 | FR-001, FR-002, FR-003, FR-006, FR-009, FR-011 |
| Should Have | 4 | FR-004, FR-005, FR-007, FR-008 |
| Could Have | 2 | FR-010, FR-012 |

### Non-Functional Requirements

| Priority | Count | Requirements |
|----------|-------|-------------|
| Must Have | 4 | NFR-001, NFR-003, NFR-005, NFR-007, NFR-008 |
| Should Have | 3 | NFR-002, NFR-004, NFR-006 |

### Research References

| Source | Key Contribution | Relevance |
|--------|-----------------|-----------|
| MDPI — Weighted AU-Zone Analysis (2025) | Weighted facial region scoring, 92.72% accuracy | Direct implementation in FR-003 |
| Exploration Medicine — AI Pain Detection Review (2025) | Model survey, dataset catalogue, clinical gap analysis | Model selection, dataset strategy |
| PMC — AAP 2025 Workshop | TabPFN outperforms DL on small data, Tiny-BioMoE for biosignals | Architecture decisions, future multimodal |
| ACM — AI4Pain Grand Challenge 2025 | Physiological signal classification (EDA, BVP, SPO2) | Phase 2 multimodal foundation |
