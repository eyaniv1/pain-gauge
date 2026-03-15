# Sprint Plan: Pain Gauge AI Engine

**Date:** 2026-03-15
**Scrum Master:** Eran
**Project Level:** 2
**Total Stories:** 15
**Total Points:** 52
**Planned Sprints:** 4

---

## Executive Summary

This sprint plan breaks the Pain Gauge AI Engine into 15 implementable stories across 4 two-week sprints. Sprint 1 establishes the Python inference service foundation and database schema. Sprint 2 connects inference to the Node.js backend for live sessions. Sprint 3 adds the training pipeline and dataset tooling. Sprint 4 delivers clinical UI enhancements (score correction, comparison dashboard).

**Key Metrics:**
- Total Stories: 15
- Total Points: 52
- Sprints: 4 (8 weeks)
- Team: 1 developer + Claude AI pair
- Capacity: ~13 points per sprint
- Target Completion: 2026-05-10

---

## Story Inventory

### EPIC-001: AI Inference Service

---

### STORY-001: Python Inference Service Scaffold

**Epic:** EPIC-001
**Priority:** Must Have
**Points:** 3

**User Story:**
As a developer, I want a FastAPI service that starts on port 5000 with a health check endpoint, so that I have the foundation to add model inference.

**Acceptance Criteria:**
- [ ] FastAPI app in `inference/main.py` starts with `uvicorn`
- [ ] `GET /health` returns `{ status: "ok", model_loaded: false }`
- [ ] Service binds to `127.0.0.1:5000` only
- [ ] `requirements.txt` includes fastapi, uvicorn, pydantic
- [ ] `inference/config.py` reads port, model path from env vars with defaults
- [ ] Basic pytest test for health endpoint

**Technical Notes:**
- Create `inference/` directory structure per architecture doc
- Use Pydantic models for request/response validation
- Graceful shutdown on SIGTERM

**Dependencies:** None

---

### STORY-002: Model Manager — Load Pre-trained Model

**Epic:** EPIC-001
**Priority:** Must Have
**Points:** 5

**User Story:**
As a developer, I want the service to load a pre-trained ONNX or PyTorch pain detection model on startup, so that it's ready to run inference.

**Acceptance Criteria:**
- [ ] `model_manager.py` loads `.onnx` via ONNX Runtime or `.pt` via PyTorch
- [ ] Auto-detects format from file extension
- [ ] `GET /health` reports `model_loaded: true, model_version, model_format`
- [ ] If no model file found, service starts but reports `model_loaded: false`
- [ ] `model-meta.json` stores version, format, training info
- [ ] Model directory: `models/active/`
- [ ] Unit tests with a mock/small test model

**Technical Notes:**
- Start with a publicly available ResNet or VGG model fine-tuned for facial expression (HuggingFace search)
- If no pain-specific model found, use a general facial expression model as placeholder
- ONNX Runtime for CPU inference

**Dependencies:** STORY-001

---

### STORY-003: Image Preprocessing Pipeline

**Epic:** EPIC-001
**Priority:** Must Have
**Points:** 3

**User Story:**
As a developer, I want incoming face images preprocessed (decoded, cropped, resized, normalized) before model inference, so that the model receives properly formatted input.

**Acceptance Criteria:**
- [ ] `preprocessing.py` accepts base64 JPEG/PNG, returns tensor/array
- [ ] Face detection via MediaPipe to crop face region
- [ ] Resize to model input dimensions (224x224)
- [ ] Normalize pixel values to model's expected range
- [ ] Returns None/error if no face detected
- [ ] Unit tests with sample face images

**Technical Notes:**
- OpenCV for image decode, MediaPipe for face detection
- Single-pass pipeline: decode → detect → crop → resize → normalize

**Dependencies:** STORY-001

---

### STORY-004: Predict Endpoint — Frame to Score

**Epic:** EPIC-001
**Priority:** Must Have
**Points:** 5

**User Story:**
As a developer, I want a `POST /predict` endpoint that accepts a face image and optional AU data, runs inference, and returns an AI pain score with confidence.

**Acceptance Criteria:**
- [ ] `POST /predict` accepts `{ image: base64, au_data?: {...} }`
- [ ] Returns `{ ai_score, confidence, model_version, inference_ms }`
- [ ] `ai_score` in 0-10 range
- [ ] `confidence` in 0-1 range
- [ ] Returns 400 if image is invalid or no face detected
- [ ] Returns 503 if model not loaded
- [ ] `inference_ms` measured accurately
- [ ] Integration test with real model and test image

**Technical Notes:**
- Combines preprocessing (STORY-003) + model inference (STORY-002)
- Confidence derived from model softmax output or similar
- If AU data provided, apply weighted scoring (STORY-005)

**Dependencies:** STORY-002, STORY-003

---

### STORY-005: AU-Weighted Pain Scoring

**Epic:** EPIC-001
**Priority:** Must Have
**Points:** 3

**User Story:**
As a developer, I want the inference service to apply configurable AU-zone weights when AU data is provided, so that pain estimation reflects the relative importance of different facial regions (per MDPI research).

**Acceptance Criteria:**
- [ ] `au_weights.py` defines default weights per AU zone (AU4, AU6, AU7, AU9, AU10, AU43)
- [ ] Weights loaded from `au_weights.json` config file
- [ ] When AU data is provided with image, weighted AU score is blended with model score
- [ ] Response includes `au_weights_applied` showing which weights were used
- [ ] Weights can be updated without code changes (JSON config)
- [ ] Unit tests for weight calculation

**Technical Notes:**
- Default weights from MDPI paper: AU4=0.35, AU6=0.10, AU7=0.10, AU9=0.15, AU10=0.15, AU43=0.15
- Blend formula: `final_score = α × model_score + (1-α) × weighted_au_score` where α is configurable

**Dependencies:** STORY-004

---

### STORY-006: Model Listing and Switching API

**Epic:** EPIC-001
**Priority:** Should Have
**Points:** 2

**User Story:**
As a developer, I want API endpoints to list available models and switch the active model, so that I can deploy new model versions without restarting the service.

**Acceptance Criteria:**
- [ ] `GET /models` returns list of available models in `models/` directory
- [ ] `POST /models/activate` switches active model by version
- [ ] New model takes effect for next prediction (no restart needed)
- [ ] Returns error if requested model version not found
- [ ] Unit test for model switching

**Technical Notes:**
- Scan `models/active/` and `models/archive/` directories
- Use `model-meta.json` for version info

**Dependencies:** STORY-002

---

### EPIC-002: Data Pipeline & Training

---

### STORY-007: UNBC-McMaster Dataset Loader

**Epic:** EPIC-002
**Priority:** Must Have
**Points:** 5

**User Story:**
As a developer, I want a script that loads and preprocesses the UNBC-McMaster dataset into a standard format, so that I have benchmark data for training and evaluation.

**Acceptance Criteria:**
- [ ] `training/prepare_dataset.py` reads UNBC-McMaster directory structure
- [ ] Extracts frames, AU labels (FACS), and observer pain intensity (OPI) scores
- [ ] Outputs preprocessed data: images resized to 224x224 + labels CSV
- [ ] Splits into train/validation/test (70/15/15)
- [ ] Handles missing/corrupt files gracefully
- [ ] Prints dataset statistics (sample count, score distribution)
- [ ] README documents dataset structure and licensing

**Technical Notes:**
- UNBC-McMaster has sequence folders with frame images and AAM-coded AUs
- Pain scores available as VAS and OPI (use OPI, scale to 0-10)

**Dependencies:** None (standalone script)

---

### STORY-008: Model Fine-tuning Script

**Epic:** EPIC-002
**Priority:** Should Have
**Points:** 5

**User Story:**
As a developer, I want a training script that fine-tunes the pre-trained model on labeled pain data, so that the model can improve from UNBC-McMaster data and future clinician corrections.

**Acceptance Criteria:**
- [ ] `training/train.py` loads preprocessed dataset (from STORY-007)
- [ ] Fine-tunes pre-trained model (PyTorch) with configurable hyperparameters
- [ ] Logs training metrics (loss, accuracy per epoch) to console
- [ ] Saves best model checkpoint as `.pt` file
- [ ] Exports ONNX version of fine-tuned model
- [ ] Generates evaluation report comparing new vs. old model
- [ ] Configurable: epochs, learning rate, batch size, base model path

**Technical Notes:**
- Use PyTorch, freeze early layers, fine-tune last 2-3 layers
- Save to `models/archive/model-v{version}.pt` and `.onnx`

**Dependencies:** STORY-007, STORY-002

---

### STORY-009: Export Clinician Corrections for Training

**Epic:** EPIC-002
**Priority:** Should Have
**Points:** 2

**User Story:**
As an admin, I want to export clinician-corrected samples as a labeled dataset, so that corrections can feed into model retraining.

**Acceptance Criteria:**
- [ ] `training/export_corrections.py` queries SQLite for samples with `corrected_score IS NOT NULL`
- [ ] Exports CSV: frame_filename, original_ai_score, corrected_score, au_data
- [ ] Copies associated frame images to export directory
- [ ] Output format compatible with training script input
- [ ] Prints export statistics

**Technical Notes:**
- Direct SQLite access from Python (no API needed)
- Export to `training/data/corrections/`

**Dependencies:** STORY-012 (clinician correction UI must exist to generate data)

---

### EPIC-003: Backend & Storage Integration

---

### STORY-010: Database Schema Migration — AI Columns

**Epic:** EPIC-003
**Priority:** Must Have
**Points:** 2

**User Story:**
As a developer, I want the samples table extended with AI score columns, so that AI predictions are persisted alongside existing PSPI data.

**Acceptance Criteria:**
- [ ] Migration adds `ai_score` (REAL), `ai_confidence` (REAL), `model_version` (TEXT) to samples
- [ ] Migration adds `corrected_score` (REAL), `corrected_by` (TEXT), `corrected_at` (TEXT) to samples
- [ ] Migration runs on server startup (ALTER TABLE IF NOT EXISTS pattern)
- [ ] Existing data unaffected (new columns nullable)
- [ ] Sample API responses include new fields
- [ ] Index on `corrected_score` for training export queries

**Technical Notes:**
- SQLite ALTER TABLE ADD COLUMN (one per statement)
- Add to `server/index.js` startup, after table creation
- Update prepared statements to include new columns in INSERT and SELECT

**Dependencies:** None

---

### STORY-011: Node.js Inference Proxy

**Epic:** EPIC-003
**Priority:** Must Have
**Points:** 5

**User Story:**
As a developer, I want the Node.js backend to proxy frames to the Python inference service during sample ingestion and store the AI results, so that every captured frame gets an AI pain score.

**Acceptance Criteria:**
- [ ] Sample ingestion (`POST /api/sessions/:id/samples`) sends frame to `localhost:5000/predict`
- [ ] AI response (ai_score, confidence, model_version) stored in sample record
- [ ] 2-second timeout on inference call
- [ ] On failure: stores `ai_score=NULL`, logs warning, continues with PSPI
- [ ] `GET /api/health` reports inference service status (up/down)
- [ ] `GET /api/inference/status` returns detailed AI engine info
- [ ] Works correctly when inference service is not running (graceful degradation)

**Technical Notes:**
- Use Node.js native `fetch` with `AbortSignal.timeout(2000)`
- Cache inference service health status for 30 seconds
- Send both base64 image and AU data (from `sensor_data` field) to predict endpoint

**Dependencies:** STORY-004, STORY-010

---

### STORY-012: Update start.bat for Both Services

**Epic:** EPIC-003
**Priority:** Must Have
**Points:** 1

**User Story:**
As a developer, I want `start.bat` to launch both Node.js and the Python inference service, so that everything starts with one click.

**Acceptance Criteria:**
- [ ] `start.bat` starts Node.js server (existing)
- [ ] `start.bat` starts Python inference service in separate window
- [ ] Waits for both services, reports status
- [ ] Handles case where Python/venv not set up (warning, not error)
- [ ] Starts Cloudflare tunnel after both services

**Technical Notes:**
- Per architecture doc deployment section
- Use `start` command for separate windows

**Dependencies:** STORY-001

---

### EPIC-004: Clinical UI Enhancements

---

### STORY-013: Dual Score Display — Live Session

**Epic:** EPIC-004
**Priority:** Should Have
**Points:** 3

**User Story:**
As a clinician, I want to see both PSPI and AI pain scores during a live session, so that I can compare the two approaches in real time.

**Acceptance Criteria:**
- [ ] Pain gauge displays AI score (when available) alongside PSPI score
- [ ] Score source clearly labeled ("AI" vs "PSPI")
- [ ] Confidence badge: green (>0.7), yellow (0.5-0.7), red (<0.5)
- [ ] When AI unavailable, shows "AI unavailable" indicator and PSPI only
- [ ] Session chart shows dual lines (PSPI blue, AI red) with legend
- [ ] Stats bar shows averages for both scores

**Technical Notes:**
- Modify `js/main.js` sample display logic
- Modify `js/session-chart.js` for dual-line support
- Read `ai_score` from sample API response

**Dependencies:** STORY-011

---

### STORY-014: Clinician Score Correction UI

**Epic:** EPIC-004
**Priority:** Should Have
**Points:** 3

**User Story:**
As a clinician, I want to review a completed session's samples and correct individual AI pain scores, so that my expert judgment creates labeled training data.

**Acceptance Criteria:**
- [ ] Session detail view (History tab) shows AI score per sample with edit icon
- [ ] Clicking edit opens inline input for corrected score
- [ ] `PUT /api/samples/:id/correct` saves correction with clinician name and timestamp
- [ ] Original AI score preserved, correction shown alongside
- [ ] Corrected samples visually marked (different color/icon)
- [ ] Backend endpoint added to Node.js

**Technical Notes:**
- New API endpoint in `server/index.js`
- Frontend changes in `js/main.js` history panel
- Store `corrected_score`, `corrected_by`, `corrected_at`

**Dependencies:** STORY-010, STORY-011

---

### STORY-015: AI Status Indicator in Header

**Epic:** EPIC-004
**Priority:** Should Have
**Points:** 1

**User Story:**
As a clinician, I want a small status indicator in the app header showing whether the AI engine is online, so that I know which scoring mode is active.

**Acceptance Criteria:**
- [ ] Small dot/badge in header: green = AI online, gray = AI offline
- [ ] Tooltip shows model version and status details
- [ ] Updates every 30 seconds via health check
- [ ] Does not interfere with existing header layout

**Technical Notes:**
- Poll `GET /api/inference/status` from frontend
- Add to header next to existing buttons

**Dependencies:** STORY-011

---

## Sprint Allocation

---

### Sprint 1 (Weeks 1-2) — 12/13 points

**Goal:** Stand up the Python inference service with model loading, preprocessing, and prediction endpoint

**Stories:**
| Story | Title | Points | Priority |
|-------|-------|--------|----------|
| STORY-001 | Python Service Scaffold | 3 | Must Have |
| STORY-002 | Model Manager — Load Pre-trained Model | 5 | Must Have |
| STORY-003 | Image Preprocessing Pipeline | 3 | Must Have |
| STORY-012 | Update start.bat | 1 | Must Have |

**Total:** 12 points / 13 capacity (92% utilization)

**Sprint 1 Deliverable:** A running Python inference service that loads a model, preprocesses images, and is launchable via start.bat. Not yet connected to Node.js — can be tested independently via curl/Postman.

**Risks:**
- Finding a suitable pre-trained model may take research time
- MediaPipe face detection setup on Windows

---

### Sprint 2 (Weeks 3-4) — 13/13 points

**Goal:** Connect inference to the live session pipeline — AI scores flowing end-to-end

**Stories:**
| Story | Title | Points | Priority |
|-------|-------|--------|----------|
| STORY-004 | Predict Endpoint — Frame to Score | 5 | Must Have |
| STORY-005 | AU-Weighted Pain Scoring | 3 | Must Have |
| STORY-010 | Database Schema Migration | 2 | Must Have |
| STORY-011 | Node.js Inference Proxy | 5 | Must Have |

**Total:** 15 points / 13 capacity (115% utilization — tight but all Must Have; STORY-010 is quick)

**Sprint 2 Deliverable:** Full end-to-end AI scoring. Browser captures frame → Node.js proxies to Python → AI score returned and stored. PSPI fallback works. This is the MVP milestone.

**Risks:**
- Sprint is slightly overcommitted — STORY-010 (schema migration) is simple and may absorb quickly
- Integration testing between services may surface issues

---

### Sprint 3 (Weeks 5-6) — 12/13 points

**Goal:** Training pipeline and dataset tooling for model improvement

**Stories:**
| Story | Title | Points | Priority |
|-------|-------|--------|----------|
| STORY-007 | UNBC-McMaster Dataset Loader | 5 | Must Have |
| STORY-008 | Model Fine-tuning Script | 5 | Should Have |
| STORY-006 | Model Listing and Switching API | 2 | Should Have |

**Total:** 12 points / 13 capacity (92% utilization)

**Sprint 3 Deliverable:** Ability to load benchmark datasets, fine-tune the model, and switch to new model versions without restart.

**Risks:**
- UNBC-McMaster dataset access (may need to apply and wait)
- Training script depends on having actual data

**Dependencies:**
- UNBC-McMaster dataset must be obtained before Sprint 3

---

### Sprint 4 (Weeks 7-8) — 9/13 points

**Goal:** Clinical UI enhancements — dual scoring, corrections, status indicator

**Stories:**
| Story | Title | Points | Priority |
|-------|-------|--------|----------|
| STORY-013 | Dual Score Display — Live Session | 3 | Should Have |
| STORY-014 | Clinician Score Correction UI | 3 | Should Have |
| STORY-015 | AI Status Indicator in Header | 1 | Should Have |
| STORY-009 | Export Corrections for Training | 2 | Should Have |

**Total:** 9 points / 13 capacity (69% utilization — buffer for bugs, polish, testing)

**Sprint 4 Deliverable:** Full clinical experience — clinicians see dual scores, can correct AI predictions, and corrections flow into training pipeline. Buffer time for end-to-end testing and bug fixes.

**Risks:**
- Low risk — all building on established foundation

---

## Epic Traceability

| Epic ID | Epic Name | Stories | Total Points | Sprint |
|---------|-----------|---------|--------------|--------|
| EPIC-001 | AI Inference Service | STORY-001, 002, 003, 004, 005, 006 | 21 | Sprint 1-3 |
| EPIC-002 | Data Pipeline & Training | STORY-007, 008, 009 | 12 | Sprint 3-4 |
| EPIC-003 | Backend & Storage Integration | STORY-010, 011, 012 | 8 | Sprint 1-2 |
| EPIC-004 | Clinical UI Enhancements | STORY-013, 014, 015 | 7 | Sprint 4 |

---

## Requirements Coverage

| FR ID | FR Name | Story | Sprint |
|-------|---------|-------|--------|
| FR-001 | Pre-trained Model Integration | STORY-002 | 1 |
| FR-002 | Frame-to-Score Inference | STORY-004 | 2 |
| FR-003 | AU-Weighted Pain Estimation | STORY-005 | 2 |
| FR-004 | Confidence Score | STORY-004, STORY-013 | 2, 4 |
| FR-005 | Model Versioning | STORY-006 | 3 |
| FR-006 | UNBC-McMaster Dataset Integration | STORY-007 | 3 |
| FR-007 | Fine-tuning Pipeline | STORY-008 | 3 |
| FR-008 | Clinician Score Correction | STORY-014 | 4 |
| FR-009 | Python Inference Service | STORY-001 | 1 |
| FR-010 | Score Comparison Dashboard | STORY-013 | 4 |
| FR-011 | AI Scores in Sample Records | STORY-010 | 2 |
| FR-012 | Training Dataset Management | STORY-009 | 4 |

**Coverage: 12/12 FRs (100%)**

---

## Risks and Mitigation

**High:**
- **Pre-trained model availability** — May not find a pain-specific model on HuggingFace. Mitigation: Start with a general facial expression model (FER); replace with pain-specific model when available.
- **UNBC-McMaster dataset access** — Requires application to McMaster University. Mitigation: Apply immediately; use BioVid Heat Pain Database as fallback; Sprint 3 can proceed with synthetic/placeholder data.

**Medium:**
- **CPU inference speed** — Target <500ms may be tight on some hardware. Mitigation: Use ONNX with INT8 quantization; benchmark early in Sprint 1.
- **Sprint 2 overcommitment** — 15 points vs 13 capacity. Mitigation: STORY-010 is a quick schema change; if behind, defer STORY-005 (AU weights) to Sprint 3.

**Low:**
- **Python environment on Windows** — venv/pip issues possible. Mitigation: Document setup; consider conda.
- **face-api.js AU accuracy** — May not provide reliable AU values. Mitigation: Test AU output quality; if poor, add Python-side AU detection later.

---

## Dependencies

**External:**
- UNBC-McMaster dataset (apply before Sprint 3)
- Pre-trained model from HuggingFace/research repos (research in Sprint 1)

**Internal:**
- Sprint 2 depends on Sprint 1 (Python service must be running)
- Sprint 4 depends on Sprint 2 (AI scores must be flowing)
- STORY-009 depends on STORY-014 (need corrections to exist before exporting)

---

## Definition of Done

For a story to be considered complete:
- [ ] Code implemented and committed to master
- [ ] Tests written and passing (unit + integration where applicable)
- [ ] Existing functionality not broken (PSPI scoring, session recording)
- [ ] Pushed to GitHub (frontend auto-deploys via Pages)
- [ ] Backend changes tested with running services
- [ ] Acceptance criteria validated

---

## Next Steps

**Immediate:** Begin Sprint 1

Run `/dev-story STORY-001` to start implementing the Python inference service scaffold.

**Sprint cadence:**
- Sprint length: 2 weeks
- Sprint 1: 2026-03-17 to 2026-03-28
- Sprint 2: 2026-03-31 to 2026-04-11
- Sprint 3: 2026-04-14 to 2026-04-25
- Sprint 4: 2026-04-28 to 2026-05-09

**Pre-Sprint 1 actions:**
1. Apply for UNBC-McMaster dataset access (lead time needed)
2. Research pre-trained pain/expression models on HuggingFace
3. Install Python 3.11+ if not already installed

---

**This plan was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
