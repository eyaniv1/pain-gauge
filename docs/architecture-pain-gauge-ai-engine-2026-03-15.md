# System Architecture: Pain Gauge AI Engine

**Date:** 2026-03-15
**Architect:** Eran
**Version:** 1.0
**Project Type:** AI/ML Integration — Pain Detection
**Project Level:** 2
**Status:** Draft

---

## Document Overview

This document defines the system architecture for the Pain Gauge AI Engine. It provides the technical blueprint for implementation, addressing all functional and non-functional requirements from the PRD.

**Related Documents:**
- Product Requirements Document: `docs/prd-pain-gauge-ai-engine-2026-03-15.md`

---

## Executive Summary

The Pain Gauge AI Engine adds a Python-based inference microservice alongside the existing Node.js backend. The two services communicate via HTTP on localhost. The Python service loads a pre-trained pain detection model (ONNX/PyTorch), accepts face images, and returns AI pain scores with confidence values. The Node.js backend proxies inference requests, stores results, and serves the frontend. Graceful degradation ensures the system falls back to PSPI scoring when the AI service is unavailable.

The architecture follows a **Sidecar Service** pattern — minimal, decoupled, and independently deployable — appropriate for a Level 2 project that needs ML capabilities without overengineering.

---

## Architectural Drivers

These requirements heavily influence architectural decisions:

| Driver | NFR | Impact |
|--------|-----|--------|
| Decoupled Services | NFR-008 | Python and Node.js are separate processes communicating via HTTP |
| Graceful Degradation | NFR-005 | Node.js must detect AI service failure and fallback to PSPI |
| Inference Latency | NFR-001 | Model format (ONNX preferred), preprocessing pipeline, no network hops |
| Data Privacy | NFR-003 | All inference local — no cloud ML APIs, images stay on-machine |

---

## System Overview

### High-Level Architecture

The system consists of four layers:

1. **Browser (Frontend)** — Existing HTML/JS app running on GitHub Pages. Captures webcam frames, sends to backend, displays scores.
2. **Node.js Backend** — Existing Express server. Manages patients, sessions, samples. Proxies frames to inference service. Stores AI scores.
3. **Python Inference Service** — New FastAPI microservice. Loads pre-trained model, processes frames, returns predictions.
4. **Storage** — SQLite database (existing) + model files + frame images on local filesystem.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (GitHub Pages)                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐             │
│  │ Camera   │  │ Chart    │  │ History   │             │
│  │ Panel    │  │ Panel    │  │ Panel     │             │
│  └────┬─────┘  └──────────┘  └───────────┘             │
│       │ frames + AU data                                 │
└───────┼─────────────────────────────────────────────────┘
        │ HTTPS (Cloudflare Tunnel)
        ▼
┌───────────────────────────────────────────────────────┐
│              Node.js Backend (port 3000)               │
│  ┌──────────────┐  ┌───────────┐  ┌───────────────┐  │
│  │ Patient/     │  │ Sample    │  │ Inference     │  │
│  │ Session API  │  │ Ingestion │  │ Proxy         │  │
│  └──────────────┘  └─────┬─────┘  └───────┬───────┘  │
│                          │                 │           │
│                          ▼                 │           │
│                    ┌──────────┐            │           │
│                    │ SQLite   │            │           │
│                    │ Database │            │           │
│                    └──────────┘            │           │
└───────────────────────────────────────────┼───────────┘
                                            │ HTTP localhost:5000
                                            ▼
┌───────────────────────────────────────────────────────┐
│           Python Inference Service (port 5000)         │
│  ┌──────────────┐  ┌───────────┐  ┌───────────────┐  │
│  │ FastAPI      │  │ Model     │  │ Preprocessing │  │
│  │ Server       │  │ Manager   │  │ Pipeline      │  │
│  └──────────────┘  └─────┬─────┘  └───────────────┘  │
│                          │                             │
│                    ┌─────┴──────┐                      │
│                    │ Model Files│                      │
│                    │ (.onnx/.pt)│                      │
│                    └────────────┘                      │
└───────────────────────────────────────────────────────┘
```

### Architectural Pattern

**Pattern:** Sidecar Service (Python inference alongside Node.js backend)

**Rationale:** This is the simplest architecture that satisfies all requirements:
- Two independent processes on localhost — no container orchestration needed
- HTTP communication — language-agnostic, easy to debug, easy to replace
- Single machine deployment — matches the local-first privacy requirement
- Independent lifecycle — Python service can be updated/restarted without touching Node.js
- Future-proof — the sidecar can evolve into a full microservice if needed

---

## Technology Stack

### Frontend

**Choice:** Vanilla HTML/CSS/JavaScript (existing)

**Rationale:** Already built and deployed on GitHub Pages. No change needed for the AI engine — the frontend just receives an additional `ai_score` field in sample data.

**Trade-offs:** Simple and fast; limited component reuse vs. React/Vue, but adequate for current scope.

### Backend

**Choice:** Node.js + Express (existing, port 3000)

**Rationale:** Already handles all CRUD operations, session management, and Cloudflare tunnel exposure. Adding inference proxy is minimal code.

**Trade-offs:** Not ideal for ML workloads (hence the Python sidecar), but excellent for API routing and database operations.

### AI Inference Service

**Choice:** Python 3.11+ with FastAPI (new, port 5000)

**Rationale:**
- FastAPI: High-performance async framework, automatic OpenAPI docs, type validation with Pydantic
- Python: Required for PyTorch/ONNX Runtime, OpenCV, and the ML ecosystem
- Uvicorn: Production-grade ASGI server

**Trade-offs:** Adds a second runtime (Python) to the deployment. Justified because ML tooling is Python-exclusive.

### ML Framework

**Choice:** ONNX Runtime (primary) + PyTorch (training/fine-tuning)

**Rationale:**
- ONNX Runtime: 2-5x faster inference than PyTorch on CPU, optimized for production
- PyTorch: Required for fine-tuning and training pipeline
- Both formats supported per NFR-006

**Trade-offs:** Maintaining two formats adds complexity, but ONNX's performance advantage on CPU is critical for NFR-001 (<500ms latency).

### Database

**Choice:** SQLite via better-sqlite3 (existing)

**Rationale:** Already in use. New columns added to samples table. No need to change — SQLite handles the data volume for a local clinical tool.

**Trade-offs:** Single-writer limitation is acceptable for local deployment with low concurrency.

### Infrastructure

**Choice:** Local machine + Cloudflare Tunnel (existing)

**Rationale:** Privacy requirement (NFR-003) mandates local inference. Cloudflare Tunnel provides HTTPS access from GitHub Pages frontend.

**Trade-offs:** Tied to a single machine; acceptable for Phase 1 clinical tool.

### Third-Party Services

| Service | Purpose | Justification |
|---------|---------|---------------|
| Cloudflare Tunnel | HTTPS exposure | Existing; exposes both Node.js and (proxied) inference |
| UNBC-McMaster Dataset | Training/benchmarking | Gold standard pain expression dataset |
| face-api.js | Frontend AU detection | Existing; provides AU values to frontend |

### Development & Deployment

| Tool | Purpose |
|------|---------|
| Git + GitHub | Version control, GitHub Pages deployment |
| npm | Node.js dependency management |
| pip + venv | Python dependency management |
| start.bat | Single-click startup for all services |
| pytest | Python testing |

---

## System Components

### Component 1: Node.js Backend (existing, extended)

**Purpose:** API gateway, data persistence, inference proxy

**Responsibilities:**
- Patient/session/sample CRUD (existing)
- Receive sample submissions with frame data
- Proxy frames to Python inference service for AI scoring
- Store AI scores in sample records
- Health check for both self and inference service
- Fallback to PSPI-only when inference service is down

**Interfaces:**
- REST API on port 3000 (HTTPS via Cloudflare Tunnel)

**Dependencies:**
- SQLite database
- Python Inference Service (optional — graceful degradation)

**FRs Addressed:** FR-002, FR-005, FR-011

---

### Component 2: Python Inference Service (new)

**Purpose:** ML model loading, preprocessing, and pain score prediction

**Responsibilities:**
- Load pre-trained model (ONNX or PyTorch) on startup
- Preprocess incoming face images (resize, normalize, face detection)
- Run inference and return pain score + confidence
- Implement weighted AU-zone scoring
- Expose health check with model status
- Support model hot-swapping (load new version without restart)

**Interfaces:**
- REST API on port 5000 (localhost only — not exposed externally)

**Dependencies:**
- Model files on local filesystem
- ONNX Runtime and/or PyTorch
- OpenCV for image preprocessing

**FRs Addressed:** FR-001, FR-002, FR-003, FR-004, FR-009

---

### Component 3: Model Manager (within Python service)

**Purpose:** Model lifecycle management

**Responsibilities:**
- Load model from file on startup
- Track model version metadata
- Support switching between model versions via API
- Auto-detect model format (.onnx vs .pt)
- Report model status (loaded, version, format, device)

**Interfaces:**
- Internal Python module, exposed via FastAPI endpoints

**Dependencies:**
- Model files directory
- ONNX Runtime / PyTorch

**FRs Addressed:** FR-001, FR-005

---

### Component 4: Preprocessing Pipeline (within Python service)

**Purpose:** Image preparation for model inference

**Responsibilities:**
- Decode incoming JPEG/PNG image
- Detect and crop face region (using MediaPipe or dlib)
- Resize to model input dimensions (e.g., 224x224)
- Normalize pixel values to model's expected range
- Apply AU-zone weighting per FR-003

**Interfaces:**
- Internal Python module called by inference endpoint

**Dependencies:**
- OpenCV, MediaPipe/dlib

**FRs Addressed:** FR-002, FR-003

---

### Component 5: Training Pipeline (CLI tool)

**Purpose:** Dataset preparation and model fine-tuning

**Responsibilities:**
- Load and preprocess UNBC-McMaster dataset
- Split into train/validation/test sets
- Fine-tune pre-trained model on labeled data
- Evaluate model performance (accuracy, loss, confusion matrix)
- Export fine-tuned model in ONNX and PyTorch formats
- Generate comparison report (new vs. previous model)

**Interfaces:**
- CLI scripts (not a running service)

**Dependencies:**
- PyTorch, ONNX, dataset files

**FRs Addressed:** FR-006, FR-007, FR-012

---

### Component 6: SQLite Database (existing, extended)

**Purpose:** Persistent storage for all application data

**Responsibilities:**
- Store patients, sessions, samples (existing)
- Store AI scores, confidence, model version per sample (new)
- Store clinician corrections (new)

**FRs Addressed:** FR-011, FR-008

---

## Data Architecture

### Data Model

**Existing entities (unchanged):**
- **Patient** (id, name, dob, patient_id, notes, created_at)
  - Has many: Sessions
- **Session** (id, patient_id, start_time, end_time, baseline_pain_level, calibration_data, settings_snapshot, notes, created_at)
  - Belongs to: Patient
  - Has many: Samples

**Extended entity:**
- **Sample** (existing + new fields)
  - Existing: id, session_id, timestamp_ms, score, raw_score, pspi, sensor_data, frame_filename, created_at
  - New: `ai_score` (REAL), `ai_confidence` (REAL), `model_version` (TEXT)
  - New: `corrected_score` (REAL), `corrected_by` (TEXT), `corrected_at` (TEXT)

### Database Design

**Migration SQL:**
```sql
-- Add AI scoring columns to samples table
ALTER TABLE samples ADD COLUMN ai_score REAL;
ALTER TABLE samples ADD COLUMN ai_confidence REAL;
ALTER TABLE samples ADD COLUMN model_version TEXT;

-- Add clinician correction columns
ALTER TABLE samples ADD COLUMN corrected_score REAL;
ALTER TABLE samples ADD COLUMN corrected_by TEXT;
ALTER TABLE samples ADD COLUMN corrected_at TEXT;

-- Index for querying corrected samples (training data export)
CREATE INDEX IF NOT EXISTS idx_samples_corrected ON samples(corrected_score) WHERE corrected_score IS NOT NULL;
```

**Model files directory:**
```
server/
  models/
    active/
      model.onnx            ← currently active model
      model-meta.json        ← { version, format, trained_on, accuracy }
    archive/
      model-v1.0.onnx
      model-v1.1.pt
```

### Data Flow

**Real-time inference flow:**
```
Browser captures frame
  → POST /api/sessions/:id/samples (with frame base64)
  → Node.js receives sample
  → Node.js sends frame to Python service: POST localhost:5000/predict
    → Python preprocesses image
    → Python runs model inference
    → Returns { ai_score, confidence, model_version }
  → Node.js stores sample with ai_score, ai_confidence, model_version
  → Response to browser includes ai_score
```

**Fallback flow (AI service down):**
```
Browser captures frame
  → POST /api/sessions/:id/samples (with frame base64)
  → Node.js receives sample
  → Node.js tries POST localhost:5000/predict → FAILS (timeout/connection refused)
  → Node.js stores sample with ai_score=NULL, ai_confidence=NULL
  → Response to browser includes pspi score only
  → UI shows "AI unavailable" indicator
```

**Training data export flow:**
```
CLI: python train/export_corrections.py
  → Queries samples WHERE corrected_score IS NOT NULL
  → Exports { frame_filename, corrected_score, original_ai_score, au_data }
  → Outputs CSV + frame images in training format
```

---

## API Design

### API Architecture

- **Node.js API (port 3000):** REST, JSON, existing endpoints extended with AI fields
- **Python Inference API (port 5000):** REST, JSON, localhost-only, not externally exposed
- **Authentication:** None (local deployment). Model management endpoints protected by localhost-only binding.
- **Versioning:** Not needed at this scale; endpoints are additive.

### Node.js API Endpoints (existing + new)

**Existing (unchanged):**
- `GET /api/health` — Health check
- `GET /api/patients` — List patients
- `POST /api/patients` — Create patient
- `GET /api/patients/:id` — Get patient
- `PUT /api/patients/:id` — Update patient
- `DELETE /api/patients/:id` — Delete patient
- `GET /api/patients/:pid/sessions` — List sessions
- `POST /api/patients/:pid/sessions` — Create session
- `GET /api/sessions/:id` — Get session
- `PUT /api/sessions/:id` — End session
- `DELETE /api/sessions/:id` — Delete session
- `POST /api/sessions/batch-delete` — Batch delete sessions
- `GET /api/sessions/:id/samples` — Get samples

**Modified:**
- `POST /api/sessions/:id/samples` — Now also proxies frame to inference service; stores ai_score, ai_confidence, model_version
- `GET /api/health` — Now also reports inference service status

**New:**
- `PUT /api/samples/:id/correct` — Submit clinician correction for a sample
  - Body: `{ corrected_score: 7.5, corrected_by: "Dr. Smith" }`
  - Response: Updated sample record
- `GET /api/inference/status` — Detailed AI engine status (model version, uptime, avg latency)
- `GET /api/training/export` — Export corrected samples as JSON for training pipeline

### Python Inference API Endpoints (port 5000, localhost only)

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/health` | Service health + model status | — | `{ status, model_version, model_format, device, uptime_s }` |
| POST | `/predict` | Run inference on a face image | `{ image: base64, au_data?: {...} }` | `{ ai_score, confidence, model_version, inference_ms }` |
| GET | `/models` | List available models | — | `[{ version, format, path, active }]` |
| POST | `/models/activate` | Switch active model | `{ version: "1.1" }` | `{ ok, active_model }` |

**Predict endpoint detail:**

Request:
```json
{
  "image": "base64-encoded JPEG",
  "au_data": {
    "AU4": 2.1,
    "AU6": 1.3,
    "AU7": 0.8,
    "AU9": 0.5,
    "AU10": 1.2,
    "AU43": 0.0
  }
}
```

Response:
```json
{
  "ai_score": 6.3,
  "confidence": 0.87,
  "model_version": "1.0.0",
  "inference_ms": 142,
  "au_weights_applied": {
    "AU4": 0.35,
    "AU6": 0.10,
    "AU7": 0.10,
    "AU9": 0.15,
    "AU10": 0.15,
    "AU43": 0.15
  }
}
```

### Authentication & Authorization

- **Node.js API:** No authentication (local clinical tool, same as current)
- **Python Inference API:** Bound to `127.0.0.1` only — not accessible from outside the machine
- **Model management endpoints:** Localhost-only binding provides sufficient access control for Phase 1
- **Future:** If multi-user access is needed, add JWT auth to Node.js API

---

## Non-Functional Requirements Coverage

### NFR-001: Performance — Inference Latency

**Requirement:** AI inference < 500ms per frame (CPU); < 100ms (GPU)

**Architecture Solution:**
- ONNX Runtime as primary inference engine (2-5x faster than PyTorch on CPU)
- Image preprocessing pipeline optimized: single face crop, resize to 224x224, minimal transforms
- Model quantization (INT8) available for further CPU speedup
- No network hops — Python service on localhost
- Connection pooling from Node.js to Python service (keep-alive HTTP)

**Implementation Notes:**
- Benchmark during development: log `inference_ms` for every prediction
- If > 500ms on target hardware, apply INT8 quantization or reduce model size
- Consider batching if multiple frames arrive simultaneously

**Validation:**
- Automated benchmark: 100 frames, measure p95 latency
- Log inference_ms in production for ongoing monitoring

---

### NFR-002: Performance — Concurrent Sessions

**Requirement:** Support 3 concurrent sessions with < 1s inference per frame

**Architecture Solution:**
- FastAPI with Uvicorn workers (async, can handle concurrent requests)
- Request queuing — inference is sequential per-worker but multiple workers can run
- Start with 1 worker; scale to 2-3 if needed (configurable via `--workers` flag)

**Implementation Notes:**
- Default: `uvicorn main:app --workers 1` (sufficient for most cases)
- For 3 concurrent sessions: `--workers 2` with model loaded per-worker

**Validation:**
- Load test: 3 concurrent clients sending frames at 2/second

---

### NFR-003: Security — Patient Data Privacy

**Requirement:** All inference local; no data sent to external cloud ML services

**Architecture Solution:**
- Python service binds to `127.0.0.1:5000` — physically cannot be reached from network
- No outbound network calls from Python service (model loaded from local files)
- No cloud ML SDK dependencies (no boto3, no google-cloud-ai, no openai)
- Frame images stored only in `server/frames/` directory

**Implementation Notes:**
- Dependency audit: ensure no packages phone home
- Firewall rule: Python service port 5000 blocked on all network interfaces (localhost exempt)

**Validation:**
- Network packet capture during inference session — zero external calls

---

### NFR-004: Security — Model Access Control

**Requirement:** Only admin can modify models or trigger retraining

**Architecture Solution:**
- Model directory (`server/models/`) has restricted file permissions
- Model management API endpoints (`/models/activate`) bound to localhost only
- Training pipeline is CLI-only (no web API for training)

**Implementation Notes:**
- Set `chmod 700 server/models/` on deployment
- Training scripts require local terminal access

**Validation:**
- Verify model endpoints return 403/404 from external network

---

### NFR-005: Reliability — Graceful Degradation

**Requirement:** System falls back to PSPI when AI service is unavailable

**Architecture Solution:**
- Node.js inference proxy uses timeout + try/catch around inference call
- Timeout: 2 seconds (generous but bounded)
- On failure: set `ai_score=NULL`, continue with PSPI score
- Health check polls inference service every 30 seconds; caches status
- Frontend checks `ai_score` presence to show/hide AI indicator

**Implementation Notes:**
```javascript
// In Node.js sample ingestion
let aiResult = null;
try {
    const resp = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: frameBase64, au_data: auData }),
        signal: AbortSignal.timeout(2000)
    });
    if (resp.ok) aiResult = await resp.json();
} catch (e) {
    // AI service unavailable — continue with PSPI only
    console.warn('Inference service unavailable:', e.message);
}
```

**Validation:**
- Kill Python service during active session — verify PSPI continues uninterrupted
- Restart Python service — verify AI scores resume automatically

---

### NFR-006: Compatibility — Model Formats

**Requirement:** Support ONNX and PyTorch model formats

**Architecture Solution:**
- Model Manager auto-detects format from file extension
- ONNX: loaded via `onnxruntime.InferenceSession`
- PyTorch: loaded via `torch.load()` + `model.eval()`
- Unified inference interface: both return same output format

**Implementation Notes:**
```python
class ModelManager:
    def load(self, path):
        if path.endswith('.onnx'):
            self.session = ort.InferenceSession(path)
            self.format = 'onnx'
        elif path.endswith('.pt') or path.endswith('.pth'):
            self.model = torch.load(path, map_location='cpu')
            self.model.eval()
            self.format = 'pytorch'
```

**Validation:**
- Load same model in both formats; verify outputs differ by < 0.1 on test images

---

### NFR-007: Usability — Transparent Scoring

**Requirement:** UI clearly indicates AI vs PSPI score source and confidence

**Architecture Solution:**
- API responses include `ai_score` (nullable) alongside existing `score` (PSPI)
- Frontend displays both when AI is available; PSPI-only with indicator when not
- Confidence shown as color-coded badge (green > 0.7, yellow 0.5-0.7, red < 0.5)
- Session chart supports dual-line mode (PSPI line + AI line)

**Validation:**
- Visual inspection: AI available vs. unavailable states clearly distinguishable

---

### NFR-008: Maintainability — Modular Architecture

**Requirement:** Python and Node.js services fully decoupled, independently updatable

**Architecture Solution:**
- Separate processes, separate codebases, communicate only via HTTP
- No shared files (except model directory, read-only from Node.js perspective)
- Python service has its own `requirements.txt` and virtual environment
- Independent health checks for each service
- `start.bat` manages both but they can be started/stopped independently

**Validation:**
- Restart Python service while Node.js is running — verify Node.js unaffected
- Restart Node.js while Python is running — verify Python unaffected

---

## Security Architecture

### Authentication

- **Phase 1:** No authentication (local clinical tool, single-user deployment)
- **Future consideration:** JWT-based auth if multi-user or remote access is needed

### Authorization

- **Python service:** Bound to localhost only — inherently protected
- **Model management:** CLI access only for training; localhost API for model switching
- **Patient data:** Same access model as current (whoever has backend URL access)

### Data Encryption

- **In transit:** HTTPS via Cloudflare Tunnel (browser to Node.js). Localhost traffic (Node.js to Python) is unencrypted but never leaves the machine.
- **At rest:** SQLite database and frame images on local filesystem. Rely on OS-level disk encryption (BitLocker on Windows).
- **Model files:** No encryption needed (not sensitive data).

### Security Best Practices

- Input validation on all API endpoints (Pydantic models in FastAPI, Express middleware)
- Image size limits (max 10MB per frame) to prevent memory attacks
- No `eval()` or dynamic code execution on user input
- Dependency pinning (`requirements.txt` with exact versions)
- No secrets in code (model paths via environment variables)

---

## Scalability & Performance

### Scaling Strategy

**Current (Phase 1):** Single machine, single instance of each service. Adequate for 1-3 concurrent sessions.

**Future scaling path:**
1. **Vertical:** Add GPU to deployment machine → inference drops from ~400ms to ~50ms
2. **Horizontal (if needed):** Run multiple Python inference workers behind a local load balancer
3. **Cloud (Phase 3+):** If multi-location deployment needed, containerize both services with Docker

### Performance Optimization

- **ONNX Runtime** with CPU execution provider (default) or CUDA provider (if GPU available)
- **Model quantization:** INT8 quantization reduces model size ~4x and inference time ~2x on CPU
- **Image preprocessing:** Single-pass pipeline (decode → crop → resize → normalize) — no redundant operations
- **Connection reuse:** Node.js keeps persistent HTTP connection to Python service (keep-alive)

### Caching Strategy

- **Model:** Loaded once on startup, kept in memory. No per-request loading.
- **Face detection:** No caching needed — each frame is unique
- **Inference service status:** Node.js caches health check result for 30 seconds to avoid polling on every request

### Load Balancing

Not needed for Phase 1 (single machine). If multiple workers are used, Uvicorn handles internal load balancing.

---

## Reliability & Availability

### High Availability Design

- **Primary mechanism:** Graceful degradation (NFR-005). The system is always available because PSPI fallback requires no external service.
- **Python service auto-restart:** `start.bat` can be enhanced to use a process manager (PM2 or equivalent) for auto-restart on crash.
- **No single point of failure for core functionality** — PSPI scoring is entirely in-browser.

### Disaster Recovery

- **RPO:** Near-zero. SQLite database with WAL mode provides crash recovery.
- **RTO:** < 5 minutes. Restart services via `start.bat`.
- **Backups:** Daily copy of `paingauge.db` and `models/` directory. Can be automated via scheduled task.

### Backup Strategy

```
# Daily backup (add to Windows Task Scheduler)
copy server\paingauge.db backups\paingauge-%date%.db
xcopy server\models backups\models-%date% /E /I
```

### Monitoring & Alerting

- **Inference latency:** Logged per-request (`inference_ms` in response)
- **Service health:** `/api/health` reports both Node.js and Python service status
- **Error logging:** Both services log to stdout/stderr (visible in terminal windows from `start.bat`)
- **Future:** Structured logging to file with rotation; simple dashboard showing avg latency, error rate, uptime

---

## Integration Architecture

### External Integrations

| System | Direction | Protocol | Purpose |
|--------|-----------|----------|---------|
| Cloudflare Tunnel | Outbound | HTTPS | Expose Node.js API to internet |
| GitHub Pages | N/A | Static hosting | Serve frontend |
| UNBC-McMaster Dataset | Import (one-time) | File download | Training data |

### Internal Integrations

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| Browser | Node.js | HTTPS (tunnel) | All API calls |
| Node.js | Python Service | HTTP localhost:5000 | Inference requests |
| Training CLI | Model files | Filesystem | Export trained models |
| Training CLI | SQLite | Direct access | Export corrected samples |

### Message/Event Architecture

Not applicable for Phase 1. All communication is synchronous request/response.

---

## Development Architecture

### Code Organization

```
Pain Gauge/
├── index.html                    # Frontend entry point
├── css/
│   └── styles.css
├── js/
│   ├── main.js                   # App logic
│   ├── api-client.js             # Backend API client
│   └── session-chart.js          # Chart rendering
├── server/
│   ├── index.js                  # Node.js backend (extended)
│   ├── paingauge.db              # SQLite database
│   ├── frames/                   # Captured frame images
│   ├── start.bat                 # Startup script (both services)
│   └── package.json
├── inference/                    # NEW: Python inference service
│   ├── main.py                   # FastAPI application
│   ├── model_manager.py          # Model loading and management
│   ├── preprocessing.py          # Image preprocessing pipeline
│   ├── au_weights.py             # AU-zone weight configuration
│   ├── config.py                 # Service configuration
│   ├── requirements.txt          # Python dependencies
│   └── tests/
│       ├── test_predict.py
│       ├── test_model_manager.py
│       └── test_preprocessing.py
├── training/                     # NEW: Training pipeline (CLI)
│   ├── prepare_dataset.py        # UNBC-McMaster preprocessing
│   ├── train.py                  # Fine-tuning script
│   ├── evaluate.py               # Model evaluation
│   ├── export_corrections.py     # Export clinician corrections
│   └── requirements.txt
├── models/                       # NEW: Model storage
│   ├── active/
│   │   ├── model.onnx
│   │   └── model-meta.json
│   └── archive/
└── docs/
    ├── prd-pain-gauge-ai-engine-2026-03-15.md
    └── architecture-pain-gauge-ai-engine-2026-03-15.md
```

### Module Structure

**Python Inference Service modules:**

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI app, endpoint definitions, startup/shutdown hooks |
| `model_manager.py` | Load/switch models, format detection, version tracking |
| `preprocessing.py` | Image decode, face crop, resize, normalize |
| `au_weights.py` | Weighted AU-zone scoring configuration and calculation |
| `config.py` | Environment variables, defaults, model paths |

### Testing Strategy

**Python Inference Service:**
- **Unit tests:** `preprocessing.py`, `au_weights.py`, `model_manager.py` — test with sample images and mock models
- **Integration tests:** Full predict endpoint with a small test model
- **Benchmark tests:** Measure inference latency on test images
- **Coverage target:** 80%+ for inference service

**Node.js Backend:**
- **Integration tests:** Sample ingestion with/without AI service available
- **Fallback tests:** Verify PSPI-only mode when AI service is down

**Training Pipeline:**
- **Unit tests:** Dataset loading, preprocessing, export
- **Validation:** Model accuracy on UNBC-McMaster test set

### CI/CD Pipeline

**Phase 1 (simple):**
1. Manual testing locally
2. `git push` to GitHub → GitHub Pages auto-deploys frontend
3. Restart `start.bat` on server to pick up backend changes

**Future:**
- GitHub Actions: Run Python tests on push
- Automated model benchmark on new model upload

---

## Deployment Architecture

### Environments

| Environment | Purpose | Services |
|-------------|---------|----------|
| Development | Local development and testing | Node.js + Python (both localhost) |
| Production | Clinical use | Node.js + Python + Cloudflare Tunnel (same machine) |

Single environment for Phase 1 — development machine IS production machine.

### Deployment Strategy

**Updated `start.bat`:**
```bat
@echo off
echo Starting Pain Gauge Backend...
cd /d "%~dp0"

REM Kill leftover processes
taskkill /F /IM cloudflared.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1

timeout /t 2 /noq >nul

REM Start Node.js backend
start "Pain Gauge Server" cmd /k "node index.js"

REM Start Python inference service
start "AI Inference" cmd /k "cd ..\inference && python -m uvicorn main:app --host 127.0.0.1 --port 5000"

echo Waiting for services to start...
timeout /t 8 /noq >nul

REM Verify services
curl -s http://localhost:3000/api/health >nul 2>&1
if errorlevel 1 (echo WARNING: Node.js server not ready)
curl -s http://localhost:5000/health >nul 2>&1
if errorlevel 1 (echo WARNING: AI inference service not ready - will use PSPI fallback)

echo Starting Cloudflare Tunnel...
echo Backend: https://paingauge.sidetalk.io
cloudflared tunnel run paingauge
```

### Infrastructure as Code

Not applicable for Phase 1 (single machine deployment). Future: Dockerfile for each service if containerization is needed.

---

## Requirements Traceability

### Functional Requirements Coverage

| FR ID | FR Name | Components | Status |
|-------|---------|------------|--------|
| FR-001 | Pre-trained Model Integration | Python Service, Model Manager | Covered |
| FR-002 | Frame-to-Score Inference | Python Service, Node.js Proxy | Covered |
| FR-003 | AU-Weighted Pain Estimation | Python Service (au_weights.py) | Covered |
| FR-004 | Confidence Score | Python Service, Frontend | Covered |
| FR-005 | Model Versioning | Model Manager, Node.js API | Covered |
| FR-006 | UNBC-McMaster Dataset Integration | Training Pipeline | Covered |
| FR-007 | Fine-tuning Pipeline | Training Pipeline | Covered |
| FR-008 | Clinician Score Correction | Node.js API, Frontend, SQLite | Covered |
| FR-009 | Python Inference Service | Python Service (FastAPI) | Covered |
| FR-010 | Score Comparison Dashboard | Frontend (session-chart.js) | Covered |
| FR-011 | AI Scores in Sample Records | SQLite Schema, Node.js API | Covered |
| FR-012 | Training Dataset Management | Training Pipeline, Node.js API | Covered |

**Coverage: 12/12 FRs (100%)**

### Non-Functional Requirements Coverage

| NFR ID | NFR Name | Solution | Validation |
|--------|----------|----------|------------|
| NFR-001 | Inference Latency | ONNX Runtime, localhost, no hops | Benchmark 100 frames, p95 < 500ms |
| NFR-002 | Concurrent Sessions | Uvicorn workers, async FastAPI | Load test 3 streams |
| NFR-003 | Patient Data Privacy | Localhost-only inference, no cloud | Network audit |
| NFR-004 | Model Access Control | Localhost binding, file permissions | External access test |
| NFR-005 | Graceful Degradation | Try/catch proxy, 2s timeout, PSPI fallback | Kill service mid-session |
| NFR-006 | Model Formats | ONNX + PyTorch dual support | Load both, compare outputs |
| NFR-007 | Transparent Scoring | Dual display, confidence badge, fallback indicator | Visual inspection |
| NFR-008 | Modular Architecture | Separate processes, HTTP API only | Independent restart test |

**Coverage: 8/8 NFRs (100%)**

---

## Trade-offs & Decision Log

### Decision 1: FastAPI over Flask

**Choice:** FastAPI for Python inference service
**Trade-off:**
- Gain: Async support, automatic OpenAPI docs, Pydantic validation, better performance
- Lose: Flask's larger ecosystem and simpler learning curve
**Rationale:** Performance and type safety matter for an inference service. FastAPI is the modern standard.

### Decision 2: ONNX Runtime as primary, PyTorch for training

**Choice:** Dual framework approach
**Trade-off:**
- Gain: Optimized CPU inference (ONNX) + flexible training (PyTorch)
- Lose: Must maintain model export pipeline (PyTorch → ONNX)
**Rationale:** 2-5x inference speedup on CPU justifies the conversion step.

### Decision 3: Proxy through Node.js (not direct browser-to-Python)

**Choice:** Node.js proxies inference requests to Python service
**Trade-off:**
- Gain: Single external endpoint, centralized auth (future), Python service stays localhost-only
- Lose: One extra network hop (~1ms on localhost — negligible)
**Rationale:** Security (NFR-003) and simplicity. Browser only talks to one backend.

### Decision 4: SQLite over PostgreSQL

**Choice:** Keep existing SQLite
**Trade-off:**
- Gain: Zero setup, existing data preserved, adequate for local single-user tool
- Lose: No concurrent write support, harder to scale
**Rationale:** Level 2 project, single-machine deployment. PostgreSQL is overkill.

### Decision 5: AU data from frontend (face-api.js) passed to Python, not re-computed

**Choice:** Python service receives AU data from frontend via Node.js, does not run its own AU detection
**Trade-off:**
- Gain: No duplicate face detection, lower Python service resource usage, simpler pipeline
- Lose: Dependent on face-api.js accuracy; can't use superior AU detection (OpenFace) in Python
**Rationale:** Start simple. If face-api.js AU quality is insufficient, add Python-side AU detection as an enhancement (open question from PRD).

---

## Open Issues & Risks

| # | Issue | Impact | Mitigation |
|---|-------|--------|------------|
| 1 | UNBC-McMaster dataset access/licensing | Cannot train or benchmark without it | Apply for access early; BioVid as fallback |
| 2 | Pre-trained model availability | May need to train from scratch | Survey HuggingFace, PapersWithCode for pain detection models |
| 3 | face-api.js AU accuracy | Weighted AU scoring depends on reliable AU values | Benchmark against OpenFace; add Python-side detection if needed |
| 4 | CPU inference speed on target hardware | May exceed 500ms on older machines | ONNX quantization, smaller model architecture |
| 5 | Python environment management on Windows | venv/pip issues common on Windows | Document setup clearly; consider conda as alternative |

---

## Assumptions & Constraints

**Assumptions:**
1. Deployment machine has Python 3.11+ installed (or can install it)
2. 8GB+ RAM available for running both services + model in memory
3. UNBC-McMaster dataset will be obtainable for research use
4. A suitable pre-trained model exists or can be trained within reasonable time
5. face-api.js provides sufficiently accurate AU values for weighted scoring

**Constraints:**
1. All processing must be local (no cloud ML) — NFR-003
2. Must not break existing functionality (PSPI scoring, session recording)
3. Single machine deployment (no distributed systems)
4. Windows as primary deployment OS

---

## Future Considerations

1. **Multimodal fusion (Phase 2):** Add Apple Watch physiological signals (heart rate, HRV, EDA). The Python service becomes the fusion layer — accepting both facial and physiological inputs.

2. **Docker containerization:** Package both services as Docker containers for easier deployment on any machine.

3. **GPU acceleration:** ONNX Runtime with CUDA provider for 5-10x speedup. No architecture changes needed — just install CUDA and update config.

4. **Cloud deployment:** If needed for multi-site use, containerize and deploy to cloud with proper security (VPN, auth, encrypted storage).

5. **Model marketplace:** Support downloading community-trained models from a registry (HuggingFace-style).

6. **Real-time video pipeline:** Replace frame-by-frame with WebRTC stream processing for higher throughput.

---

## Approval & Sign-off

**Review Status:**
- [ ] Product Owner (Eran)
- [ ] Technical Lead (Eran)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-15 | Eran | Initial architecture |

---

## Next Steps

### Phase 4: Sprint Planning & Implementation

Run `/sprint-planning` to:
- Break epics into detailed user stories
- Estimate story complexity
- Plan sprint iterations
- Begin implementation following this architectural blueprint

**Key Implementation Principles:**
1. Follow component boundaries defined in this document
2. Implement NFR solutions as specified
3. Use technology stack as defined
4. Follow API contracts exactly
5. Adhere to security and performance guidelines

---

**This document was created using BMAD Method v6 - Phase 3 (Solutioning)**

*To continue: Run `/workflow-status` to see your progress and next recommended workflow.*

---

## Appendix A: Technology Evaluation Matrix

| Category | Option A | Option B | Choice | Why |
|----------|----------|----------|--------|-----|
| Python Framework | FastAPI | Flask | FastAPI | Async, auto-docs, Pydantic, faster |
| ML Inference | ONNX Runtime | PyTorch (direct) | ONNX Runtime | 2-5x CPU speedup |
| ML Training | PyTorch | TensorFlow | PyTorch | Simpler API, better research ecosystem |
| Face Detection (Python) | MediaPipe | dlib | MediaPipe | Faster, maintained by Google, no cmake |
| Image Processing | OpenCV | Pillow | OpenCV | Industry standard for CV, faster |

---

## Appendix B: Capacity Planning

| Resource | Estimate | Notes |
|----------|----------|-------|
| Model in memory | 100-500MB | Depends on architecture (ResNet-50 ~100MB, VGG16 ~500MB) |
| Python service RAM | 500MB-1GB | Model + runtime overhead |
| Node.js RAM | 100-200MB | Existing, unchanged |
| Disk (models) | 1-2GB | Active + 2-3 archived versions |
| Disk (frames) | ~5KB/frame | JPEG thumbnails, ~50MB per 100-sample session |
| Total RAM | 1-2GB | Both services running |

---

## Appendix C: Cost Estimation

| Item | Cost | Notes |
|------|------|-------|
| Infrastructure | $0 | Local machine, existing hardware |
| Cloudflare Tunnel | $0 | Free tier |
| GitHub Pages | $0 | Free for public repos |
| Python libraries | $0 | Open source (PyTorch, ONNX, FastAPI) |
| UNBC-McMaster Dataset | $0 | Free for research use (application required) |
| GPU (optional) | $0-300 | Only if purchasing dedicated GPU for inference |
| **Total Phase 1** | **$0** | All open source, local deployment |
