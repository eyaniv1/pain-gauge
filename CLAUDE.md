# Algoscope - System Architecture Reference

## Overview

Algoscope is a browser-based multimodal pain detection system. It combines facial action unit analysis (AU), CNN-based facial expression inference, heart rate/HRV monitoring, and body motion analysis to produce a real-time pain score (0-10). It has a Node.js backend for data persistence and a FastAPI microservice for CNN inference.

## File Map

| File | Role |
|------|------|
| `index.html` | Two-screen UI: calibration + main (tabs: Camera, Chart, History) |
| `js/main.js` | Orchestrator: settings, session lifecycle, data flow, UI updates |
| `js/pain-engine.js` | Rules-based facial AU pain scoring (PSPI formula) |
| `js/body-motion.js` | MediaPipe Pose body motion pain detection |
| `js/session-chart.js` | Canvas time-series chart with 6 series + legend checkboxes |
| `js/gauge.js` | Semicircular pain gauge needle visualization |
| `js/api-client.js` | HTTP client for backend (retry with backoff on 429/529) |
| `js/ble-heart-rate.js` | BLE heart rate sensor via Web Bluetooth API |
| `css/styles.css` | Responsive layout (flexbox, mobile breakpoints) |
| `server/index.js` | Express.js + SQLite backend (patients, sessions, samples, frames) |
| `inference/main.py` | FastAPI CNN inference service (ONNX/PyTorch) |
| `training/train.py` | Model training (SimplePainModel / ComparativePainModel) |

## Pain Score Pipeline

```
MediaPipe FaceMesh (468 pts)    BLE Sensor     MediaPipe Pose (33 pts)
        |                           |                   |
        v                           v                   v
  pain-engine.js              HR/HRV data        body-motion.js
  (PSPI AU rules)                   |           (guarding, bracing,
        |                           |            restlessness, freezing)
        v                           v                   v
  facePain (0-10)            hrPain (0-10)       bodyPain (0-10)
        |                           |                   |
        +------ Fusion Weights -----+-------------------+
        |   (proportional normalization, sum = 1.0)
        v
   Total Score (0-10, smoothed)
```

### Face Score = AU + CNN Blend

The face score is a weighted blend of two engines:
- **AU engine** (rules-based): `pain-engine.js` processes FaceMesh landmarks using PSPI formula
- **CNN engine** (deep learning): async score from backend inference service

```
faceScore = auScore * (1 - cnnWeight) + cnnScore * cnnWeight
```

`cnnWeight` default is 0.4. CNN scores arrive asynchronously via `flushSamples()` response.

### Fusion Weights

Three weights (`faceWeight`, `hrWeight`, `bodyWeight`) are proportionally normalized at runtime. When a modality has no data or weight=0, the remaining weights scale up proportionally. The UI sliders are interdependent - changing one auto-redistributes the others to maintain sum = 1.0.

### Calibration

1. User selects current pain level (0-10) - stored as `baselinePainLevel`
2. ~45 frames captured to establish neutral face baseline
3. HR baseline captured if sensor connected
4. All subsequent scores are deviations from this baseline
5. Face pain anchored bidirectionally: positive deviation scales toward 10, negative toward 0

## DEFAULTS (in main.js)

```
Face AU: au4Sensitivity=5, au67Sensitivity=4, au910Sensitivity=5,
         au43Threshold=0.55, au43Sensitivity=8, talkingSuppression=0.5
CNN:     cnnWeight=0.4
HR:      hrElevationSensitivity=1.0, hrvSuppressionSensitivity=1.0
Body:    guardingSensitivity=1.0, bracingSensitivity=1.0,
         restlessnessSensitivity=1.0, freezingSensitivity=1.0
Fusion:  faceWeight=0.5, hrWeight=0.3, bodyWeight=0.2
General: smoothingSize=8, sampleIntervalMs=500, backendUrl=''
```

Settings sync: AU/HR/fusion params → `engine` instance, body params → `bodyMotion` instance, `cnnWeight` stays in `settings` only (used in `main.js` blending logic).

## Session Chart Series

| Key | Label | Color | Style | Default Visible |
|-----|-------|-------|-------|----------------|
| total | Total | #1a1a2e | solid 3px | yes |
| face | Face | #8e44ad | solid 2px | no |
| au | AU Engine | #2980b9 | dashed 1.5px | no |
| cnn | CNN | #c0392b | dashed 1.5px | no |
| body | Body | #27ae60 | solid 2px | no |
| hr | HR Pain | #e67e22 | solid 2px | no |

`addSample(totalScore, faceScore, auScore, bodyScore, hrPainScore, frame)` - called from `maybeRecordSample()`. CNN samples added separately via `addCnnSample(time, score)`.

## CNN Inference Flow

```
main.js: sampleBuffer accumulates samples
  → every 5s: flushSamples() → api-client.js POST /api/sessions/{id}/samples
  → server/index.js: stores samples, forwards frames to inference service
  → inference/main.py: runs CNN, returns ai_score
  → server returns ai_results in response
  → main.js: chart.addAiSample(), lastCnnScore = ai_score
  → next sample: blended into faceScore via cnnWeight
```

## Backend API Endpoints

- `GET /api/health` - connection check
- `GET/POST /api/patients` - list/create patients
- `GET/PUT/DELETE /api/patients/{id}` - patient CRUD
- `GET/POST /api/patients/{id}/sessions` - list/create sessions
- `GET/PUT/DELETE /api/sessions/{id}` - session CRUD
- `POST /api/sessions/batch-delete` - bulk delete
- `GET/POST /api/sessions/{id}/samples` - get/send samples
- `PUT /api/samples/{id}/correct` - clinician score correction
- `GET /api/inference/status` - inference service health
- `POST /api/inference/calibrate` - send calibration frame
- `POST /api/inference/clear-calibration` - reset AI calibration
- `GET /api/frames/{filename}` - serve captured frame images

## Key Architectural Notes

- **No build step**: vanilla JS, loaded via script tags in index.html
- **Offline-capable**: works without backend (AU engine + body motion only)
- **CNN weight (`cnnWeight`) is NOT set on the engine** - it's only in `settings`, used in `main.js` blending logic
- **Body motion auto-calibrates** from first ~30 frames of pose detection
- **HR pain scoring** estimates resting HR/HRV by subtracting pain component from baseline (~3 bpm and ~8% HRV per pain level)
- **Fusion weight redistribution**: when slider changes, others adjust proportionally to maintain sum = 1.0
- **Chart resets visibility** to total-only on each new session start
- **Frames stored as base64 data URLs** in sample buffer, saved as JPEG files by server
