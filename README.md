# Vision Incident Response Kit (VIRK)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Beta-green)]()

**Production diagnostics for vision model failures.**

</div>

VIRK is a lightweight "flight recorder" for computer vision pipelines. It sits alongside your inference service, detects conceptual drift (blur, lighting, camera shifts), diagnoses the root cause, and automatically bundles "incident packs" for reproducible debugging.

## Architecture

```mermaid
graph LR
    Inference[Inference Service] -->|Embeddings| MW[Middleware]
    MW -->|Sync Check| Detector{Drift?}
    Detector -->|No| Metrics[Prometheus/OTel]
    Detector -->|Yes| Queue[(Async Queue)]

    subgraph Background Worker
        Queue --> Fingerprinter
        Queue --> Slicer
        Fingerprinter --> Bundler
    end

    Bundler -->|Zip| Storage[S3 / Local]
    Bundler -->|Alert| Metrics
```

## Key Features

- **🔎 Scalable Drift Detection**: Uses Linear MMD with subsampling. Complexity is **O(k²)** (where k=subsample size, e.g. 1000) regardless of total stream volume N. O(1) relative to traffic.
- **⚡ Async & Non-Blocking**: Heavy diagnostics (Fingerprinting, Bundling) run in a background thread thread pool (default queue size=100). If the queue fills up under heavy load, new drift events are dropped (Load Shedding) to protect inference latency.
- **🧬 Cause Fingerprinting**: Tells you _why_ it failed (e.g., "Motion Blur detected", "Brightness shift").
- **📦 Incident Bundler**: Automatically creates zip bundles with images, metadata, and a replay script for local reproduction.

## Installation

```bash
git clone https://github.com/krishna-dhulipalla/Vision-Incident-Response-Kit--VIRK-.git
cd Vision-Incident-Response-Kit--VIRK-
pip install -e .
```

### Optional Dependencies

- **S3 Storage**: `pip install -e .[s3]`
- **Prometheus**: `pip install -e .[prometheus]`
- **OpenTelemetry**: `pip install -e .[otel]`

## Integration Guide

### 1. The Easy Way: `VirkMiddleware.create()`

Wrap your prediction logic with our middleware. The `create()` factory uses sensible defaults.

```python
from virk.integration.middleware import VirkMiddleware

# 1. Initialize (One Line)
# Note: 'extractor' must be a Callable or Model that takes images and returns embeddings.
monitor = VirkMiddleware.create(
    reference_embeddings=ref_data,  # Your training/baseline embeddings
    extractor=model,                 # Feature extractor (e.g. TIMM model interface)
    output_dir="/tmp/bundles",       # Where to save incidents
    metrics_type="prometheus",       # 'prometheus', 'otel', or 'none'. Otel falls back to Prom if missing.
    async_processing=True            # Run heavy tasks in background
)

# 2. Inside your inference loop
def predict(images):
    embeddings = model.encode(images)

    # 🔗 Hook! (Non-blocking)
    monitor.process_batch(
        embeddings=embeddings,
        image_paths=[img.path for img in images],
        metadata=[{"cam": img.cam_id} for img in images]
    )
```

### 2. Calibration (FPR Targeting)

Don't guess your drift threshold. Use `calibrate` to find a threshold that guarantees a specific False Positive Rate (e.g., 5%) on your clean data.

```bash
virk calibrate --dataset path/to/clean_data --fpr 0.05
```

**Output:**

```text
==================================================
 VIRK THRESHOLD CALIBRATION
==================================================
Target FPR:          5.0%
Max Clean Score:     0.0421
--------------------------------------------------
RECOMMENDED THRESHOLD: 0.038512
==================================================
```

## DemoService

VIRK comes with a production-ready demo service you can run immediately:

```bash
# Runs uvicorn with the demo app
virk demo-prod --port 8080
```

## CLI Tools

### Incident Summary

Instant Root Cause Analysis (RCA) on a captured bundle:

```bash
virk incident summarize incident_1234abcd.zip
```

### Benchmarking

Verify VIRK's performance on standard datasets:

```bash
virk eval --dataset-name cifar10
```

## Contributing

Running tests:

```bash
pytest tests/
```
