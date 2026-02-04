# Vision Incident Response Kit (VIRK)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Beta-green)]()

**Production diagnostics for vision model failures.**

</div>

VIRK is a lightweight "flight recorder" for computer vision pipelines. It sits alongside your inference service, detects conceptual drift (blur, lighting, camera shifts), diagnoses the root cause, and automatically bundles "incident packs" for reproducible debugging.

## Key Features

- **🔎 Automated Drift Detection**: Scalable MMD-based detection (O(1) complexity) to spot distribution shifts instantly.
- **🧬 Cause Fingerprinting**: Tells you _why_ it failed (e.g., "Motion Blur detected", "Brightness shift").
- **🍰 Metadata Slicing**: Identifies _where_ it failed (e.g., "Camera 02 at Night").
- **📦 Incident Bundler**: Automatically creates zip bundles with images, metadata, and a replay script for local reproduction.
- **🔌 Production Middleware**: Drop-in wrapper for your inference loop.

## Installation

### From Source (Recommended)

VIRK is currently in active development. We recommend installing in **editable mode** to access the CLI tools:

```bash
git clone https://github.com/krishna-dhulipalla/Vision-Incident-Response-Kit--VIRK-.git
cd Vision-Incident-Response-Kit--VIRK-
pip install -e .
```

### Optional Dependencies

- **S3 Storage**: `pip install -e .[s3]`
- **Plotting**: `pip install -e .[plot]`
- **OpenTelemetry**: `pip install -e .[otel]`

## Quick Start: Benchmarking

Verify VIRK's performance on standard datasets using the built-in evaluation harness.

```bash
# Evaluate on CIFAR-10 (Standard)
virk eval --dataset-name cifar10

# Evaluate on Flowers-102 (Fine-grained)
virk eval --dataset-name flowers102
```

_Results will be saved to `eval_out/report_<dataset>_<timestamp>.html`._

## Integration Guide

### 1. The Easy Way: `VirkMiddleware`

Wrap your prediction logic with our middleware. It handles drift detection, metric emission, and incident bundling automatically.

```python
from virk.integration.middleware import VirkMiddleware
from virk.monitor.window import WindowedDriftDetector
from virk.repro.bundler import ReproBundler
from virk.repro.storage import S3Storage

# 1. Setup Components
detector = WindowedDriftDetector(ref_embeddings, window_size=1000)
bundler = ReproBundler(output_dir="/tmp/bundles")
storage = S3Storage(bucket="my-incident-bucket")

# 2. Initialize Middleware
monitor = VirkMiddleware(
    drift_detector=detector,
    bundler=bundler,
    storage=storage,
    incident_threshold=0.05, # Sensitivity
    service_name="vision-service"
)

# 3. Inside your inference loop
def predict(images):
    embeddings = model.encode(images)
    # ... logic ...

    # 🔗 Hook! (Non-blocking usually recommended)
    monitor.process_batch(
        embeddings=embeddings,
        image_paths=[img.path for img in images],
        metadata=[{"cam": img.cam_id} for img in images]
    )
```

### 2. See it in Action (FastAPI Demo)

Run a complete end-to-end demo service that simulates drift and triggers an incident response:

```bash
python examples/fastapi_service.py
```

## CLI Tools

### Incident Summary

Instant Root Cause Analysis (RCA) on a captured bundle:

```bash
virk incident summarize incident_1234abcd.zip
```

**Output:**

```text
==================================================
 VIRK INCIDENT SUMMARY: 1234abcd
==================================================
Timestamp:   2024-03-20T10:00:00
Drift Mag:   0.1542 (CRITICAL)

Probable Root Cause:
  > MOTION_BLUR (Score: 0.82)
  > (Secondary) DEFOCUS_BLUR (Score: 0.12)

Affected Slices (Targeting Guidance):
  - camera_id:cam_02 (Contribution: 0.082)
--------------------------------------------------
Action: Run 'python replay.py' inside the unzipped bundle.
==================================================
```

## Contributing

Running tests:

```bash
pytest tests/
```
