# Vision Incident Response Kit (VIRK)

**Diagnostic tooling for vision model failures in production.**

VIRK is a Python library designed for ML platform engineers. It provides drift detection, root cause fingerprinting, and deterministic reproduction bundles for image classification pipelines. It is designed to sit alongside your inference service (FastAPI/Triton), not replace it.

## Features

- **Drift Monitor (`timm` compatible)**: Detects distribution shift using Maximum Mean Discrepancy (MMD) on feature embeddings.
- **Calibrated Fingerprinting**: Identifies shift types (Blur, Noise, Brightness, JPEG) using Z-score normalization against a clean baseline.
- **Multi-dimensional Slicing**: Attributes drift to metadata intersections (e.g., `camera_id=02` AND `time=night`).
- **Incident Bundling**: Creates deterministic, hashed zip bundles containing the specific images and a `replay.py` script for local reproduction.
- **Operational Integration**:
  - **Middleware**: Drop-in wrapper for inference loops.
  - **Prometheus**: Native exporter for drift magnitude and top-cause counters.
  - **S3 Support**: Automatically offload incident bundles to scalable storage.

## Installation

```bash
pip install virk
# Optional: Install S3 and Plotting support
pip install virk[s3, plot]
```

## Quick Start (Local)

```python
import numpy as np
from virk.monitor.extractor import TimmExtractor
from virk.fingerprint.matcher import Fingerprinter

# 1. Initialize with Reference Data (Clean Baseline)
# VIRK uses a pretrained ResNet18 by default for feature extraction
extractor = TimmExtractor(model_name='resnet18', pretrained=True)
ref_paths = ["./data/clean_1.jpg", "./data/clean_2.jpg", ...]
ref_embs = extractor.extract(ref_paths)

# 2. Calibrate Fingerprinter (Establishes noise floor)
fingerprinter = Fingerprinter(extractor, ref_paths)
fingerprinter.calibrate()

# 3. Diagnose Production Batch
prod_embs = extractor.extract(prod_paths)
diagnosis = fingerprinter.diagnose(prod_embs)

print(f"Top Shift Cause: {diagnosis.top_cause} (Z-Score: {diagnosis.top_score:.2f})")
# Output: Top Shift Cause: MOTION_BLUR (Z-Score: 12.45)
```

## Production Integration

### Middleware Pattern

Wrap your inference logic to automatically track drift and log incidents.

```python
from virk.integration.middleware import VirkMiddleware
from virk.monitor.window import WindowedDriftDetector

# Initialize components
detector = WindowedDriftDetector(ref_embs, window_size=1000)
middleware = VirkMiddleware(detector, fingerprinter, service_name="inspection-v1")

# In your inference loop
def predict(batch_images, metadata):
    embeddings = model.encode(batch_images)

    # Async hook: updates metrics, logs drift, uploads incidents if critical
    middleware.process_batch(embeddings, metadata)

    return model.classify(embeddings)
```

### Prometheus Metrics

VIRK exposes standard metrics on port 8000 by default when using `virk.integration.prometheus`.

- `virk_drift_magnitude{ref_id="baseline"}`: Gauge (0.0 - 1.0)
- `virk_drift_detected_total{cause="motion_blur"}`: Counter

## CLI Tools

### Evaluation Harness

Verify VIRK's sensitivity on your dataset before deploying.

```bash
# Run standard corruption suite on your clean data
python -m virk.cli eval --dataset ./cifar10_clean --output-dir ./report
```

_Outputs `report.html` with AUROC and Confusion Matrices._

### Incident Summarizer

Get instant Root Cause Analysis (RCA) from a downloaded repro bundle.

```bash
virk incident summarize incident_a1b2.zip
```

**Output:**

```text
VIRK INCIDENT SUMMARY: incident_a1b2
==================================================
Timestamp:   2026-02-02 14:30:00
Drift Mag:   0.0921 (CRITICAL)
Data Hash:   e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

Probable Root Cause:
  > MOTION_BLUR (Score: 12.45)
  > (Secondary) DEFOCUS_BLUR (Score: 8.10)

Affected Slices (Targeting Guidance):
  - camera_id:cam_02 (Contribution: 0.045)
--------------------------------------------------
Action: Run 'python replay.py' inside the unzipped bundle for reproduction.
```

## Performance Benchmarks

Validated on **CIFAR-10** (Natural Images) with synthetic corruptions (Severity 3).

| Corruption Type      | Detection AUROC | Feature                                             |
| :------------------- | :-------------- | :-------------------------------------------------- |
| **JPEG Compression** | **1.00**        | Perfect separation from clean baseline.             |
| **Scale / Resize**   | **1.00**        | Robustly detected as resolution artifacts.          |
| **Motion Blur**      | **0.75**        | Detectable; may alias with Defocus Blur in low-res. |
| **Gaussian Noise**   | **1.00**        | Highly sensitive to sensor noise.                   |

## License

Apache 2.0
