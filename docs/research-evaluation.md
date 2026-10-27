# Research evaluation and reproducibility

## Isolated CPU environment

Use Python 3.11 for the tested research environment. The generated `requirements/research.txt` pins direct and transitive packages, including a compatible [PyTorch 2.5.1 / torchvision 0.20.1 pair](https://pytorch.org/get-started/previous-versions/), and selects CPU wheels on Windows/Linux x86-64. The library's general dependency ranges remain flexible. These pins specify a reproducible environment; they are not a claim that these are the latest releases.

PowerShell:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements/research.txt
.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python.exe -m pip check
$env:VIRK_MODEL_SMOKE = "1"
.venv/Scripts/python.exe -m pytest -q
```

On Linux, use `python3.11 -m venv .venv` and replace `.venv/Scripts/python.exe` with `.venv/bin/python`. Set `VIRK_MODEL_SMOKE=1` in the shell when enabling the model test. That test uses a real ResNet-18 with `pretrained=False`, so it exercises extraction, calibration, saved weights and replay without downloading weights or claiming pretrained-model accuracy. CI is configured to run it on Windows and Linux; the lightweight test matrix also covers Python 3.9–3.11.

Run the remaining examples from the activated environment (`.venv/Scripts/Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux), or invoke `.venv/Scripts/virk.exe` / `.venv/bin/virk` directly.

To intentionally refresh pins, edit `requirements/research.in`, regenerate with the command in the generated file header, and rerun both the unit and real-model tests. Keep separate environments for optional demo tools such as Transformers and experiment trackers.

## Independent clean-data evaluation

```bash
virk calibrate --dataset data/clean_calibration --test-dataset data/clean_test --batch-size 64 --num-batches 100 --seed 42 --output calibration.json
```

`clean_calibration` is internally split 50/50 into reference and threshold-fitting images. `clean_test` is never used to choose the threshold. Full batches are scored using the fixed reference and the strict `score > threshold` comparison. The CLI rejects path overlap and byte-identical copies between the two directories. It does not detect semantically duplicated frames or establish independence: prepare separate subjects, cameras or time periods as required by your use case.

The output distinguishes calibration FPR from independent test FPR. The test report includes the number of false-positive batches, denominator, and a 95% [Wilson score interval](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm). This interval assumes independent representative Bernoulli batch outcomes. Non-overlapping batches alone do not establish independence if images are temporally correlated. Zero observed false positives still has a positive upper confidence limit; neither the estimate nor interval guarantees future performance. Do not tune the threshold after inspecting the test result and continue to call that set independent.

The default CLI extractor is pretrained ResNet-18. Programmatic calibration accepts `extractor=your_extractor`. Keep the baseline, extractor, normalization, kernel, subsampling, and window size consistent with deployment. Threshold fitting uses biased RBF MMD-squared with `gamma = 1 / embedding_dimension`, not a distribution-free significance test.

## Full evaluation reports

```bash
virk eval --dataset-name cifar10 --batch-size 16 --window-size 64 --fpr 0.05 --seed 42 --sample-interval 0.1 --benchmark-inference
```

The evaluation harness shuffles once with the supplied seed and creates disjoint 20/40/40 reference/calibration/test splits. It fits a threshold using complete clean calibration windows, evaluates it on complete independent test windows, then generates corruptions only from test images. Partial scoring windows are excluded and counted. Too-small datasets and failed corruption generation cause explicit errors rather than fabricated metrics. A random image split is a benchmark protocol; use group/time holdouts for real deployment evidence.

HTML and JSON reports include the threshold, AUROC, FPR/TPR at that threshold, Wilson test interval, fingerprint confusion matrix, split counts, model configuration and seed. Fingerprint labels describe matches to known synthetic corruption signatures and do not establish causality or performance on novel shifts.

Each shift also creates a clean-then-shifted embedding stream. Calibration/test windows have `window_size` samples. Streaming updates contain `batch_size` samples, allowing overlap when the batch is smaller than the window. Every streaming check waits for a full window. Supply a representative sample cadence; the default one second per sample is only a simulation parameter.

## Recorded traffic and operational metrics

```bash
virk benchmark-stream --reference reference.npy --embeddings ordered_traffic.npy --labels shifted_labels.npy --threshold 0.0385 --window-size 1000 --batch-size 32 --sample-interval 0.1 --output stream_report.json
```

Reference and traffic arrays must have shapes `(N, D)` and `(M, D)` with finite real values. Labels must be a vector of `M` boolean/0/1 values marking shifted samples. Preserve traffic order and use a threshold selected on separate calibration data. This evaluator assumes a constant sampling interval; resample or evaluate runs separately if arrival cadence changes substantially.

- A false-positive check requires its entire recent window to be clean. Windows containing shifted samples during recovery are excluded from the clean denominator.
- Every flagged clean check counts as a false alert, matching the current middleware's repeated-event behavior. No cooldown or alert deduplication is simulated. False alerts per clean hour use only the intervals associated with eligible clean checks, excluding warmup and contaminated windows.
- A shift episode is a contiguous run of shifted labels. Delay runs from its first sample to the first flagged check while the episode is active, at the time of the last sample in that checked batch. Undetected episodes remain in the report with null delays. Delay percentiles summarize detected episodes only; read them together with the missed count. This definition measures time to a threshold violation, not causal attribution.
- Per-check traces expose sample endpoints, clean-window labels, scores and decisions. Overlapping checks are dependent, so the streaming report deliberately omits binomial confidence intervals.
- Monitoring latency measures window updates and drift checks on prerecorded embeddings, after warmup. It excludes inference, feature extraction and background incident processing. These measurements describe the machine running the experiment.

## Measure actual inference overhead

`virk eval --benchmark-inference` compares TIMM extraction alone against extraction plus the middleware hook on the same clean test batches. It alternates the measurement order, excludes initial model/monitor warmup, and reports baseline, monitored and paired-difference p50/p95 timing. The monitor has no image paths, fingerprinting or remote storage in this benchmark. It therefore measures the monitoring hook, not the full cost of production incident handling.

For your own model:

```python
from virk.eval.streaming import measure_inference_overhead

# predict_with_embeddings(batch) should perform the inference work you want to
# measure and return the exact embeddings passed to monitor.process_batch.
try:
    timings = measure_inference_overhead(
        predict_with_embeddings, monitor, representative_batches, repeats=5
    )
finally:
    monitor.shutdown()
```

Synchronize GPU work inside the predictor when measuring GPU wall time. Background completion is excluded from timed calls and drained separately. Negative paired differences can occur from timing noise. Report runtime, hardware, batch size, window size and load alongside latency results; these microbenchmarks do not establish service throughput or tail latency under contention.

## Replay configuration and exact drift decisions

```python
monitor = VirkMiddleware.create(
    reference_embeddings=reference,
    extractor=extractor,
    drift_threshold=validated_threshold,
    incident_threshold=validated_threshold,
    window_size=64,
    min_window_size=64,
    seed=42,
    capture_model_state=True,
)
```

Middleware incidents record the exact post-subsampling reference/current embeddings used for the decision, kernel gamma and threshold, window configuration, extractor/model identity, preprocessing, installed package versions, and artifact hashes. Evidence is captured before diagnostics are queued, so later detector calls cannot replace the incident's operands. The default image cap remains 20.

When processing asynchronously, keep image files available until queued incident work completes. Embedding evidence is copied at detection time; image files are read later by the worker.

For TIMM, `capture_model_state=True` saves the current state dictionary once when the monitor is created and includes it in incidents. Keep the inference model frozen after monitor creation; recreate the monitor when changing weights or preprocessing. For a custom extractor, supply `replay_config={"extractor": ...}` and `model_state_path="weights.pt"`, or implement `describe()` and `save_model_state(path)`. A supplied checkpoint is copied and hashed; VIRK cannot infer whether it corresponds to a custom model.

After extracting a bundle:

```bash
python replay.py
python replay.py --verify-drift
```

Default inspection only needs Python. Drift replay needs NumPy and scikit-learn and verifies saved artifact hashes, recomputes the MMD score, and checks the decision using the recorded threshold. Recorded model weights are optional and can substantially increase bundle size. Hashes check accidental changes relative to the manifest, not authenticity against a party who can rewrite both.

Exact drift replay operates on saved embeddings. Re-running model inference for the whole window additionally requires all original input images and application preprocessing, which capped incident samples do not provide. For TIMM checkpoint restoration, create the recorded architecture with `pretrained=False`, load `model_state.pt` with `torch.load(..., weights_only=True)`, and use the recorded preprocessing configuration. Older inspection-only bundles remain readable and report explicitly when drift replay context is absent.
