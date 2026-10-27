"""Replay labeled embedding streams and measure monitoring costs."""

from collections import deque
from time import perf_counter

import numpy as np

from virk.monitor.window import WindowedDriftDetector


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=float) * 1000
    return {
        "observations": len(values),
        "p50_ms": float(np.percentile(values, 50)) if len(values) else None,
        "p95_ms": float(np.percentile(values, 95)) if len(values) else None,
        "mean_ms": float(np.mean(values)) if len(values) else None,
    }


class StreamingEvaluator:
    """Score a fixed threshold on ordered traffic; labels mark shifted samples.

    A clean check requires its entire window to be clean. Detection delay is
    measured from episode onset to the first flagged check while the episode
    is still active. Missed episodes remain in the report with null delays.
    """

    def __init__(
        self,
        reference_embeddings,
        threshold,
        window_size=1000,
        max_samples=1000,
        seed=42,
    ):
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        self.reference = np.asarray(reference_embeddings)
        self.threshold = threshold
        self.window_size = window_size
        self.max_samples = max_samples
        self.seed = seed

    def run(self, embeddings, shifted, batch_size=32, sample_interval_seconds=1.0):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        embeddings = WindowedDriftDetector._validate_embeddings(embeddings)
        shifted = np.asarray(shifted)
        if (
            shifted.ndim != 1
            or len(shifted) != len(embeddings)
            or not np.isin(shifted, [0, 1]).all()
        ):
            raise ValueError("shifted must contain one boolean/0/1 label per sample")
        shifted = shifted.astype(bool)
        detector = WindowedDriftDetector(
            self.reference,
            window_size=self.window_size,
            min_window_size=self.window_size,
            max_samples=self.max_samples,
            random_state=self.seed,
        )
        label_window = deque(maxlen=self.window_size)
        trace, timings = [], []
        for start in range(0, len(embeddings), batch_size):
            stop = min(start + batch_size, len(embeddings))
            label_window.extend(shifted[start:stop])
            began = perf_counter()
            detector.update(embeddings[start:stop])
            profile = detector.detect_trend(threshold=self.threshold)
            elapsed = perf_counter() - began
            if len(label_window) < self.window_size:
                continue
            timings.append(elapsed)
            trace.append(
                {
                    "samples_seen": stop,
                    "window_start": max(0, stop - self.window_size),
                    "clean_window": not any(label_window),
                    "is_alert": profile.is_drift_detected,
                    "score": profile.drift_magnitude,
                    "batch_samples": stop - start,
                }
            )
        if not trace:
            raise ValueError("Stream must contain enough samples to fill the window")

        clean_checks = [item for item in trace if item["clean_window"]]
        false_checks = sum(item["is_alert"] for item in clean_checks)
        clean_exposure = (
            sum(item["batch_samples"] for item in clean_checks)
            * sample_interval_seconds
        )
        episodes = []
        changes = np.diff(np.r_[False, shifted, False].astype(int))
        for start, stop in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        ):
            first = next(
                (
                    item
                    for item in trace
                    if item["is_alert"] and start < item["samples_seen"] <= stop
                ),
                None,
            )
            delay_samples = first["samples_seen"] - 1 - int(start) if first else None
            episodes.append(
                {
                    "start_sample": int(start),
                    "stop_sample": int(stop),
                    "detected": first is not None,
                    "delay_samples": delay_samples,
                    "delay_seconds": (
                        delay_samples * sample_interval_seconds if first else None
                    ),
                }
            )
        delays = [e["delay_seconds"] for e in episodes if e["detected"]]
        return {
            "settings": {
                "threshold": self.threshold,
                "batch_size": batch_size,
                "window_size": self.window_size,
                "min_window_size": self.window_size,
                "max_samples": self.max_samples,
                "seed": self.seed,
                "sample_interval_seconds": sample_interval_seconds,
            },
            "num_samples": len(embeddings),
            "scored_checks": len(trace),
            "clean_checks": len(clean_checks),
            "false_alert_checks": false_checks,
            "empirical_clean_check_fpr": (
                false_checks / len(clean_checks) if clean_checks else None
            ),
            "clean_exposure_seconds": clean_exposure,
            "false_alerts_per_clean_hour": (
                false_checks * 3600 / clean_exposure if clean_exposure else None
            ),
            "shift_episodes": len(episodes),
            "detected_episodes": len(delays),
            "missed_episodes": len(episodes) - len(delays),
            "delay_p50_seconds": float(np.percentile(delays, 50)) if delays else None,
            "delay_p95_seconds": float(np.percentile(delays, 95)) if delays else None,
            "monitoring_latency": latency_summary(timings),
            "timing_scope": "Window update and MMD checks only; excludes inference, extraction, diagnostics and warmup.",
            "rate_scope": "Every flagged clean check counts as an alert; no cooldown. Exposure includes only eligible clean check intervals.",
            "uncertainty": "Overlapping windows are dependent; no binomial confidence interval is reported.",
            "episodes": episodes,
            "trace": trace,
        }


def measure_inference_overhead(predict, monitor, batches, repeats=3):
    """Compare inference alone with inference plus process_batch on real inputs.

    ``predict`` returns embeddings. Alternate measurement order to reduce order
    bias. Synchronize GPU work inside predict when measuring GPU wall time.
    Background completion is excluded; call monitor.shutdown() separately.
    """
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    batches = list(batches)
    if not batches:
        raise ValueError("batches cannot be empty")
    # Exclude model and monitor warmup from measurements.
    monitor.process_batch(predict(batches[0]))
    baseline, monitored, differences = [], [], []
    for repeat in range(repeats):
        for index, batch in enumerate(batches):
            pair = {}
            for with_monitor in (
                (False, True) if (repeat + index) % 2 == 0 else (True, False)
            ):
                start = perf_counter()
                embeddings = predict(batch)
                if with_monitor:
                    monitor.process_batch(embeddings)
                pair[with_monitor] = perf_counter() - start
            baseline.append(pair[False])
            monitored.append(pair[True])
            differences.append(pair[True] - pair[False])
    return {
        "baseline_inference": latency_summary(baseline),
        "inference_with_monitor": latency_summary(monitored),
        "paired_overhead": latency_summary(differences),
        "mean_overhead_percent": (
            100 * sum(differences) / sum(baseline) if sum(baseline) else None
        ),
        "timing_scope": "Inference plus inline process_batch; async completion excluded. Negative differences reflect timing noise.",
    }
