from unittest.mock import Mock

import numpy as np
import pytest

from virk.eval.streaming import StreamingEvaluator, measure_inference_overhead


def test_stream_detects_shift_and_excludes_contaminated_recovery_windows():
    reference = np.zeros((20, 2))
    traffic = np.r_[np.zeros((12, 2)), np.full((8, 2), 5.0), np.zeros((12, 2))]
    labels = np.r_[np.zeros(12), np.ones(8), np.zeros(12)]
    report = StreamingEvaluator(reference, threshold=0.1, window_size=4).run(
        traffic, labels, batch_size=2, sample_interval_seconds=0.5
    )
    assert report["false_alert_checks"] == 0
    assert report["empirical_clean_check_fpr"] == 0
    assert report["detected_episodes"] == 1
    assert report["episodes"][0]["delay_samples"] == 1
    assert report["episodes"][0]["delay_seconds"] == 0.5
    recovery = next(row for row in report["trace"] if row["samples_seen"] == 22)
    assert not recovery["clean_window"]
    assert report["monitoring_latency"]["p95_ms"] >= 0
    assert "dependent" in report["uncertainty"]


def test_missed_events_remain_censored_in_report():
    traffic = np.zeros((20, 2))
    labels = np.r_[np.zeros(8), np.ones(4), np.zeros(8)]
    report = StreamingEvaluator(np.zeros((8, 2)), threshold=1, window_size=4).run(
        traffic, labels, batch_size=2
    )
    assert report["missed_episodes"] == 1
    assert report["episodes"][0]["delay_seconds"] is None
    assert report["delay_p50_seconds"] is None


def test_false_alert_exposure_and_no_clean_observations():
    report = StreamingEvaluator(np.zeros((4, 2)), threshold=0.1, window_size=4).run(
        np.ones((8, 2)) * 5,
        np.zeros(8),
        batch_size=2,
        sample_interval_seconds=0.5,
    )
    assert report["false_alert_checks"] == 3
    assert report["clean_exposure_seconds"] == 3
    assert report["false_alerts_per_clean_hour"] == 3600
    all_shift = StreamingEvaluator(np.zeros((4, 2)), threshold=0.1, window_size=4).run(
        np.ones((8, 2)) * 5,
        np.ones(8),
        batch_size=2,
    )
    assert all_shift["empirical_clean_check_fpr"] is None
    assert all_shift["false_alerts_per_clean_hour"] is None


@pytest.mark.parametrize("labels", [np.zeros(7), np.full(8, 2), np.full(8, np.nan)])
def test_stream_label_validation(labels):
    with pytest.raises(ValueError):
        StreamingEvaluator(np.zeros((4, 2)), 0.1, window_size=4).run(
            np.zeros((8, 2)), labels
        )


def test_inference_overhead_runs_real_prediction_and_hook_pairs():
    predict, monitor = Mock(side_effect=lambda batch: batch), Mock()
    report = measure_inference_overhead(predict, monitor, [np.zeros((4, 2))], repeats=2)
    assert predict.call_count == 5  # warmup plus two paired measurements
    assert monitor.process_batch.call_count == 3
    assert report["paired_overhead"]["observations"] == 2


def test_window_update_owns_data_and_waits_for_full_window():
    from virk.monitor.window import WindowedDriftDetector

    detector = WindowedDriftDetector(np.zeros((4, 2)), window_size=4, min_window_size=4)
    batch = np.zeros((2, 2))
    detector.update(batch)
    batch[:] = 100
    assert not detector.detect_trend().is_drift_detected
    detector.update(np.zeros((2, 2)))
    assert detector.detect_trend().drift_magnitude == pytest.approx(0)
