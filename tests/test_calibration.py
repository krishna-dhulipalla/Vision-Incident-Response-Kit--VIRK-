from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from virk.eval.calibration import CalibrationEngine


class FakeExtractor:
    """Deterministic features without model downloads or image decoding."""

    def __init__(self):
        self.calls = []

    def extract(self, paths):
        self.calls.append(list(paths))
        return np.array([[int(Path(p).stem), 1.0] for p in paths])


def make_dataset(tmp_path, count):
    nested = tmp_path / "class_a"
    nested.mkdir()
    for i in range(count):
        (nested / f"{i}.JPEG").touch()


def test_calibration_includes_last_full_batch_and_repeats_split(tmp_path):
    make_dataset(tmp_path, 12)
    first, second = FakeExtractor(), FakeExtractor()
    result = CalibrationEngine(str(tmp_path), extractor=first).calibrate(
        batch_size=2, num_batches=3, seed=123
    )
    repeated = CalibrationEngine(str(tmp_path), extractor=second).calibrate(
        batch_size=2, num_batches=3, seed=123
    )
    assert result == repeated
    assert first.calls == second.calls
    assert result["num_batches"] == 3
    reference = {p for batch in first.calls[:3] for p in batch}
    validation = {p for batch in first.calls[3:] for p in batch}
    assert len(reference) == len(validation) == 6
    assert reference.isdisjoint(validation)
    assert all(len(batch) <= 2 for batch in first.calls)


def test_calibration_honors_batch_limit(tmp_path):
    make_dataset(tmp_path, 20)
    extractor = FakeExtractor()
    result = CalibrationEngine(str(tmp_path), extractor=extractor).calibrate(
        batch_size=2, num_batches=2
    )
    assert result["num_batches"] == 2
    assert sum(map(len, extractor.calls)) == 10 + 4


def test_one_complete_batch_warns_and_returns_usable_result(tmp_path):
    make_dataset(tmp_path, 4)
    with pytest.warns(UserWarning, match="Only 1 full clean validation batches"):
        result = CalibrationEngine(str(tmp_path), extractor=FakeExtractor()).calibrate(
            batch_size=2
        )
    assert result["num_batches"] == 1
    assert result["empirical_fpr"] == 0
    assert result["recommended_threshold"] == result["max_clean_score"]


def test_small_sample_fpr_uses_conservative_threshold(tmp_path, monkeypatch):
    make_dataset(tmp_path, 8)
    detector = Mock()
    detector.detect.side_effect = [
        Mock(drift_magnitude=score) for score in [0.1, 0.2, 0.3, 0.4]
    ]
    monkeypatch.setattr(
        "virk.eval.calibration.DriftDetector", Mock(return_value=detector)
    )
    result = CalibrationEngine(str(tmp_path), extractor=FakeExtractor()).calibrate(
        fpr_target=0.05, batch_size=1, num_batches=4
    )
    assert result["recommended_threshold"] == 0.4
    assert result["empirical_fpr"] == 0.0


@pytest.mark.parametrize("fpr", [0.0, 0.25, 0.5, 1.0])
def test_tied_scores_are_not_counted_as_false_positives(tmp_path, monkeypatch, fpr):
    make_dataset(tmp_path, 8)
    detector = Mock()
    detector.detect.return_value = Mock(drift_magnitude=0.2)
    monkeypatch.setattr(
        "virk.eval.calibration.DriftDetector", Mock(return_value=detector)
    )
    result = CalibrationEngine(str(tmp_path), extractor=FakeExtractor()).calibrate(
        fpr_target=fpr, batch_size=1, num_batches=4
    )
    assert result["recommended_threshold"] == 0.2
    assert result["empirical_fpr"] == 0


@pytest.mark.parametrize("count", [0, 1, 2])
def test_insufficient_data_fails_before_loading_model(tmp_path, count):
    make_dataset(tmp_path, count)
    engine = CalibrationEngine(str(tmp_path))
    with pytest.raises(ValueError, match="Not enough clean images"):
        engine.calibrate(batch_size=2)
    assert engine.extractor is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fpr_target": -0.1},
        {"fpr_target": 1.1},
        {"fpr_target": float("nan")},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"num_batches": 0},
    ],
)
def test_invalid_calibration_parameters(tmp_path, kwargs):
    with pytest.raises(ValueError):
        CalibrationEngine(str(tmp_path)).calibrate(**kwargs)


def test_missing_dataset(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        CalibrationEngine(str(tmp_path / "missing")).calibrate()


def test_independent_test_cannot_influence_selected_threshold(tmp_path, monkeypatch):
    calibration, test = tmp_path / "calibration", tmp_path / "test"
    calibration.mkdir()
    test.mkdir()
    for i in range(8):
        (calibration / f"{i}.png").write_bytes(f"calibration-{i}".encode())
    for i in range(4):
        (test / f"{100 + i}.png").write_bytes(f"test-{i}".encode())
    detector = Mock()
    detector.detect.side_effect = [
        Mock(drift_magnitude=value)
        for value in [0.1, 0.2, 0.3, 0.4, 9.0, 10.0, 11.0, 12.0]
    ]
    monkeypatch.setattr(
        "virk.eval.calibration.DriftDetector", Mock(return_value=detector)
    )
    result = CalibrationEngine(str(calibration), extractor=FakeExtractor()).calibrate(
        batch_size=1, num_batches=4, fpr_target=0.05, test_dataset_path=str(test)
    )
    assert result["recommended_threshold"] == 0.4
    assert result["empirical_fpr"] == 0
    assert result["independent_test"]["fpr"] == 1
    assert result["independent_test"]["num_batches"] == 4
    assert result["independent_test"]["ci_lower"] < 1
    assert all(
        call.kwargs["threshold"] == 0.4 for call in detector.detect.call_args_list[4:]
    )


def test_same_test_directory_rejected(tmp_path):
    make_dataset(tmp_path, 8)
    with pytest.raises(ValueError, match="disjoint"):
        CalibrationEngine(str(tmp_path)).calibrate(
            batch_size=1, test_dataset_path=str(tmp_path)
        )


def test_renamed_test_copy_rejected(tmp_path):
    calibration, test = tmp_path / "calibration", tmp_path / "test"
    calibration.mkdir()
    test.mkdir()
    (calibration / "0.png").write_bytes(b"same image")
    (test / "renamed.png").write_bytes(b"same image")
    with pytest.raises(ValueError, match="copies"):
        CalibrationEngine(str(calibration)).calibrate(
            batch_size=1, test_dataset_path=str(test)
        )
