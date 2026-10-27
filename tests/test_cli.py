import json
import sys
import zipfile
from unittest.mock import Mock

import pytest

from virk.cli import main


def test_calibration_cli_reports_empirical_fpr(monkeypatch, capsys):
    engine = Mock()
    engine.calibrate.return_value = {
        "target_fpr": 0.05,
        "empirical_fpr": 0.04,
        "num_batches": 50,
        "max_clean_score": 0.1,
        "recommended_threshold": 0.08,
    }
    monkeypatch.setattr(
        "virk.eval.calibration.CalibrationEngine", Mock(return_value=engine)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "virk",
            "calibrate",
            "--dataset",
            "clean",
            "--seed",
            "7",
            "--num-batches",
            "50",
        ],
    )
    main()
    engine.calibrate.assert_called_once_with(
        fpr_target=0.05, batch_size=32, num_batches=50, seed=7
    )
    output = capsys.readouterr().out
    assert "Empirical FPR:       4.0%" in output
    assert "future false positive rates may differ" in output


def test_calibration_cli_rejects_small_dataset(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["virk", "calibrate", "--dataset", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "Not enough clean images" in capsys.readouterr().err


@pytest.mark.parametrize("as_zip", [False, True])
def test_summary_reads_hash_from_manifest_metadata(
    tmp_path, monkeypatch, capsys, as_zip
):
    manifest = {"metadata": {"data_hash": "abc123"}}
    if as_zip:
        bundle = tmp_path / "incident.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
    else:
        bundle = tmp_path
        (bundle / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["virk", "incident", "summarize", str(bundle)])
    main()
    assert "Data Hash:   abc123" in capsys.readouterr().out


def test_benchmark_stream_cli_writes_auditable_json(tmp_path, monkeypatch, capsys):
    import numpy as np

    np.save(tmp_path / "reference.npy", np.zeros((8, 2)))
    np.save(tmp_path / "stream.npy", np.r_[np.zeros((8, 2)), np.full((8, 2), 5.0)])
    np.save(tmp_path / "labels.npy", np.r_[np.zeros(8), np.ones(8)])
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "virk",
            "benchmark-stream",
            "--reference",
            str(tmp_path / "reference.npy"),
            "--embeddings",
            str(tmp_path / "stream.npy"),
            "--labels",
            str(tmp_path / "labels.npy"),
            "--threshold",
            "0.1",
            "--window-size",
            "4",
            "--batch-size",
            "2",
            "--sample-interval",
            "0.1",
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["detected_episodes"] == 1
    assert report["trace"]
    assert "Monitoring p95" in capsys.readouterr().out


def test_calibration_cli_independent_report_and_json(tmp_path, monkeypatch, capsys):
    engine = Mock()
    result = {
        "target_fpr": 0.05,
        "empirical_fpr": 0.04,
        "num_batches": 50,
        "max_clean_score": 0.1,
        "recommended_threshold": 0.08,
        "independent_test": {
            "fpr": 0.02,
            "false_positives": 1,
            "num_batches": 50,
            "confidence": 0.95,
            "ci_lower": 0.0035,
            "ci_upper": 0.105,
        },
    }
    engine.calibrate.return_value = result
    monkeypatch.setattr(
        "virk.eval.calibration.CalibrationEngine", Mock(return_value=engine)
    )
    output = tmp_path / "calibration.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "virk",
            "calibrate",
            "--dataset",
            "clean",
            "--test-dataset",
            "independent",
            "--output",
            str(output),
        ],
    )
    main()
    assert engine.calibrate.call_args.kwargs["test_dataset_path"] == "independent"
    assert json.loads(output.read_text()) == result
    assert "95% Wilson interval" in capsys.readouterr().out
