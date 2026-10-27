from pathlib import Path

import numpy as np
import pytest

from virk.core.types import ShiftFingerprint
from virk.eval.engine import EvalEngine
from virk.eval.reporter import ReportGenerator


class PathExtractor:
    def extract(self, paths):
        # Shifted paths add a known offset; clean embeddings have low variance.
        return np.array(
            [[int(Path(p).stem) / 100, 5.0 if "shifted" in p else 0] for p in paths]
        )


class Generator:
    def generate(self, paths):
        return {
            "brightness_s3": [
                str(Path(p).parent / "shifted" / Path(p).name) for p in paths
            ]
        }


class Fingerprinter:
    def diagnose(self, batch):
        return ShiftFingerprint({"brightness": 4.0})


def test_evaluation_has_disjoint_splits_and_matching_window_settings(tmp_path):
    for i in range(40):
        (tmp_path / f"{i}.png").touch()
    engine = EvalEngine(
        str(tmp_path),
        str(tmp_path / "output"),
        batch_size=2,
        window_size=4,
        extractor=PathExtractor(),
        fingerprinter=Fingerprinter(),
        generator=Generator(),
        benchmark_inference=True,
    )
    reference, calibration, test = map(
        set, (engine.ref_paths, engine.calibration_paths, engine.test_paths)
    )
    assert reference.isdisjoint(calibration)
    assert reference.isdisjoint(test)
    assert calibration.isdisjoint(test)
    report = engine.run()
    assert report["independent_clean_test"]["num_batches"] == 4
    assert report["calibration"]["num_batches"] == 4
    stream = report["per_shift_streaming"]["brightness_s3"]
    assert stream["settings"]["window_size"] == 4
    assert stream["settings"]["batch_size"] == 2
    assert stream["settings"]["threshold"] == report["calibration"]["threshold"]
    assert stream["detected_episodes"] == 1
    assert report["inference_overhead"]["paired_overhead"]["observations"] == 24
    report_path = tmp_path / "report.html"
    ReportGenerator().generate(report, str(report_path))
    html = report_path.read_text(encoding="utf-8")
    assert "Independent Clean Test" in html
    assert "Wilson interval" in html
    assert "Ordered Stream Evaluation" in html
    assert "Measured Inference Overhead" in html


def test_evaluation_rejects_small_dataset_before_loading_model(tmp_path):
    (tmp_path / "one.png").touch()
    with pytest.raises(ValueError, match="enough images"):
        EvalEngine(str(tmp_path), str(tmp_path / "output"))


def test_missing_shifted_images_cannot_silently_bias_evaluation(tmp_path):
    for i in range(40):
        (tmp_path / f"{i}.png").touch()
    generator = Generator()
    generator.generate = lambda paths: {"brightness_s3": paths[:-1]}
    engine = EvalEngine(
        str(tmp_path),
        str(tmp_path / "output"),
        batch_size=2,
        window_size=4,
        extractor=PathExtractor(),
        fingerprinter=Fingerprinter(),
        generator=generator,
    )
    with pytest.raises(ValueError, match="Incomplete shifted dataset"):
        engine.run()
