"""Evaluation with disjoint reference, calibration and clean test splits."""

import os
import platform
from pathlib import Path

import numpy as np

from virk.eval.metrics import MetricCalculator
from virk.eval.statistics import false_positive_summary
from virk.eval.streaming import StreamingEvaluator, measure_inference_overhead
from virk.monitor.drift import DriftDetector


class _BatchedExtractor:
    def __init__(self, extractor, batch_size):
        self.extractor, self.batch_size = extractor, batch_size

    def extract(self, images):
        return np.concatenate(
            [
                self.extractor.extract(images[i : i + self.batch_size])
                for i in range(0, len(images), self.batch_size)
            ]
        )


class EvalEngine:
    """Calibrate once, then evaluate fixed thresholds on untouched test data."""

    def __init__(
        self,
        dataset_path,
        output_dir,
        batch_size=32,
        window_size=32,
        fpr_target=0.05,
        seed=42,
        sample_interval_seconds=1.0,
        benchmark_inference=False,
        extractor=None,
        fingerprinter=None,
        generator=None,
    ):
        for name, value in (("batch_size", batch_size), ("window_size", window_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(fpr_target) or not 0 <= fpr_target <= 1:
            raise ValueError("fpr_target must be between 0 and 1")
        if not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.dataset_path, self.output_dir = dataset_path, output_dir
        self.batch_size, self.window_size = batch_size, window_size
        self.fpr_target, self.seed = fpr_target, seed
        self.sample_interval_seconds = sample_interval_seconds
        self.benchmark_inference = benchmark_inference
        self.image_paths = sorted(
            str(p)
            for p in Path(dataset_path).rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        np.random.default_rng(seed).shuffle(self.image_paths)
        reference_end, calibration_end = int(len(self.image_paths) * 0.2), int(
            len(self.image_paths) * 0.6
        )
        self.ref_paths = self.image_paths[:reference_end]
        calibration_paths = self.image_paths[reference_end:calibration_end]
        test_paths = self.image_paths[calibration_end:]
        self.calibration_paths = calibration_paths[
            : len(calibration_paths) // window_size * window_size
        ]
        self.test_paths = test_paths[: len(test_paths) // window_size * window_size]
        if (
            not self.ref_paths
            or not self.calibration_paths
            or len(self.test_paths) < max(window_size, batch_size)
        ):
            raise ValueError(
                "Need enough images for 20% reference, 40% calibration and 40% test splits, including full scoring windows and a fingerprint batch"
            )
        self.excluded_images = (
            len(calibration_paths)
            - len(self.calibration_paths)
            + len(test_paths)
            - len(self.test_paths)
        )
        if extractor is None:
            from virk.monitor.extractor import TimmExtractor

            extractor = TimmExtractor(model_name="resnet18")
        self.extractor = extractor
        self.batched_extractor = _BatchedExtractor(extractor, batch_size)
        if generator is None:
            from virk.eval.generator import ShiftGenerator

            generator = ShiftGenerator(os.path.join(output_dir, "shifted_data"))
        self.generator = generator
        self.ref_embeddings = self.batched_extractor.extract(self.ref_paths)
        self.detector = DriftDetector(self.ref_embeddings, random_state=seed)
        if fingerprinter is None:
            from virk.fingerprint.matcher import Fingerprinter

            fingerprinter = Fingerprinter(self.batched_extractor, self.ref_paths)
            fingerprinter.calibrate()
        self.fingerprinter = fingerprinter

    def _scores(self, embeddings):
        return [
            self.detector.detect(embeddings[i : i + self.window_size]).drift_magnitude
            for i in range(0, len(embeddings) - self.window_size + 1, self.window_size)
        ]

    def run(self):
        calibration_embeddings = self.batched_extractor.extract(self.calibration_paths)
        calibration_scores = self._scores(calibration_embeddings)
        threshold = float(
            np.quantile(calibration_scores, 1 - self.fpr_target, method="higher")
        )
        clean_embeddings = self.batched_extractor.extract(self.test_paths)
        clean_scores = self._scores(clean_embeddings)
        independent_test = false_positive_summary(clean_scores, threshold)
        shifted_data = self.generator.generate(self.test_paths)
        if not shifted_data:
            raise ValueError("No shifted datasets generated")
        drift_scores_all, true_causes, predicted_causes = [], [], []
        shift_results, streaming_results = {}, {}
        for shift_key, paths in shifted_data.items():
            if len(paths) != len(self.test_paths):
                raise ValueError(
                    f"Incomplete shifted dataset {shift_key}: {len(paths)}/{len(self.test_paths)} images"
                )
            embeddings = self.batched_extractor.extract(paths)
            scores = self._scores(embeddings)
            drift_scores_all.extend(scores)
            shift_results[shift_key] = MetricCalculator.evaluate_detection(
                clean_scores, scores, threshold=threshold
            )
            streaming_results[shift_key] = StreamingEvaluator(
                self.ref_embeddings,
                threshold,
                window_size=self.window_size,
                seed=self.seed,
            ).run(
                np.concatenate([clean_embeddings, embeddings]),
                np.r_[
                    np.zeros(len(clean_embeddings), dtype=bool),
                    np.ones(len(embeddings), dtype=bool),
                ],
                batch_size=self.batch_size,
                sample_interval_seconds=self.sample_interval_seconds,
            )
            for i in range(0, len(embeddings) - self.batch_size + 1, self.batch_size):
                fingerprint = self.fingerprinter.diagnose(
                    embeddings[i : i + self.batch_size]
                )
                true_causes.append(shift_key.rsplit("_", 1)[0])
                predicted_causes.append(
                    max(fingerprint.shift_types, key=fingerprint.shift_types.get)
                )

        report = {
            "detection_overall": MetricCalculator.evaluate_detection(
                clean_scores, drift_scores_all, threshold=threshold
            ),
            "fingerprint_overall": MetricCalculator.evaluate_fingerprinting(
                true_causes, predicted_causes
            ),
            "per_shift_detection": shift_results,
            "calibration": {
                "threshold": threshold,
                "target_fpr": self.fpr_target,
                "empirical_fpr": float(
                    np.mean(np.asarray(calibration_scores) > threshold)
                ),
                "num_batches": len(calibration_scores),
                "scores": calibration_scores,
            },
            "independent_clean_test": independent_test,
            "per_shift_streaming": streaming_results,
            "experiment": {
                "seed": self.seed,
                "batch_size": self.batch_size,
                "window_size": self.window_size,
                "reference_images": len(self.ref_paths),
                "calibration_images": len(self.calibration_paths),
                "test_images": len(self.test_paths),
                "excluded_images": self.excluded_images,
                "split": "Seeded random 20/40/40; partial scoring windows excluded",
                "sample_interval_seconds": self.sample_interval_seconds,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "extractor": (
                    self.extractor.describe()
                    if hasattr(self.extractor, "describe")
                    else {"class": type(self.extractor).__name__}
                ),
                "limitations": "Wilson intervals assume independent representative clean batches. Synthetic shift episodes and supplied arrival cadence do not establish production performance.",
            },
        }
        if self.benchmark_inference:
            from virk.integration.middleware import VirkMiddleware
            from virk.monitor.window import WindowedDriftDetector

            monitor = VirkMiddleware(
                WindowedDriftDetector(
                    self.ref_embeddings,
                    window_size=self.window_size,
                    min_window_size=self.window_size,
                    random_state=self.seed,
                ),
                drift_threshold=threshold,
            )
            batches = [
                self.test_paths[i : i + self.batch_size]
                for i in range(0, len(self.test_paths), self.batch_size)
            ]
            try:
                report["inference_overhead"] = measure_inference_overhead(
                    self.extractor.extract, monitor, batches
                )
            finally:
                monitor.shutdown()
        return report

    def cleanup(self):
        import shutil

        shift_dir = Path(self.output_dir) / "shifted_data"
        if shift_dir.exists():
            shutil.rmtree(shift_dir)
