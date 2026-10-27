import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from virk.monitor.drift import DriftDetector
from virk.eval.statistics import false_positive_summary


class CalibrationEngine:
    """Estimate a drift threshold from held-out clean batch scores."""

    def __init__(
        self, dataset_path: str, extractor_name: str = "resnet18", extractor=None
    ):
        self.dataset_path = dataset_path
        self.extractor_name = extractor_name
        self.extractor = extractor

    def calibrate(
        self,
        fpr_target: float = 0.05,
        num_batches: int = 50,
        batch_size: int = 32,
        seed: Optional[int] = 42,
        test_dataset_path: Optional[str] = None,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Target an empirical batch FPR, without a future-data guarantee.

        Split clean images into disjoint reference and validation sets, then
        score up to ``num_batches`` full validation batches. The observed FPR
        is measured on the same scores used to select the threshold; it is
        not an independent estimate of generalization performance.
        """
        if not np.isfinite(fpr_target) or not 0 <= fpr_target <= 1:
            raise ValueError("fpr_target must be between 0 and 1.")
        for name, value in (("num_batches", num_batches), ("batch_size", batch_size)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        num_batches, batch_size = int(num_batches), int(batch_size)

        dataset = Path(self.dataset_path)
        if not dataset.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset}")
        image_paths = sorted(
            str(path)
            for path in dataset.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        test_paths = []
        if test_dataset_path is not None:
            test_root = Path(test_dataset_path)
            if not test_root.is_dir():
                raise ValueError(f"Test dataset directory does not exist: {test_root}")
            test_paths = sorted(
                str(path)
                for path in test_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if {Path(p).resolve() for p in image_paths} & {
                Path(p).resolve() for p in test_paths
            }:
                raise ValueError(
                    "Test images must be disjoint from reference/calibration images"
                )
            if len(test_paths) < batch_size:
                raise ValueError("Test dataset must contain at least one full batch")
            if not 0 < confidence < 1:
                raise ValueError("confidence must be between 0 and 1")
            # Catch renamed copies as well as overlapping directories.
            import hashlib

            def digest(path):
                hasher = hashlib.sha256()
                with open(path, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                return hasher.digest()

            clean_hashes = {digest(p) for p in image_paths}
            if any(digest(p) in clean_hashes for p in test_paths):
                raise ValueError(
                    "Test dataset contains copies of reference/calibration images"
                )
        split = len(image_paths) // 2
        available_batches = (len(image_paths) - split) // batch_size
        if split == 0 or available_batches == 0:
            raise ValueError(
                "Not enough clean images for a nonempty reference split and "
                f"one full validation batch of {batch_size}."
            )
        if available_batches < num_batches:
            warnings.warn(
                f"Only {available_batches} full clean validation batches available "
                f"(requested {num_batches}); empirical FPR resolution is "
                f"{1 / available_batches:.1%}. Use more clean data for a more stable estimate.",
                UserWarning,
                stacklevel=2,
            )
        num_batches = min(num_batches, available_batches)
        np.random.default_rng(seed).shuffle(image_paths)
        ref_paths = image_paths[:split]
        val_paths = image_paths[split : split + num_batches * batch_size]

        # Validate the dataset before loading a model or downloading weights.
        if self.extractor is None:
            from virk.monitor.extractor import TimmExtractor

            self.extractor = TimmExtractor(model_name=self.extractor_name)

        print("Extracting Reference Embeddings...")
        ref_embs = np.concatenate(
            [
                self.extractor.extract(ref_paths[i : i + batch_size])
                for i in range(0, len(ref_paths), batch_size)
            ]
        )
        detector = DriftDetector(ref_embs, random_state=seed)

        print(f"Running Validation on {len(val_paths)} images...")
        scores = []
        for i in range(0, len(val_paths), batch_size):
            batch = self.extractor.extract(val_paths[i : i + batch_size])
            scores.append(detector.detect(batch).drift_magnitude)

        # 'higher' uses an observed score and is conservative for finite samples.
        # Detection uses score > threshold, including when scores are tied.
        scores = np.asarray(scores)
        threshold = float(np.quantile(scores, 1 - fpr_target, method="higher"))
        result = {
            "recommended_threshold": threshold,
            "target_fpr": float(fpr_target),
            "empirical_fpr": float(np.mean(scores > threshold)),
            "num_batches": num_batches,
            "batch_size": batch_size,
            "mean_clean_score": float(np.mean(scores)),
            "std_clean_score": float(np.std(scores)),
            "max_clean_score": float(np.max(scores)),
            "seed": seed,
            "reference_images": len(ref_paths),
            "calibration_scores": scores.tolist(),
            "extractor": (
                self.extractor.describe()
                if hasattr(self.extractor, "describe")
                else {"class": type(self.extractor).__name__}
            ),
        }
        if test_dataset_path is not None:
            # Freeze the threshold before touching independent clean test scores.
            test_scores = []
            for i in range(0, len(test_paths) - batch_size + 1, batch_size):
                embeddings = self.extractor.extract(test_paths[i : i + batch_size])
                test_scores.append(
                    detector.detect(embeddings, threshold=threshold).drift_magnitude
                )
            result["independent_test"] = false_positive_summary(
                test_scores, threshold, confidence
            )
            result["independent_test"]["scores"] = test_scores
            result["independent_test"]["excluded_images"] = len(test_paths) % batch_size
        return result
