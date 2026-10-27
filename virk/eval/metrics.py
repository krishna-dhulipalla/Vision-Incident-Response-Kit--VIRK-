from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score


class MetricCalculator:
    """Calculates accuracy metrics for Drift Detection and Fingerprinting."""

    @staticmethod
    def evaluate_detection(
        clean_scores: List[float], drifted_scores: List[float], threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Evaluate drift detection performance using AUROC.

        Args:
            clean_scores: List of drift magnitudes for clean (in-distribution) samples.
            drifted_scores: List of drift magnitudes for shifted (out-of-distribution) samples.

        Returns:
            Dict containing AUROC score and summary stats.
        """
        clean_scores = np.asarray(clean_scores, dtype=float)
        drifted_scores = np.asarray(drifted_scores, dtype=float)
        if (
            clean_scores.ndim != 1
            or drifted_scores.ndim != 1
            or not len(clean_scores)
            or not len(drifted_scores)
        ):
            raise ValueError("Detection metrics require both clean and shifted scores")
        if not np.isfinite(clean_scores).all() or not np.isfinite(drifted_scores).all():
            raise ValueError("Detection scores must be finite")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")

        y_true = [0] * len(clean_scores) + [1] * len(drifted_scores)
        y_scores = np.concatenate([clean_scores, drifted_scores])

        auroc = roc_auc_score(y_true, y_scores)

        return {
            "auroc": float(auroc),
            "clean_mean": float(np.mean(clean_scores)),
            "shifted_mean": float(np.mean(drifted_scores)),
            "clean_std": float(np.std(clean_scores)),
            "shifted_std": float(np.std(drifted_scores)),
            # FPR at specific threshold (e.g. 0.05 implies drift)
            "fpr_0_05": float(np.mean(np.array(clean_scores) > 0.05)),
            "threshold": float(threshold),
            "fpr_at_threshold": float(np.mean(np.asarray(clean_scores) > threshold)),
            "tpr_at_threshold": float(np.mean(np.asarray(drifted_scores) > threshold)),
        }

    @staticmethod
    def evaluate_fingerprinting(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Evaluate fingerprinting accuracy using Confusion Matrix.

        Args:
            y_true: List of ground-truth shift names.
            y_pred: List of predicted shift names.

        Returns:
            Dict containing accuracy and confusion matrix.
        """
        labels = sorted(list(set(y_true) | set(y_pred)))
        if not labels or not y_true or len(y_true) != len(y_pred):
            raise ValueError("Fingerprint metrics require matching nonempty labels")

        cm = (
            np.array([[len(y_true)]])
            if len(labels) == 1
            else confusion_matrix(y_true, y_pred, labels=labels)
        )
        acc = accuracy_score(y_true, y_pred)

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        }
