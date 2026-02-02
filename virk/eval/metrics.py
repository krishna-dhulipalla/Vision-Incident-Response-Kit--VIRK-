from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

class MetricCalculator:
    """Calculates accuracy metrics for Drift Detection and Fingerprinting."""
    
    @staticmethod
    def evaluate_detection(clean_scores: List[float], drifted_scores: List[float]) -> Dict[str, float]:
        """
        Evaluate drift detection performance using AUROC.
        
        Args:
            clean_scores: List of drift magnitudes for clean (in-distribution) samples.
            drifted_scores: List of drift magnitudes for shifted (out-of-distribution) samples.
            
        Returns:
            Dict containing AUROC score and summary stats.
        """
        if not clean_scores or not drifted_scores:
            return {"auroc": 0.0, "clean_mean": 0.0, "shifted_mean": 0.0}
            
        y_true = [0] * len(clean_scores) + [1] * len(drifted_scores)
        y_scores = clean_scores + drifted_scores
        
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auroc = 0.5 # Fail gracefully if only one class present
            
        return {
            "auroc": float(auroc),
            "clean_mean": float(np.mean(clean_scores)),
            "shifted_mean": float(np.mean(drifted_scores)),
            "clean_std": float(np.std(clean_scores)),
            "shifted_std": float(np.std(drifted_scores))
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
        if not labels:
            return {"accuracy": 0.0}
            
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        acc = accuracy_score(y_true, y_pred)
        
        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "labels": labels
        }
