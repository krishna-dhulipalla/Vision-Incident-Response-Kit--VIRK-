import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from virk.core.types import DriftProfile
from datetime import datetime

class DriftDetector:
    """Detects distribution drift between reference and current embeddings."""
    
    def __init__(self, reference_embeddings: np.ndarray, max_samples: int = 1000):
        """
        Args:
            reference_embeddings: (N, D) array of baseline embeddings.
            max_samples: Maximum number of samples to use for MMD calculation. 
                         Larger sets are subsampled.
        """
        self.max_samples = max_samples
        
        # Subsample reference immediately if needed
        if len(reference_embeddings) > max_samples:
            indices = np.random.choice(len(reference_embeddings), max_samples, replace=False)
            self.reference_embeddings = reference_embeddings[indices]
        else:
            self.reference_embeddings = reference_embeddings
        
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between two sets of samples using RBF kernel.
        """
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)
        
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def detect(self, current_embeddings: np.ndarray, threshold: float = 0.02) -> DriftProfile:
        """
        Check if current batch has drifted from reference.
        Subsamples current_embeddings if larger than max_samples.
        
        Args:
            current_embeddings: (M, D) array of new embeddings.
            threshold: MMD threshold to flag drift.
            
        Returns:
            DriftProfile object.
        """
        # Subsample current if needed
        target = current_embeddings
        if len(target) > self.max_samples:
             indices = np.random.choice(len(target), self.max_samples, replace=False)
             target = target[indices]

        # Simple heuristic for gamma: 1 / num_features
        gamma = 1.0 / self.reference_embeddings.shape[1] if self.reference_embeddings.shape[1] > 0 else 1.0
        
        mmd_score = self._compute_mmd(self.reference_embeddings, target, gamma=gamma)
        
        return DriftProfile(
            is_drift_detected=bool(mmd_score > threshold),
            drift_magnitude=float(mmd_score),
            reference_id="baseline",
            timestamp=datetime.now().isoformat()
        )
