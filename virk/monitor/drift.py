import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from virk.core.types import DriftProfile, DetectionEvidence
from datetime import datetime


class DriftDetector:
    """Detects distribution drift between reference and current embeddings."""

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        max_samples: int = 1000,
        random_state=None,
    ):
        """
        Args:
            reference_embeddings: (N, D) array of baseline embeddings.
            max_samples: Maximum number of samples to use for MMD calculation.
                         Larger sets are subsampled.
            random_state: Optional seed for reproducible subsampling. If omitted,
                          preserves the global NumPy random-state behavior.
        """
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, (int, np.integer))
            or max_samples < 1
        ):
            raise ValueError("max_samples must be a positive integer")
        reference_embeddings = self._validate_embeddings(reference_embeddings)
        self.max_samples = max_samples
        self._rng = (
            np.random if random_state is None else np.random.default_rng(random_state)
        )

        # Subsample reference immediately if needed
        if len(reference_embeddings) > max_samples:
            indices = self._rng.choice(
                len(reference_embeddings), max_samples, replace=False
            )
            self.reference_embeddings = reference_embeddings[indices]
        else:
            self.reference_embeddings = reference_embeddings.copy()

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2 or 0 in embeddings.shape:
            raise ValueError("embeddings must be a nonempty (N, D) matrix")
        if (
            not np.issubdtype(embeddings.dtype, np.number)
            or not np.isrealobj(embeddings)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("embeddings must contain finite real numbers")
        return embeddings

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between two sets of samples using RBF kernel.
        """
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)

        # Biased MMD-squared is nonnegative; suppress floating-point roundoff.
        return max(0.0, float(XX.mean() + YY.mean() - 2 * XY.mean()))

    def detect(
        self, current_embeddings: np.ndarray, threshold: float = 0.02
    ) -> DriftProfile:
        """
        Check if current batch has drifted from reference.
        Subsamples current_embeddings if larger than max_samples.

        Args:
            current_embeddings: (M, D) array of new embeddings.
            threshold: MMD threshold to flag drift.

        Returns:
            DriftProfile object.
        """
        return self._detect(current_embeddings, threshold)[0]

    def detect_with_evidence(
        self, current_embeddings, threshold=0.02, only_on_drift=False
    ):
        profile, target = self._detect(current_embeddings, threshold)
        if only_on_drift and not profile.is_drift_detected:
            return profile, None
        return profile, DetectionEvidence(
            self.reference_embeddings.copy(),
            target.copy(),
            float(threshold),
            1.0 / self.reference_embeddings.shape[1],
        )

    def _detect(self, current_embeddings, threshold):
        target = self._validate_embeddings(current_embeddings)
        if target.shape[1] != self.reference_embeddings.shape[1]:
            raise ValueError(
                "current and reference embeddings must have the same feature dimension"
            )
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        # Subsample current if needed
        if len(target) > self.max_samples:
            indices = self._rng.choice(len(target), self.max_samples, replace=False)
            target = target[indices]

        # Simple heuristic for gamma: 1 / num_features
        gamma = (
            1.0 / self.reference_embeddings.shape[1]
            if self.reference_embeddings.shape[1] > 0
            else 1.0
        )

        mmd_score = self._compute_mmd(self.reference_embeddings, target, gamma=gamma)

        profile = DriftProfile(
            is_drift_detected=bool(mmd_score > threshold),
            drift_magnitude=float(mmd_score),
            reference_id="baseline",
            timestamp=datetime.now().isoformat(),
        )
        return profile, target
