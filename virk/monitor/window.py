import numpy as np
from collections import deque
from virk.monitor.drift import DriftDetector
from virk.core.types import DriftProfile
from datetime import datetime

class WindowedDriftDetector(DriftDetector):
    """
    Detects drift using a sliding window approach to track gradual shifts.
    Maintains a fixed baseline (training data) and a sliding window (recent traffic).
    """
    
    def __init__(self, reference_embeddings: np.ndarray, window_size: int = 1000):
        super().__init__(reference_embeddings)
        self.window_size = window_size
        # Store embeddings in a deque for O(1) pops
        self.window = deque(maxlen=window_size)
        self.window_array = None # Cache for numpy array version
        
    def update(self, new_embeddings: np.ndarray):
        """Add new embeddings to the sliding window."""
        for emb in new_embeddings:
            self.window.append(emb)
        # Invalidate cache
        self.window_array = None
            
    def _get_window_array(self) -> np.ndarray:
        if self.window_array is None:
            if not self.window:
                return np.empty((0, self.reference_embeddings.shape[1]))
            # Convert deque to array (efficient enough for reasonable window sizes)
            self.window_array = np.array(self.window)
        return self.window_array

    def detect_trend(self, threshold: float = 0.02) -> DriftProfile:
        """
        Check for drift in the CURRENT window against the FIXED baseline.
        This smooths out noise from small batches.
        """
        current_window = self._get_window_array()
        if len(current_window) < self.window_size // 4: # Wait for partial fill
             return DriftProfile(False, 0.0, "baseline", datetime.now().isoformat())
             
        return self.detect(current_window, threshold)
    
    def detect_sudden_shift(self, current_batch: np.ndarray, threshold: float = 0.02) -> DriftProfile:
        """
        Check if current batch is significantly different from the RECENT window.
        Useful for spotting sudden spikes (e.g. camera obstruction) vs gradual drift.
        """
        recent_window = self._get_window_array()
        if len(recent_window) < 100:
             return DriftProfile(False, 0.0, "recent_window", datetime.now().isoformat())
        
        # We temporarily use recent_window as 'reference' for this check
        # Reuse MMD logic
        gamma = 1.0 / recent_window.shape[1] if recent_window.shape[1] > 0 else 1.0
        mmd = self._compute_mmd(recent_window, current_batch, gamma=gamma)
        
        return DriftProfile(
            is_drift_detected=bool(mmd > threshold),
            drift_magnitude=float(mmd),
            reference_id="recent_window",
            timestamp=datetime.now().isoformat()
        )
