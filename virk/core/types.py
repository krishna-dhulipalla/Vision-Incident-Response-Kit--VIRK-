from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class DriftProfile:
    """Stores the result of a drift detection analysis."""
    is_drift_detected: bool
    drift_magnitude: float
    reference_id: str
    timestamp: str
    
@dataclass
class ShiftFingerprint:
    """Stores the attribution of drift to specific shift types."""
    shift_types: Dict[str, float] = field(default_factory=dict)
    # e.g. {"blur": 0.8, "brightness": 0.2}

@dataclass
class SliceAttribution:
    """Stores the attribution of drift to specific data slices."""
    slice_name: str  # e.g., "camera_id:cam_01"
    contribution_score: float

@dataclass
class Incident:
    """Represents a detected incident bundle."""
    incident_id: str
    timestamp: str
    drift_profile: DriftProfile
    fingerprint: ShiftFingerprint
    top_slices: List[SliceAttribution]
    affected_sample_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "drift_profile": self.drift_profile.__dict__,
            "fingerprint": self.fingerprint.__dict__,
            "top_slices": [s.__dict__ for s in self.top_slices],
            "affected_samples": self.affected_sample_ids,
            "metadata": self.metadata
        }
