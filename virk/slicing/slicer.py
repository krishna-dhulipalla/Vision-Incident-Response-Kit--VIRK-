from typing import List, Dict, Any
import numpy as np
from virk.core.types import SliceAttribution
from virk.monitor.drift import DriftDetector

class SliceEngine:
    """Identifies metadata slices responsible for drift."""
    
    def __init__(self, detector: DriftDetector):
        self.detector = detector

    def analyze(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], top_k: int = 5, max_dims: int = 2) -> List[SliceAttribution]:
        """
        Find top_k slices with highest drift, supporting interactions (e.g. cam_1 AND night).
        
        Args:
            embeddings: (N, D) array of current embeddings.
            metadata: List of N dicts.
            top_k: Number of slices to return.
            max_dims: Maximum combination size (1=single, 2=pairs).
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
            
        N = len(embeddings)
        
        # 1. Candidate Generation
        # We start with 1D slices
        candidates: Dict[str, List[int]] = {}
        dimensions = set()
        
        for idx, meta in enumerate(metadata):
            for key, val in meta.items():
                slice_key = f"{key}:{val}"
                if slice_key not in candidates:
                    candidates[slice_key] = []
                candidates[slice_key].append(idx)
                dimensions.add(key)
                
        # 2. Score 1D Slices
        scored_slices = []
        high_impact_slices = [] # Keep indices for 2D interaction search
        
        for slice_name, indices in candidates.items():
            if len(indices) < 5 or len(indices) == N: # Skip too small or practically all
                continue
                
            slice_embs = embeddings[indices]
            # Use raw MMD as contribution score
            result = self.detector.detect(slice_embs, threshold=0.0)
            score = result.drift_magnitude
            
            scored_slices.append(SliceAttribution(slice_name=slice_name, contribution_score=score))
            
            if score > 0.005: # Heuristic threshold for "interesting" enough to cross
                high_impact_slices.append((slice_name, set(indices)))

        # 3. Greedy 2D Interactions (if requested)
        if max_dims >= 2 and len(high_impact_slices) > 1:
            # Pair up high impact slices from DIFFERENT dimensions
            # Optimization: Only pair if they have intersection
            
            for i in range(len(high_impact_slices)):
                name_a, indices_a = high_impact_slices[i]
                dim_a = name_a.split(":")[0]
                
                for j in range(i+1, len(high_impact_slices)):
                    name_b, indices_b = high_impact_slices[j]
                    dim_b = name_b.split(":")[0]
                    
                    if dim_a == dim_b: # Don't cross "cam_1" with "cam_2" (disjoint usually)
                        continue
                        
                    # Compute intersection
                    # Using sets is O(N), bitmasks would be faster but Python sets are optimized
                    intersection = list(indices_a.intersection(indices_b))
                    
                    if len(intersection) < 5:
                        continue
                        
                    # Score the interaction slice
                    slice_name = f"{name_a} & {name_b}"
                    slice_embs = embeddings[intersection]
                    result = self.detector.detect(slice_embs, threshold=0.0)
                    
                    # We only care if interaction score is HIGHER than the parents
                    # O(N^2) detect calls? Maybe slow. But usually few high impact slices.
                    scored_slices.append(SliceAttribution(slice_name=slice_name, contribution_score=result.drift_magnitude))

        # 4. Sort and return top_k
        scored_slices.sort(key=lambda x: x.contribution_score, reverse=True)
        return scored_slices[:top_k]
