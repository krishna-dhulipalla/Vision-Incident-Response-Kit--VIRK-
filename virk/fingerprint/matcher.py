from typing import Dict, List, Union
import numpy as np
from PIL import Image
from virk.monitor.extractor import EmbeddingExtractor
from virk.fingerprint.corruptions import Corruptions
from virk.core.types import ShiftFingerprint
from scipy.spatial.distance import cosine

class Fingerprinter:
    """Diagnoses the cause of drift by matching against known corruption signatures."""
    
    def __init__(self, extractor: EmbeddingExtractor, reference_images: List[str]):
        """
        Args:
            extractor: specific model wrapper to reuse.
            reference_images: small set of healthy images to generate signatures from.
        """
        self.extractor = extractor
        self.reference_images = reference_images
        self.signatures: Dict[str, np.ndarray] = {} # shift_name -> mean_embedding_vector
        self.baseline_mean: np.ndarray = None
        
    def _load_images(self) -> List[Image.Image]:
        imgs = []
        for p in self.reference_images:
            imgs.append(Image.open(p).convert('RGB'))
        return imgs

    def calibrate(self):
        """Generates signatures and calibrates sensitivity using Z-scores."""
        # 1. Compute baseline mean
        imgs = self._load_images()
        base_embs = self.extractor.extract(imgs)
        self.baseline_mean = np.mean(base_embs, axis=0)
        
        # 2. Compute mean shift for each corruption (The Signature)
        print("Calibrating signatures...")
        self.signatures = {}
        for name, func in Corruptions.get_all().items():
            # Apply corruption level 3
            corrupted_imgs = [func(img.copy(), severity=3) for img in imgs]
            corr_embs = self.extractor.extract(corrupted_imgs)
            corr_mean = np.mean(corr_embs, axis=0)
            
            # Signature vector
            sig_vector = corr_mean - self.baseline_mean
            # Normalize signature vector to unit length for cleaner projection
            norm_sig = sig_vector / (np.linalg.norm(sig_vector) + 1e-9)
            self.signatures[name] = norm_sig

        # 3. Compute baseline statistics (Noise Floor)
        # We project the CLEAN reference embeddings onto each signature
        # to see what "normal" variation looks like.
        print("Calibrating noise baseline...")
        self.calibration_stats = {} # name -> (mean, std)
        
        # Center the base embeddings
        centered_base = base_embs - self.baseline_mean
        
        for name, sig_vector in self.signatures.items():
            # Project centered clean data onto signature: dot product
            # (N, D) dot (D,) -> (N,)
            scores = np.dot(centered_base, sig_vector)
            
            self.calibration_stats[name] = (float(np.mean(scores)), float(np.std(scores)) + 1e-9)

    def diagnose(self, problem_embeddings: np.ndarray) -> ShiftFingerprint:
        """
        Diagnose using Z-score calibrated matching.
        """
        if self.baseline_mean is None:
            raise RuntimeError("Fingerprinter not calibrated. Call calibrate() first.")
            
        problem_mean = np.mean(problem_embeddings, axis=0)
        observed_drift_vector = problem_mean - self.baseline_mean
        
        scores = {}
        z_scores = {}
        
        for name, sig_vector in self.signatures.items():
            # Raw projection score (how much of this drift is in the direction of the signature?)
            raw_score = np.dot(observed_drift_vector, sig_vector)
            
            # Z-Score Normalization
            # "How many standard deviations is this score away from the clean baseline?"
            mu, sigma = self.calibration_stats[name]
            z = (raw_score - mu) / sigma
            
            z_scores[name] = z
            
            # We only care if meaningful drift occurred (positive Z)
            scores[name] = max(0.0, float(z))

        # Check if top score is significant (e.g., Z > 3)
        if not scores: # empty if all negative
             return ShiftFingerprint(shift_types={"unknown": 1.0})

        top_shift, top_score = max(scores.items(), key=lambda x: x[1])
        if top_score < 3.0: # Threshold for "significant" drift match
             # If nothing is above 3 sigma, likely Unknown or just generic OOD
             # But let's verify if we want to suppress everything.
             # For now, append an "unknown" score inversely prop to top_score?
             # Or just include "unknown" as a class.
             scores["unknown"] = 3.0 # if everything is low, unknown wins if < 3
        else:
             scores["unknown"] = 0.0

        # Optional: Softmax-like normalization for readability?
        # Or just return raw Z-scores. Z-scores are interpretable (sigmas).
        # Let's keep Z-scores but maybe cap/scale for the User Interface.
        # User asked for "probability/score". Let's stick to Z-scores in the object,
        # but maybe the UI/CLI normalizes them.
        
        return ShiftFingerprint(shift_types=scores)
