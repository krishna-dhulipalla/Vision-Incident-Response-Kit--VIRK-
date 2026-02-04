import numpy as np
from typing import List, Dict, Any
from virk.monitor.drift import DriftDetector
from virk.monitor.extractor import TimmExtractor

class CalibrationEngine:
    """
    Helps Determine suitable drift thresholds for a specific dataset
    based on a target False Positive Rate (FPR).
    """
    
    def __init__(self, dataset_path: str, extractor_name: str = 'resnet18'):
        self.dataset_path = dataset_path
        self.extractor = TimmExtractor(model_name=extractor_name)
        
    def calibrate(self, fpr_target: float = 0.05, num_batches: int = 50, batch_size: int = 32) -> Dict[str, float]:
        """
        Run detection on clean data held-out to find the threshold 
        where only `fpr_target` fraction of batches drift.
        
        Returns:
            Dict with recommended threshold and stats.
        """
        import glob
        import os
        
        # Load all images
        image_paths = glob.glob(os.path.join(self.dataset_path, "*.jpg")) + \
                      glob.glob(os.path.join(self.dataset_path, "*.png"))
                      
        if len(image_paths) < (num_batches * batch_size):
            print(f"Warning: Not enough images ({len(image_paths)}) for robust calibration. Using what we have.")
            num_batches = len(image_paths) // batch_size
            
        # Shuffle
        np.random.shuffle(image_paths)
        
        # Split: 50% Reference (Baseline), 50% Validation (Test)
        split = len(image_paths) // 2
        ref_paths = image_paths[:split]
        val_paths = image_paths[split:]
        
        print("Extracting Reference Embeddings...")
        ref_embs = self.extractor.extract(ref_paths)
        detector = DriftDetector(ref_embs)
        
        print(f"Running Validation on {len(val_paths)} images...")
        val_embs = self.extractor.extract(val_paths)
        
        scores = []
        for i in range(0, len(val_embs) - batch_size, batch_size):
            batch = val_embs[i:i+batch_size]
            result = detector.detect(batch)
            scores.append(result.drift_magnitude)
            
        if not scores:
            return {"threshold": 0.0, "max_clean_score": 0.0}
            
        # Calculate Threshold
        # If we want FPR <= 0.05, we need the 95th percentile of CLEAN scores.
        # i.e., 95% of clean data falls BELOW this score.
        percentile = (1.0 - fpr_target) * 100
        threshold = np.percentile(scores, percentile)
        
        return {
            "recommended_threshold": float(threshold),
            "target_fpr": fpr_target,
            "mean_clean_score": float(np.mean(scores)),
            "std_clean_score": float(np.std(scores)),
            "max_clean_score": float(np.max(scores))
        }
