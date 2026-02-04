import os
import glob
import numpy as np
from typing import Dict, Any, List
from virk.eval.generator import ShiftGenerator
from virk.eval.metrics import MetricCalculator
from virk.monitor.extractor import TimmExtractor
from virk.monitor.drift import DriftDetector
from virk.fingerprint.matcher import Fingerprinter

class EvalEngine:
    """Orchestrates the evaluation process."""
    
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.generator = ShiftGenerator(os.path.join(output_dir, "shifted_data"))
        self.extractor = TimmExtractor(model_name='resnet18') # Default for eval
        
        # Load baseline images
        self.image_paths = glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                           glob.glob(os.path.join(dataset_path, "*.png"))
        if not self.image_paths:
            raise ValueError(f"No images found in {dataset_path}")
            
        # Split: 20% ref (for drift baseline), 80% test (to generating clean/shifted scores)
        split_idx = int(len(self.image_paths) * 0.2)
        self.ref_paths = self.image_paths[:split_idx]
        self.test_paths = self.image_paths[split_idx:]
        
        print(f"Loaded {len(self.image_paths)} images. Ref: {len(self.ref_paths)}, Test: {len(self.test_paths)}")
        
        # Initialize components
        print("Computing reference embeddings...")
        self.ref_embeddings = self.extractor.extract(self.ref_paths)
        self.detector = DriftDetector(self.ref_embeddings)
        self.fingerprinter = Fingerprinter(self.extractor, self.ref_paths)
        self.fingerprinter.calibrate()

    def run(self) -> Dict[str, Any]:
        report = {}
        
        # 1. Generate Drift Baseline (Clean Test Data)
        print("Evaluating Clean Baseline...")
        clean_embeddings = self.extractor.extract(self.test_paths)
        # Evaluate in batches to simulate separate "incidents", or just one big batch?
        # For ROC, we need distribution of scores. Let's do batch-wise or per-sample.
        # Detector is designed for batches. Let's split test set into mini-batches.
        batch_size = 32
        clean_scores = []
        
        for i in range(0, len(clean_embeddings), batch_size):
            batch = clean_embeddings[i:i+batch_size]
            result = self.detector.detect(batch, threshold=0)
            clean_scores.append(result.drift_magnitude)
            
        # 2. Generate and Evaluate Shifts
        print("Generating Shifts...")
        shifted_data = self.generator.generate(self.test_paths)
        
        drift_scores_all = [] # Accumulate all shifted scores for aggregate ROC
        y_true_fingerprint = []
        y_pred_fingerprint = []
        
        shift_results = {}
        
        for shift_key, paths in shifted_data.items():
            shift_type = shift_key.rsplit("_", 1)[0] # "brightness_s3" -> "brightness"
            
            print(f"Evaluating {shift_key}...")
            embeddings = self.extractor.extract(paths)
            
            # A. Drift Detection
            batch_scores = []
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size]
                result = self.detector.detect(batch, threshold=0)
                batch_scores.append(result.drift_magnitude)
                
            drift_scores_all.extend(batch_scores)
            
            # Calc per-shift AUROC
            metrics = MetricCalculator.evaluate_detection(clean_scores, batch_scores)
            shift_results[shift_key] = metrics
            
            # B. Fingerprinting
            # For fingerprinting, we typically treat the whole batch as one incident
            # and diagnose the CAUSE of that incident.
            # So we take the MEAN of the batch and classify it.
            # OR we classify per sample? 
            # The logic in fingerprinter.diagnose() takes a batch and computes ONE fingerprint.
            # So "Accuracy" here is: did we get the right top-1 cause for this batch?
            # Since we have ONE batch (the whole shifted dataset for this key), we get ONE prediction.
            # To get a confusion matrix, we probably want to split into mini-batches too.
            
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size]
                fingerprint = self.fingerprinter.diagnose(batch)
                
                # Get top 1 prediction
                best_shift = max(fingerprint.shift_types, key=fingerprint.shift_types.get)
                y_true_fingerprint.append(shift_type)
                y_pred_fingerprint.append(best_shift)

        # 3. Aggregate Metrics
        print("Calculating Aggregate Metrics...")
        total_detection = MetricCalculator.evaluate_detection(clean_scores, drift_scores_all)
        fingerprint_metrics = MetricCalculator.evaluate_fingerprinting(y_true_fingerprint, y_pred_fingerprint)
        
        report["detection_overall"] = total_detection
        report["fingerprint_overall"] = fingerprint_metrics
        report["per_shift_detection"] = shift_results
        
        return report

    def cleanup(self):
        """Remove generated shifted data to save space."""
        shift_dir = os.path.join(self.output_dir, "shifted_data")
        if os.path.exists(shift_dir):
            import shutil
            shutil.rmtree(shift_dir)
            print(f"Cleaned up generated data in {shift_dir}")
