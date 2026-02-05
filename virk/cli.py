import argparse
import sys
import os
from virk.eval.engine import EvalEngine
from virk.eval.reporter import ReportGenerator
from virk.eval.datasets import DATASET_REGISTRY

def eval_command(args):
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    data_dir = args.data_dir or f"{dataset_name}_eval"
    
    print(f"Starting evaluation on {dataset_name}...")
    
    # 0. Setup Data
    if dataset_name in DATASET_REGISTRY:
        DATASET_REGISTRY[dataset_name](data_dir)
    else:
        # Assume data_dir is a custom path if name not in registry
        if not os.path.exists(data_dir):
            print(f"Error: Dataset {dataset_name} not supported and directory {data_dir} does not exist.")
            return

    # 1. Create Output Dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Run Engine
    engine = EvalEngine(data_dir, output_dir)
    results = engine.run()
    
    # 3. Generate Report
    reporter = ReportGenerator()
    import time
    ts = int(time.time())
    report_path = os.path.join(output_dir, f"report_{dataset_name}_{ts}.html")
    reporter.generate(results, report_path)
    print(f"Report generated at: {report_path}")
    
    # 4. Cleanup
    if args.keep_data:
        print(f"Intermediate data kept in {output_dir}/shifted_data")
    else:
        engine.cleanup()

def main():
    parser = argparse.ArgumentParser(description="VIRK CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Eval Command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation harness")
    eval_parser.add_argument("--dataset-name", default="cifar10", choices=["cifar10", "flowers102", "custom"], help="Dataset to use")
    eval_parser.add_argument("--data-dir", help="Path to data (optional if using built-in datasets)")
    eval_parser.add_argument("--output-dir", default="eval_out", help="Directory for output artifacts")
    eval_parser.add_argument("--keep-data", action="store_true", help="Keep generated shifted images after evaluation")
    
    # Incident Sub-commands
    incident_parser = subparsers.add_parser("incident", help="Incident management")
    inc_subparsers = incident_parser.add_subparsers(dest="inc_command")
    
    # virk incident summarize
    inc_sum_parser = inc_subparsers.add_parser("summarize", help="Summarize incident bundle")
    inc_sum_parser.add_argument("bundle", help="Path to incident zip or folder")
    
    # Deprecated: Top-level summarize
    summary_parser = subparsers.add_parser("summarize", help="[DEPRECATED] Use 'virk incident summarize'")
    summary_parser.add_argument("bundle", help="Path to incident zip or folder")
    
    # Calibrate Command
    cal_parser = subparsers.add_parser("calibrate", help="Recommend drift thresholds")
    cal_parser.add_argument("--dataset", required=True, help="Path to clean dataset")
    cal_parser.add_argument("--fpr", type=float, default=0.05, help="Target False Positive Rate (default 0.05)")
    cal_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for simulation")

    # Demo Prod Command
    demo_parser = subparsers.add_parser("demo-prod", help="Run production demo service")
    demo_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    demo_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    
    args = parser.parse_args()
    
    if args.command == "eval":
        eval_command(args)
    elif args.command == "incident":
        if args.inc_command == "summarize":
            summarize_command(args)
        else:
            incident_parser.print_help()
    elif args.command == "summarize":
        print("Warning: 'virk summarize' is deprecated. Please use 'virk incident summarize'.")
        summarize_command(args)
    elif args.command == "calibrate":
        calibrate_command(args)
    elif args.command == "demo-prod":
        import uvicorn
        print(f"Starting VIRK Demo Service on http://{args.host}:{args.port}")
        uvicorn.run("virk.examples.fastapi_service:app", host=args.host, port=args.port, reload=False)
    else:
        parser.print_help()

def calibrate_command(args):
    from virk.eval.calibration import CalibrationEngine
    
    print(f"Running Calibration on {args.dataset}")
    engine = CalibrationEngine(args.dataset)
    
    result = engine.calibrate(fpr_target=args.fpr, batch_size=args.batch_size)
    
    print("\n" + "="*50)
    print(" VIRK THRESHOLD CALIBRATION")
    print("="*50)
    print(f"Target FPR:          {result['target_fpr']*100:.1f}%")
    print(f"Max Clean Score:     {result['max_clean_score']:.6f}")
    print("-" * 50)
    print(f"RECOMMENDED THRESHOLD: {result['recommended_threshold']:.6f}")
    print("="*50 + "\n")

def summarize_command(args):
    """
    Unzips (if needed) and summarizes the incident manifest.
    """
    import zipfile
    import json
    import tempfile
    import shutil
    from pathlib import Path
    
    path = Path(args.bundle)
    
    if not path.exists():
        print(f"Error: {path} not found.")
        return

    cleanup_dir = None
    manifest_path = None
    
    try:
        if path.suffix == ".zip":
            # Extract manifest to temp
            tmp_dir = tempfile.mkdtemp()
            cleanup_dir = tmp_dir
            with zipfile.ZipFile(path, 'r') as zf:
                # Find manifest
                if "manifest.json" in zf.namelist():
                    zf.extract("manifest.json", tmp_dir)
                    manifest_path = Path(tmp_dir) / "manifest.json"
                # Check for nested (incident_X/manifest.json)
                else:
                    for n in zf.namelist():
                        if n.endswith("manifest.json"):
                            zf.extract(n, tmp_dir)
                            manifest_path = Path(tmp_dir) / n
                            break
        elif path.is_dir():
             manifest_path = path / "manifest.json"
             
        if not manifest_path or not manifest_path.exists():
            print("Error: manifest.json not found in bundle.")
            return
            
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            
        # RCA Output
        print("\n" + "="*50)
        print(f" VIRK INCIDENT SUMMARY: {data.get('incident_id', 'Unknown')}")
        print("="*50)
        
        # 1. Critical Stats
        drift = data.get('drift_profile', {})
        mag = drift.get('drift_magnitude', 0.0)
        print(f"Timestamp:   {drift.get('timestamp', 'N/A')}")
        print(f"Drift Mag:   {mag:.4f} " + ("(CRITICAL)" if mag > 0.05 else "(DETECTED)"))
        print(f"Data Hash:   {data.get('data_hash', 'N/A')}")
        
        # 2. Causality
        fp = data.get('fingerprint', {}).get('shift_types', {})
        if fp:
            top_cause = max(fp.items(), key=lambda x: x[1])
            print("\nProbable Root Cause:")
            print(f"  > {top_cause[0].upper()} (Score: {top_cause[1]:.2f})")
            
            # Show secondary
            sorted_fp = sorted(fp.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_fp) > 1:
                sec = sorted_fp[1]
                if sec[1] > 5.0: # Heuristic
                    print(f"  > (Secondary) {sec[0]} ({sec[1]:.2f})")
        
        # 3. Slicing
        slices = data.get('top_slices', [])
        if slices:
            print("\nAffected Slices (Targeting Guidance):")
            for s in slices[:3]:
                print(f"  - {s['slice_name']} (Contribution: {s.get('contribution_score', 0):.3f})")

        print("\n" + "-"*50)
        print("Action: Run 'python replay.py' inside the unzipped bundle for reproduction.")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error reading bundle: {e}")
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir)

if __name__ == "__main__":
    main()
