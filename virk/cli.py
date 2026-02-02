import argparse
import sys
import os
from virk.eval.engine import EvalEngine
from virk.eval.reporter import ReportGenerator

def eval_command(args):
    print(f"Starting evaluation on {args.dataset}...")
    
    # 1. Create Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Run Engine
    engine = EvalEngine(args.dataset, args.output_dir)
    results = engine.run()
    
    # 3. Generate Report
    reporter = ReportGenerator()
    report_path = os.path.join(args.output_dir, "report.html")
    reporter.generate(results, report_path)

def main():
    parser = argparse.ArgumentParser(description="VIRK CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Eval Command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation harness")
    eval_parser.add_argument("--dataset", required=True, help="Path to clean image dataset")
    eval_parser.add_argument("--output-dir", default="eval_out", help="Directory for output artifacts")
    
    # Incident Summary Command
    summary_parser = subparsers.add_parser("summarize", help="Summarize incident bundle")
    summary_parser.add_argument("bundle", help="Path to incident zip or folder")
    
    args = parser.parse_args()
    
    if args.command == "eval":
        eval_command(args)
    elif args.command == "summarize":
        summarize_command(args)
    else:
        parser.print_help()

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
