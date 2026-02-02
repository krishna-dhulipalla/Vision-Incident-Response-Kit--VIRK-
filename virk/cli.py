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
    
    args = parser.parse_args()
    
    if args.command == "eval":
        eval_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
