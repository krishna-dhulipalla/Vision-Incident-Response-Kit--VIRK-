import argparse
import sys
import os


def eval_command(args):
    from virk.eval.engine import EvalEngine
    from virk.eval.reporter import ReportGenerator
    from virk.eval.datasets import DATASET_REGISTRY

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
            print(
                f"Error: Dataset {dataset_name} not supported and directory {data_dir} does not exist."
            )
            return

    # 1. Create Output Dir
    os.makedirs(output_dir, exist_ok=True)

    # 2. Run Engine
    engine = EvalEngine(
        data_dir,
        output_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        fpr_target=args.fpr,
        seed=args.seed,
        sample_interval_seconds=args.sample_interval,
        benchmark_inference=args.benchmark_inference,
    )
    results = engine.run()

    # 3. Generate Report
    reporter = ReportGenerator()
    import time

    ts = int(time.time())
    report_path = os.path.join(output_dir, f"report_{dataset_name}_{ts}.html")
    reporter.generate(results, report_path)
    _write_json(results, os.path.splitext(report_path)[0] + ".json")
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
    eval_parser.add_argument(
        "--dataset-name",
        default="cifar10",
        choices=["cifar10", "flowers102", "custom"],
        help="Dataset to use",
    )
    eval_parser.add_argument(
        "--data-dir", help="Path to data (optional if using built-in datasets)"
    )
    eval_parser.add_argument(
        "--output-dir", default="eval_out", help="Directory for output artifacts"
    )
    eval_parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep generated shifted images after evaluation",
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=32, help="Inference/stream update batch size"
    )
    eval_parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Scoring window used for calibration, test and streaming",
    )
    eval_parser.add_argument("--fpr", type=float, default=0.05)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Assumed seconds per sample in synthetic streams",
    )
    eval_parser.add_argument(
        "--benchmark-inference",
        action="store_true",
        help="Measure real extractor inference with/without the monitor hook",
    )

    # Incident Sub-commands
    incident_parser = subparsers.add_parser("incident", help="Incident management")
    inc_subparsers = incident_parser.add_subparsers(dest="inc_command")

    # virk incident summarize
    inc_sum_parser = inc_subparsers.add_parser(
        "summarize", help="Summarize incident bundle"
    )
    inc_sum_parser.add_argument("bundle", help="Path to incident zip or folder")

    # Deprecated: Top-level summarize
    summary_parser = subparsers.add_parser(
        "summarize", help="[DEPRECATED] Use 'virk incident summarize'"
    )
    summary_parser.add_argument("bundle", help="Path to incident zip or folder")

    # Calibrate Command
    cal_parser = subparsers.add_parser("calibrate", help="Recommend drift thresholds")
    cal_parser.add_argument("--dataset", required=True, help="Path to clean dataset")
    cal_parser.add_argument(
        "--fpr",
        type=float,
        default=0.05,
        help="Target False Positive Rate (default 0.05)",
    )
    cal_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for simulation"
    )
    cal_parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Maximum number of full validation batches",
    )
    cal_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for splitting and detector subsampling",
    )
    cal_parser.add_argument(
        "--test-dataset",
        help="Separate clean test directory for independent FPR evaluation",
    )
    cal_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for independent test Wilson interval",
    )
    cal_parser.add_argument(
        "--output",
        help="Write calibration statistics and independent test results as JSON",
    )

    stream_parser = subparsers.add_parser(
        "benchmark-stream",
        help="Evaluate ordered embedding traffic and monitoring latency",
    )
    stream_parser.add_argument(
        "--reference", required=True, help="Baseline embeddings .npy"
    )
    stream_parser.add_argument(
        "--embeddings", required=True, help="Ordered traffic embeddings .npy"
    )
    stream_parser.add_argument(
        "--labels", required=True, help="One 0/1 shifted label per traffic sample, .npy"
    )
    stream_parser.add_argument(
        "--threshold", type=float, required=True, help="Previously selected threshold"
    )
    stream_parser.add_argument("--window-size", type=int, default=1000)
    stream_parser.add_argument("--batch-size", type=int, default=32)
    stream_parser.add_argument("--max-samples", type=int, default=1000)
    stream_parser.add_argument("--seed", type=int, default=42)
    stream_parser.add_argument(
        "--sample-interval",
        type=float,
        required=True,
        help="Seconds per sample; used for exposure and delay",
    )
    stream_parser.add_argument("--output", required=True, help="Output JSON report")

    # Demo Prod Command
    demo_parser = subparsers.add_parser("demo-prod", help="Run production demo service")
    demo_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    demo_parser.add_argument("--port", type=int, default=8080, help="Port to bind")

    args = parser.parse_args()

    if args.command == "eval":
        try:
            eval_command(args)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.command == "incident":
        if args.inc_command == "summarize":
            summarize_command(args)
        else:
            incident_parser.print_help()
    elif args.command == "summarize":
        print(
            "Warning: 'virk summarize' is deprecated. Please use 'virk incident summarize'."
        )
        summarize_command(args)
    elif args.command == "calibrate":
        try:
            calibrate_command(args)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.command == "benchmark-stream":
        try:
            benchmark_stream_command(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.command == "demo-prod":
        import uvicorn

        print(f"Starting VIRK Demo Service on http://{args.host}:{args.port}")
        uvicorn.run(
            "virk.examples.fastapi_service:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    else:
        parser.print_help()


def calibrate_command(args):
    from virk.eval.calibration import CalibrationEngine

    print(f"Running Calibration on {args.dataset}")
    engine = CalibrationEngine(args.dataset)

    options = {}
    if args.test_dataset is not None:
        options.update(test_dataset_path=args.test_dataset, confidence=args.confidence)
    result = engine.calibrate(
        fpr_target=args.fpr,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
        **options,
    )

    print("\n" + "=" * 50)
    print(" VIRK THRESHOLD CALIBRATION")
    print("=" * 50)
    print(f"Target FPR:          {result['target_fpr']*100:.1f}%")
    print(f"Empirical FPR:       {result['empirical_fpr']*100:.1f}%")
    print(f"Validation batches:  {result['num_batches']}")
    print(f"Max Clean Score:     {result['max_clean_score']:.6f}")
    print("-" * 50)
    print(f"RECOMMENDED THRESHOLD: {result['recommended_threshold']:.6f}")
    print("Empirical calibration only; future false positive rates may differ.")
    if "independent_test" in result:
        test = result["independent_test"]
        print(
            f"Independent test FPR: {test['fpr']:.1%} ({test['false_positives']}/{test['num_batches']} batches)"
        )
        print(
            f"{test['confidence']:.0%} Wilson interval: [{test['ci_lower']:.1%}, {test['ci_upper']:.1%}]"
        )
        print("Interval assumes independent representative clean batches.")
    print("=" * 50 + "\n")
    if args.output:
        _write_json(result, args.output)


def _write_json(data, path):
    import json
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report written to: {target}")


def benchmark_stream_command(args):
    import numpy as np
    from virk.eval.streaming import StreamingEvaluator

    evaluator = StreamingEvaluator(
        np.load(args.reference, allow_pickle=False),
        args.threshold,
        window_size=args.window_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    report = evaluator.run(
        np.load(args.embeddings, allow_pickle=False),
        np.load(args.labels, allow_pickle=False),
        batch_size=args.batch_size,
        sample_interval_seconds=args.sample_interval,
    )
    _write_json(report, args.output)
    print(
        f"False clean alerts: {report['false_alert_checks']}/{report['clean_checks']} checks"
    )
    print(
        f"Shift episodes detected: {report['detected_episodes']}/{report['shift_episodes']}"
    )
    print(f"Monitoring p95: {report['monitoring_latency']['p95_ms']:.3f} ms")


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
            with zipfile.ZipFile(path, "r") as zf:
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

        with open(manifest_path, "r") as f:
            data = json.load(f)

        # RCA Output
        print("\n" + "=" * 50)
        print(f" VIRK INCIDENT SUMMARY: {data.get('incident_id', 'Unknown')}")
        print("=" * 50)

        # 1. Critical Stats
        drift = data.get("drift_profile", {})
        mag = drift.get("drift_magnitude", 0.0)
        print(f"Timestamp:   {drift.get('timestamp', 'N/A')}")
        print(
            f"Drift Mag:   {mag:.4f} " + ("(CRITICAL)" if mag > 0.05 else "(DETECTED)")
        )
        print(
            f"Data Hash:   {data.get('metadata', {}).get('data_hash', data.get('data_hash', 'N/A'))}"
        )

        # 2. Causality
        fp = data.get("fingerprint", {}).get("shift_types", {})
        if fp:
            top_cause = max(fp.items(), key=lambda x: x[1])
            print("\nProbable Root Cause:")
            print(f"  > {top_cause[0].upper()} (Score: {top_cause[1]:.2f})")

            # Show secondary
            sorted_fp = sorted(fp.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_fp) > 1:
                sec = sorted_fp[1]
                if sec[1] > 5.0:  # Heuristic
                    print(f"  > (Secondary) {sec[0]} ({sec[1]:.2f})")

        # 3. Slicing
        slices = data.get("top_slices", [])
        if slices:
            print("\nAffected Slices (Targeting Guidance):")
            for s in slices[:3]:
                print(
                    f"  - {s['slice_name']} (Contribution: {s.get('contribution_score', 0):.3f})"
                )

        print("\n" + "-" * 50)
        print(
            "Action: Run 'python replay.py' inside the unzipped bundle for reproduction."
        )
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"Error reading bundle: {e}")
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir)


if __name__ == "__main__":
    main()
