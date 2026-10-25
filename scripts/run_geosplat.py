#!/usr/bin/env python3
"""Run the established FSGS quality or uniform-L1 fast-training workflow.

This supervisor uses only the standard library. Model code runs in explicitly
configured environments; it never installs dependencies or retries a worker.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MIDAS_SHA256 = "501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853"
CONFIG_KEYS = {"python", "fsgs_python", "colmap_python", "cuda_home", "gsplat_source",
               "fsgs_source", "midas_source", "midas_weights", "small_source",
               "small_source_manifest", "small_weights", "metric_weights"}


def load_config(path):
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ValueError("Runtime config must contain only the documented location keys")
    # Keep virtualenv interpreter symlinks intact: resolve() would select base Python.
    return {key: Path(os.path.abspath(path.parent / Path(value).expanduser()))
            for key, value in config.items() if value is not None}


def step(name, argv, products, gpu=False):
    return {"name": name, "argv": [str(value) for value in argv],
            "products": [str(value) for value in products], "gpu": gpu}


def build_steps(args, config, root=ROOT):
    required = {"python", "cuda_home", "metric_weights"}
    if args.action == "run":
        required |= ({"fsgs_python", "fsgs_source", "midas_source", "midas_weights"}
                     if args.mode == "fsgs" else {"gsplat_source"})
        if args.mode == "l1" and args.depth_priors is None:
            required |= {"colmap_python", "small_source", "small_source_manifest", "small_weights"}
    missing = required - config.keys()
    if missing:
        raise ValueError("Missing runtime locations: " + ", ".join(sorted(missing)))
    for key in required:
        if not config[key].exists():
            raise ValueError(f"Missing {key}: {config[key]}")
    for key in required & {"python", "fsgs_python", "colmap_python"}:
        if not config[key].is_file() or not os.access(config[key], os.X_OK):
            raise ValueError(f"Not an executable interpreter: {config[key]}")
    for directory in [args.train_data, args.test_data]:
        if not (directory / "cameras.json").is_file() or not (directory / "images").is_dir():
            raise ValueError(f"Expected a prepared split: {directory}")
    if args.train_data == args.test_data or args.train_data.name != "train" or args.test_data.name != "test":
        raise ValueError("Use separate prepared train/ and test/ directories")
    protected = [args.train_data, args.test_data]
    protected += [p for p in [args.depth_priors, args.checkpoint] if p is not None]
    for path in protected:
        if args.output == path or args.output in path.parents or path in args.output.parents:
            raise ValueError("Output must be separate from input directories and checkpoints")
    manifest = args.train_data / "input-manifest.json"
    if args.action == "run" and not manifest.is_file():
        raise ValueError("Expected frozen train/input-manifest.json; see docs/usage.md")
    if args.action == "evaluate" and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise ValueError("evaluate requires an existing gsplat-format --checkpoint")
    if args.depth_priors is not None and not (args.depth_priors / "manifest.json").is_file():
        raise ValueError("--depth-priors requires an existing aligned-prior manifest")
    python = config["python"]
    scripts = root / "scripts"
    out = args.output
    steps = []
    checkpoint = args.checkpoint
    if args.action == "run" and args.mode == "fsgs":
        steps.append(step("export-fsgs", [python, scripts / "prepare_fsgs_scene.py",
            "--train-dir", args.train_data, "--output", out / "fsgs-data"],
            [out / "fsgs-data/adapter-manifest.json"]))
        checkpoint = out / "model/model.pt"
        native = out / "train/chkpnt10000.pth"
        steps.append(step("train-fsgs", [config["fsgs_python"], scripts / "run_fsgs_reference.py",
            "train", "--source", config["fsgs_source"], "--train-data", out / "fsgs-data",
            "--midas-source", config["midas_source"], "--weights", config["midas_weights"],
            "--weights-sha256", MIDAS_SHA256, "--output", out / "train", "--seed", args.seed],
            [native, out / "train/training-complete.json"], gpu=True))
        steps.append(step("convert-fsgs", [python, scripts / "convert_fsgs_checkpoint.py",
            "--checkpoint", native, "--output", checkpoint], [checkpoint]))
    elif args.action == "run":
        priors = args.depth_priors or out / "priors"
        if args.depth_priors is None:
            steps.append(step("export-anchors", [config["colmap_python"],
                scripts / "export_train_anchors.py", "--data", args.train_data,
                "--scene-manifest", manifest, "--output", out / "anchors"],
                [out / "anchors/manifest.json"]))
            steps.append(step("prepare-priors", [python, scripts / "prepare_depth_priors.py",
                "--data", args.train_data, "--scene-manifest", manifest,
                "--anchors", out / "anchors", "--source", config["small_source"],
                "--source-manifest", config["small_source_manifest"], "--weights", config["small_weights"],
                "--output", priors], [priors / "manifest.json"], gpu=True))
        checkpoint = out / "train/checkpoint.pt"
        steps.append(step("train-l1", [python, scripts / "train_baseline.py",
            "--data", args.train_data, "--scene-manifest", manifest, "--depth-priors", priors,
            "--depth-lambda", "0.01", "--depth-start", "1000", "--depth-loss", "l1",
            "--max-steps", "30000", "--seed", args.seed, "--source", config["gsplat_source"],
            "--output", out / "train"], [checkpoint, out / "train/resources.json"], gpu=True))
    profile = "fsgs-native" if args.mode == "fsgs" else "standard"
    steps.append(step("evaluate", [python, scripts / "evaluate_baseline.py",
        "--checkpoint", checkpoint, "--train-data", args.train_data, "--test-data", args.test_data,
        "--weights", config["metric_weights"], "--render-profile", profile,
        "--label", "FSGS quality" if args.mode == "fsgs" else "Uniform L1 fast training",
        "--output", out / "evaluation"], [out / "evaluation/metrics.json",
        out / "evaluation/novel-view-path.mp4"], gpu=True))
    return steps


def save_report(output, report):
    temporary = output / "workflow.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "workflow.json")


def query_board_memory():
    result = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
    return float(result.stdout.strip())


def stop_worker(process):
    # Include descendants (CUDA loaders and helper workers) in cancellation.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may have exited while a descendant still owns the group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_steps(steps, output, inherited, budget, env, board_limit=11500,
              memory_query=query_board_memory, metadata=None):
    if not math.isfinite(inherited) or not math.isfinite(budget) or not 0 <= inherited < budget:
        raise ValueError("Require finite 0 <= inherited seconds < cumulative budget seconds")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    report = {"schema": "geosplat-workflow-v1", **(metadata or {}), "status": "running",
              "inherited_seconds": inherited, "budget_seconds": budget,
              "charged_seconds": inherited, "steps": [], "error": None}
    save_report(output, report)
    charged = inherited
    try:
        for planned in steps:
            if charged >= budget:
                raise RuntimeError("Cumulative wall-time budget exhausted")
            record = {**planned, "status": "running", "wall_seconds": 0., "exit_code": None,
                      "board_peak_sampled_mib": None, "board_samples": 0,
                      "reserved_until_seconds": budget}
            report["steps"].append(record)
            process = None
            started = time.monotonic()
            try:
                with (output / "logs" / (planned["name"] + ".log")).open("xb") as log, \
                     (output / "logs" / (planned["name"] + "-memory.csv")).open("x") as memory:
                    memory.write("elapsed_seconds,board_memory_mib\n")

                    def sample():
                        if planned["gpu"]:
                            used = memory_query()
                            if not math.isfinite(used) or used < 0:
                                raise RuntimeError("Invalid board memory reading")
                            record["board_samples"] += 1
                            record["board_peak_sampled_mib"] = max(used, record["board_peak_sampled_mib"] or 0)
                            memory.write(f"{time.monotonic()-started:.6f},{used:.3f}\n")
                            memory.flush()
                            if used > board_limit:
                                raise RuntimeError(f"GPU board memory {used:g} MiB exceeds {board_limit} MiB")

                    sample()
                    process = subprocess.Popen(planned["argv"], stdout=log, stderr=subprocess.STDOUT,
                        cwd=ROOT, env=env, start_new_session=True)
                    record["pid"] = process.pid
                    while True:
                        elapsed = time.monotonic() - started
                        record["wall_seconds"] = elapsed
                        report["charged_seconds"] = charged + elapsed
                        save_report(output, report)
                        if charged + elapsed >= budget:
                            raise RuntimeError("Cumulative wall-time budget exhausted")
                        try:
                            code = process.wait(timeout=min(1., budget - charged - elapsed))
                            break
                        except subprocess.TimeoutExpired:
                            sample()
                    sample()
                    record["exit_code"] = code
                    if code:
                        raise RuntimeError(f"{planned['name']} exited with status {code}; see its log")
                    missing = [p for p in planned["products"] if not Path(p).is_file()]
                    if missing:
                        raise RuntimeError("Worker returned without expected products: " + ", ".join(missing))
                    record["status"] = "passed"
            except BaseException:
                record["status"] = "failed"
                raise
            finally:
                if process is not None:
                    stop_worker(process)
                    record["exit_code"] = process.returncode
                record["wall_seconds"] = time.monotonic() - started
                charged += record["wall_seconds"]
                report["charged_seconds"] = charged
                save_report(output, report)
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        save_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "evaluate"])
    parser.add_argument("--mode", choices=["fsgs", "l1"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="evaluate: existing gsplat-format checkpoint")
    parser.add_argument("--depth-priors", type=Path, help="l1 run: reuse frozen aligned priors")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--inherited-seconds", type=float, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True, help="Cumulative cap, including inherited use")
    parser.add_argument("--plan", action="store_true", help="Print commands without creating outputs or using CUDA")
    args = parser.parse_args(argv)
    try:
        for key in ["config", "train_data", "test_data", "output", "checkpoint", "depth_priors"]:
            if getattr(args, key) is not None:
                setattr(args, key, getattr(args, key).expanduser().resolve())
        if args.action == "run" and args.checkpoint is not None:
            raise ValueError("run starts a new model; --checkpoint belongs to evaluate")
        if args.depth_priors is not None and (args.mode != "l1" or args.action != "run"):
            raise ValueError("--depth-priors belongs to l1 run")
        if not math.isfinite(args.inherited_seconds) or not math.isfinite(args.budget_seconds) or not 0 <= args.inherited_seconds < args.budget_seconds:
            raise ValueError("Require finite 0 <= inherited seconds < cumulative budget seconds")
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        config = load_config(args.config)
        steps = build_steps(args, config)
        source_paths = {Path(s["argv"][1]) for s in steps}
        source_paths.add(Path(__file__).resolve())
        source_paths |= set((ROOT / "src/geosplat_r").glob("*.py"))
        metadata = {"action": args.action, "mode": args.mode,
            "training_seed": args.seed if args.action == "run" else None,
            "train_data": str(args.train_data), "test_data": str(args.test_data),
            "runtime_locations": {k: str(v) for k, v in config.items()},
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(source_paths)},
            "render_profile": "fsgs-native" if args.mode == "fsgs" else "standard",
            "recipe_optimizer_updates": 9999 if args.mode == "fsgs" else 30000,
            "budget_scope": "all worker wall seconds, including failed workers and CPU preparation; inherited total is never reset",
            "board_limit_mib": 11500, "gpu_index": 0,
            "memory_coverage": "periodic whole-board samples, not an unsampled continuous peak",
            "planned_steps": steps}
        if args.plan:
            print(json.dumps({**metadata, "inherited_seconds": args.inherited_seconds,
                "budget_seconds": args.budget_seconds, "steps": steps}, indent=2))
            return 0
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
            PYTHONHASHSEED=str(args.seed), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0",
            HF_HUB_OFFLINE="1", MPLCONFIGDIR=str(args.output / "cache/matplotlib"),
            TORCH_HOME=str(config["metric_weights"]), CUDA_HOME=str(config["cuda_home"]))
        env["PATH"] = str(config["cuda_home"] / "bin") + os.pathsep + env.get("PATH", "")
        # Prevent an inherited PYTHONPATH from importing a different checkout.
        env["PYTHONPATH"] = str(ROOT / "src")
        report = run_steps(steps, args.output, args.inherited_seconds, args.budget_seconds, env, metadata=metadata)
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "charged_seconds": report["charged_seconds"]}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; partial outputs and charged budget retained", file=sys.stderr)
        return 130


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    sys.exit(main())
