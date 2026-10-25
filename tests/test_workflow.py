"""Exercise workflow boundaries and real subprocess supervision without CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workflow", ROOT / "scripts/run_geosplat.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "new-results"

    def worker(self, name, program, products=(), gpu=False):
        return workflow.step(name, [sys.executable, "-c", program], products, gpu)

    def read_report(self):
        return json.loads((self.out / "workflow.json").read_text())

    def run_workers(self, steps, inherited=5, budget=15, **kwargs):
        return workflow.run_steps(steps, self.out, inherited, budget, dict(os.environ), **kwargs)

    def test_completion_and_second_run_refuses_to_touch_results(self):
        artifact = self.root / "model.pt"
        job = self.worker("finish", f"from pathlib import Path; Path({str(artifact)!r}).write_text('model')", [artifact])
        result = self.run_workers([job])
        self.assertEqual(result["status"], "passed")
        self.assertGreater(result["charged_seconds"], 5)
        before = (self.out / "workflow.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_workers([job])
        self.assertEqual((self.out / "workflow.json").read_bytes(), before)
        self.assertEqual(artifact.read_text(), "model")

    def test_worker_failure_retains_partial_and_never_starts_evaluation(self):
        partial, forbidden = self.root / "partial", self.root / "test-assets-read"
        child_pid = self.root / "child-pid"
        bad = self.worker("train", f"from pathlib import Path; import subprocess,sys; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); Path({str(child_pid)!r}).write_text(str(child.pid)); Path({str(partial)!r}).write_text('partial'); raise SystemExit(7)")
        later = self.worker("evaluate", f"from pathlib import Path; Path({str(forbidden)!r}).touch()")
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            self.run_workers([bad, later])
        report = self.read_report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["steps"][0]["exit_code"], 7)
        self.assertEqual(partial.read_text(), "partial")
        self.assertFalse(forbidden.exists())
        self.assertGreater(report["charged_seconds"], 5)
        status = Path("/proc") / child_pid.read_text() / "stat"
        for _ in range(50):
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(.01)
        else:
            self.fail("Failed worker left a running descendant")

    def test_timeout_stops_worker_and_records_failure_cost(self):
        job = self.worker("slow", "import time; time.sleep(10)")
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            self.run_workers([job], budget=5.08)
        report = self.read_report()
        self.assertEqual(report["status"], "failed")
        self.assertIsNotNone(report["steps"][0]["exit_code"])
        self.assertGreaterEqual(report["charged_seconds"], 5.08)

    def test_gpu_limit_stops_before_launch_and_sample_is_retained(self):
        marker = self.root / "unexpected-launch"
        job = self.worker("gpu", f"from pathlib import Path; Path({str(marker)!r}).touch()", gpu=True)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            self.run_workers([job], memory_query=lambda: 11501)
        self.assertFalse(marker.exists())
        self.assertEqual(self.read_report()["steps"][0]["board_peak_sampled_mib"], 11501)

    def test_success_exit_without_model_cannot_advance(self):
        job = self.worker("missing-model", "pass", [self.root / "missing.pt"])
        with self.assertRaisesRegex(RuntimeError, "expected products"):
            self.run_workers([job])
        self.assertEqual(self.read_report()["steps"][0]["status"], "failed")

    def test_nan_and_exhausted_budgets_cannot_create_outputs(self):
        for inherited, budget in [(0, float("nan")), (float("nan"), 10), (10, 10), (-1, 10)]:
            with self.assertRaises(ValueError):
                self.run_workers([], inherited=inherited, budget=budget)
            self.assertFalse(self.out.exists())

    def test_both_recipes_keep_test_assets_out_of_training_commands(self):
        train, test = self.root / "prepared/train", self.root / "prepared/test"
        for folder in [train, test]:
            (folder / "images").mkdir(parents=True)
            (folder / "cameras.json").write_text("[]")
        (train / "input-manifest.json").write_text("{}")
        priors = self.root / "priors"
        priors.mkdir()
        (priors / "manifest.json").write_text("{}")
        config = {k: self.root for k in workflow.CONFIG_KEYS}
        for key in ["python", "fsgs_python", "colmap_python"]:
            config[key] = Path(sys.executable)
        for mode, prior in [("fsgs", None), ("l1", priors), ("l1", None)]:
            args = SimpleNamespace(action="run", mode=mode, train_data=train, test_data=test,
                output=self.out, depth_priors=prior, checkpoint=None, seed=44)
            steps = workflow.build_steps(args, config)
            for training_step in steps[:-1]:
                self.assertNotIn(str(test), training_step["argv"])
                self.assertNotIn("--test-data", training_step["argv"])
            evaluation = steps[-1]["argv"]
            self.assertIn(str(test), evaluation)
            self.assertEqual(evaluation[evaluation.index("--render-profile")+1],
                             "fsgs-native" if mode == "fsgs" else "standard")
            self.assertEqual(len(steps), 4 if mode == "fsgs" or prior is None else 2)
        args.output = train / "new-results"
        with self.assertRaisesRegex(ValueError, "Output must be separate"):
            workflow.build_steps(args, config)

    def test_runtime_config_keeps_virtualenv_interpreter_symlink(self):
        interpreter = self.root / "env/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        config = self.root / "runtime.json"
        config.write_text(json.dumps({"python": "env/bin/python"}))
        self.assertEqual(workflow.load_config(config)["python"], interpreter)
        self.assertNotEqual(workflow.load_config(config)["python"], interpreter.resolve())


if __name__ == "__main__":
    unittest.main()
