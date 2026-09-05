"""Contract and orchestration tests; no Rust build or large render required.

Run: python -m unittest discover -s scripts -p test_simsat_case_score.py -v
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("simsat-case-score.py")
SPEC = importlib.util.spec_from_file_location("simsat_case_score", MODULE_PATH)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def report_fixture(product="abi-band13", empty_splits=False):
    metric = "brightness_temperature_kelvin" if product == "abi-band13" else "luminance"
    regimes = {}
    for name, count, bias in (("valid", 100, 4.0), ("strict_clear", 30, -2.0),
                              ("cloudy", 50, 10.0), ("both_cloudy", 20, 8.0),
                              ("observed_only_cloudy", 30, 11.0), ("synthetic_only_cloudy", 10, -5.0)):
        if empty_splits and name in harness.SPLITS:
            count = 0
        values = {"count": count}
        if count:
            values.update(bias=bias, mae=abs(bias) + 1, correlation=0.25,
                          observed={"min": 193.0, "mean": 250.0},
                          synthetic={"min": 206.0, "mean": 254.0})
        regimes[name] = {"count": count, "fraction_of_valid": count / 100, metric: values}
    return {"regimes": regimes, "cold_threshold_diagnostics": {"thresholds": [
        {"threshold_kelvin": 235.0, "area": {"observed_fraction": .12, "synthetic_fraction": .07},
         "fss": [{"neighborhood_width_pixels": 9, "fss": .2},
                 {"neighborhood_width_pixels": 27, "fss": .3}]}]}}


class MetricsTests(unittest.TestCase):
    def test_original_metrics_and_operator_forecast_splits_survive(self):
        result = harness.extract_scores(report_fixture(), "abi-band13")
        self.assertEqual(result["bias"], 4)
        self.assertEqual(result["clear_bias"], -2)
        self.assertEqual(result["both_cloudy_bias"], 8)
        self.assertEqual(result["observed_only_cloudy_count"], 30)
        self.assertEqual(result["observed_only_cloudy_fraction"], .3)
        self.assertEqual(result["coldest_obs"], 193)
        self.assertEqual(result["fss81km"], .3)

    def test_empty_regimes_are_null_not_zero(self):
        result = harness.extract_scores(report_fixture(empty_splits=True), "abi-band13")
        self.assertIsNone(result["both_cloudy_bias"])
        self.assertEqual(result["both_cloudy_count"], 0)

    def test_broken_reports_cannot_look_like_success(self):
        for mutation in (lambda r: r["regimes"].pop("both_cloudy"),
                         lambda r: r["regimes"]["valid"]["brightness_temperature_kelvin"].update(bias=float("nan")),
                         lambda r: r["regimes"]["valid"].update(count=0)):
            report = report_fixture()
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises((KeyError, ValueError)):
                harness.extract_scores(report, "abi-band13")

    def test_constant_field_correlation_can_be_null(self):
        report = report_fixture()
        report["regimes"]["valid"]["brightness_temperature_kelvin"]["correlation"] = None
        self.assertIsNone(harness.extract_scores(report, "abi-band13")["corr"])


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = harness.parser().parse_args([
            "--data-root", str(self.root), "--bin", str(self.root / "bin"),
            "--out", str(self.root / "output"), "--hours", "12"])
        self.args.bin.mkdir()
        for stem in ("simsat-render-ir", "simsat-render-frame"):
            (self.args.bin / (stem + ".exe")).write_bytes(b"mock binary, never executed")
        frame = self.root / "data/d01v/wrfout_d01_2026-09-04_12_00_00"
        ref = self.root / "out/simsat/goesfd-12z/abi-reference-aligned.npz"
        for path in (frame, ref):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        harness.write_json(ref.parent / "source-manifest.json", {"target_time": "2026-09-04T12:00:00Z"})
        self.commands = []

    def runner(self, command, stdout, **kwargs):
        self.commands.append(command)
        if "--product" in command:
            product = command[command.index("--product") + 1]
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir()
            harness.write_json(destination / "validation.json", report_fixture(product))
            stdout.write("validation fixture completed\n")
        else:
            args = dict(item.split("=", 1) for item in command[1:])
            for key in ("out", "bt-out", "cloud-mask-out", "rgb-reflectance-out"):
                if key in args:
                    Path(args[key]).write_bytes(b"render fixture")
            stdout.write("IRSUMMARY cold_top_bt=206 median_bt=254\nSUMMARY sun_elev=83 cloud_frac=0.3\n")
        return subprocess.CompletedProcess(command, 0)

    def test_both_intents_use_this_worktree_validator_and_same_cloud_mask(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = harness.run_case_library(self.args, self.runner)
        manifest = harness.read_json(self.args.out / "run.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(self.commands), 6)
        self.assertEqual(set(harness.GROUPS) - set(result["12"]), set())
        ir = dict(item.split("=", 1) for item in self.commands[0][1:])
        # Exact baseline renderer options, with only the new mask output added.
        self.assertEqual(set(ir), {"input", "out", "bt-out", "cloud-mask-out", "view", "resolution",
                                  "enhancement", "sensor", "cache"})
        self.assertEqual(ir["sensor"], "goes-r-abi-band13-fm4")
        for command in self.commands[1::2]:
            self.assertIn(str(harness.VALIDATOR), command)
            self.assertEqual(command[command.index("--synthetic-cloud-mask") + 1], ir["cloud-mask-out"])
        for command, intent in zip((self.commands[2], self.commands[4]), ("display", "sensor-fast-gray")):
            args = dict(item.split("=", 1) for item in command[1:])
            self.assertEqual(args["intent"], intent)
            self.assertEqual(args["view"], "topdown")
            self.assertEqual(args["resolution"], "native")
            self.assertNotIn("exposure", args)
        self.assertIn("f32le-rgb", self.commands[3])
        self.assertIn("f32le-rgb-unclipped", self.commands[5])
        self.assertEqual(len(list((self.args.out / "logs").glob("*.log"))), 6)
        self.assertIn("not like-for-like ABI", manifest["science_note"])

    def test_failed_later_stage_keeps_completed_ir_and_error_log(self):
        def fail_visible(command, stdout, **kwargs):
            if command[0].endswith("simsat-render-frame.exe"):
                stdout.write("fixture renderer exploded\n")
                return subprocess.CompletedProcess(command, 17)
            return self.runner(command, stdout, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exit 17"):
            harness.run_case_library(self.args, fail_visible)
        checkpoint = harness.read_json(self.args.out / "scores.json")
        self.assertIn("ir", checkpoint["12"])
        self.assertNotIn("vis", checkpoint["12"])
        manifest = harness.read_json(self.args.out / "run.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stages"][-1]["returncode"], 17)
        self.assertIn("exploded", Path(manifest["stages"][-1]["log"]).read_text())
        with self.assertRaisesRegex(ValueError, "partial checkpoints"):
            harness.load_completed_scores(self.args.out)

    def test_missing_hour_fails_before_creating_output_or_rendering(self):
        self.args.hours = [12, 15]
        with self.assertRaisesRegex(FileNotFoundError, "15Z"):
            harness.run_case_library(self.args, self.runner)
        self.assertFalse(self.args.out.exists())
        self.assertEqual(self.commands, [])

    def test_reference_from_another_day_is_rejected_before_render(self):
        manifest = self.root / "out/simsat/goesfd-12z/source-manifest.json"
        harness.write_json(manifest, {"target_time": "2026-09-03T12:00:00Z"})
        with self.assertRaisesRegex(ValueError, "does not match requested"):
            harness.run_case_library(self.args, self.runner)
        self.assertFalse(self.args.out.exists())
        self.assertEqual(self.commands, [])

    def test_existing_baseline_is_untouched(self):
        self.args.out.mkdir()
        marker = self.args.out / "scores.json"
        marker.write_text("baseline sentinel")
        with self.assertRaises(FileExistsError):
            harness.run_case_library(self.args, self.runner)
        self.assertEqual(marker.read_text(), "baseline sentinel")
        self.assertEqual(self.commands, [])


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / "a", self.root / "b"
        self.a.mkdir()
        self.b.mkdir()
        self.baseline = {str(hour): {"ir": {"clear_bias": -2}, "vis": {"clear_bias": -.005}}
                         for hour in (12, 15, 18, 21)}
        self.candidate = copy.deepcopy(self.baseline)

    def compare(self, hours=(12, 15, 18, 21)):
        harness.write_json(self.a / "scores.json", self.baseline)
        harness.write_json(self.b / "scores.json", self.candidate)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = harness.compare(self.a, self.b, hours)
        return result, output.getvalue()

    def test_limits_measure_absolute_bias_not_signed_change(self):
        self.candidate["12"]["ir"]["clear_bias"] = -2.5
        self.candidate["21"]["vis"]["clear_bias"] = -.025
        self.assertEqual(self.compare()[0], 0)
        self.candidate["12"]["ir"]["clear_bias"] = -2.5001
        self.assertEqual(self.compare()[0], 1)
        self.candidate["12"]["ir"]["clear_bias"] = 1
        self.candidate["21"]["vis"]["clear_bias"] = .0251
        self.assertEqual(self.compare()[0], 1)

    def test_missing_hour_and_null_gate_do_not_pass(self):
        self.candidate.pop("15")
        code, text = self.compare()
        self.assertEqual(code, 1)
        self.assertIn("15Z missing", text)
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate["21"]["ir"]["clear_bias"] = None
        self.assertEqual(self.compare()[0], 1)

    def test_gray_rgb_is_not_compared_to_display_and_new_metrics_are_visible(self):
        self.candidate["18"]["vis_sensor_fast_gray"] = {"clear_bias": .9}
        self.candidate["18"]["ir"]["both_cloudy_bias"] = 8
        code, text = self.compare()
        self.assertEqual(code, 0)
        row = next(line for line in text.splitlines() if "vis_sensor_fast_gray.clear_bias" in line)
        self.assertIn("n/a", row)
        self.assertIn("+0.9000", row)
        self.assertIn("ir.both_cloudy_bias", text)

    def test_nonfinite_json_is_rejected(self):
        (self.a / "scores.json").write_text('{"12": {"ir": {"clear_bias": NaN}}}')
        with self.assertRaisesRegex(ValueError, "non-finite JSON"):
            harness.load_completed_scores(self.a)


if __name__ == "__main__":
    unittest.main()
