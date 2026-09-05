#!/usr/bin/env python3
"""Reproducible, immutable SimSat collocation scoring on an aligned model grid.

This is a repository-owned replacement for soma-render-work/tools/simsat_validate.py.
It retains the original ``ir`` and ``vis`` score groups; ``vis`` means the raw RGB
reflectance diagnostic rendered with intent=display. The sensor-fast-gray intent
has its own group. Neither RGB luminance is an instrument-SRF-integrated ABI band.
The raw RGB dump precedes the final PNG exposure/tonemap, but the display intent
still changes upstream optics, including the default cloud-extinction scale 0.15.

Only exact model-grid topdown/native output is currently scored. A geostationary
raster cannot be compared pixel-for-pixel until it has been resampled onto that
grid. No RGB channel is relabelled C01, C02 or C03.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent.parent
VALIDATOR = REPO / "scripts" / "simsat-validate-goes.py"
GROUPS = ("ir", "vis", "vis_sensor_fast_gray")
SPLITS = ("both_cloudy", "observed_only_cloudy", "synthetic_only_cloudy")
# Regression limits come from CODEX-SIMSAT-NEXTGEN.md section 4, item 4.
CLEAR_IR_TOLERANCE_K = 0.5
AFTERNOON_CLEAR_VIS_TOLERANCE = 0.02
SCIENCE_NOTE = (
    "Visible scores are gray RGB luminance diagnostics, not like-for-like ABI "
    "C01/C02/C03 reflectance validation. vis uses intent=display; "
    "vis_sensor_fast_gray uses unscaled model extinction and neutral display "
    "shaping. Both raw RGB dumps precede the final PNG exposure/tonemap. "
    "The legacy display diagnostic clips reflectance to [0,1]; sensor-fast-gray "
    "scores unclipped simulation and observation, retaining glint above one. "
    "IR uses the official GOES-R ABI band-13 FM4 SRF. Both-cloudy isolates "
    "collocated cloud columns, but does not remove cloud-structure forecast error."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Replace this run's checkpoint atomically; never replace another run."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON value {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(log_path, tag):
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(tag + " "):
            return dict(token.split("=", 1) for token in line[len(tag) + 1:].split() if "=" in token)
    return {}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def extract_scores(report, product):
    """Fail on broken metric contracts; retain null for legitimately empty regimes."""
    metric = "brightness_temperature_kelvin" if product == "abi-band13" else "luminance"
    regimes = report["regimes"]
    for name in ("valid", "strict_clear", "cloudy", *SPLITS):
        entry = regimes[name]
        count = entry["count"]
        values = entry[metric]
        if not isinstance(count, int) or count < 0 or values["count"] != count:
            raise ValueError(f"invalid {name} sample count")
        if count:
            for key in ("bias", "mae"):
                if not finite_number(values[key]):
                    raise ValueError(f"non-finite {name}.{key}")
            correlation = values["correlation"]
            if correlation is not None and not finite_number(correlation):
                raise ValueError(f"invalid {name}.correlation")
    if regimes["valid"]["count"] == 0:
        raise ValueError("validator found no valid collocated pixels")

    def value(name, key):
        values = regimes[name][metric]
        return values[key] if values["count"] else None

    valid = regimes["valid"][metric]
    result = {
        "bias": value("valid", "bias"), "mae": value("valid", "mae"),
        "corr": value("valid", "correlation"),
        "clear_bias": value("strict_clear", "bias"),
        "cloudy_bias": value("cloudy", "bias"),
    }
    if product == "abi-band13":
        threshold = next(t for t in report["cold_threshold_diagnostics"]["thresholds"]
                         if t["threshold_kelvin"] == 235.0)
        fss = {f["neighborhood_width_pixels"]: f["fss"] for f in threshold["fss"]}
        result.update(
            cold235_obs=threshold["area"]["observed_fraction"],
            cold235_sim=threshold["area"]["synthetic_fraction"],
            fss27km=fss.get(9), fss81km=fss.get(27),
            coldest_obs=valid["observed"]["min"], coldest_sim=valid["synthetic"]["min"],
        )
    else:
        result.update(cloudy_corr=value("cloudy", "correlation"),
                      mean_obs=valid["observed"]["mean"], mean_sim=valid["synthetic"]["mean"])
    for name in SPLITS:
        entry = regimes[name]
        result.update({f"{name}_{key}": value(name, source)
                       for key, source in (("bias", "bias"), ("mae", "mae"), ("corr", "correlation"))})
        result[f"{name}_count"] = entry["count"]
        result[f"{name}_fraction"] = entry["fraction_of_valid"]
    for key, value_ in result.items():
        if value_ is not None and not finite_number(value_):
            raise ValueError(f"invalid score {key}: {value_!r}")
    return result


def binary(bin_dir, stem):
    for suffix in (".exe", ""):
        candidate = bin_dir / (stem + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {stem}[.exe] in {bin_dir}; build the release binaries first")


def preflight(args):
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}; choose a fresh directory")
    bins = {"ir": binary(args.bin, "simsat-render-ir"),
            "visible": binary(args.bin, "simsat-render-frame")}
    if not VALIDATOR.is_file():
        raise FileNotFoundError(f"worktree validator missing: {VALIDATOR}")
    cases = {}
    for hour in args.hours:
        substitutions = {"date": args.date, "hour": hour}
        frame = args.data_root / args.frame_template.format(**substitutions)
        reference = args.data_root / args.reference_template.format(**substitutions)
        for path in (frame, reference):
            if not path.is_file():
                raise FileNotFoundError(f"{hour:02d}Z required input missing: {path}; no hours are silently skipped")
        reference_manifest = reference.parent / "source-manifest.json"
        if not reference_manifest.is_file():
            raise FileNotFoundError(f"{hour:02d}Z reference source-manifest.json is required to verify time")
        target = read_json(reference_manifest).get("target_time")
        expected = datetime.fromisoformat(f"{args.date}T{hour:02d}:00:00+00:00")
        try:
            actual = datetime.fromisoformat(target.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid reference target_time in {reference_manifest}") from error
        if actual.tzinfo is None or actual != expected:
            raise ValueError(f"reference target_time {target} does not match requested {expected.isoformat()}")
        cases[hour] = {"frame": frame.resolve(), "reference": reference.resolve(),
                       "reference_manifest": reference_manifest.resolve()}
    return bins, cases


def run_case_library(args, runner=subprocess.run):
    bins, cases = preflight(args)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "cache").mkdir()
    logs = args.out / "logs"
    logs.mkdir()
    scores = {}
    manifest = {
        "schema_version": 1, "status": "running", "started_utc": utc_now(),
        "requested_hours": args.hours, "date": args.date, "science_note": SCIENCE_NOTE,
        "geometry": {"view": "topdown", "resolution": "native"},
        "data_root": str(args.data_root), "worktree": str(REPO),
        "validator": {"path": str(VALIDATOR), "sha256": sha256(VALIDATOR)},
        "binaries": {key: {"path": str(path), "sha256": sha256(path)} for key, path in bins.items()},
        "cases": {str(hour): {key: str(path) for key, path in case.items()}
                  for hour, case in cases.items()},
        "stages": [],
    }

    def checkpoint():
        write_json(args.out / "scores.json", scores)
        write_json(args.out / "run.json", manifest)

    def stage(name, command):
        command = [str(part) for part in command]
        log = logs / f"{len(manifest['stages']) + 1:02d}-{name}.log"
        entry = {"name": name, "command": command, "log": str(log),
                 "started_utc": utc_now(), "status": "running"}
        manifest["stages"].append(entry)
        checkpoint()
        print(f"{name}: running; log {log}", flush=True)
        start = time.monotonic()
        try:
            with log.open("x", encoding="utf-8") as stream:
                stream.write("command: " + json.dumps(command) + "\n")
                stream.flush()
                result = runner(command, stdout=stream, stderr=subprocess.STDOUT,
                                cwd=REPO, check=False)
            entry["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{name} failed (exit {result.returncode}); see {log}")
            entry["status"] = "complete"
        except BaseException as error:
            entry["status"] = "failed"
            entry["error"] = str(error)
            raise
        finally:
            entry["elapsed_seconds"] = round(time.monotonic() - start, 3)
            entry["finished_utc"] = utc_now()
            checkpoint()
        return log

    def validate(hour, product, plane, cloud_mask, folder, input_kind=None):
        destination = args.out / folder
        command = [sys.executable, "-X", "utf8", VALIDATOR,
                   "--product", product, "--synthetic", plane, "--reference", cases[hour]["reference"],
                   "--synthetic-cloud-mask", cloud_mask, "--output-dir", destination]
        if input_kind:
            command.extend(["--input-kind", input_kind])
        stage(f"{hour:02d}z-{folder}", command)
        return extract_scores(read_json(destination / "validation.json"), product)

    checkpoint()
    print(SCIENCE_NOTE, flush=True)
    try:
        for hour, case in cases.items():
            tag = f"{hour:02d}Z"
            ir_png = args.out / f"ir13_d01_{tag}.png"
            ir_plane = args.out / f"ir13_d01_{tag}.bin"
            mask = args.out / f"cloud-mask_d01_{tag}.bin"
            ir_log = stage(f"{tag.lower()}-render-ir", [bins["ir"], f"input={case['frame']}",
                           f"out={ir_png}", f"bt-out={ir_plane}", f"cloud-mask-out={mask}",
                           "view=topdown", "resolution=native", "enhancement=cimss",
                           "sensor=goes-r-abi-band13-fm4", f"cache={args.out / 'cache'}"])
            scores[str(hour)] = {"ir": validate(hour, "abi-band13", ir_plane, mask, f"validate-{tag.lower()}")}
            scores[str(hour)]["ir_summary"] = summary(ir_log, "IRSUMMARY")
            checkpoint()
            for intent, group, infix in (("display", "vis", ""),
                                          ("sensor-fast-gray", "vis_sensor_fast_gray", "_sensor-fast-gray")):
                png = args.out / f"vis_d01_{tag}{infix}_topdown.png"
                plane = args.out / f"vis_d01_{tag}{infix}_rho.bin"
                log = stage(f"{tag.lower()}-render-{intent}", [bins["visible"],
                            f"input={case['frame']}", f"out={png}", f"rgb-reflectance-out={plane}",
                            f"intent={intent}", "sat=goes-east", "view=topdown", "resolution=native",
                            f"cache={args.out / 'cache'}"])
                scores[str(hour)][group] = validate(hour, "visible", plane, mask,
                                                    f"validate-vis{infix}-{tag.lower()}",
                                                    "f32le-rgb-unclipped" if intent == "sensor-fast-gray" else "f32le-rgb")
                scores[str(hour)][group + "_summary"] = summary(log, "SUMMARY")
                checkpoint()
            ir, vis, gray = (scores[str(hour)][group] for group in GROUPS)
            print(f"{tag} IR clear={fmt(ir['clear_bias'])} K both-cloudy={fmt(ir['both_cloudy_bias'])} K; "
                  f"RGB diagnostic clear display={fmt(vis['clear_bias'])}, "
                  f"sensor-fast-gray={fmt(gray['clear_bias'])}", flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = utc_now()
        checkpoint()
    return scores


def fmt(value):
    return f"{value:+.4f}" if finite_number(value) else "n/a"


def load_completed_scores(folder):
    metadata = folder / "run.json"
    if metadata.exists():
        manifest = read_json(metadata)
        if manifest.get("status") != "complete":
            raise ValueError(f"{folder} is {manifest.get('status')}; partial checkpoints cannot pass regression")
    result = read_json(folder / "scores.json")
    if not result:
        raise ValueError(f"empty scoreboard: {folder}")
    return result


def compare(a, b, hours):
    baseline, candidate = load_completed_scores(a), load_completed_scores(b)
    problems = []
    print(SCIENCE_NOTE)
    print(f"{'hour':5s} {'metric':44s} {'baseline':>12s} {'candidate':>12s} {'B-A':>12s}")
    for hour in hours:
        key = str(hour)
        if key not in baseline or key not in candidate:
            problems.append(f"{hour:02d}Z missing from baseline or candidate")
            continue
        for group in GROUPS:
            old, new = baseline[key].get(group, {}), candidate[key].get(group, {})
            for metric in sorted(set(old) | set(new)):
                before, after = old.get(metric), new.get(metric)
                delta = after - before if finite_number(before) and finite_number(after) else None
                print(f"{hour:02d}Z   {group + '.' + metric:44s} {fmt(before):>12s} {fmt(after):>12s} {fmt(delta):>12s}")
        for group in ("ir", "vis"):
            if group not in candidate[key]:
                problems.append(f"{hour:02d}Z candidate missing {group}")
        checks = [("ir", "clear_bias", CLEAR_IR_TOLERANCE_K)]
        if hour == 21:
            checks.append(("vis", "clear_bias", AFTERNOON_CLEAR_VIS_TOLERANCE))
        for group, metric, tolerance in checks:
            before = baseline[key].get(group, {}).get(metric)
            after = candidate[key].get(group, {}).get(metric)
            if not finite_number(before) or not finite_number(after):
                problems.append(f"{hour:02d}Z {group}.{metric} unavailable; cannot evaluate regression")
            elif abs(after) - abs(before) > tolerance + 1e-12:
                problems.append(f"{hour:02d}Z {group}.{metric}: absolute bias worsened by "
                                f"{abs(after) - abs(before):.4f} > {tolerance:g}")
    print("Regression limits: |candidate bias| - |baseline bias| <= 0.5 K for clear IR at every requested hour; "
          "<= 0.02 for display-intent RGB clear bias at 21Z (15:00 local for this case).")
    print("Other metrics are reported for review; these two gates alone do not establish physical improvement.")
    if problems:
        for problem in problems:
            print("FAIL: " + problem)
        return 1
    print("PASS: declared regression gates; n/a means unavailable, never zero.")
    return 0


def parse_hours(value):
    try:
        hours = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("hours must be comma-separated integers") from error
    if not hours or any(hour < 0 or hour > 23 for hour in hours) or len(set(hours)) != len(hours):
        raise argparse.ArgumentTypeError("hours must be unique UTC hours in 0..23")
    return hours


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog="Examples:\n  python scripts/simsat-case-score.py --data-root C:/Users/drew/soma-render-work "
                                           "--bin target/release --out C:/Users/drew/soma-render-work/out/simsat-nextgen\n"
                                           "  python scripts/simsat-case-score.py --compare <baseline> <candidate>\n"
                                           "Exit codes: 0 complete/pass; 1 regression failure; 2 input/stage/metric error.\n"
                                           "scores.json is checkpointed after each validated product; run.json must say complete.\n"
                                           "No files in an existing output directory are removed or overwritten.")
    result.add_argument("--data-root", type=Path, help="root containing data/d01v and out/simsat aligned references")
    result.add_argument("--bin", type=Path, help="directory with both release renderer binaries")
    result.add_argument("--out", type=Path, help="new output directory; even an existing empty directory is rejected")
    result.add_argument("--hours", type=parse_hours, default=parse_hours("12,15,18,21"),
                        help="required UTC hours, comma separated (default: 12,15,18,21); never silently skipped")
    result.add_argument("--date", default="2026-09-04", help="case date YYYY-MM-DD (default: 2026-09-04)")
    result.add_argument("--frame-template", default="data/d01v/wrfout_d01_{date}_{hour:02d}_00_00",
                        help="input path relative to data root; supports {date} and {hour:02d}")
    result.add_argument("--reference-template", default="out/simsat/goesfd-{hour:02d}z/abi-reference-aligned.npz",
                        help="already aligned reference path relative to data root; supports {date} and {hour:02d}")
    result.add_argument("--compare", type=Path, nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="print every available metric and enforce the brief's declared regression limits")
    return result


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    try:
        date.fromisoformat(args.date)
        if args.compare:
            return compare(*args.compare, args.hours)
        for key in ("data_root", "bin", "out"):
            if getattr(args, key) is None:
                arg_parser.error("render mode requires --data-root, --bin and --out")
            setattr(args, key, getattr(args, key).resolve())
        run_case_library(args)
        return 0
    except (OSError, ValueError, KeyError, StopIteration, RuntimeError) as error:
        print(f"simsat-case-score: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("simsat-case-score: interrupted; completed scores and stage logs are preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
