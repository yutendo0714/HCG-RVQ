#!/usr/bin/env python3
"""Build mixed-domain EF-LIC perceptual selector CSVs.

The output is designed for analyze_e353_eflic_perceptual_learned_selector_split.py:
all calibration images are prefixed with ``cal_`` and held-out CLIC images with
``eval_`` so alphabetical splitting by ``--cal-count`` is deterministic.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("run must be risk=path.csv")
    name, raw = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("empty risk name")
    return name, Path(raw)


def read_csv(path: Path) -> Tuple[List[str], Dict[str, dict]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"missing header: {path}")
        rows = {str(row["image"]): dict(row) for row in reader}
        return list(reader.fieldnames), rows


def prefixed(row: dict, image: str, source: str, split: str) -> dict:
    out = dict(row)
    out["source_image"] = str(row["image"])
    out["source_dataset"] = source
    out["source_split"] = split
    out["image"] = image
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clic-run", action="append", type=parse_run, required=True)
    ap.add_argument("--kodak-run", action="append", type=parse_run, required=True)
    ap.add_argument("--clic-cal-count", type=int, default=20)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    clic_specs = dict(args.clic_run)
    kodak_specs = dict(args.kodak_run)
    if set(clic_specs) != set(kodak_specs):
        raise ValueError("CLIC and Kodak risk names must match")

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    outputs = {}
    risk_counts = {}
    for risk in sorted(clic_specs):
        clic_fields, clic_rows = read_csv(clic_specs[risk])
        kodak_fields, kodak_rows = read_csv(kodak_specs[risk])
        clic_images = sorted(clic_rows)
        kodak_images = sorted(kodak_rows)
        cal_clic = clic_images[: args.clic_cal_count]
        eval_clic = clic_images[args.clic_cal_count :]

        merged = []
        for image in cal_clic:
            merged.append(prefixed(clic_rows[image], f"cal_clic_{image}", "clic_pro", "cal"))
        for image in kodak_images:
            merged.append(prefixed(kodak_rows[image], f"cal_kodak_{image}", "kodak24", "cal"))
        for image in eval_clic:
            merged.append(prefixed(clic_rows[image], f"eval_clic_{image}", "clic_pro", "eval"))

        fieldnames = []
        for name in ["image", "source_image", "source_dataset", "source_split"] + clic_fields + kodak_fields:
            if name not in fieldnames:
                fieldnames.append(name)
        out_path = out_prefix.parent / f"{out_prefix.name}_{risk}.csv"
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged)
        outputs[risk] = str(out_path)
        risk_counts[risk] = {
            "cal_clic": len(cal_clic),
            "cal_kodak": len(kodak_images),
            "cal_total": len(cal_clic) + len(kodak_images),
            "eval_clic": len(eval_clic),
            "total": len(merged),
        }

    manifest = {
        "outputs": outputs,
        "risk_counts": risk_counts,
        "clic_cal_count": args.clic_cal_count,
        "cal_count_for_e353": next(iter(risk_counts.values()))["cal_total"],
    }
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = ["# E355 EF-LIC Mixed Perceptual Teacher CSVs", ""]
    lines.append(f"CLIC calibration images: {args.clic_cal_count}")
    lines.append(f"E353 cal-count: {manifest['cal_count_for_e353']}")
    lines.append("")
    for risk, counts in sorted(risk_counts.items()):
        lines.append(f"- {risk}: {counts}; csv={outputs[risk]}")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {json_path}, {md_path}")
    for risk, out in sorted(outputs.items()):
        print(f"{risk}={out}")


if __name__ == "__main__":
    main()
