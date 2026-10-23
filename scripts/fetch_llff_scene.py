"""Fetch the explicitly frozen, small LLFF room or fortress subset.

The earlier flower/leaves command and its approved-plan hash remain unchanged.
Only selected RGBs, public camera metadata and poses are fetched via ZIP ranges.
"""
import argparse
import json
from pathlib import Path

from fetch_llff_subset import fetch_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    scene = plan["scene"]
    supported = {("llff-fixed-generalization-input-v1", "room"),
                 ("llff-fixed-independent-input-v1", "fortress")}
    if (plan["protocol"], scene["scene"]) not in supported or len(scene["train_indices"]) != 3:
        raise ValueError("Explicit approved three-view scene plan required")
    if set(scene["train_indices"]) & set(scene["test_indices"]):
        raise ValueError("Train and test selections overlap")
    prefix = scene["scene"]
    expected = {f"{prefix}/images_8/{name}" for name in scene["train_names"] + scene["test_names"]}
    expected |= {f"{prefix}/poses_bounds.npy", f"{prefix}/sparse/0/cameras.bin", f"{prefix}/sparse/0/images.bin"}
    members = scene["selected_members"]
    if len(members) != len(expected) or {entry["name"] for entry in members} != expected:
        raise ValueError("ZIP member list differs from the frozen RGB/calibration subset")
    transfer = sum(entry["compressed_bytes"] + 30 for entry in members)
    if transfer > min(plan["download_cap_bytes"], 8_000_000):
        raise ValueError("Subset exceeds the 8 MB transfer cap")
    fetch_scene(scene, args.output)


if __name__ == "__main__":
    main()
