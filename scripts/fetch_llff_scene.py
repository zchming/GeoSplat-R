"""Fetch the explicitly frozen, small LLFF room generalization subset.

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
    if plan["protocol"] != "llff-fixed-generalization-input-v1":
        raise ValueError("Explicit generalization input plan required")
    scene = plan["scene"]
    if scene["scene"] != "room" or len(scene["train_indices"]) != 3:
        raise ValueError("Only the approved three-view room subset is supported")
    if set(scene["train_indices"]) & set(scene["test_indices"]):
        raise ValueError("Train and test selections overlap")
    expected = {f"room/images_8/{name}" for name in scene["train_names"] + scene["test_names"]}
    expected |= {"room/poses_bounds.npy", "room/sparse/0/cameras.bin", "room/sparse/0/images.bin"}
    members = scene["selected_members"]
    if len(members) != len(expected) or {entry["name"] for entry in members} != expected:
        raise ValueError("ZIP member list differs from the frozen RGB/calibration subset")
    transfer = sum(entry["compressed_bytes"] + 30 for entry in members)
    if transfer > min(plan["download_cap_bytes"], 8_000_000):
        raise ValueError("Subset exceeds the 8 MB transfer cap")
    fetch_scene(scene, args.output)


if __name__ == "__main__":
    main()
