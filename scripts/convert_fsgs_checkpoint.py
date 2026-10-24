#!/usr/bin/env python3
"""Convert a completed native FSGS checkpoint for the existing gsplat evaluator.

Gaussian means, log scales, wxyz quaternions, opacity logits and SH coefficients
retain their original coordinate system and values. This does not retrain them.
"""
import argparse
import codecs
import hashlib
import json
from pathlib import Path

import torch
import numpy as np


def convert(checkpoint, output):
    completion = json.loads((checkpoint.parent / "training-complete.json").read_text())
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha != completion["checkpoint_sha256"] or completion["native_iterations"] != 10000:
        raise ValueError("Expected an unchanged completed native 10k checkpoint")
    if completion.get("pixel_origin") != "gsplat rays: native projected pixels shifted by -0.5":
        raise ValueError("Checkpoint must use the gsplat ray convention")
    torch.serialization.add_safe_globals([codecs.encode, np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))])
    capture, iteration = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if iteration != 10000 or capture[0] != 3:
        raise ValueError("Expected final checkpoint with active SH degree 3")
    splats = dict(zip(["means", "sh0", "shN", "scales", "quats", "opacities"], capture[1:7]))
    splats = {key: tensor.detach().contiguous() for key, tensor in splats.items()}
    count = len(splats["means"])
    shapes = {"means": (count, 3), "sh0": (count, 1, 3), "shN": (count, 15, 3),
              "scales": (count, 3), "quats": (count, 4), "opacities": (count, 1)}
    for key, tensor in splats.items():
        if tuple(tensor.shape) != shapes[key] or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid Gaussian parameter: {key}")
    if not count or (splats["quats"].norm(dim=1) <= 1e-12).any():
        raise ValueError("Empty cloud or degenerate quaternion")
    splats["opacities"] = splats["opacities"].flatten()
    metadata = {"source_checkpoint_sha256": sha, "native_iterations": iteration,
                "optimizer_updates": completion["optimizer_steps"], "seed": completion["seed"],
                "gaussians": count, "method": "FSGS native training; gsplat rendering",
                "coordinate_conversion": "none", "parameter_value_conversion": "opacity [N,1] reshaped to [N]; all numeric values retained"}
    # Exclusive creation protects all previous checkpoints and outputs.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save({"splats": splats, **metadata}, stream)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
