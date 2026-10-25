#!/usr/bin/env python3
"""Export a frozen train-only scene to FSGS's COLMAP directory layout.

The initial cloud stays sparse: the directory name dense/fused.ply is required
by upstream and does not imply that MVS or additional images were used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(train_dir, output):
    cameras_path = train_dir / "cameras.json"
    points_path = train_dir / "points.npz"
    manifest_path = train_dir / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cameras = json.loads(cameras_path.read_text())
    if manifest["split"] != "train":
        raise ValueError("Only a train manifest is accepted")
    for key, path in [("cameras_sha256", cameras_path), ("points_sha256", points_path)]:
        if manifest[key] != digest(path):
            raise ValueError(f"Frozen asset changed: {path.name}")
    expected = {c["name"]: c["sha256"] for c in manifest["train_images"]}
    if len(cameras) != len(expected) or {c["name"] for c in cameras} != set(expected):
        raise ValueError("Camera/image manifest mismatch")
    cameras = sorted(cameras, key=lambda c: c["name"])
    with np.load(points_path, allow_pickle=False) as archive:
        points, rgb = archive["points"].copy(), archive["rgb"].copy()
    if points.ndim != 2 or points.shape[1] != 3 or rgb.shape != points.shape:
        raise ValueError("Expected matching [N,3] points and RGB")
    if not np.isfinite(points).all() or len(points) != manifest["initial_point_count"]:
        raise ValueError("Invalid frozen initialization")

    # Validate before creating the output. Existing directories are never reused.
    rows, poses_bounds, camera_lines, image_lines = [], [], [], []
    for index, camera in enumerate(cameras, 1):
        name = camera["name"]
        if Path(name).name != name:
            raise ValueError("Image name must be a basename")
        image_path = train_dir / "images" / name
        if digest(image_path) != expected[name]:
            raise ValueError(f"Image changed: {name}")
        width, height = camera["width"], camera["height"]
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError("Image/camera dimensions differ")
        k, c2w = np.array(camera["K"]), np.array(camera["c2w"])
        if k.shape != (3, 3) or c2w.shape != (4, 4):
            raise ValueError("Invalid K or c2w shape")
        if not np.isfinite(k).all() or not np.isfinite(c2w).all():
            raise ValueError("Nonfinite camera")
        w2c = np.linalg.inv(c2w)
        if not np.allclose(w2c[:3, :3].T @ w2c[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError("Camera rotation is not orthonormal")
        # Derive pseudo-view bounds solely from in-frustum train-SfM points.
        # This is a projection test, not an occlusion/visibility estimate.
        camera_points = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = camera_points[:, 2]
        projected = camera_points @ k.T
        uv = projected[:, :2] / np.where(z[:, None] > 0, z[:, None], 1)
        visible = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        visible &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
        if visible.sum() < 3:
            raise ValueError(f"Insufficient visible train points: {name}")
        near, far = np.percentile(z[visible], [0.1, 99.9])
        if not 0 < near < far:
            raise ValueError("Degenerate train-only bounds")
        llff_pose = c2w[:3, [1, 0, 2, 3]].copy()
        llff_pose[:, 2] *= -1
        hwf = np.array([height, width, k[0, 0]])[:, None]
        poses_bounds.append(np.r_[np.c_[llff_pose, hwf].ravel(), near, far])
        q_xyzw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        q = q_xyzw[[3, 0, 1, 2]]
        camera_lines.append(f"{index} PINHOLE {width} {height} " +
                            " ".join(format(v, ".17g") for v in (k[0, 0], k[1, 1], k[0, 2], k[1, 2])))
        pose = " ".join(format(v, ".17g") for v in np.r_[q, w2c[:3, 3]])
        image_lines.append(f"{index} {pose} {index} {name}\n\n")
        rows.append({"name": name, "in_frustum_train_points": int(visible.sum()),
                     "near_far": [float(near), float(far)],
                     # Native ndc2Pix(v,S)=((v+1)*S-1)/2 uses this center.
                     "principal_offset_from_upstream_center": [float(k[0, 2] - (width - 1) / 2),
                                                               float(k[1, 2] - (height - 1) / 2)]})

    output.mkdir(parents=True, exist_ok=False)
    sparse = output / "sparse" / "0"
    sparse.mkdir(parents=True)
    (output / "images").mkdir()
    for camera in cameras:
        shutil.copy2(train_dir / "images" / camera["name"], output / "images" / camera["name"])
    (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "images.txt").write_text("".join(image_lines))
    (sparse / "points3D.txt").write_text("# Initialization is in the explicit PLY below.\n")
    np.save(output / "poses_bounds.npy", np.stack(poses_bounds))
    cloud = output / f"{len(cameras)}_views" / "dense" / "fused.ply"
    cloud.parent.mkdir(parents=True)
    with cloud.open("w") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        for field in ["x", "y", "z", "nx", "ny", "nz"]:
            stream.write(f"property float {field}\n")
        for field in ["red", "green", "blue"]:
            stream.write(f"property uchar {field}\n")
        stream.write("end_header\n")
        for point, color in zip(points, rgb):
            stream.write(" ".join(format(float(v), ".9g") for v in point) +
                         " 0 0 0 " + " ".join(str(int(v)) for v in color) + "\n")
    report = {"protocol": "fsgs-train-only-sparse-adapter-v1", "scene": manifest["scene"],
              "source_manifest_sha256": digest(manifest_path), "train_views": rows,
              "test_images_present": False, "initial_point_count": len(points),
              "initialization": "existing sparse train-only SfM; no MVS",
              "bounds": "per-view in-frustum train-SfM Z percentiles [0.1,99.9]; no occlusion test",
              "upstream_cli": ["--images", "images", "--resolution", "1", "--n_views", str(len(cameras))],
              "eval_flag": "omit: this directory already contains only frozen training views",
              "raster_origin_adapter": "run_fsgs_reference.py shifts native projections by -0.5 pixel to match gsplat sampling at pixel index +0.5",
              "requires_exact_principal_point_adapter": any(
                  abs(v) > 1e-8 for row in rows for v in row["principal_offset_from_upstream_center"]),
              "output_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "adapter-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_dir.resolve(), args.output.resolve()), indent=2))
