#!/usr/bin/env python3
"""Run the pinned FSGS training recipe or independently evaluate its checkpoint.

Use a separate environment containing FSGS's CUDA extensions. This entry point
never downloads a model, changes the source tree, or overwrites an output folder.
The native 10k recipe performs 9,999 optimizer steps (upstream skips the last).
"""
import argparse
import codecs
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import types


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def align_native_pixel_origin():
    """Match gsplat's rays at pixel index + .5 using native integer samples.

    For the supported centered K, native ndc2Pix already subtracts .5 from
    W/2. One more -.5 shift converts the gsplat sampling convention. Both
    real and pseudo cameras need this coordinate conversion.
    """
    from scene import cameras
    for camera_type in (cameras.Camera, cameras.PseudoCamera):
        original = camera_type.__init__

        def initialize(self, *pos, _original=original, **kw):
            _original(self, *pos, **kw)
            self.projection_matrix[2, 0] -= 1.0 / self.image_width
            self.projection_matrix[2, 1] -= 1.0 / self.image_height
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        camera_type.__init__ = initialize


def train(args):
    import numpy as np
    import torch

    adapter = json.loads((args.train_data / "adapter-manifest.json").read_text())
    if len(adapter["train_views"]) != 3 or adapter["requires_exact_principal_point_adapter"]:
        raise ValueError("This native recipe supports exactly three centered training cameras")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.90)
    sys.path.insert(0, str(args.midas_source))
    from midas.dpt_depth import DPTDepthModel

    if sha256(args.weights) != args.weights_sha256:
        raise ValueError("MiDaS weight hash mismatch")
    model = DPTDepthModel(path=None, backbone="vitb_rn50_384", non_negative=True)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)

    def local_hub(repo, name, *pos, **kw):
        if repo != "intel-isl/MiDaS":
            raise RuntimeError(f"Unexpected hub request: {repo}/{name}")
        if name == "DPT_Hybrid":
            return model
        if name == "transforms":
            # Upstream assigns this transform but estimate_depth implements its
            # own interpolation/normalization and never invokes this variable.
            return types.SimpleNamespace(dpt_transform=None)
        raise RuntimeError(f"Unexpected MiDaS model: {name}")

    torch.hub.load = local_hub
    sys.path.insert(0, str(args.source))
    spec = importlib.util.spec_from_file_location("fsgs_train", args.source / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    align_native_pixel_origin()
    parser = argparse.ArgumentParser()
    lp = module.ModelParams(parser)
    op = module.OptimizationParams(parser)
    pp = module.PipelineParams(parser)
    native = parser.parse_args(["--source_path", str(args.train_data), "--model_path", str(args.output),
                               "--images", "images", "--resolution", "1", "--n_views", "3",
                               "--iterations", "10000", "--sample_pseudo_interval", "1"])
    native.test_iterations = []
    native.save_iterations = [10000]
    native.checkpoint_iterations = [10000]
    native.start_checkpoint = None
    native.debug_from = -1
    native.train_bg = False
    # Same user seed as the preserved baseline; native optimizer/densification
    # and sampling algorithms are retained and differ from gsplat's.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    (args.output / "run-config.json").write_text(json.dumps(vars(native), indent=2))
    start = time.monotonic()
    module.training(lp.extract(native), op.extract(native), pp.extract(native), native)
    torch.cuda.synchronize()
    report = {"completed": True, "native_iterations": 10000, "optimizer_steps": 9999,
              "seed": args.seed, "wall_seconds": time.monotonic() - start,
              "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
              "checkpoint_sha256": sha256(args.output / "chkpnt10000.pth"),
              "test_views_loaded": 0, "midas_weights_sha256": args.weights_sha256,
              "pixel_origin": "gsplat rays: native projected pixels shifted by -0.5"}
    (args.output / "training-complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def evaluate(args):
    import numpy as np
    import torch
    from PIL import Image
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    torch.set_num_threads(4)
    if not (args.checkpoint.parent / "training-complete.json").is_file():
        raise ValueError("Independent evaluation requires completed training")
    sys.path.insert(0, str(args.source))
    # scene/__init__.py imports camera_utils, which otherwise constructs MiDaS
    # at import time. Pure Gaussian evaluation needs no depth-network instance.
    depth_module = types.ModuleType("utils.depth_utils")

    def no_depth_in_evaluation(*pos, **kw):
        raise RuntimeError("Depth-network inference is forbidden in evaluation")

    depth_module.estimate_depth = no_depth_in_evaluation
    sys.modules["utils.depth_utils"] = depth_module
    from scene.gaussian_model import GaussianModel
    from scene.cameras import PseudoCamera
    from gaussian_renderer import render
    align_native_pixel_origin()

    # Upstream stores a NumPy float64 learning rate inside the Adam state.
    # Keep restricted loading, explicitly admitting that scalar representation.
    torch.serialization.add_safe_globals([codecs.encode, np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))])
    capture, iteration = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    if iteration != 10000:
        raise ValueError("Expected the final native checkpoint")
    gaussian = GaussianModel(types.SimpleNamespace(sh_degree=3))
    gaussian.active_sh_degree = capture[0]
    (gaussian._xyz, gaussian._features_dc, gaussian._features_rest,
     gaussian._scaling, gaussian._rotation, gaussian._opacity) = capture[1:7]
    gaussian.confidence = torch.ones_like(gaussian._opacity)
    pipe = types.SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                                 debug=False, use_confidence=False, use_color=True)
    if sha256(args.metric_weights / "hub/checkpoints/alexnet-owt-7be5be79.pth") != "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02":
        raise ValueError("AlexNet metric weights mismatch")
    import os
    os.environ["TORCH_HOME"] = str(args.metric_weights)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    ssim = dict(gaussian_kernel=True, sigma=1.5, kernel_size=11, data_range=1., k1=.01, k2=.03)
    records = []
    start = time.monotonic()
    with torch.no_grad():
        for camera in json.loads((args.test_data / "cameras.json").read_text()):
            k = np.asarray(camera["K"])
            width, height = camera["width"], camera["height"]
            if not np.allclose(k[:2, 2], [(width - 1) / 2, (height - 1) / 2]):
                raise ValueError("Native projection requires centered principal point")
            w2c = np.linalg.inv(camera["c2w"])
            view = PseudoCamera(w2c[:3, :3].T, w2c[:3, 3],
                                2 * np.arctan(width / (2 * k[0, 0])),
                                2 * np.arctan(height / (2 * k[1, 1])), width, height)
            prediction = render(view, gaussian, pipe, torch.zeros(3, device="cuda"))["render"].clamp(0, 1)[None]
            image_path = args.test_data / "images" / camera["name"]
            if sha256(image_path) != camera["image_sha256"]:
                raise ValueError("Test image identity mismatch")
            truth = torch.tensor(np.array(Image.open(image_path).convert("RGB")), device="cuda", dtype=torch.float32).permute(2, 0, 1)[None] / 255.
            score = {"image": camera["name"],
                     "psnr": float(-10 * torch.log10((prediction - truth).square().mean())),
                     "ssim": float(structural_similarity_index_measure(prediction, truth, **ssim)),
                     "lpips_alex": float(lpips(prediction, truth))}
            lpips.reset()
            records.append(score)
            pixels = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(args.output / camera["name"])
    torch.cuda.synchronize()
    result = {"per_image": records, "mean": {key: float(np.mean([r[key] for r in records]))
                                             for key in ["psnr", "ssim", "lpips_alex"]},
              "wall_seconds": time.monotonic() - start, "point_count": len(gaussian._xyz),
              "checkpoint_sha256": sha256(args.checkpoint), "renderer": "FSGS native CUDA",
              "metric_protocol": "same float RGB, torchmetrics SSIM options and LPIPS-Alex normalize=True as baseline"}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--midas-source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--metric-weights", type=Path)
    args = parser.parse_args()
    required = ["train_data", "midas_source", "weights", "weights_sha256"] if args.mode == "train" else ["checkpoint", "test_data", "metric_weights"]
    for key in required:
        if getattr(args, key) is None:
            parser.error(f"--{key.replace('_', '-')} is required for {args.mode}")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
