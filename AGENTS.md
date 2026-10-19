# GeoSplat-R working rules

- Phase 1 is authorized only for the fern 3-view vanilla baseline: strict train/test
  isolation, real metrics, visualizations, GPU memory and timing. Stop and report after
  this substep. No other scenes, depth regularization or Phase 2 without new authorization.
- Authorized future remote: https://github.com/zchming/GeoSplat-R.git. Do not commit or push automatically.
- This is independent of VisionGeometryLab / GeoRecon-Fusion. Never modify that project.
- Use codex/ branches. No automatic commits, pushes, remote repository creation or license selection.
- Before a commit, show full diff, file list, actual checks and skip reasons; obtain confirmation.
- Before large dataset downloads or deleting/condensing historical artifacts, provide exact
  scope and effects and obtain confirmation. No paid/cloud resources.
- Keep data, weights, vendor checkouts, environments and raw logs outside tracked files.
  Preserve failures, OOMs, resource-profile changes and all comparable runs.
- GPU use is limited to the local 12 GB card. For future authorized training, prioritize
  maximum practical GPU utilization and training speed within available VRAM without OOM.
  Do not retain an arbitrary low memory cap when a measured faster configuration fits;
  keep measured headroom for desktop, CUDA workspaces and densification peaks.
  Prefer safe performance optimizations; record resource-profile changes. Batch size,
  precision, resolution, iteration count or algorithm changes must not silently alter
  the frozen protocol or invalidate comparisons. Filling VRAM is not itself a speed goal.
  This preference does not authorize new training or crossing phase boundaries.
- Separate training inputs from evaluation inputs, and development scenes from final test scenes.
  Never use test images/depth/points as training priors or initialization.
- New entry points accept explicit relevant locations and refuse to overwrite results.
- No performance, novelty or project-success claims without matched, multi-scene evidence.

## Acceptance leadership

- Codex leads the technical acceptance recommendation for each authorized substep.
  State pass, conditional pass or fail based on actual checks, metrics, failures,
  reproducibility and resource evidence; identify limitations and the next highest-value action.
- Do not delegate technical acceptance judgment back to the user or treat a completed run,
  high training fit or low loss as proof of quality, correctness or algorithm contribution.
- Distinguish implementation/engineering acceptance from evaluation-protocol credibility
  and research-performance acceptance. If a gate is unresolved, recommend the smallest
  diagnostic that can resolve it and do not expand algorithm complexity.
- Acceptance leadership is not blanket execution authorization. Existing phase/substep
  boundaries and approval requirements for large downloads, commits and pushes remain.

## Current authorized substep

- Continue Phase 1 with fern 3-view baseline diagnosis: check camera/image mapping,
  train-only initialization coverage and upstream implementation parity; run targeted
  reference validation if warranted. Preserve the original 30k run and frozen split.
  No depth priors, other scenes, commits or pushes are authorized by this continuation.

## Authorized on 2026-10-19

- Validate a distortion-corrected Phase 1 fern baseline using the same three train
  and three test views, 504x378, seed42 and 30k updates. Update intrinsics consistently,
  retriangulate using train RGB only, validate geometry/isolation and evaluate separately.
  Preserve every old input/run. Do not call cross-protocol metric differences gains.
  Stop and give acceptance/next-step/push recommendations; no Phase 2, commits or pushes.

## Current submission-review authorization

- Prepare the first engineering commit candidate, complete diff, proposed file list and
  actual validation report. Documentation and packaging checks only; no new training,
  Phase 2, commits or pushes. Preserve all historical files and external evidence.
