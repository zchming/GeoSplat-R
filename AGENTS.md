# GeoSplat-R working rules

- Phase 1 is authorized only for the fern 3-view vanilla baseline: strict train/test
  isolation, real metrics, visualizations, GPU memory and timing. Stop and report after
  this substep. No other scenes, depth regularization or Phase 2 without new authorization.
- Authorized future remote: https://github.com/zchming/GeoSplat-R.git. Do not commit or push automatically.
- This is independent of VisionGeometryLab / GeoRecon-Fusion. Never modify that project.
- For future work and commits, use descriptive branches such as feat/, fix/ or
  experiment/; never create or commit further work on a codex/-prefixed branch.
  Switch to an appropriately named branch before the next commit. Historical
  branch names below describe completed actions and are not naming instructions.
  No automatic commits, pushes, remote repository creation or license selection
  without applicable user authorization.
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

## Current Phase 2 authorization (2026-10-19)

- User approved pushing only 49f0522 to the existing codex branch; this is complete.
- Implement and run the minimal uniform depth-regularization comparison on the frozen
  fern-undistorted-v1 protocol, same initialization, split, seed and 30k budget.
  Training priors/alignment use train images and train SfM only. No reliability module,
  robust-loss extension, new scenes or test-based tuning. Stop and report the result.
- Exact Small weights (99,218,434 bytes) were separately approved for download.
  Keep all data, priors, weights, source mirrors, checkpoints and raw logs external.
  No automatic commits or pushes of Phase 2 changes.

## Current seed-stability authorization (2026-10-19)

- Run the zero-weight/uniform-depth pair for seeds43 and44, exactly 30k updates
  each, reusing the frozen fern inputs, train-only priors, lambda=0.01 and start=1000.
  Keep training and evaluation separate; combine with the preserved seed42 pair.
  No reliability module, hyperparameter changes, new scenes, commits or pushes.
- Report baseline variability, paired seed/image improvements and degradations,
  resources, failures, acceptance and the next recommended substep; then stop.
- All user-facing conversational responses start with 张总 and default to Chinese.

## Current RGB-path diagnostic authorization (2026-10-19)

- User approved the recommended same-implementation RGB-only seed42/43/44
  three-run 30k diagnostic. Reuse the frozen cameras, images, initialization,
  sampling, optimizers and budget; omit priors and depth rendering for these runs.
- Compare with all preserved RGB+ED zero-weight and uniform-depth runs, using
  separate evaluation and identical GT/metric protocol. Path differences also
  include the existing prior cache/diagnostics; do not claim isolated causality.
- Do not change the frozen training/evaluation implementation, tune coefficients,
  expand scenes, implement reliability, commit or push. Complete checks and report.

## Current cumulative submission-review authorization (2026-10-19)

- Review all accumulated Phase 2 changes against 49f0522, the complete diff,
  candidate file list, provenance and actual checks; recommend whether to commit.
- Documentation/packaging corrections are allowed; preserve frozen experiment
  source hashes and all historical evidence. No new training, flower/leaves work,
  downloads, staging, commits or pushes are authorized by this review.

## Current publication and development-data planning authorization (2026-10-19)

- User approved pushing only 42e297e to codex/phase2-uniform-depth; verify the
  remote ref after pushing. Do not push new planning documents automatically.
- Inspect flower/leaves file listings, exact sizes and the frozen experiment
  proposal using metadata only; do not retrieve dataset member payloads, prepare
  data, implement new input adapters, train or enter reliability in this substep.
- Report verified facts, pending input/calibration gates and next-step judgment.

## Current development-data preparation authorization (2026-10-19)

- User approved the 21 named flower/leaves members totaling 11,049,045 compressed
  payload bytes and minimal explicit input adaptation plus data-preparation checks.
- Preserve frozen split/math, train/test isolation, old fern entry paths and all
  historical inputs, models and evidence. Preserve failures and resource records.
- Prepare train-only geometry/anchors/Small priors and validate inputs; report
  whether the fixed 18-run experiment should be released. No Gaussian training,
  reliability module, automatic staging, commits or pushes are authorized.

## Current development-scene training authorization (2026-10-19)

- User approved the frozen flower/leaves 18-run matrix and evaluation only after
  all 18 training runs finish. Verify preparation/source hashes before execution.
- Keep split, initialization, depth lambda/schedule and all settings frozen.
  Preserve every attempt and degradation; stop and investigate anomalies first.
- No reliability module, automatic staging, commits or pushes. Complete checks,
  report actual metrics/resources/limitations and the next acceptance judgment.

## Current development-scene submission-review authorization (2026-10-19)

- Review accumulated flower/leaves changes against 42e297e, the complete diff,
  proposed file list, real checks and preserved experiment evidence.
- Documentation and packaging corrections are allowed. Keep frozen source/input
  hashes and all historical failures, runs and reproduction paths unchanged.
- Recommend whether to make one engineering commit; stop and report. No staging,
  commits, pushes, new training/evaluation, downloads or reliability work.

## Current leaves seed44 environment-control authorization (2026-10-19)

- The reviewed 25-file commit 2e1e679 was separately approved and pushed to the
  existing codex/phase2-uniform-depth branch; preserve its evidence and history.
- User approved two additional leaves seed44 RGB-only/ED-zero 30k controls.
  Preflight found another environment restart after the previous uniform retry;
  user then explicitly approved one additional uniform 30k run as well.
- Run the three paths sequentially with the same frozen inputs, initialization,
  coefficient/schedule and settings. Check boot identity and source/input hashes;
  complete all three training runs before fresh evaluation. Preserve every attempt.
- This is a single-seed diagnostic, not a replacement for the original matrix or
  proof of a causal reboot effect. No downloads, reliability, commits or pushes.
- The uniform attempt stopped at 21k after a manual chat-turn interruption;
  user explicitly approved a full 30k retry in a new directory. Preserve both.

## Current leaves seed44 submission-review authorization (2026-10-20)

- Review the five-file diagnostic diff against 2e1e679, actual checks, all paired
  regressions and preserved interrupted/retried attempts. Prepare one submission
  candidate and recommend whether to commit; keep the review package external.
- Documentation/packaging corrections are allowed; no frozen implementation,
  input, historical result or recovery-evidence changes. No staging, commits,
  pushes, training, fresh evaluation, downloads or reliability implementation.

## Current train-only diagnostic planning authorization (2026-10-20)

- The reviewed five-file document commit 2754e1a was separately approved and
  pushed to the existing codex branch; preserve its result and source history.
- Prepare and review a train-only error/coverage diagnostic proposal with
  hypotheses, explicit input boundaries, checks and acceptance criteria.
- Source/schema review and metadata/hash inventory of existing training files
  are allowed; do not compute folds, residuals, coverage or correlations yet.
- No diagnostic implementation/execution, checkpoint/model loads, rendering,
  training, reliability weights, downloads, staging, commits or pushes.

## Current D1 implementation and offline execution authorization (2026-10-20)

- User approved the recommended D1 minimal CPU entry, checks and one frozen
  leaves train-only offline diagnostic. Preserve the reviewed proposal bytes.
- Use only the fourteen approved train/anchor/prior files; exclude test assets,
  evaluation caches, checkpoints, models, rendering and all optimizer updates.
- Keep global point-group folds, fitting/normalization, bins and unknown/failure
  handling frozen. Complete the synthetic/isolation checks before real execution.
- Preserve all failures and partial outputs. Report actual results, resources,
  limitations and acceptance after completion. No reliability weights, new
  training, downloads, staging, commits or pushes.

## Current D1 submission-review authorization (2026-10-20)

- Review the nine-file D1 diff against 2754e1a, complete pre-commit checks and
  prepare one engineering commit candidate with an acceptance recommendation.
- Documentation and packaging corrections are allowed; preserve frozen plan,
  diagnostic source/input hashes, prior results and external recovery evidence.
- Keep the full diff, file inventory and actual check logs outside Git. No staging,
  commits, pushes, extra diagnostic runs, training, downloads or reliability work.

## Current cross-view diagnostic planning authorization (2026-10-20)

- User requested the next recommended step after the D1 engineering commit.
  Prepare and review one train-only cross-view prior diagnostic proposal.
- Source/schema inspection and exact existing training-file size/hash inventory
  are allowed. Preserve D1 results, frozen code/configuration and all evidence.
- No new diagnostic implementation, folds/scores/correlations, other scenes,
  reliability weights, model loads, inference, rendering, training or downloads.
- Keep fbc61f2 local. No staging, commits or pushes; complete the plan checks,
  report the acceptance recommendation and stop.

## Current D2 implementation and offline execution authorization (2026-10-20)

- User explicitly approved the frozen D2 minimal CPU entry, checks before real
  execution, and exactly one offline leaves train-only diagnostic.
- Preserve both proposal files and D1 evidence. Use only the fourteen approved
  training assets, global point-group folds and fixed scoring/analysis/gates.
- Retain every failed or unavailable target/direction and all execution evidence.
  No reliability module, extra training, downloads, staging, commits or pushes.
- Complete the implementation/checks and one frozen run, report actual results,
  resources, limitations and acceptance/next-step recommendation, then stop.

## Current D2 submission-review authorization (2026-10-20)

- User approved reviewing the nine-file D2 diff against fbc61f2, completing
  pre-commit checks and preparing one engineering commit recommendation.
- Documentation/packaging corrections are allowed. Preserve frozen plans,
  runtime/test sources, input hashes, negative results and historical evidence.
- Keep the full diff, file inventory and actual check logs outside Git. No staging,
  commits, pushes, new diagnostic runs, training, downloads or reliability work.

## Current publication and robust-depth planning authorization (2026-10-20)

- User explicitly approved pushing both fbc61f2 and 9987bf4 to the existing
  codex/phase2-uniform-depth branch. Completed and remote ref verified at 9987bf4.
- The next recommended work is review of a simple residual-robust depth control
  proposal only: inspect existing source, training-asset hashes and public loss
  definitions, freeze the candidate/controls and declare implementation gates.
- No loss implementation, autograd/model/GPU work, rendering, training, new test
  evaluation or reliability modules. Preserve old source, inputs and evidence.
- Do not stage, commit or push these new planning changes; report plan acceptance
  and the next bounded recommendation, then stop.

## Current robust-depth CPU implementation authorization (2026-10-20)

- User requested the next recommended step: implement the minimal optional
  residual objective/CLI and CPU value, gradient, isolation and legacy checks.
- Preserve the reviewed plan bytes, original depth.py and renderer/evaluator,
  historical inputs/results/source snapshots. Default L1 must reconstruct its
  historical loop/metadata; explicitly check the opt-in extensions separately.
- No CUDA context/GPU preflight, model/checkpoint loads, rendering, optimization,
  four-run pilot, reliability module, downloads, staging, commits or pushes.
- Preserve all failed checks and report actual CPU evidence and remaining
  execution gates. CPU acceptance is not authorization to train.

## Current robust-depth execution-review authorization (2026-10-20)

- User approved cumulative full-diff review and freezing four comparison argv
  arrays, actual source/upstream/runtime identities and evaluator-only assets.
- Static checks and documentation/manifest additions only; preserve frozen
  proposal bytes, CPU acceptance sources and all old evidence. Test assets may
  be hashed for evaluation provenance; never pass them to the trainer.
- Recommend GPU preflight/formal-training release separately. This step does
  not execute GPU preflight, models, rendering, training or evaluation.
- No reliability module, downloads, staging, commits or pushes. Report checks,
  remaining gates and acceptance, then stop.

## Current robust-depth zero-update GPU authorization (2026-10-20)

- User approved preparing and reviewing the loss-specific zero-update GPU entry,
  CPU checks first, followed by exactly one preflight if checks pass.
- Preserve frozen training/evaluation code, matrix, coefficients and schedule.
  Preflight reads only ten training assets, never real test assets, evaluation
  weights, diagnostic scores or old checkpoints. Preserve every attempt.
- Initialize splats and run forward/backward probes only; no optimizer, scheduler
  or densification updates. No formal training, evaluation, downloads, reliability,
  staging, commits or pushes. Report acceptance and four-run recommendation.
- User requests autonomous routine technical decisions and fewer progress reports;
  this does not lift the explicit current no-training/commit/push boundaries.

## Current robust-depth four-run authorization (2026-10-20)

- User explicitly approved frozen A1_l1 -> B_weak_l1 -> C_cauchy -> A2_l1_repeat,
  four sequential full 30k runs and unchanged independent evaluation only after
  all four complete. Verify code, inputs and same boot/driver/environment, keep
  coefficients/schedule and train/test isolation; preserve failures/degradation.
- Stop on anomalies and investigate; no automatic retries, extra matrix arms,
  hyperparameter changes or new downloads. Codex leads routine acceptance.
- User explicitly authorizes Codex to decide whether to commit/push the reviewed
  engineering result and whether evidence warrants reliability work. This
  supersedes earlier confirmation requirements for this result only; complete
  full-diff/file-list/actual-check/publication review before any chosen commit.
- Choose next-stage recommendations from evidence, not a successful process or
  training fit. Preserve all negative/unknown results and historical evidence.

## Four-run acceptance decision (2026-10-20)

- All four frozen 30k runs and independent post-completion evaluation finished;
  80 related checks passed. Preserve every arm, bookend, per-view regression and
  the initial supervisor interpreter failure (no training started in that failure).
- Engineering acceptance passed. Cauchy was worse than all three controls on
  each test mean and failed all declared direction gates. Stop this fixed
  candidate, do not tune c/lambda or add seeds/scenes automatically.
- Do not enter reliability: existing D1/D2 score routes remain unsupported, and
  this ordinary residual-loss pilot does not establish geometric confidence.
- User delegated publication decision. One cohesive engineering commit/push
  may proceed after complete diff/file/provenance/test/publication checks.

## Current history integration authorization (2026-10-20)

- User requested continuation of the reviewed branch integration plan after
  removing codex/ prefixes from the published stage branches. Use
  chore/integrate-project-history to preserve both independent roots and all
  stage commits; retain the original main description and full project README.
- Review the full diff, file list, actual CPU checks and source/result hashes
  before publishing the integration branch and opening a pull request to main.
  Integrating engineering history does not change research acceptance.
- Preserve all stage branches and historical evidence. No force push, history
  rewriting, branch deletion, new training or reliability implementation.
- Future approved work starts from current main on descriptive feat/, fix/,
  experiment/ or chore/ branches; retain historical names in provenance.

## Current CPU CI and branch protection authorization (2026-10-20)

- User requested the next recommended step: portable CPU continuous integration
  and main protection. Use chore/cpu-ci-and-main-protection; preserve frozen
  training/evaluation code, configurations, results and all historical evidence.
- Review full diff, actual checks, explicit skips and dependency boundaries;
  publish through a PR. Bind main protection only to a real successful CPU
  regression check, require PR integration and apply checks to administrators.
- Use the public repository's standard CPU Actions runner only; no paid runner,
  GPU training, data/model downloads, new algorithm or reliability work.
- Preserve existing branches and commit histories. No force push or deletion.
