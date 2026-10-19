# fern 当前基线的复现入口

当前协议为 `fern-undistorted-v1`，固定三张训练图、三张测试图、504×378、seed42、30k。
以下是人工复现说明，本次提交候选审阅没有重新下载、初始化、训练或评价。
相机来自公开标定，仅初始化点由训练 RGB 得到；这不是三视图相机恢复。

## 环境与外部资产

Linux/WSL、NVIDIA GPU；本机验证 RTX 4070 SUPER 12 GB。
按 [Phase 0 环境说明](phase0.md#6-环境复建说明) 建立 Python 3.10.22 训练环境和 CUDA 12.4.1 编译组件，
依赖见 `requirements-phase0.txt`。另以 Python 3.12.3 安装 `requirements-colmap.txt`，避免两个 pycolmap 实现互相覆盖。
这些是主依赖版本和本机实测组合，尚未验证全新机器安装，也不是完整传递依赖锁。

下文在仓库根目录执行。先把变量改为明确的绝对路径，环境、源码、权重、数据与输出均放在仓库外：

```bash
TRAIN_PY=/absolute/train-env/bin/python
COLMAP_PY=/absolute/colmap-env/bin/python
GSPLAT_SOURCE=/absolute/upstream/gsplat
FERN_WORK=/absolute/new/fern-reproduction
EVAL_WEIGHTS=/absolute/evaluation-weights
```

训练器要求源码是干净的 Git checkout，HEAD 精确为
`937e29912570c372bed6747a5c9bf85fed877bae`；安装的 gsplat wheel 另由依赖文件固定。
可从 `https://github.com/nerfstudio-project/gsplat.git` 获取指定版本；不使用自动更新的 main。

评价前须人工准备 PyTorch 官方 AlexNet 文件：
`https://download.pytorch.org/models/alexnet-owt-7be5be79.pth`，244,408,911 bytes，
放到 `$EVAL_WEIGHTS/hub/checkpoints/alexnet-owt-7be5be79.pth`。
SHA256 为 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`。
评价入口先验证此哈希，不自动下载；LPIPS 标定权重随 torchmetrics 1.9.0 安装并在评价报告记录哈希。
本次审阅不执行下载。使用数据/权重仍需遵守其来源条款，仓库不再分发它们。

## 数据、标定、去畸变、训练与评价

每个叶级 `--output` 都必须不存在；失败目录留存，重试使用新路径。
数据源归档总计 370,385,516 bytes，入口强制 HTTP Range，服务器不支持时拒绝回退全量下载。
第一步只取六张 images_8 和 poses_bounds.npy，成员压缩载荷 2,027,686 bytes。
第三步另读 images.bin（压缩 1,887,022 bytes）与 cameras.bin（43 bytes），加上 ZIP 索引请求。
相机审计跳过所有特征观测，不保留这两个原始二进制，也不读源 points3D；元数据只用于相机核查。

```bash
"$TRAIN_PY" scripts/fetch_fern.py \
  --url https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip \
  --archive-bytes 370385516 --output "$FERN_WORK/raw"

"$COLMAP_PY" scripts/prepare_fern.py \
  --source "$FERN_WORK/raw" --output "$FERN_WORK/legacy-prepared"

"$TRAIN_PY" scripts/audit_fern.py \
  --url https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip \
  --archive-bytes 370385516 --raw "$FERN_WORK/raw" \
  --data "$FERN_WORK/legacy-prepared" --output "$FERN_WORK/camera-audit"

"$TRAIN_PY" scripts/undistort_fern.py \
  --source "$FERN_WORK/raw" \
  --calibration "$FERN_WORK/camera-audit/camera-and-coverage.json" \
  --output "$FERN_WORK/undistorted-source"

"$COLMAP_PY" scripts/prepare_fern.py \
  --source "$FERN_WORK/undistorted-source" \
  --intrinsics "$FERN_WORK/undistorted-source/intrinsics.json" \
  --output "$FERN_WORK/undistorted-prepared"

"$TRAIN_PY" scripts/train_baseline.py \
  --data "$FERN_WORK/undistorted-prepared/train" --source "$GSPLAT_SOURCE" \
  --output "$FERN_WORK/train-30k" --max-steps 30000 --seed 42

"$TRAIN_PY" scripts/evaluate_baseline.py \
  --checkpoint "$FERN_WORK/train-30k/checkpoint.pt" \
  --train-data "$FERN_WORK/undistorted-prepared/train" \
  --test-data "$FERN_WORK/undistorted-prepared/test" \
  --weights "$EVAL_WEIGHTS" --output "$FERN_WORK/evaluation"
```

legacy-prepared 仅为原针孔相机审计所需的训练三角化，不需要重新训练旧模型。
审计入口对照的是旧 K，不能直接改用去畸变后的 prepared。
所有训练入口仅接收 train 目录；独立评价才接收 test 目录。
配置 JSON 是冻结协议说明，入口不读取它自动配置；请保留上面的显式参数，训练后核对 run/config.json。
旧协议/上游参照的命令及证据恢复路径保留在 [Phase 1 报告](phase1-fern.md)。

## 检查与复现限度

无需真实数据即可运行单元检查，但会明确跳过真实资产检查：

```bash
"$TRAIN_PY" -m unittest discover -s tests -v
```

测试仍需安装训练环境依赖，不能用未安装 Torch/gsplat 的空 Python 环境代替。
共有 16 项：无资产时 8 项执行、8 项跳过，不能称为“16 项全部通过”。
针对当前复现资产可设置以下变量后运行同一命令：

```bash
export GEOSPLAT_FERN_RAW="$FERN_WORK/raw"
export GEOSPLAT_FERN_DATA="$FERN_WORK/legacy-prepared"
export GEOSPLAT_CAMERA_AUDIT="$FERN_WORK/camera-audit/camera-and-coverage.json"
export GEOSPLAT_UNDISTORT_SOURCE="$FERN_WORK/undistorted-source"
export GEOSPLAT_UNDISTORT_DATA="$FERN_WORK/undistorted-prepared"
export GEOSPLAT_UNDISTORT_RUN="$FERN_WORK/train-30k"
"$TRAIN_PY" -m unittest discover -s tests -v
```

这会保留原始训练与上游参照两项跳过。要复查本机全部历史证据，再显式提供：

| 变量 | 要求的已完成资产 |
|---|---|
| `GEOSPLAT_RUN` | 原始未去畸变 30k run |
| `GEOSPLAT_TRAINING_SOURCE` | 上述 run 原哈希对应的归档 baseline.py；新 run 有 source/baseline.py 时可省略 |
| `GEOSPLAT_REFERENCE` | 同原始输入的已完成上游 Runner.train 参照 run |

原始审计测试还固定检查历史 373 点与原哈希。全新三角化/CUDA 即使固定 seed 也未保证逐位复现；
这些历史证据检查不等于通用跨机器验收，不能放宽断言或更换旧资产来制造通过。
本机此次使用全部历史与当前资产得到 16 项通过、0 skip；当前检查日志随本机审阅包保留。

期望结果及全部哈希见 `results/fern-3view-undistorted.json`：测试约 15.020 dB / 0.3780 / 0.4491。
单场景单 seed 仍严重过拟合；没有保证复跑逐位相同，也没有跨机器或多种子误差范围。
重采样 GT、K 与初始化均不同于旧协议，不能把差值作为算法收益。
计时保留多个时钟口径，已观察到 perf_counter 与墙钟差异，原因尚未验证。
