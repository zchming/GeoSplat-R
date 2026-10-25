# 使用流程

质量优先选择 FSGS；需要更快完成训练时选择统一 L1。先填写[环境配置](environment.md)，质量与成本比较见[实验结果](results.md)。

## 1. 准备输入

统一入口接受已有的 `prepared/train` 和 `prepared/test`：

```text
prepared/
  train/  cameras.json, points.npz, input-manifest.json, images/, triangulated/
  test/   cameras.json, images/
```

适用范围是已经冻结的 LLFF 3-view、504×378、已知相机协议。通用清单支持 flower、leaves、room、fortress、horns、orchids、trex；历史 fern 入口继续保留。L1 生成锚点时使用 `triangulated/`。训练的初始化、锚点及深度先验都来自训练图；测试图在最后的独立评价阶段读取。

已有准备结果直接复用。原始 LLFF 输入使用 `configs/fsgs-room-input.json` 或 `configs/fsgs-final-*-input.json` 中的固定划分。以下以 room 为例，所有输出均须使用新位置；下载只获取指定成员：

```bash
"$PYTHON" scripts/fetch_llff_scene.py --plan configs/fsgs-room-input.json --output "$NEW_RAW"
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); json.dump({"scenes":[p["scene"]]},open(sys.argv[2],"x"))' configs/fsgs-room-input.json "$SCENE_SELECTION"
"$PYTHON" scripts/undistort_llff_subset.py --source "$NEW_RAW" --plan "$SCENE_SELECTION" --scene room --output "$NEW_UNDISTORTED"
"$COLMAP_PYTHON" scripts/prepare_fern.py --source "$NEW_UNDISTORTED" --intrinsics "$NEW_UNDISTORTED/intrinsics.json" --output "$NEW_PREPARED"
"$PYTHON" scripts/freeze_scene_input.py --data "$NEW_PREPARED/train" --scene room --output "$NEW_PREPARED/train/input-manifest.json"
```

`PYTHON`、`COLMAP_PYTHON` 分别指向训练和COLMAP环境中的解释器。准备过程使用训练图重建稀疏初始化，不运行全场景稠密MVS。

若工作目录还保留其他未提交实验，建议先导出已交付版本到仓库外再运行，避免把本地实验代码混入配方。此操作复制已跟踪版本，不移动或覆盖工作目录：

```bash
mkdir "$NEW_RUNTIME"
git archive HEAD | tar -x -C "$NEW_RUNTIME"
cd "$NEW_RUNTIME"
```

## 2. 运行两条路径

设置 `CONFIG` 为本机环境 JSON，`DATA` 为已准备场景目录，`SPENT` 为当前账本的累计已用秒数，`CAP` 为包含已有消耗的总上限。两条路径比较时，种子和输入保持相同。新输出目录必须尚不存在。

FSGS 质量路径：

```bash
python3 scripts/run_geosplat.py run --mode fsgs --config "$CONFIG" \
  --train-data "$DATA/train" --test-data "$DATA/test" \
  --output "$NEW_FSGS_RUN" --seed 44 \
  --inherited-seconds "$SPENT" --budget-seconds "$CAP"
```

自动依次执行：训练输入导出 → 原生固定 10k 循环 → 检查点无数值改动转换 → `fsgs-native` 评价。初始化仍为训练稀疏 SfM 点；上游必需的 `3_views/dense/fused.ply` 路径不表示执行了稠密 MVS。

L1 快训路径：

```bash
python3 scripts/run_geosplat.py run --mode l1 --config "$CONFIG" \
  --train-data "$DATA/train" --test-data "$DATA/test" \
  --output "$NEW_L1_RUN" --depth-priors "$EXISTING_PRIORS" --seed 44 \
  --inherited-seconds "$SPENT" --budget-seconds "$CAP"
```

没有先验时省略 `--depth-priors`，入口先导出训练锚点、生成并对齐 Small 先验。L1 保持 30k 更新、λ=0.01、start=1000、`standard` 评价；“快训”指已测场景相对 FSGS 更快，不是缩减更新数。上一步完成后，用其 `workflow.json` 的 `charged_seconds` 更新下一步的 `SPENT`。

## 3. 复用模型和查看结果

已有模型可直接重新生成评价交付物，不启动训练。FSGS 使用转换后的 `model.pt`，L1 使用 `checkpoint.pt`：

```bash
python3 scripts/run_geosplat.py evaluate --mode fsgs --config "$CONFIG" \
  --checkpoint "$EXISTING_FSGS_MODEL" \
  --train-data "$DATA/train" --test-data "$DATA/test" \
  --output "$NEW_EVALUATION_RUN" \
  --inherited-seconds "$SPENT" --budget-seconds "$CAP"
```

评价 L1 时改成 `--mode l1` 和对应检查点。FSGS 自动使用中心 Z≤0.2 剔除的原生兼容渲染，L1 保留 near=0.01；FSGS 的 standard 敏感性对照仍可通过底层评价入口显式运行，旧结果全部保留。

| 输出位置 | 内容 |
|---|---|
| `workflow.json` | 配方、源码散列、实际命令、完成状态、退出码、累计预算和采样显存 |
| `logs/` | 每步原始日志与整卡显存采样；失败目录同样保留 |
| `train/` | 原始训练记录及检查点；evaluation-only 运行不生成此目录 |
| `model/model.pt` | FSGS 转换后的 gsplat 模型 |
| `evaluation/metrics.json` | 全部训练/测试视角指标及渲染配置 |
| `evaluation/test-comparison.png` | 全部测试视角 GT / 预测 / 误差 / 深度对比 |
| `evaluation/scene.ply`、`scene-3d.png` | 模型点云与相机/高斯示意 |
| `evaluation/novel-view-path.mp4` | 48 帧新视角视频 |

预算按全部子进程墙时累加，包含 CPU 准备和失败进程，不清零继承消耗；首次编译也会占用该次进程时限。超时、超显存或工作进程失败即结束本次流程，无自动重训。普通中断会终止子进程组并写回账本；若系统强制终止监督器，保留的 `running` 记录及 `reserved_until_seconds` 表示预算尚未结算，确认子进程状态前按该上限保守继承。显存是周期采样值，连续峰值以模型自身 Torch 记录另行报告。

需要查看命令时在上述命令后附加 `--plan`：不创建输出、不初始化 CUDA，不作为运行前置环节。新增流程的 CPU 检查为 `python3 -m unittest discover -s tests -p test_workflow.py`。
