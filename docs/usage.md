# GeoSplat-R 使用流程

**质量优先选择 FSGS；需要更快完成训练时选择统一 L1。** 两条路径已在同一批冻结输入上完成对照。horns、orchids、trex 的 FSGS 测试 PSNR 平均提高 4.456 dB，高斯参数存储减少 55.5%—78.7%；训练函数耗时约 625—659 秒，L1 为 207—356 秒。完整逐图结果、资源和失败记录见[独立场景验证](fsgs-final-three-scenes.md)。这是两套完整配方的取舍，更新次数分别为 9,999 和 30,000。

## 1. 配置已有环境

把 [runtime.example.json](../configs/runtime.example.json) 复制到仓库外，填入本机位置。路径可为绝对路径，也可相对配置文件；虚拟环境的 `bin/python` 路径会原样保留。入口使用标准库，配置中的 Python 才负责模型计算。

| 配置键 | 使用范围 |
|---|---|
| `python`、`cuda_home`、`metric_weights` | 两条路径及独立评价 |
| `fsgs_python`、`fsgs_source`、`midas_source`、`midas_weights` | FSGS 训练 |
| `gsplat_source` | L1 训练 |
| `colmap_python`、`small_source`、`small_source_manifest`、`small_weights` | L1 尚无对齐先验时；复用先验可省略 |

已验证环境是 Linux/WSL、Python 3.10.22、Torch 2.4.1+cu124、CUDA Toolkit 12.4.1、gsplat 1.5.3；COLMAP 使用独立 Python 3.12.3 / pycolmap 4.2.1。基础依赖见 [requirements-phase0.txt](../requirements-phase0.txt)、[requirements-colmap.txt](../requirements-colmap.txt)和[环境记录](phase0.md)。当前入口使用本地 GPU 0，整卡采样上限 11,500 MiB；依赖、CUDA 扩展和权重须提前准备，运行时不安装或下载。

固定源码与权重：

| 资产 | 身份 |
|---|---|
| gsplat 源码 | `937e29912570c372bed6747a5c9bf85fed877bae` |
| FSGS 源码 | `a536a64c5b366b1088be64eeadf9e791ca26897c`，应用下述兼容补丁 |
| MiDaS | `v3_1`，`dpt_hybrid_384.pt`；SHA256 `501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853` |
| Depth Anything V2 Small | 源码 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`；权重 SHA256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` |
| 指标权重目录 | 包含 `hub/checkpoints/alexnet-owt-7be5be79.pth`；SHA256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |

`small_source_manifest` 是已有的 `source-selection.json`，记录 Small 源码文件散列。FSGS 不读取 Small 先验。

已有 FSGS 环境可直接复用。重建环境时，保持基础 Torch 栈，单独安装 `timm==0.9.12`、`plyfile==1.1.3`、`scipy==1.15.3`、`tqdm==4.70.1`，并在同一 FSGS 环境中编译两个 CUDA 扩展。下面是准备新源码的命令，变量指向已设置的位置：

```bash
git clone --recursive https://github.com/VITA-Group/FSGS.git "$NEW_FSGS_SOURCE"
git -C "$NEW_FSGS_SOURCE" checkout a536a64c5b366b1088be64eeadf9e791ca26897c
git -C "$NEW_FSGS_SOURCE" submodule update --init --recursive
patch -d "$NEW_FSGS_SOURCE" -p1 < "$PROJECT/patches/fsgs-cuda12.patch"
CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST=8.9 "$FSGS_PYTHON" -m pip install --no-build-isolation \
  "$NEW_FSGS_SOURCE/submodules/diff-gaussian-rasterization-confidence" \
  "$NEW_FSGS_SOURCE/submodules/simple-knn"
```

`8.9` 对应本次使用的 RTX 4070 SUPER；其他显卡需使用对应架构。补丁只补 `<cstdint>` 并移除两个未使用的 Open3D 导入，与已验证运行源码一致。像素采样修正和本地 MiDaS 加载由项目训练适配器完成。[FSGS](https://github.com/VITA-Group/FSGS/tree/a536a64c5b366b1088be64eeadf9e791ca26897c)、MiDaS 及其 CUDA 子模块保留各自许可，使用及分发须遵循原许可。

## 2. 选择已准备的输入

统一入口接受已有的 `prepared/train` 和 `prepared/test`：

```text
prepared/
  train/  cameras.json, points.npz, input-manifest.json, images/, triangulated/
  test/   cameras.json, images/
```

适用范围是已经冻结的 LLFF 3-view、504×378、已知相机协议。通用清单支持 flower、leaves、room、fortress、horns、orchids、trex；历史 fern 入口继续保留。L1 生成锚点时使用 `triangulated/`。训练的初始化、锚点及深度先验都来自训练图；测试图在最后的独立评价阶段读取。

已有准备结果直接复用。需要准备原始 LLFF 数据时，沿用 [room 输入准备](fsgs-generalization.md#复现入口)或最终三场景的固定输入配置，不重新运行全场景 SfM/MVS。数据准备独立于训练入口，下载范围和冻结划分保持显式。

若工作目录还保留其他未提交实验，建议先导出已交付版本到仓库外再运行，避免把本地实验代码混入配方。此操作复制已跟踪版本，不移动或覆盖工作目录：

```bash
mkdir "$NEW_RUNTIME"
git archive HEAD | tar -x -C "$NEW_RUNTIME"
cd "$NEW_RUNTIME"
```

## 3. 运行两条路径

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

## 4. 复用模型和查看结果

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

需要查看命令时在上述命令后附加 `--plan`：不创建输出、不初始化 CUDA，不作为运行前置环节。新增流程的 CPU 检查为 `python3 -m unittest discover -s tests -p test_workflow.py`；交付验收见[工程收尾](engineering-closeout.md)。
