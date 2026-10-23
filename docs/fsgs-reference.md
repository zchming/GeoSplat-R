# FSGS 接入与三开发场景确认

**FSGS 完整训练路径已接通现有 gsplat 评价/可视化；固定配方的 fern、flower、leaves 开发场景确认全部通过。** 三场景使用同一 seed44，每个 FSGS 模型完成 9,999 次参数更新；下表使用未修改的 gsplat 评价器。

| 场景 | 统一 L1 PSNR | FSGS→gsplat PSNR | ΔPSNR | FSGS SSIM | FSGS LPIPS-Alex |
|---|---:|---:|---:|---:|---:|
| fern | 15.651364 | 21.946962 | +6.295598 | 0.690963 | 0.203203 |
| flower | 16.234055 | 21.310167 | +5.076113 | 0.656716 | 0.174667 |
| leaves | 12.298660 | 17.368537 | +5.069877 | 0.566035 | 0.171683 |

三场景等权平均 ΔPSNR +5.480529 dB、ΔSSIM +0.309602、ΔLPIPS −0.207336。12 张测试图共 36 个逐图/指标比较全部改善，未筛除图像。fern/flower 的对照是历史同协议 seed44 统一 L1；leaves 是同次启动的 L1 首尾重复均值。对照均为 30k 更新，不能把这项完整配方比较写成单模块贡献或等更新数加速。

完整指标、逐图对照、原生/gsplat 差异及资源见[三开发场景结果](../results/fsgs-development-confirmation.json)和[执行配置](../configs/fsgs-development-confirmation.json)。新增两场景训练先全部完成，随后独立评价；没有根据测试结果调整参数。三模型分别为 115,352 / 232,051 / 359,092 个高斯，成功训练进程总计 33.46 分钟，整卡采样峰值 4071 MiB。两种渲染器最大逐图 PSNR 差为 fern 的 0.015123 dB，原生结果同样完整保留。

新增 fern/flower 阶段用时 1345.653/1800 秒；继承此前 4911.190 秒，连续阶段累计记账 6256.843 秒，不代表项目全部历史成本。前一阶段的 90 分钟预算和本次新增阶段的 30 分钟预算分别记账，所有消耗和中止记录均保留。原生与转换后检查点、RGB/深度图、三维点图及新视角视频均位于仓库外。对比图固定取各场景 image000，没有按指标挑选视角；leaves 中间图展示 A1，表格报告 A1/A2 均值。

当前证据覆盖三个已使用的开发场景、一个 FSGS seed。工程接入和开发场景质量改善成立；最终未调参场景、多种子稳定性与交付收尾尚未完成。本阶段未启动最终独立 benchmark，实验执行期间未自动提交或推送。

本阶段接入[官方 FSGS](https://github.com/VITA-Group/FSGS/tree/a536a64c5b366b1088be64eeadf9e791ca26897c)的原生渲染、增密及虚拟视角深度监督，使用 leaves 冻结训练/测试图像、相机和 274 个训练 SfM 点。没有重新估计相机、下载数据或引入测试图像训练。配置与来源身份见 [配置](../configs/fsgs-reference.json)。

## 运行范围

- 3 张训练图、4 张独立测试图，504×378，seed44。
- 保留官方 LLFF 10k iteration 配方及 `sample_pseudo_interval=1`；原代码最后一次不执行 optimizer.step，因此实际为 9,999 次参数更新。
- 官方 MiDaS DPT-Hybrid 权重，MiDaS v3_1 源码及 timm 0.9.12；没有用现有 Depth Anything V2 替代。
- 初始化保持现有稀疏点，没有使用官方主要配方的稠密 MVS 点。虚拟相机 near/far 只从投影落入各训练相机视锥的 SfM 点正 Z 取 0.1/99.9 百分位，不将视锥内点等同于已验证无遮挡的可见点。
- 原生 CUDA 光栅器在整数像素位置采样，gsplat 在像素索引 +0.5 处采样。当前 K 为居中主点；真实和虚拟相机的原生投影都额外平移 −0.5 像素，使每个输出像素对应的射线与冻结 gsplat 基线一致。只核对投影均值与 K 数值不足以建立这种一致性。
- 运行适配只补齐缺失的 `<cstdint>`，去掉两处未使用的 Open3D 导入，并显式从本地加载深度网络。保留第三方许可及原始源码归档。

这是完整训练路径的工程对照，不是论文指标复现或单模块消融：相较当前 gsplat 统一 L1，渲染器、优化器、增密、SH、深度网络和日程都不同，旧基线为 30k 更新。采用相同测试图和指标定义比较最终质量与实测成本，不把差异归因于某一个组件。

## 接入与复现

独立环境复用已有 Torch 2.4.1+cu124；安装 timm 0.9.12，并使用 CUDA 12.4.1 为 sm89 编译官方 `diff-gaussian-rasterization-confidence` 和 `simple-knn`。不修改已有训练环境。

用[输入转换入口](../scripts/prepare_fsgs_scene.py)创建新目录：

```bash
python scripts/prepare_fsgs_scene.py --train-dir "$TRAIN_DATA" --output "$NEW_FSGS_DATA"
```

入口校验原训练清单中的相机、点和 RGB 身份，导出 COLMAP 文本、训练点 PLY 及 train-only bounds。输出目录中没有测试资产。`3_views/dense/fused.ply` 是上游必需路径名，其内容仍为原始稀疏点。

[原生训练与评价入口](../scripts/run_fsgs_reference.py)接受明确的源码、数据、权重及新输出位置，拒绝覆盖：

```bash
python scripts/run_fsgs_reference.py train \
  --source "$PATCHED_FSGS_SOURCE" --train-data "$NEW_FSGS_DATA" \
  --midas-source "$MIDAS_SOURCE" --weights "$MIDAS_WEIGHTS" \
  --weights-sha256 501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853 \
  --output "$NEW_TRAIN_OUTPUT" --seed 44

python scripts/run_fsgs_reference.py evaluate \
  --source "$PATCHED_FSGS_SOURCE" --checkpoint "$NEW_TRAIN_OUTPUT/chkpnt10000.pth" \
  --test-data "$TEST_DATA" --metric-weights "$METRIC_WEIGHTS" --output "$NEW_EVAL_OUTPUT"
```

[检查点转换入口](../scripts/convert_fsgs_checkpoint.py)将完整 FSGS 模型接到现有 gsplat 评价和可视化流程。均值、log-scale、wxyz 四元数、opacity logits 及 SH 系数保持数值不变，只展平 opacity 维度；不训练、不重新拟合。转换后仍使用未修改的 `scripts/evaluate_baseline.py`，原生和 gsplat 两套渲染结果都保留。

```bash
python scripts/convert_fsgs_checkpoint.py \
  --checkpoint "$NEW_TRAIN_OUTPUT/chkpnt10000.pth" --output "$NEW_CONVERTED_CHECKPOINT"
python scripts/evaluate_baseline.py \
  --checkpoint "$NEW_CONVERTED_CHECKPOINT" --train-data "$TRAIN_DATA" \
  --test-data "$TEST_DATA" --weights "$METRIC_WEIGHTS" \
  --output "$NEW_GSPLAT_EVAL_OUTPUT" --label "FSGS trained / gsplat rendered"
```

开启本轮测试指标前固定选择规则：三个测试均值均超过对应指标上较好的 L1 首尾重复结果，才作为下一阶段的单场景质量候选；所有逐图变化仍完整报告。原生结果与 gsplat 接入结果分别判定，不在评价后挑选较好的渲染器隐藏另一套结果。

本次执行由外部 supervisor 计时和采样整卡显存。继承此前 3653.198913/5400 秒记账，初次训练进程上限 1656.801087 秒，预留评价 90 秒，整卡上限 11500 MiB。发现像素采样约定差异后停止初次尝试，保留其 557.861572 秒消耗和全部日志；修正运行从 4211.060485 秒继续记账，新训练上限 1098.939515 秒。该入口本身不启动 supervisor；复现时须使用等价资源限制。预算计训练/评价进程墙时，CPU 编译和下载另计，不重置此前消耗。

## 已完成的输入核对

实际上游读取函数成功加载 3 张训练图、0 张测试图和 274 点；相机 R/T 相对冻结 OpenCV c2w 的最大误差为 1.11e-16。原始点按 float32 完全一致，官方函数生成的 10,000 个虚拟位姿全部有限。发现像素约定差异后，对 100 个投影点核对两种采样网格的光栅位移，修正后最大误差为 0。上述 CPU 核对不初始化 CUDA，不代替真实训练结果。

原始克隆 TLS 失败、CUDA 12 缺头文件编译失败、初次 CPU 核对脚本的 PIL 命名遮蔽及其修正均保留在外部证据中。两个导出目录及首次中止训练均保留，图像、位姿、点及 bounds 未更改。未对中止模型评价；修正运行从头初始化。独立评价不构建或调用 MiDaS，避免上游包导入时的自动下载。

## leaves 实测结果

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| 统一 L1 首尾两次均值，30k 更新 | 12.298660 | 0.217038 | 0.370363 |
| FSGS 训练、原生渲染，9,999 更新 | 17.368485 | 0.566042 | 0.171667 |
| 同一 FSGS 模型、现有 gsplat 渲染 | 17.368537 | 0.566035 | 0.171683 |

接入现有 gsplat 后相对统一 L1 两次平均，PSNR +5.069877 dB、SSIM +0.348997、LPIPS −0.198680。四张测试图的三个指标均改善，三个均值也全部超过对应指标较好的 L1 首尾重复。逐图结果和原生/gsplat 差异完整保留在[结果 JSON](../results/fsgs-reference.json)；两种渲染器逐图指标最大差异为 PSNR 0.000174 dB、SSIM 0.000016、LPIPS 0.000041。

成功训练进程耗时 684.55 秒，其中原生 training 函数 679.70 秒；整卡采样峰值 4071 MiB，最终 359,092 个高斯。原 L1 两次 training 函数的 perf-counter 计时分别为 807.47/816.03 秒，点数 982,111/925,165；这是不同完整配方的实测成本，不是等更新数加速比较。相同 gsplat 评价器测得 FSGS 模型平均 RGB+ED 渲染 1.986 ms/图。

本轮累计记账 4911.190367/5400 秒，包含先前实验的 3653.198913 秒、首次中止训练及所有评价修复。评价兼容修复包括 NumPy 标量/字节编码的显式受限读取，以及采用 Torch 2.4 支持的 `add_safe_globals`；两次失败评价没有重训模型，目录和日志保留。

**工程接入和 leaves 单场景质量对照通过。** 完整 FSGS 路线与当前 gsplat 训练输出格式已接通；随后固定配方完成了 fern/flower 开发场景确认，结果见本页开头。新增阶段独立限时，不清零此前累计消耗。

累计贡献是可复用的 train-only 官方实现适配、独立评价及无数值改动的模型导入，并在现有渲染流程中验证了质量收益。实验执行期间未自动提交或推送。
