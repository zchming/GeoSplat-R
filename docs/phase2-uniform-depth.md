# Phase 2：fern 最小统一深度正则对照

本页保留首次seed42结果和当时判断；新增seed43/44及三种子结论见 [种子复核报告](phase2-seed-stability.md)。

执行日期：2026-10-19。范围仅为冻结的 fern 3-view/seed42/30k 去畸变协议。
用户已批准 49f0522 推送到 `codex/phase1-fern-undistortion`，远端核对一致；main 未改变。
Phase 2 在本地 `codex/phase2-uniform-depth` 实施，后续修改不自动提交或推送。
没有新增场景、鲁棒损失、几何可靠性模块或超参数搜索。

## 冻结对照

沿用 [当前相机/图像协议](phase1-fern.md) 与 `configs/fern-3view-undistorted.json`：
训练 image001/010/019，测试 image000/008/016，504×378，batch=1、float32、seed42、30k，
SH3、同一 DefaultStrategy、同一 RGB 损失、优化器和学习率。训练从原来的 488 个训练三角化点重新初始化，
不从已收敛基线继续训练，不重做相机、去畸变或初始化。
相机 JSON 与点云内容哈希保持不变，两个组的初始参数哈希也逐项核对。

| 组别 | 深度系数 | 从零基 step 1000 起 | 渲染 |
|---|---:|---|---|
| 匹配零权重对照 | 0 | 仍为 0 | RGB+ED |
| 最小统一深度 | 0.01 | 固定 0.01 | RGB+ED |

0.01 参考固定 gsplat 示例的深度系数量级；本项目另有显式相对尺度归一化，因此不声称与上游稀疏深度项等价。
1000 步预热与系数在本轮任何测试评价之前固定，只运行这一组设置，不根据 fern 测试分数选择系数或调度。
两组均加载相同先验并计算深度诊断；零权重组不将该项加入反向目标。
两组使用 RGB+ED，以隔离渲染通道及记录方式的差异。旧 RGB 基线保留作为历史参照，主结论用此次匹配零权重组。
冻结配置见 `configs/fern-3view-uniform-depth.json`；项目外 frozen-experiment.json 记录训练前配置/源码/先验哈希。

## 深度来源与隔离

选择 [Depth Anything V2 官方 Small](https://github.com/DepthAnything/Depth-Anything-V2)，
按[论文 5.2 节](https://arxiv.org/html/2406.09414v2#S5.SS2)将其输出作为仿射不变的相对逆深度，
不能将原始输出直接作为米制或场景 Z 相减。

- 权重 `depth_anything_v2_vits.pth`：99,218,434 bytes；用户事先明确批准后才下载。
- SHA256：`715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`。
- Hugging Face revision：`03876f8651c73a60fe4c2c48294e09fcb6838fcf`。
- 官方代码 revision：`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`；仅获取 15 个推理/来源文件共 73,964 bytes，逐个校验 Git blob。
- Small 依官方说明使用 Apache-2.0；代码及权重镜像保留在项目外，未复制进本项目，也未替本项目选择许可证。
- 没有下载新数据集、额外 backbone 权重、更大模型、Gradio 或 xFormers。

推理采用官方 infer_image、input_size=518、float32、OpenCV 流程；实际网络输入 1×3×518×686，
输出插值回 378×504。此 518 仅为离线模型输入，不改变 3DGS 或评价分辨率。
xFormers 不可用，使用官方 attention 回退路径；不修改上游源码。
只对三张训练 RGB 推理，不生成测试深度，也不把测试 RGB/深度用于对齐、损失或参数选择。
数据读取审计禁止访问 prepared 的兄弟 test 目录；同时记录实际 Python 文件读取与内容哈希。
读取审计是该入口的可审计证据，不是针对任意恶意原生代码的安全沙箱。

## 对齐、有效性与损失

从已有 train/triangulated 模型导出观测锚点；验证只含三张训练图，按相同训练相机归一化后，
与冻结 488 点最近距离最大约 1.05e-6。仅用该图实际观测到的训练轨迹，正 Z、重投影残差≤2 px，
剔除边界，在同一个舍入像素上只留最近点。每图分别保留 208、398、393 个锚点。
这是明确的基础有效性处理，不是额外学习匹配、可靠性估计或新 SfM 初始化。

对每张训练图，用其锚点的相机轴向逆深度拟合无权重普通最小二乘：

```text
(a,b) = argmin sum_i (a * raw_inverse_depth(u_i) + b - 1/Z_sfm_i)^2
target(u) = a * raw_inverse_depth(u) + b
normalizer = median_i(1/Z_sfm_i)
L_depth = mean_valid |1/ED_render(u) - target(u)| / normalizer
L_total = 0.8 L1_RGB + 0.2 (1 - SSIM_valid) + lambda * L_depth
```

raw 在投影锚点处双线性采样。锚点不足20、常量预测、非正斜率或过少正目标时直接失败，不切换模型或偷偷反转深度。
固定目标掩码是有限正逆深度加2像素边界；训练时再排除非有限/非正 ED 和 detached alpha<0.5。
所有剩余像素权重相同，用当步有效像素数求均值；记录实际有效比例，不按置信度或残差重新加权。
掩码不传梯度，ED 的倒数和 L1 残差保持可微。空有效集返回有穷零损失。
该实现没有独立遮挡真值或多视图遮挡检验；SfM 观测与每像素最近锚点只避免部分明显无效锚点，
alpha 也只是渲染覆盖门槛，不能当成准确率、遮挡正确性或可靠性证据。

| 训练图 | a | b | 锚点相关系数 | 归一化拟合 L1 |
|---|---:|---:|---:|---:|
| image001 | 0.0169284 | 0.0422844 | 0.8216 | 0.0720 |
| image010 | 0.0136824 | 0.0503308 | 0.6483 | 0.0976 |
| image019 | 0.0141552 | 0.0534928 | 0.6436 | 0.0998 |

三图目标有效比例均约 98.16%。拟合残差和相关性只说明训练锚点上的兼容性，不证明密集深度准确。
已实际查看全部训练先验图：大致区分花盆/植物前景和远背景，但细小叶片与空隙常被连接成片；
玻璃、反射和纹理薄结构的深度分层仅靠 RGB 无法定量判真。没有可靠密集真值，不能宣称哪些像素已经证实正确或错误。

## 实际结果与验收

| 测试组别 | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| 本轮零权重 | 13.9014 | 0.334741 | 0.479944 |
| 统一深度 | 15.9755 | 0.393947 | 0.436121 |

本轮差值（统一−零权重）：PSNR +2.0741 dB，SSIM +0.059206，LPIPS -0.043823。

| 测试图 | 零权重 PSNR | 统一深度 PSNR | 差值 dB |
|---|---:|---:|---:|
| image000.png | 14.8334 | 17.3449 | +2.5115 |
| image008.png | 16.2430 | 17.6310 | +1.3880 |
| image016.png | 10.6279 | 12.9506 | +2.3227 |

三张测试图的三个指标均改善，没有逐图指标恶化被平均值掩盖。
训练均值从零权重的 47.9787 dB/0.998431/0.001174，变为统一深度的 43.3101/0.994939/0.006230；
训练拟合下降符合目标的取舍，不能只展示测试收益而隐藏训练退化。
已实际查看三图 GT/预测/Error/Depth：黑色缺失区域减少，但 image016 仍有严重枝叶混叠、重影、背景扭曲，
仅 12.951 dB。统一组测试 alpha≥0.5 比例均为1，并不证明几何或遮挡正确。

**零权重波动必须披露：** 前次相同 GT 的 RGB 基线为 15.0197/0.378023/0.449066，
此次零权重只有 13.9014/0.334741/0.479944，PSNR 低 1.1182 dB。
统一深度相对旧 RGB 基线的 PSNR 差仅 +0.9558 dB，不能只突出相对较弱对照的 +2.0741。
旧/新零权重有渲染通道和诊断路径差异，并非严格重复；这些因素和非确定性尚未通过完整训练消融拆分，
不能把差异原因直接认定为 CUDA 或深度通道。实际两组第1步损失相同，初始参数哈希相同，
但尚未开启深度的第1000步已分别为3449/3567个高斯、损失0.089706/0.088537。
额外 CUDA 小场景检查的 RGB 与 RGB+ED 前向/纯RGB梯度在容差内一致，
它排除了该检查条件下的明显通道差错，不保证长程增密轨迹逐位一致。

| 训练资源 | 零权重 | 统一深度 |
|---|---:|---:|
| time.time 墙钟秒 | 374.181 | 374.994 |
| perf_counter 秒 | 334.439 | 338.869 |
| CUDA stream event 秒 | 371.609 | 376.531 |
| 显卡采样跨度秒 | 374.157 | 374.909 |
| Torch allocated 峰值 MiB | 711.593 | 724.845 |
| Torch reserved 峰值 MiB | 1094.000 | 1012.000 |
| 整卡采样峰值 MiB | 2196.000 | 2141.000 |
| GPU 平均利用率 % | 89.534 | 89.365 |
| GPU 峰值利用率 % | 94.000 | 94.000 |
| 最终高斯数 | 447350.000 | 457281.000 |

两次正式训练均成功、各30000更新，另有2000步短跑成功；本小步训练失败0、OOM0。
先验准备整段墙钟2.366秒（含模型加载/首次推理等），锚点导出0.079秒；
单图推理0.521/0.028/0.029秒，第一张包含首次执行开销。
先验准备 Torch allocated峰值283.55MiB，整卡峰值1541MiB；权重下载12.667秒另记，不混进训练时间。
两组墙钟训练时间接近，但最终高斯数不同，多时钟仍不一致，不能把差值当成孤立深度损失开销或算法加速。
完整计时与逐图评价/渲染时间见 results/fern-3view-uniform-depth.json。

首次评价继承了“Vanilla”图例；只新增 --label 参数和显示元数据，在独立 v2 目录重新评价，
原始评价完整保留。所有训练/测试 PSNR、SSIM、LPIPS 与首次评价差异小于1e-6，
未改评价公式、GT、clamp、裁切/掩码或 LPIPS 权重。主对照图明确标记 zero/uniform。
检查全部25项通过、0 skip；另有4个相关入口帮助和覆盖保护检查通过。
包括仿射方向与退化拒绝、均值/掩码/梯度、真实先验来源、固定输入/初值/预算、
实际读取隔离、旧实验回归以及额外 CUDA 通道一致性检查。


## 复现与证据

在 [Phase 1 环境与资产](reproduce-fern.md) 完成的基础上，三个新输出必须是不存在的独立目录：

```bash
"$COLMAP_PY" scripts/export_train_anchors.py \
  --data "$FERN_WORK/undistorted-prepared/train" --output /absolute/new/train-anchors

"$TRAIN_PY" scripts/prepare_depth_priors.py \
  --data "$FERN_WORK/undistorted-prepared/train" --anchors /absolute/new/train-anchors \
  --source /absolute/pinned/depth-anything-v2 --source-manifest /absolute/source-selection.json \
  --weights /absolute/weights/depth_anything_v2_vits.pth --output /absolute/new/train-priors

"$TRAIN_PY" scripts/train_baseline.py \
  --data "$FERN_WORK/undistorted-prepared/train" --source "$GSPLAT_SOURCE" \
  --output /absolute/new/control-zero-30k --seed 42 --max-steps 30000 \
  --depth-priors /absolute/new/train-priors --depth-lambda 0 --depth-start 1000

"$TRAIN_PY" scripts/train_baseline.py \
  --data "$FERN_WORK/undistorted-prepared/train" --source "$GSPLAT_SOURCE" \
  --output /absolute/new/uniform-30k --seed 42 --max-steps 30000 \
  --depth-priors /absolute/new/train-priors --depth-lambda 0.01 --depth-start 1000
```

两个正式 run 各使用原有独立 evaluate_baseline.py，参数为各自 checkpoint、同一训练/测试目录、
同一评价权重，以及各自新的评价输出目录。评价方法、GT、指标实现与评价掩码未改；评价入口仅新增显示标签参数，见上述 v2 核对。
source-selection.json 结构为 `{sha: 固定代码revision, files: [{path: 相对源码路径, sha256: 内容哈希, ...}]}`；
本机原始文件及逐文件 Git blob 记录保存在项目外 phase2/source-selection.json。
新机器须按固定 revision 获取同一官方文件并重新校验；不把本机绝对路径照抄为机器无关配置。

完整证据位于项目外 `~/.local/share/geosplat-r/phase2/`，包括下载/源码来源、冻结配置、锚点、
先验数组/可视化、2000步短跑、两组30k、各自评价与对照图、资源采样、源码副本及读取审计。
全部旧原始/诊断/去畸变输入与模型保留；完整生成物不随 Git 发布。

完整历史检查之外，设置 GEOSPLAT_DEPTH_PRIORS 指向 train-priors，GEOSPLAT_DEPTH_PAIR 指向 phase2，
并设置 GEOSPLAT_CUDA_CHECK=1，再运行 `python -m unittest discover -s tests -v`。缺少显式资产时明确跳过对应测试。

## 判断与停止

**工程/协议验收通过；研究效果仅为单场景单种子的初步正信号。**
已有最低限度证据支持继续验证统一深度先验，不足以宣称稳定收益、独立几何准确率改善或新算法贡献。
主比较中三个测试视角均改善，但零权重基线波动明显，当前不建议进入可靠性模块或扩大方法复杂度。

下一步建议先以同一冻结系数和调度补充 seed43/44 的两组配对，核实收益与零权重波动；
不根据测试分数重新调参。该额外训练仍需另行授权，本次停止。
Phase 2 改动暂不提交或推送；若要形成下一次工程提交，先审阅完整 Diff、实际检查与文件清单，再获确认。

累计成果增加了 Small 训练先验、显式逆Z对齐/有效性规则、可微统一损失、匹配零权重对照和负结果披露。
已观察到一组正向新视角结果，但尚无跨种子/跨场景稳定性结论；未实现几何可靠性模块。
