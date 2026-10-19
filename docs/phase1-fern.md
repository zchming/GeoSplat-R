# Phase 1：fern 3-view Vanilla 3DGS

本页按实验历史保留原始基线、诊断与去畸变复核；当前结论见文末“2026-10-19：去畸变协议与基线复核”。
当前协议的顺序复现入口见 [reproduce-fern.md](reproduce-fern.md)。前文资源、缓存和验收意见属于相应历史运行。

仅完成用户授权的 fern、3 张训练视角、seed42 原始基线。没有进入深度正则、可靠性模块、其他场景或退化实验。
Git 目标为 https://github.com/zchming/GeoSplat-R.git，当前仅配置 origin，没有提交或推送。

## 范围与协议

这是一条 **gsplat 对 Vanilla 3DGS 的最小适配基线**，不是 Inria 原实现的逐位复刻，也不是深度监督方法。
使用固定 gsplat v1.5.3 SHA `937e29912570c372bed6747a5c9bf85fed877bae`，直接调用其
`create_splats_with_optimizers`、CUDA 光栅器、DefaultStrategy 和 fused SSIM。
训练循环遵循上游损失、Adam、指数位置学习率、SH 调度及更新后增密次序。
本地差异是训练输入隔离、3 张 CPU 图像缓存、独立评价、资源采样和保存全部更新后的检查点；
不复用上游在最后更新前保存 checkpoint 的时序。训练相机按 seed42 的独立随机发生器逐轮排列；
这与上游 DataLoader 的随机数消费次序不同，故不声称逐位相同。归属及 Apache-2.0 副本在 `docs/licenses/`。
没有为项目自行选择许可证。

| 项目 | 冻结设置 |
|---|---|
| 训练 | image001.png、image010.png、image019.png |
| 测试 | image000.png、image008.png、image016.png |
| 其他图像 | 14 张未使用，RGB 未下载 |
| 视角选择 | 按 20 张文件名排序，零基每 8 张留测；从剩余索引按 rint(linspace(0,M-1,3)) 选训练 |
| 输入与评价分辨率 | 504×378，images_8；无裁剪/额外掩码 |
| 训练 | 30,000 次优化器更新；seed42；float32；batch=1 |
| 表示与光栅 | SH3，每 1000 步增加一级；classic、packed=True |
| 目标 | 0.8 L1 + 0.2 (1 − fused SSIM valid) |
| 其他模块 | 无深度、相机、曝光、外观优化或附加正则 |
| 初始点 | 仅训练 RGB 上的固定相机 COLMAP 三角化，373 点 |

**校准来源限制：** 输入相机来自公开 LLFF/NeRF 冻结标定，可能由完整视角 SfM 得到。
严格隔离的是本次训练 RGB、初始化点、归一化统计与测试 RGB/几何；不是端到端只用 3 张图估计相机。
不把这一结果解释成“3-view 相机恢复”或与未知相机流程等价。

## 隔离与初始化证据

数据来自 [NeRF 官方示例下载入口](https://github.com/bmild/nerf/blob/master/download_example_data.sh)
指向的 `nerf_example_data.zip`（370,385,516 bytes）。服务器支持 HTTP Range，实际只取六张
images_8 PNG 和 poses_bounds.npy；压缩成员有效载荷共 2,027,686 bytes（另有约 0.16 MB ZIP 索引和成员头请求）；
每个成员验证 ZIP CRC32 并记录 SHA256。
没有下载官方 points3D.bin、images.bin 特征、database.db、原分辨率 RGB 或其他场景。
只读取 poses_bounds 前 15 个相机字段，未将其由全点云得到的 near/far bounds 用作训练输入。
数据使用条款仍按来源处理，不把图片或原始数据纳入 Git，也不把代码许可证当成数据许可证。

初始化环境单独使用 Python 3.12.3 / PyCOLMAP 4.2.1 / NumPy 2.2.6，避免覆盖 gsplat 的旧 parser fork。
CPU SIFT、穷举匹配、已知相机三角化，允许两视图轨迹；相机位姿和内参保持固定，未做 MVS。
COLMAP 数据库精确包含三张训练图，所有点轨迹的 image_id 均属于该集合。
平均训练重投影误差约 0.378 px，只说明初始化拟合，不是独立几何准确率。
归一化中心与尺度仅由三台训练相机中心确定，并将冻结变换应用到测试相机。

训练入口只接收 `prepared/train`，该目录仅有三张训练图与训练相机/点。
读文件审计拒绝访问 prepared 下的任何兄弟目录，保存实际访问清单。
训练进程没有测试数据参数；评价通过另一个进程在最终检查点上执行，不做测试反馈调参。

## 实际结果

| 集合 | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| 训练三视图 | 44.7315 | 0.998135 | 0.001904 |
| 测试三视图 | 13.0566 | 0.281176 | 0.542982 |

逐测试图：

| 测试图 | PSNR dB | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| image000.png | 14.2596 | 0.356458 | 0.479268 |
| image008.png | 15.1091 | 0.351868 | 0.480244 |
| image016.png | 9.8010 | 0.135200 | 0.669435 |

正式训练 459.495 秒（约 7 分 40 秒），含模型初始化/最终 checkpoint 写入的记录区间 459.630 秒。
Torch 峰值 allocated 1083.03 MiB / reserved 1618 MiB；
整卡每 200 ms 采样峰值 2591 MiB（2.530 GiB），2297 个样本，含桌面等其他占用。
最终 698,450 个高斯。正式训练成功 1/1、OOM 0；另有一次通过的 1000 步流程检查，耗时 4.769 秒。
独立评价/导出记录区间 3.978 秒（不含 LPIPS 初始化等前置加载）；评价 Torch allocated 峰值 583.01 MiB、整卡采样峰值 1997 MiB。
测试 RGB+ED 渲染平均每图 4.169 ms，逐视图先预热 3 次再计 20 次平均；不含编码、磁盘保存和 LPIPS。

**主要负结果：** 训练/测试 PSNR 差约 31.67 dB，训练高度拟合，测试泛化很差。
image016 最差（9.80 dB），可见黑色缺失区域、重影/拉伸和枝叶结构破坏；
image000、image008 也有明显背景与枝叶伪影。测试 alpha≥0.5 的像素比例分别约为 96.74%、97.63%、87.34%，
该比例只是渲染覆盖诊断，不能当作准确率。深度也呈碎裂/不稳定外观，但没有密集真值，不能量化几何退化。
不能只展示训练视图的好结果，更不能把这一弱基线上的后续收益直接归因为新模块。


PSNR 按每图 RGB MSE 计算 dB，再做图像算术均值；已用 NumPy float64 独立复算交叉核对。
SSIM 使用 torchmetrics 1.9.0：Gaussian kernel、sigma=1.5、kernel_size=11、k1=.01、k2=.03、data_range=1。
LPIPS 使用 AlexNet、normalize=True，每图重置；不混用 VGG 版本。
评价输入是 sRGB [0,1]，预测 clamp；没有另加遮挡掩码或裁剪。
训练使用的 fused SSIM valid 与评价 SSIM 实现明确区分。
AlexNet 权重实际下载 244,408,911 bytes，SHA256：
`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`；
LPIPS 标定权重来自已安装 torchmetrics 包，哈希记入 metrics.json。所有评价权重不参与训练。

仅一个场景、一个正式训练 seed；没有跨场景、跨种子性能结论，没有改进方法可比较。
不可把本表作为算法性能提升、几何准确率或项目最终成功的证据。

## 可视化与证据位置

所有完整产物位于项目外 `~/.local/share/geosplat-r/phase1/`：

- `fern-raw/download.json`：官方来源、成员 CRC/大小/SHA、训练/测试/未用索引。
- `fern-prepared/`：物理分离的 train/test、373 点初始云、仅三图的 COLMAP 数据库及三角化模型。
- `pilot-1000/`：相同 30k 学习率/增密调度下的 1000 步流程检查，不是独立质量基线。
- `fern-3view-seed42-30k/`：全部 30k 更新后的检查点、冻结配置、训练曲线、读文件审计及显存采样。
- `evaluation/`：逐图指标、RGB/GT/Error/Depth 对照、原始 Z/alpha/有效性图、完整 Gaussian PLY、
  高斯中心/相机三维图和训练相机间新视角轨迹视频。

Depth 是 alpha 归一化的期望相机轴向 Z，单位为归一化场景尺度；它不等于欧氏距离或表面深度真值。
黑色深度像素表示 alpha<0.5 或深度无效，颜色按每图有效深度的 2/98 分位显示，不用于指标。
Error map 是通道平均绝对 RGB 误差，所有视图统一显示 0–0.25，超出截断。
三维散点图只显示不超过 25k 个实际高斯中心，完整几何/尺度/旋转/opacity/SH 保留在 PLY 和 checkpoint；
散点图坐标范围使用展示点 1/99 分位并纳入相机位置，可能裁去外围离群点，完整模型不裁剪。
轨迹视频使用完整模型和三台训练相机的旋转/平移插值，无额外测试监督。

完整交互高斯查看可使用已安装的官方 `simple_viewer.py`（手动启动后用浏览器访问本机端口）：

```bash
~/.venvs/geosplat-r/bin/python ~/.local/share/geosplat-r/upstream/gsplat/examples/simple_viewer.py \
  --ckpt ~/.local/share/geosplat-r/phase1/fern-3view-seed42-30k/checkpoint.pt --port 8080
```

## 复现命令

仓库根目录执行；所有 `--output` 必须是新目录。先按 Phase 0 建立固定训练环境，另按
`requirements-colmap.txt` 建立初始化环境；固定源代码 SHA 和权重哈希，不自动升级。

```bash
~/.venvs/geosplat-r/bin/python scripts/fetch_fern.py \
  --url https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip \
  --archive-bytes 370385516 --output /absolute/new/fern-raw

~/.venvs/geosplat-r-colmap/bin/python scripts/prepare_fern.py \
  --source /absolute/new/fern-raw --output /absolute/new/fern-prepared

~/.venvs/geosplat-r/bin/python scripts/train_baseline.py \
  --data /absolute/new/fern-prepared/train \
  --source ~/.local/share/geosplat-r/upstream/gsplat \
  --output /absolute/new/train-run --max-steps 30000 --seed 42

~/.venvs/geosplat-r/bin/python scripts/evaluate_baseline.py \
  --checkpoint /absolute/new/train-run/checkpoint.pt \
  --train-data /absolute/new/fern-prepared/train --test-data /absolute/new/fern-prepared/test \
  --weights ~/.local/share/geosplat-r/weights --output /absolute/new/evaluation
```

复跑会拒绝覆盖既有目录；若部分失败，保留该目录和日志，以另一个新目录重试。
Gaussian 初始化与增密含 CUDA 数值敏感性；固定种子不保证逐位复现，没有用一次运行量化其波动。

## 验收与停止

- 六项输入/协议/相机/实际训练读文件测试全部通过，最终 **0 skip**；前置检查曾因正式训练尚未完成跳过一项，最终已补齐。
- 四个入口均验证已有输出目录拒绝执行；训练配置未被覆盖。
- PSNR float64 独立复算、SSIM/LPIPS 恒等图检查通过；两套环境依赖一致性通过。
- 六张 RGB/GT/Error/Depth 对照及三维图已实际打开检查；完整 PLY 顶点数与 698,450 个高斯一致。
- 整个训练结束后才启动评价；训练审计精确读到 cameras.json、points.npz 和三张训练 RGB，没有测试输入。
- 本阶段没有 OOM 或训练失败；1k 检查与正式运行初始参数哈希相同，但 1k 增密点数不同，提示数值敏感性。
  这没有用来挑选结果，两个运行都保留；不能据此估计完整训练的种子方差。
- 新视角轨迹视频已解码核验 48 帧、504×378，未做人工视觉逐帧验收；官方交互 viewer 提供复用命令，本次未启动服务。

工程运行、数据隔离和产物验收通过；**质量上是严重过拟合的初始基线，尚未证明与公开基线表现等价**。
当前没有强参考方法或多种子复跑，不直接进入复杂算法优化。


累计贡献：独立环境和框架固定；公开相机条件下的 3-view 训练数据隔离；仅训练视角初始化；
Vanilla 训练/独立评价与真实产物、资源记录。还没有深度或几何可靠性算法贡献。
下一项最重要的问题是先审查低测试分数的来源：区分稀疏视角外推/初始化覆盖不足与
潜在基线实现或协议问题，并以冻结协议核验。不能根据本次测试结果随意挑换视角或超参。
得到可信对照后再考虑统一深度正则。上述后续工作需用户另行授权，当前全部停止。

值得提交：精简源码、配置、数学/隔离检查、正式指标与本报告。
提交前仍需展示完整 diff、文件清单和实际测试，获得确认；不提交权重、原始数据、
完整 checkpoint/PLY/视频、显存 CSV、stdout/stderr 或机器环境快照。


## 后续诊断：相机模型与上游训练核心

本小步获授权后，在独立目录 `~/.local/share/geosplat-r/phase1-diagnosis/` 进行诊断，
旧训练、输入和评测产物保持原样。未更换视角、分辨率、训练步数或加入正则。

相机审计从同一官方 ZIP 额外按范围读取 images.bin（压缩 1,887,022 bytes）与 cameras.bin（43 bytes）。
images.bin 含全视角观测，解析器只提取相机外参、名称并跳过所有特征/点关联；不保存源二进制、
不下载 points3D、不将这些观测用于三角化或训练。正式基线当时未下载这些成员的记录仍成立，
它们仅在本次训练结束后的诊断中出现。

[LLFF 官方转换代码](https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py)
给出的排序和坐标约定，与官方二进制及此前转换一致：20 个外参矩阵最大差约 `8.88e-16`，
六个缩放后针孔内参也逐项核对。没有发现视图顺序、坐标轴或归一化尺度的错配。

**此前遗漏的限制：** 原相机模型是 SIMPLE_RADIAL，k=`0.0197504638`，并非原生无畸变针孔相机。
旧流程取 poses_bounds 中的焦距并采用针孔模型，没有去畸变；原始数值仍保留，并明确标记这一近似。
504×378 全图名义径向位移的中位数约 0.628 px、95 分位约 2.325 px、最大约 3.716 px。
这是相机模型计算出的位移，不是观测匹配误差；尚无消融证明它解释了多少 PSNR 损失。

373 个初始点投影的 16×12 网格占用率在训练图约 57.8%，测试图约 51.6%–58.9%；
投影点凸包约占图像 77%–81%，但凸包并不代表内部深度有密集支持。
image016 到最近训练相机的归一化距离约 2.184，另两个测试图约 0.771、0.698；
它的最近训练前向轴角差约 8.53°，另两个约 3.07°、2.57°。
这些统计提示稀疏支持和更困难的视角外推，但不单独确立低质量的因果机制。

| 训练路径 | 训练 PSNR | 测试 PSNR | 测试 SSIM | 测试 LPIPS-Alex |
|---|---:|---:|---:|---:|
| 原适配循环 | 44.7315 | 13.0566 | 0.281176 | 0.542982 |
| 上游原生循环 | 44.8217 | 13.5019 | 0.301721 | 0.513005 |

两条路径均重现严重过拟合。对照测试均值比原运行高约 0.445 dB，但采样顺序/数值轨迹不同，
不能解释为算法或实现优化收益。image016 仍最差，上游对照为 10.181 dB。
对照完成 30k 更新，初始参数哈希与原运行完全相同，最终 750,590 个高斯；
Torch allocated 峰值 1168.29 MiB / reserved 1794 MiB，
整卡采样峰值 2923 MiB（2.854 GiB），OOM 0。

**耗时口径补正：** 本次发现 perf_counter 与墙钟/显卡时间戳记录不一致，原因未验证。
原运行先前报告的 459.50 秒（7 分 40 秒）是 perf_counter 训练循环计数；
其显卡采样时间戳跨度实际约 499.419 秒（8 分 19 秒，含初始化/保存边界）。
上游对照 perf_counter 为 524.615 秒，原生 time.time 训练记录为 571.753 秒，
显卡采样区间为 571.932 秒（约 9 分 32 秒）。因此原先“实际训练耗时 7 分 40 秒”
不足以代表真实墙钟耗时，本补充更正其含义，原始日志和 JSON 数值不改写。
后续同时记录多种时钟，不能用本次记录作细微速度提升结论。


对照通过自定义训练数据适配器直接调用未修改的 `Runner.train` 和 `Runner.rasterize_splats`。
没有调用上游默认 Parser，测试数据和 LPIPS 在训练中不加载；评价仍使用同一个独立脚本。
因此该对照核查的是上游**训练核心**，不能证明共享的数据预处理正确，也不能替代对去畸变的验证。
原实现与对照使用同一初始参数，但图像抽样器不同：原实现使用独立 seed42 发生器，
上游使用默认 DataLoader 的全局随机状态。单次指标差异混合了采样顺序和 CUDA 数值敏感性，
不能直接归因为代码优劣或作为重复种子方差。

为保持参照，上游循环的数据加载方式保持原样。运行时按可用显存设置上限，保留 1 GiB 工作余量，
不增加批大小或改精度来改变实验协议。耗时包括相应的数据加载/日志开销，不是纯 CUDA 内核速度。

本次审计初次因未预期 SIMPLE_RADIAL 模型而停止；补齐模型诊断后完成。
上游包装首次因审计钩子未解码 bytes 路径而启动失败，第二次因手动构造 Runner 缺少空优化器列表
在第一次更新后失败；已分别修正，日志/目录完整保留。没有更改上游源代码，没有覆盖失败产物。
这些失败属于诊断包装，不计作一条完成的 30k 基线；仍计入本步尝试记录。

复跑诊断：

```bash
~/.venvs/geosplat-r/bin/python scripts/audit_fern.py   --url https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip   --archive-bytes 370385516 --raw /absolute/fern-raw --data /absolute/fern-prepared   --output /absolute/new/camera-audit

~/.venvs/geosplat-r/bin/python scripts/run_upstream_reference.py   --data /absolute/fern-prepared/train --source /absolute/pinned/gsplat   --output /absolute/new/upstream-reference
```

十项检查全部通过、0 skip：含独立二进制相机解析、官方内外参核对、冻结输入哈希、
两次训练初值一致、实际训练读文件清单及资源检查。新入口的输出覆盖保护通过。
前置检查曾在上游尚未完成时跳过一项，最终已补齐。上游测试对照图已实际打开检查，仍有严重伪影。

**验收建议：诊断子步骤通过；正式基线有条件通过，仅限明确标记的针孔近似协议。**
没有发现适配循环的明显训练核心错误；上游核心重跑也有相同质量问题。
相机畸变此前未覆盖，不能据此直接验证几何可靠性约束，因此**不建议进入 Phase 2，也暂不建议提交/推送**。
下一步建议只做一个因素：固定同样六个视角，明确去畸变与内参更新协议，重新构建训练三角化和基线。
重采样测试 GT 属于新数据协议，不能把它与旧数字的差值直接称为算法提升。
该验证尚未执行，不擅自改变已冻结输入，也不引入其他场景。

新增可审阅内容为两个诊断入口、回归检查、精简诊断指标与本报告补充；原始日志、模型和源证据副本
保留在项目外，旧 run 的训练源码已按原哈希额外存入 `phase1-diagnosis/source-evidence/` 供后续恢复。
累计成果增加了独立相机核查与上游训练核心参照；仍然没有算法性能贡献。



## 2026-10-19：去畸变协议与基线复核

用户授权后仅处理同样的 fern 三个训练视角与三个测试视角，继续 seed42、504×378、30k 更新。
本小步没有新增数据下载、模型权重或其他场景；所有旧输入、检查点和结果保留。
源相机 SIMPLE_RADIAL 参数由上一小步的官方记录审计取得。

[OpenCV 相机模型与去畸变接口](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html)
用于构建新针孔相机：alpha=0、主点居中，再施加预先固定的 1.001 焦距保护系数，
以保证全部目标像素具有源图内的有效双线性采样。该选择仅依赖标定和图像尺寸，没有读取测试图内容来选择参数。
新 K 为 fx=fy=413.757484793、cx=251.5、cy=188.5；输出仍为 504×378，视场因焦距变化略收窄。
没有额外裁切图像数组，没有补黑边、评价掩码或依据表现选择区域。

全图映射与独立径向投影公式最大差 `1.526e-5` px，稀疏像素逆投影往返检验通过。
源采样坐标范围为 x≈[1.440,502.560]、y≈[1.205,376.795]，所有像素有完整双线性支持。
训练/测试采用同一个仅由标定得到的 K 和映射；测试 RGB 仅做固定重采样，未用于训练初始化、损失或调参。

原始三图只读，经独立目录重采样后仅训练图重新进行 SIFT/匹配/固定相机三角化，得到 488 点。
相机内外参未优化，归一化仍仅由三台训练相机中心确定。
训练重投影均值约 0.332 px；点数增加和拟合残差下降都不能单独证明独立几何准确率提高。

新协议明确命名为 `fern-undistorted-v1`，冻结配置见 `configs/fern-3view-undistorted.json`。
本步同时更新 RGB 重采样、相机 K 和相应初始化点，测试 GT 也改变。
因此它是投影一致性维护与基线复核，不是保持 GT 不变的畸变因果消融，不能把与旧分数的差值报成算法收益。

资源维护限于安全的运行方式：三张 float32 训练 RGB 缓存在 GPU，顺序/批大小/损失/精度/更新预算不变；
显存上限由可用量扣除 1 GiB 工作余量确定。新增 mean/peak GPU 利用率与 time.time、perf_counter、
CUDA stream event、显卡时间戳四种计时。CUDA event 是流的起止时间区间，不是内核耗时之和；
时钟不一致时保留各口径，不按需挑选最快计时。
老训练源码在修改前已保存于本小步项目外 source-evidence，旧模型和日志不改写。

| 集合（新协议） | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| 训练三视图 | 45.4400 | 0.998310 | 0.001151 |
| 测试三视图 | 15.0197 | 0.378023 | 0.449066 |

| 测试图 | PSNR dB | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| image000.png | 16.9229 | 0.469715 | 0.389176 |
| image008.png | 17.2754 | 0.460207 | 0.365849 |
| image016.png | 10.8607 | 0.204149 | 0.592174 |

正式训练成功 1/1，另有 1000 步流程检查 1/1；本小步无训练失败、无 OOM。最终 441,448 个高斯。
训练 time.time 墙钟 347.185 秒（约 5 分 47 秒），
显卡采样区间 347.224 秒，CUDA stream event 346.494 秒；
perf_counter 仅 317.608 秒，时钟差异继续如实保留，原因尚未验证。
Torch allocated 峰值 699.89 MiB / reserved 1110 MiB，
整卡峰值 2062 MiB（2.014 GiB）；
GPU 全区间采样平均利用率 91.23%、峰值 96%。
测试 RGB+ED 平均渲染 2.918 ms/图（perf_counter；3次预热、20次计时，
不含图像编码/指标计算）。不把点数、输入协议与缓存方式均有差异的旧/新运行用作算法速度对比。

主要负结果仍是训练/测试巨大差距、细枝叶与背景伪影。image016 仍最差，为 10.861 dB，
可见明显拉伸、缺失和重影。已实际检查全部测试对照图；没有独立密集深度真值，深度色图仅用于诊断。
旧模型/输入原样保留，新表不与旧协议作收益排名。


完整产物位于 `~/.local/share/geosplat-r/phase1-undistorted/`：
`fern-source/` 保存派生输入及 protocol/来源哈希；`fern-prepared/` 保存隔离的 train/test 和训练点云；
`pilot-1000/` 为相同30k调度下的流程检查；`train-30k/` 保存正式模型/源码/配置/资源与访问日志；
`evaluation/` 保存逐视图评价、完整 PLY、三维图、轨迹视频、RGB/GT/Error/Depth。

复现前两步如下，之后使用前文训练和评价命令，将数据/输出路径换成对应的新目录：

```bash
~/.venvs/geosplat-r/bin/python scripts/undistort_fern.py   --source /absolute/fern-raw --calibration /absolute/camera-audit/camera-and-coverage.json   --output /absolute/new/fern-source

~/.venvs/geosplat-r-colmap/bin/python scripts/prepare_fern.py   --source /absolute/new/fern-source --intrinsics /absolute/new/fern-source/intrinsics.json   --output /absolute/new/fern-prepared
```

旧输入的复现入口仍可使用不带 --intrinsics 的 prepare_fern；旧训练代码与其哈希在项目外历史源码副本中保留。
原始运行的源码哈希检查使用显式归档源码位置，不错误地要求它匹配之后维护过的当前源码。

**本小步建议通过。** 去畸变映射、内参、固定视角、训练三角化、归一化和独立评价形成了一致且可审计的针孔协议。
16 项检查全部通过、0 skip，包含旧协议/原始源码恢复检查；前置检查跳过的正式训练项现已补齐。
新去畸变入口覆盖保护通过，旧入口继续拒绝已有输出目录；所有数据和模型保留在仓库外。
完整 PLY、六图对照、Z/alpha/有效性图、三维场景图和已解码核验的 48 帧视频已生成。

通过的是单场景基线工程/协议验收，不是渲染质量或研究成果成功。仍只有 fern 和 seed42，
fern 已作为开发场景反复检查，不能承担最终未见场景泛化证明。没有引入深度或可靠性模块。
相机一致性修正没有消除严重过拟合，不能把原低分全部归因于镜头畸变。

后续最重要的研究问题：在这个冻结协议上建立最小统一深度正则对照，核实深度先验是否有用；
该 Phase 2 仍需用户授权。本次停止，不自动扩大到多场景/多种子或复杂模块。

提交/推送建议：本步代码、关键测试、冻结配置和可信负结果值得组成首次工程提交候选。
建议下一步整理完整 Diff、拟提交文件与复现入口，审阅通过并获得确认后再提交/推送；当前暂不推送。
既有阶段材料若需压缩或归档，须先列出精确路径、依赖、恢复路线并获批准，不在本步擅自删除或精简。

累计贡献增加了标定驱动的去畸变协议、训练视角点云重建、输入隔离与多时钟/显存/利用率记录。
这些属于工程可复现性与正确性维护，没有可宣称的独立算法性能提升。
