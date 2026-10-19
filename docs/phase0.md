# GeoSplat-R Phase 0：选型、边界与下一阶段协议

核查日期：2026-10-18。仅授权 Phase 0；以下 Phase 1–4 内容是提案，尚未执行。

## 1. 决策

主框架选 **gsplat v1.5.3 + 官方 simple_trainer.py 的最小适配路线**，固定源码
`937e29912570c372bed6747a5c9bf85fed877bae`（2025-07-04）。不追随 main 自动升级。
选择较早稳定标签是为了采用官方提供的 Python/Torch/CUDA 匹配预编译包；不是声称它是最新版本。

| 候选 | 已核对事实 | 决策 |
|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat/tree/937e29912570c372bed6747a5c9bf85fed877bae) | 独立 CUDA/PyTorch 光栅器；RGB 与可微期望深度；示例有训练、增密、评测入口 | 主选，便于局部改造 |
| [官方 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) | 原始参考实现；现已提供深度正则；README 的论文质量资源要求为 24 GB，不代表 12 GB 无法运行 | 用于行为/来源参考，不作为本机主开发框架 |
| [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) | 基于 gsplat；文档给默认约 6 GB、big 约 12 GB；完整数据管理与查看框架 | 可用备选，本任务暂不需要其额外应用层 |

[gsplat 固定版许可证](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/LICENSE)
和当前 main 根许可证均为 Apache-2.0；Nerfstudio 根许可证也是 Apache-2.0。
官方 3DGS 是自定义研究/非商业许可证，不能据公开源码推定可任意商用。
这里只记录第三方条款，不为 GeoSplat-R 选择许可证。
后续复制训练器须保留归属、适用许可证并标注改动；没有把第三方源码复制进本仓库。
预编译包包含的第三方组件归属也须保留；主许可证不能覆盖所有独立依赖条款。

## 2. 环境与运行验证

独立工程目录：`~/projects/GeoSplat-R`，分支 `codex/phase0`，无提交、无远端。
独立环境：`~/.venvs/geosplat-r`，不继承已有环境包。
源码、安装包、原始日志及运行结果放在 `~/.local/share/geosplat-r/`，不纳入 Git。
工具引导环境 `~/.venvs/geosplat-r-bootstrap` 仅安装 uv。

本机实际可见 WSL 内存约 15 GiB、swap 4 GiB，磁盘预检可用约 933 GiB。
GPU 为 RTX 4070 SUPER，12282 MiB；预检整卡占用约 975 MiB。
主机驱动显示 615.78.05 / KMD 617.42、最高 CUDA 驱动接口 13.4。
**驱动接口版本不等于本地 CUDA Toolkit 或 PyTorch runtime 版本。** 初始没有 nvcc。

安装组合：Python 3.10.22、PyTorch 2.4.1+cu124、torchvision 0.19.1+cu124、
gsplat 1.5.3+pt24cu124、NumPy 1.26.4。使用 float32；没有未经验证启用 AMP。
[PyTorch 官方旧版本表](https://pytorch.org/get-started/previous-versions/)
提供此 Torch/torchvision/CUDA 组合。

官方 wheel：
<https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl>

- 大小：21,915,762 bytes。
- 本地计算 SHA256：`01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd`。
- 这是下载包校验记录，不等于从源码重建验证。

实际结果：batch=1、float32、SH3、packed、3 步预热 + 20 步测量。

| 高斯数 | 分辨率 | 平均毫秒/步 | Torch 峰值 allocated / reserved MiB | 整卡前→后 MiB | 状态 |
|---:|---|---:|---:|---:|---|
| 4,096 | 320×240 | 2.76 | 19.84 / 32 | 918 → 1014 | 通过 |
| 100,000 | 1008×756 | 5.13 | 189.03 / 228 | 890 → 1190 | 通过 |
| 500,000 | 1008×756 | 19.67 | 691.06 / 828 | 888 → 1790 | 通过 |

单点期望深度最大误差 `4.1723e-6`；Z 梯度 `0.9999997`，有限差分 `0.9999871`。
所有参数梯度有限，三组无 OOM。每次检查进程内用时约 1.8–3.6 秒（不含 Python 启动）。
没有发生 CUDA 检查失败；依赖下载发生网络超时重试，最终成功，日志完整保留。
Torch 下载准备 5 分 40 秒，其他依赖准备约 34 秒，fused-ssim 本地构建准备约 27 秒。
主虚拟环境磁盘占用约 5.8 GB，项目外源码/组件/证据约 395 MB，另有 uv 下载缓存。

额外通过：108 个包的依赖一致性；未改动的官方 `simple_trainer.py default --help`；
fused-ssim 恒等图 SSIM=1 与不同图像的有限非零反向梯度；输出覆盖拒绝检查。
fused-ssim 使用官方示例要求的 SHA `328dc9836f513d00c4b5bc38fe30478b4435cbb5`（MIT），
本地编译仅针对 sm_89、MAX_JOBS=2。pycolmap 与 nerfview 使用上游 requirements 指定的 SHA。
未安装可选 fused-bilagrid、压缩依赖，因为本基线关闭这些功能。

完整训练、数据读取到评价、LPIPS 权重加载、真实场景增密峰值和三维查看尚未验证（无数据/权重，
按 Phase 0 范围跳过）。当前通过的是选型和最小运行门槛，可申请进入 Phase 1；不是完整基线通过。


核心检查使用合成高斯，不下载数据或模型：解析单点相机 Z 深度、深度反向导数及有限差分，
RGB+ED、SH3、所有高斯参数的有限梯度，以及 Adam 更新。输出目录必须不存在。
性能计时包含同步、参数梯度检查和优化器开销，不是纯渲染 FPS。
显存区分 Torch allocated/reserved 与 nvidia-smi 整卡采样；前后整卡读数不是瞬时峰值。
合成负载没有真实图像加载、SSIM、LPIPS、增密或可见性分布，不能外推整场训练用时或峰值。

复跑示例（必须使用新输出目录）：

```bash
~/.venvs/geosplat-r/bin/python scripts/check_cuda.py \
  --output /absolute/path/to/new-result --gaussians 100000 \
  --width 1008 --height 756 --steps 20
```

完整训练器所需 parser、fused-ssim、torchmetrics、viewer 等基础依赖已安装且入口可加载；
Phase 0 的核心光栅检查不能替代真实场景基线验收。

## 3. 已有方法与贡献边界

| 直接相关工作 | 与候选方向的关系 |
|---|---|
| [DepthRegGS](https://github.com/robot0321/DepthRegularizedGS) | 已有 few-shot 深度正则方法；接入深度模型不构成独立贡献 |
| [DNGaussian](https://github.com/Fictionarry/DNGaussian) | 已有全局/局部深度归一化及稀疏视角几何约束；尺度处理不能直接称新方法 |
| [FSGS](https://github.com/VITA-Group/FSGS) | few-shot 强参考；源码读取 n_views 对应的 MVS dense/fused.ply，初始化成本和输入必须对齐 |
| [CoR-GS](https://github.com/jiaw-z/CoR-GS) | 利用两个场的点/渲染分歧识别不可靠重建；置信度思路已有先例 |
| [CDGS](https://arxiv.org/abs/2502.14684) | 直接提出置信度感知深度正则，结合单目与 SfM 多线索置信度；与暂定框架明显重叠 |
| [Confidence matters](https://arxiv.org/abs/2608.06117) | 2026 年预印本讨论多视图几何先验与置信度加权；不是本项目实测证据 |

以上是选型所需的定向核查，未声称完成系统综述或证明新颖性。
DNGaussian/CoR-GS 官方环境较旧，外部方法应使用独立环境，不能把其依赖覆盖到主环境。
当前没有复现这些方法，也不能将其论文数字与本项目不同设置直接做排名。
未来外部参考优先从 DNGaussian/FSGS 中选一个做匹配协议复跑；执行前固定源码 SHA 和各组件许可证。

源码还确认：gsplat 此版本的 `depth_loss` 是从 COLMAP 点投影出的**稀疏 Z 深度逆数 L1**，
不是现成的 Depth Anything 稠密先验基线；`test_every=8` 也不是 3/6 视角输入筛选器。
默认 Parser 读取整个点云，归一化还使用整个相机/点集合；后续必须先隔离训练资产再计算训练归一化。
不允许直接运行示例就宣称已经完成无泄漏稀疏视角基线。

## 4. 公平评测协议提案（尚未实施）

- 场景开发集固定 fern、flower、leaves；最终测试集固定 fortress、horns、orchids、room、trex。
  Phase 1 先只用 fern 做可运行性与资源验收；最终测试集在方法冻结前不跑渲染指标、不调参。
- LLFF 图像按文件名稳定排序，零基索引 `i % 8 == 0` 为干净评测视角。
  从剩余索引中按 `rint(linspace(0, M-1, k))` 选 k=3/6 个训练视角；M 不足则明确失败。
  下载后生成不可变文件名清单与内容哈希，不按效果换组合。
- 先采用 LLFF images_8（通常约 504×378，最终以实际文件尺寸为准），不裁图、不补图。
  上面的 1008×756 合成预检是额外资源负载，不能冒充相同分辨率真实训练。
  所有组使用同一图像、相机、初始化、分辨率、30k 更新、batch=1、SH3、精度及增密策略。
  先采用 DefaultStrategy、classic 光栅、packed=True、普通 Adam，关闭相机/外观/曝光/抗锯齿增强。
- 校准相机协议：允许使用公开冻结相机参数，明确它们可能来自全视角 SfM，因此不声称纯 3-view
  端到端相机恢复。测试相机只用于渲染。所有组禁止用测试 RGB/深度/特征/三维点监督训练。
- 初始化只从所选训练图像提取/匹配特征，在固定训练相机下三角化；不能直接复用全视角 points3D。
  归一化只由训练相机和训练点确定，再把同一变换应用到测试相机。
  若 3-view 三角化失败，应保留失败；随机初始化须另立统一对照配置，不能暗中替换。
- 度量 RGB 为 sRGB [0,1]，渲染统一 clamp，无遮挡掩码和额外裁边；逐图 PSNR（MSE）、SSIM、
  LPIPS-Alex，再先平均到场景。实现/版本、SSIM 参数与 LPIPS 权重校验必须在 Phase 1 固定。
  默认 torchmetrics SSIM 配置需在落地时完整保存。外部方法若采用 VGG 或不同 SSIM，重新统一评测，
  或在独立表中明确差异。训练集指标与测试集指标分开。
- 固定 seeds=42,43,44；先 seed42 检查流程，正式结论三种子配对。记录逐场景差、均值、中位数、
  恶化场景、种子波动、成功数/总数及失败原因；失败不当作缺失样本悄悄删除。
- 后续对照依次为 Vanilla、统一深度、统一鲁棒深度、几何可靠性+相同鲁棒深度。
  深度模型、对齐算法、有效像素、λ 和调度对齐；可靠性权重的归一化/有效监督总量必须可审计，
  防止仅因损失强度下降产生伪贡献。先验离线生成成本单独计入。
- 相对深度/逆深度的含义先通过模型与小样本核对，只在匹配的表示空间做尺度/偏移对齐；
  训练视角 SfM 锚点不足则报告失败，不能用测试真值补齐。低 alpha、非正深度、越界、遮挡
  及边界显式掩码。可微 ED 是相机轴向 Z 的 alpha 归一化混合，不是距离或真实表面深度。
- LLFF 没有可直接当可靠深度真值的密集参考；低拟合残差、重投影一致性或新增匹配不证明几何变好。
  Phase 3 若要评估权重识错能力，需要另获授权的小规模可靠几何参考；训练参考与独立评价分离。
- GPU 过程目标保留约 2 GB 余量，记录整卡采样峰值及 Torch allocated/reserved；30k 全过程尤其增密峰值
  才是资源验收。发生 OOM 保留失败，任何分辨率/点数/更新步数调整都新建配置并重跑对应控制组。
- 退化实验属于后续阶段。届时冻结训练图像上的模糊/曝光/视角分布协议，测试图像保持干净。
  PSNR“退化幅度下降 15%”需先定义计算式，避免把 dB 比值当线性误差比例。

## 5. 数据、权重与 Phase 1 计划

**本次没有下载任何实验数据或模型权重。**

| 阶段 | 需要的资产 | 来源与下载边界 |
|---|---|---|
| Phase 1 | LLFF fern RGB 与冻结相机；随后开发集另两个场景 | [NeRF 官方数据入口](https://github.com/bmild/nerf#running-code)；优先只取 fern。归档精确体积与独立数据使用条款尚未核验，不能用代码许可证代替数据条款；大包下载前列出精确包/体积并获批准。不打包转发数据 |
| Phase 1 | LPIPS-Alex 的预训练 AlexNet 与 LPIPS 标定权重 | 首次完整评测前显式下载并固定文件 hash/出处；当前未加载模型权重，精确总大小未核验 |
| Phase 2 | Depth Anything V2 Small，相对深度权重 | [官方模型](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/blob/main/depth_anything_v2_vits.pth)，约 99.2 MB；Small 为 Apache-2.0，Base/Large/Giant 为 CC-BY-NC-4.0；不因更大模型潜在质量而擅自升级 |
| 后续几何验证 | 获批准的 DTU 小子集、相机、评测掩码及可靠几何参考 | 暂不下载 DTU 123 GB 全集；小子集可获取性、体积和条款需另核查 |

Phase 1 获授权后：先验证真实场景加载与完整评测依赖；准备单场景与不可变划分；
实现最小输入适配和只使用训练视角的初始化；用短跑只验流程（不作为性能结论），
再固定配置进行 fern 3-view/seed42 的 30k 基线。不能使用单场景结果宣布项目优化成功。

必须交付训练/测试 PSNR、SSIM、LPIPS、RGB/GT/Error 并排、带有效性信息的深度渲染、
真实高斯/相机三维查看产物、峰值显存、训练及渲染时间、配置与版本；正式场景资源尚未实测，
不报预计分钟数作为实际结果。通过后再按授权扩展开发场景/6-view，不自动进入 Phase 2。

阶段累计贡献目前仅为：独立环境、可核查选型、核心运行检查和数据隔离方案。
尚无算法改进、真实 NVS 指标、性能增益或可靠性模块；无收益可与 Baseline 比较。
值得后续提交的是本报告、README、环境版本、运行检查及工作边界；原始日志、包、源码镜像与结果不提交。


## 6. 环境复建说明

主依赖及三个源码 SHA 见 `requirements-phase0.txt`；完整本机依赖锁仅保留在项目外的
`~/.local/share/geosplat-r/phase0/environment-full-local.txt`，不把它作为机器环境快照提交。
主依赖列表不等于经过跨机器验证的完整锁；当前仅验证本机这一组合。

CUDA 编译组件使用 [NVIDIA 12.4.1 官方 redistrib 清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.1.json)
中的 cuda_nvcc 12.4.131、cuda_cudart 12.4.127、cuda_cccl 12.4.127（合计归档 53,441,344 bytes），
每个归档均对照清单 SHA256 验证。只合并解压到项目专属 `~/.local/share/geosplat-r/cuda-12.4.1`，
没有修改系统 CUDA 或 shell 启动文件；组件的 NVIDIA 条款单独保留在该目录。

复建顺序：新建 Python 3.10 环境；从官方 cu124 索引先装 Torch/torchvision；
安装 setuptools、wheel、ninja；准备上述 CUDA Toolkit 组件；设置该次安装的 CUDA_HOME、
MAX_JOBS=2 和 TORCH_CUDA_ARCH_LIST=8.9，再用 `--no-build-isolation` 安装主依赖文件。
不要把 PyPI 的通用 gsplat 包替换进来，否则可能转入未验证的即时源码编译路径。
后续如换 Torch、CUDA、gsplat 或 GPU，必须重新验证核心渲染、梯度及 SSIM。
