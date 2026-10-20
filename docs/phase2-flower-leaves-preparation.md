# Phase 2：flower/leaves 最小输入适配与数据准备验收

本页保留训练前的原始验收结论。后续已获明确授权并完成正式矩阵，见[18模型训练与评价验收](phase2-flower-leaves-training.md)。

执行日期：2026-10-19（Asia/Shanghai）。本步仅授权清单下载、最小输入适配和数据准备验收。
**工程/数据准备验收通过，建议批准冻结的18次正式训练；当前尚未训练。**
保持[原方案](phase2-flower-leaves-plan.md)及[冻结配置](../configs/flower-leaves-phase2-plan.json)，
没有更换视图、系数、调度、相机来源、分辨率、初始化方法或深度模型。

## 实际获取与标定

21个批准成员全部获取成功，压缩载荷精确11,049,045 bytes：flower5,928,929，leaves5,120,116；
另有21个成员头共630 bytes，不含HTTP/TLS头与重定向。仅一次完整下载流程，未发生下载重试/失败，
未取369,069,196-byte完整包、未用RGB、源数据库/points3D或额外权重。
严格核对206、Content-Range、本地成员头、大小和CRC32，并计算每个解压成员的SHA256。
原始images.bin只在内存跳过特征观测并提取公开相机映射，不落盘保存，不将观测或完整点云作训练输入。
校验仍基于固定版本镜像，未宣称与官方Drive全包逐字节一致或验证完整包LFS哈希。

| 项目 | flower | leaves |
|---|---|---|
| 训练零基索引 | 1、17、33 | 1、12、25 |
| 独立评价索引 | 0、8、16、24、32 | 0、8、16、24 |
| 原生图像 | 504×378，RGB | 504×378，RGB |
| 原相机 | SIMPLE_RADIAL，4032×3024 | SIMPLE_RADIAL，4032×3024 |
| 原焦距f | 3575.058606 | 3428.475518 |
| 原径向系数k | 0.025134575 | 0.037388355 |
| 去畸变fx=fy | 453.965809 | 438.458468 |
| 去畸变cx,cy | 251.5,188.5 | 251.5,188.5 |
| 解析映射最大误差px | 0.0000152585 | 0.0000152583 |

相机名/排序/poses_bounds变换逐项一致，源尺寸与images_8一致。
保持原来alpha=0、主点居中、1.001几何保护系数和uint8 INTER_LINEAR；无裁剪、补边或额外评价mask，
全图无padding和独立解析径向投影检查通过。两场景使用自己的K/k，没有借用fern标定。
公开相机可能来自全视角SfM，不能声称三张图恢复相机。

## 训练专用几何、锚点与先验

| 项目 | flower | leaves |
|---|---:|---:|
| 训练三角化点 | 1285 | 274 |
| 训练点平均重投影误差px | 0.331109 | 0.183539 |
| 三张图有效锚点数 | 847 / 1158 / 912 | 199 / 235 / 176 |
| 三角化墙钟秒 | 0.435438 | 0.423267 |
| 先验准备墙钟秒 | 1.964063 | 1.290382 |
| 先验Torch allocated/reserved峰值MiB | 283.554 / 366 | 283.554 / 366 |
| 先验整卡采样峰值MiB | 1837 | 1883 |

保持CPU SIFT/exhaustive、固定相机三角化、允许两视图轨迹和训练相机中心归一化。
训练数据库只含3张训练图，每个点的全部观测轨迹均在这3张图内；从保存的三角化重新核对XYZ/RGB多重集。
没有在训练初始化中打开测试图像、测试特征或源完整点云。测试相机只应用训练确定的同一归一化变换。
三张训练图各自达到>=20有效锚点门槛，点数不再错误固定为fern的488。

| 图像 | 对齐a | 对齐b | 锚点相关系数 | 归一化拟合L1 |
|---|---:|---:|---:|---:|
| flower/image001 | 0.028812 | 0.015285 | 0.928396 | 0.290585 |
| flower/image017 | 0.027542 | 0.013603 | 0.923167 | 0.255412 |
| flower/image033 | 0.025236 | 0.016907 | 0.938372 | 0.207807 |
| leaves/image001 | 0.027961 | 0.014722 | 0.979628 | 0.049391 |
| leaves/image012 | 0.032892 | 0.006821 | 0.960426 | 0.097784 |
| leaves/image025 | 0.029802 | 0.009705 | 0.981055 | 0.166751 |

全部a>0，目标有效比例均为0.981565；未重新选择锚点/损失/权重来改善拟合数字。
flower约20.8%–29.1%的锚点拟合残差较fern高，保留为已知局限，不临时增加新验收阈值或改模型/系数。
相关性、拟合误差和较低重投影误差不是独立密集深度/几何准确率。
已实际查看全部六张训练先验：能大致分出前景花朵/叶片与背景，但细叶、遮挡间隙和重叠结构仍有合并、平滑。
没有密集几何真值，不能定量判定这些先验的正确区域，更不能预报训练收益。

## 最小工程适配与兼容性

新增严格Range获取、同规则去畸变、几何来源校验/训练manifest冻结、GPU输入预检四个入口。
新输入由train/input-manifest.json显式声明场景、三张训练图及内容哈希、K/点云哈希、尺寸和点数；
不接受test字段、父目录manifest或未批准视图。训练入口只接收train目录和该训练专用manifest。

已有train_baseline、export_train_anchors、prepare_depth_priors新增可选--scene-manifest；
不传仍执行原fern保护路径。六/九模型汇总支持显式scene_input_manifest，按已批准的4或5张测试图核对，
保留fern默认名字和旧清单格式。跨新场景统计仍先按每个scene/seed求均值，两个场景等权。

训练30k循环、render/load_views与原42e297e的AST一致，depth.py逐字节一致；
只适配输入验证和来源/结果元数据，没有改训练数学、优化器、增密、λ或启用步数。
新增适配确实改变了源文件哈希，后续18次必须使用本步重新冻结的新哈希，不能冒用旧历史哈希。
当前legacy fern入口也在新独立目录实跑锚点/先验准备；三张图的raw_inverse_depth、inverse_z、valid均与历史数组逐元素完全相等。
这仅是入口/先验回归，没有重训fern；新回归输出不覆盖旧先验或旧模型。

两个GPU输入预检各只读cameras.json、points.npz、input-manifest.json和3张训练RGB，
并核对同场景先验哈希；未读测试目录、未渲染、优化更新次数0。
缓存加载Torch allocated峰值各9.272MiB、reserved24MiB；这是输入加载占用，不能当作30k训练峰值。
已发生的先验推理内存/耗时另见上表；真实增密训练峰值和训练成功率仍待正式运行测量。

## 检查与保留失败

最终43项检查全部通过、0跳过；无资产模式28项执行、15项明确跳过。
包括全部原fern/三路径历史检查、Range拒绝全包/错误范围、输入来源/划分保护、
点多重集顺序/重复/颜色、可变测试视图统计、训练数学回归、两个新场景真实来源及无优化更新预检。
10个相关入口的帮助与已有输出拒绝覆盖检查通过。

flower首次input-audit误把COLMAP保存前后的哈希映射迭代顺序当作点的一一对应，检查失败。
修正为精确float32 XYZ加RGB的无序多重集，保留重复计数；重排不算数据差异，坐标/颜色/重数不同仍拒绝。
没有改点坐标、颜色、点数或重跑三角化来制造通过。原失败日志/目录保留，后续审计写入独立input-audit-v2。
本步审计失败1次；下载、三角化和先验生成失败0，OOM0，没有隐去失败。

旧fern原始/去畸变模型、六个深度对照、三个RGB-only模型、数据及原始报告继续保留。
代码旧版本可由42e297e与项目外source-before恢复，所有原数据准备和训练路径不删除、不归档、不压缩。

## 冻结复现与18次训练判断

新入口示例（始终使用不存在的新输出；本页不执行训练）：

```bash
"$TRAIN_PY" scripts/fetch_llff_subset.py --plan configs/flower-leaves-phase2-plan.json --output /absolute/new/raw
"$TRAIN_PY" scripts/undistort_llff_subset.py --plan configs/flower-leaves-phase2-plan.json --scene flower --source /absolute/new/raw/flower --output /absolute/new/flower/source
"$COLMAP_PY" scripts/prepare_fern.py --source /absolute/new/flower/source --intrinsics /absolute/new/flower/source/intrinsics.json --output /absolute/new/flower/prepared
"$COLMAP_PY" scripts/validate_llff_preparation.py --plan configs/flower-leaves-phase2-plan.json --scene flower --data /absolute/new/flower/prepared --source /absolute/new/flower/source --raw /absolute/new/raw/flower --output /absolute/new/flower/audit
"$COLMAP_PY" scripts/export_train_anchors.py --data /absolute/new/flower/prepared/train --scene-manifest /absolute/new/flower/prepared/train/input-manifest.json --output /absolute/new/flower/anchors
```

按[先验入口](phase2-uniform-depth.md)传同一显式source/source-manifest/weights，并额外传--scene-manifest；
check_scene_input另传--data、--scene-manifest、--priors及新--output。leaves流程同形，必须使用自己的场景目录。
完整环境仍见[复现说明](reproduce-fern.md)，无需新增包/权重下载。

本地frozen-execution.json已保存24个实现文件哈希、121个输入/证据文件哈希、18组训练/独立评价命令，
future-depth/rgb清单引用显式路径和新scene_input_manifest，所有未来输出目录仍不存在。
正式矩阵：flower/leaves×seed42/43/44×RGB-only/RGB+ED零权重/统一深度，各30k；
λ=0.01、零基step1000、504×378、batch1、float32和其他协议不变，18次全完成后再评价，
遇失败保留并停止依赖执行，不改设置续跑、不根据中途测试分数挑选或调参。

**建议放行这18次正式训练，但需要用户新的明确训练授权。**
支持放行的是输入、隔离、数学兼容、先验有效性和复现工程门槛通过，不是证明新场景收益。
正式训练仍需每个scene/seed核对三路径初始参数哈希、实测12GB内显存/耗时；发现输入/源码漂移就停止。
不进入可靠性模块，不获取最终保留场景，不宣称花叶场景已取得PSNR/SSIM/LPIPS改善。

精简真实来源和资源见[准备结果JSON](../results/flower-leaves-preparation.json)。
完整外部证据根目录为 `~/.local/share/geosplat-r/phase2-flower-leaves-preparation-20261019T073233Z/`，
含raw/source/prepared、锚点/先验、读文件审计、原失败、新检查、旧入口回归、源码前后副本及冻结命令。

累计贡献是在原fern深度对照上补齐两开发场景的严格输入适配、公开标定核查、训练专用几何/先验和可执行冻结矩阵；
本步正式训练次数0，没有新增渲染质量结果。所有新改动未暂存、未提交、未推送，完成本授权后停止。
