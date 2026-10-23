# FSGS 最终独立场景验证：fortress

**冻结 FSGS 完整配方在 fortress 独立场景通过预选质量判据。**

| 配方 | 更新数 | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|---:|
| 统一 L1 | 30000 | 19.977695 | 0.587009 | 0.202653 |
| FSGS→gsplat | 9999 | 24.008280 | 0.735426 | 0.104730 |

相对同阶段统一 L1，ΔPSNR +4.030585 dB、ΔSSIM +0.148417、ΔLPIPS -0.097923。六张测试图共 18/18 项逐图/指标比较改善；其余变化同样保留在[完整结果](../results/fsgs-final-fortress.json)。

fortress 是原始最终测试列表中第一个尚未用于训练比较的场景，在读取 RGB 前固定。seed44、504×378、3训练视角、每8张一张测试图及从剩余图均匀取3张的规则保持冻结。训练图为 image001/021/041，测试图为 image000/008/016/024/032/040。room 已在前一阶段观察，不计入本轮独立结果。horns、orchids、trex 未在本轮执行，本页不是完整最终测试集汇总。

两配方采用相同公开标定相机、RGB 和1094个训练图三角化初始点。公开相机可能来自全视角 SfM；训练初始化、归一化及 Small 先验对齐只使用选定训练资产。三个训练视角锚点为735/944/471个。没有使用全场景点云或测试观测初始化。

比较完整工程配方：统一 L1 λ=0.01、step1000开启、30k更新；FSGS 保留原生10k循环（9999次更新）、MiDaS、虚拟视角和像素原点修正。两个完整训练结束后才独立评价，评价器沿用既有 gsplat PSNR/SSIM/LPIPS 定义。没有根据测试结果调参或切换渲染器。

## 实际成本与选择

| 工作进程 | 耗时 / 秒 | 整卡采样峰值 / MiB | 退出码 |
|---|---:|---:|---:|
| fortress-L1-seed44-training | 169.723 | 1646 | 0 |
| fortress-FSGS-seed44-training | 665.708 | 3729 | 0 |

最终高斯数：L1 136,781，FSGS 97,745；同一评价器的 RGB+ED 测试渲染均值分别为 1.480/1.300 ms/图。这是本次固定配方实测，更新数不同，不推算等更新数训练加速。原生和 gsplat 的最大逐图 PSNR 差为 3.042427 dB。

工程建议：质量优先的固定3视图重建可继续采用 FSGS 完整配方；训练成本与渲染成本按上表取舍。

继承累计 8450.481524 秒，原总上限保持9856.843380秒；本轮工作进程用时 867.900446/1406.361856 秒，累计 9318.381970 秒。没有扩大上限或清零失败消耗。CPU 下载和三角化沿用既有独立记账口径。完整日志、检查点、图像、显存采样及失败记录保存在仓库外 `fsgs-final-fortress-20261023T113618Z`。

## 本轮发现并修复的渲染兼容问题

预先冻结的标准 gsplat 入口使用 near=0.01。本轮它与原生 FSGS 的最大逐图 PSNR 差达到3.042427 dB，原生六视角均值为24.904161 dB，标准 gsplat 为24.008280 dB。原因是原生 CUDA 的 `in_frustum` 硬编码剔除中心 Z≤0.2；相机投影矩阵的 znear 不覆盖这条规则。此前已观察场景的微小后端差异没有揭示这一情况。

新增 [FSGS 显式兼容渲染器](../src/geosplat_r/fsgs_render.py)保持所有模型参数和像素约定，按 float32 边界实现同一近裁规则。[独立诊断入口](../scripts/check_fsgs_renderer.py)完成全部6张测试图的零更新复核：兼容输出 PSNR均值24.904448 dB，最大逐图 PSNR/SSIM/LPIPS差分别为0.001125 dB、0.00003821、0.00003150。修复没有重训、优化或选择参数。

上表和主判据继续使用预先冻结的标准 gsplat 结果（ΔPSNR +4.030585 dB）。兼容结果是发现问题后的后端修复证据，不替代主对照或重新计算一个更有利的增益。全部默认输出和两套后端指标均保留，诊断消耗继续计入原累计上限。

实际导入 FSGS 模型时，可显式调用兼容入口：

```python
from geosplat_r.fsgs_render import render_native_compatible
# checkpoint 为 convert_fsgs_checkpoint.py 生成且已加载到 CUDA 的完整字典
rgb_depth, alpha, info = render_native_compatible(checkpoint, view, depth=True)
```

零更新复核命令接受明确的检查点、测试数据、原生指标和本地指标权重，新建输出目录：

```bash
python scripts/check_fsgs_renderer.py --checkpoint "$CONVERTED_CHECKPOINT" \
  --test-data "$NEW_PREPARED/test" --native-metrics "$NATIVE_METRICS" \
  --weights "$METRIC_WEIGHTS" --output "$NEW_DIAGNOSTIC_OUTPUT"
```

## 复现

[固定运行配置](../configs/fsgs-final-fortress.json)记录矩阵、预算、判据与源码身份；[输入配置](../configs/fsgs-final-fortress-input.json)固定ZIP成员和视角。沿用[room 的准备和清单生成命令](fsgs-generalization.md#复现入口)，将配置路径和 `--scene` 改为本轮 fortress；所有输出使用新目录。原生训练、转换与独立评价仍使用[同一组入口](fsgs-reference.md#接入与复现)。原来的开发场景下载及既有图像、相机、权重和失败证据保持不变。

累计工程贡献包括：成熟 FSGS 配方接入、原生/gsplat 像素采样一致性修正、完整模型无数值改动导入、四个已观察场景及 leaves 三种子确认，以及本页固定独立场景结果。图像质量指标不等同于真实三维几何精度。
