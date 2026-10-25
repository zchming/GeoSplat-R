# 实验结果

## 独立场景质量与成本

已知相机、三张训练图、504×378、seed44。比较统一L1的standard渲染与FSGS的原生兼容渲染，使用相同测试图及指标定义；两者分别完成30,000与9,999次更新。

| 场景 | L1 PSNR | FSGS PSNR | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---:|---:|---:|---:|---:|
| horns | 15.9663 | 20.3488 | +4.3825 | +0.2410 | -0.1352 |
| orchids | 15.2217 | 16.6572 | +1.4355 | +0.0840 | -0.0624 |
| trex | 13.9491 | 21.4993 | +7.5502 | +0.3757 | -0.3186 |

场景等权平均PSNR +4.456069dB、SSIM +0.233543、LPIPS -0.172073；全部19张测试图的三项指标均改善。逐图指标、最坏变化、standard敏感性对照和来源保存在[三场景结果JSON](../results/fsgs-final-three-scenes.json)。

| 场景 | L1/FSGS训练函数秒数 | L1/FSGS高斯数 | 参数存储变化 | L1/FSGS渲染ms/图 |
|---|---:|---:|---:|---:|
| horns | 206.6/624.6 | 213,302/80,966 | -62.0% | 1.568/1.237 |
| orchids | 253.3/641.8 | 289,212/128,785 | -55.5% | 2.172/1.508 |
| trex | 355.6/659.4 | 486,572/103,779 | -78.7% | 3.523/1.496 |

训练函数计时不含进程启动；参数存储按同类float32张量计算，不以含优化器状态的checkpoint文件大小代替。渲染为同一评价器预热后20次采样的RGB+ED均值。FSGS适合质量优先，L1保留更短训练等待的选择。

horns的模型和完成记录完整，但外部监控进程中断，整卡采样只覆盖训练前段，退出码未捕获。原始函数计时624.597秒；预算按900秒上限保守记账，不能把900秒当作实测训练耗时。失败、恢复及采样范围保留在结果JSON和本地原始记录中。

## 其他场景与重复验证

- [fern、flower、leaves开发场景](../results/fsgs-development-confirmation.json)：固定配方的同协议质量对照。
- [room与leaves三种子](../results/fsgs-generalization.json)：新场景和种子变化；种子统计不作为总体置信区间。
- [fortress独立场景](../results/fsgs-final-fortress.json)：标准渲染对照，以及原生近裁规则兼容诊断；不同渲染设置的均值不直接混池。

## 历史对照与负结果

- [统一深度基线](../results/phase2-acceptance.json)：fern/flower/leaves存在有限平均收益，逐视角退化完整保留。
- [Cauchy固定四臂对照](../results/leaves-robust-depth-pilot.json)：固定候选未优于预设对照，停止继续调整。
- [margin独立确认](../results/courtyard-margin-confirmation.json)：场景平均改善没有通过单图及局部分层要求，停止该候选。
- [训练锚点误差与覆盖](../results/leaves-train-error-coverage-d1.json)、[多视图先验冲突](../results/leaves-crossview-prior-d2.json)和[原始轨迹诊断](../results/leaves-raw-track-diagnostic.json)：保留未知、缺失支持及不支持继续的结论。

全部原始输入、权重、日志、模型及旧版本材料保存在本地归档；公开结果JSON和配置保留逐图指标、实验身份及负结果。
