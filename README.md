# GeoSplat-R

稀疏视角 3D Gaussian Splatting 的几何可靠性优化。

**2026-10-23 工程结果：[FSGS 新场景与多种子确认](docs/fsgs-generalization.md)已完成。**
在既有三开发场景接入基础上，新增 room 同阶段对照与 leaves seed42/43；
leaves 三种子 FSGS 测试 PSNR 为 17.754 ± 0.351 dB，
room 相对统一 L1 的 PSNR 变化为 +5.374 dB。
完整配方比较为 FSGS 9,999 更新、L1 30k；测试集和所有逐图结果保留。
此前[三开发场景结果](docs/fsgs-reference.md)及旧路线、失败记录继续保留；更广泛最终 benchmark 尚未执行。

工程集成：[CPU 检查与主分支保护](docs/cpu-ci.md)。

**[Phase 2 总验收](docs/phase2-acceptance.md)：最小统一深度基线已完成，工程与对照协议通过，研究结论有条件通过。**
fern/flower/leaves各三seed的统一深度相对预选RGB-only基线，测试PSNR平均分别提高
0.8328/0.8426/0.1869 dB；逐视角退化、环境中断和感知质量限制均保留。
收益来自接入既有深度先验，不代表几何可靠性贡献或最终独立场景目标达成；Phase 3 尚未放行。

当前研究验收：**冻结 margin 歧义候选在未开启场景确认失败，停止该候选；不进入可靠性加权或新训练。**
两已观察开发场景在保留低风险50%配对时，错误风险分别下降92.6%与91.0%；
独立 courtyard 场景均值下降35.2%，但 DSC_0304 未通过单图门，local层错误风险增加34.6%。
场景均值不能覆盖单图/分层失败，这些数字也不是PSNR或三维重建收益。
历史D1/D2/H1、Cauchy及绝对深度锚点的负结果和支持不足均继续保留。

- [两开发场景完整对比](docs/ordinal-margin-diagnostic.md) · [未开启场景确认及停止判断](docs/courtyard-margin-confirmation.md)
- [CPU入口与隔离验收](docs/ordinal-margin-cpu.md) · [冻结确认方案](configs/courtyard-margin-confirmation-plan.json)
- [原始轨迹H1](docs/leaves-raw-track-diagnostic.md) · [独立几何数据可行性](docs/independent-geometry-feasibility.md)
- [terrace序关系诊断](docs/terrace-relative-diagnostic.md) · [delivery_area序关系诊断](docs/delivery-area-relative-diagnostic.md)

几何回归检查使用生成数据，不加载模型或真实数据：
`python scripts/check_geometry_cpu_ci.py --output <新的结果JSON路径>`。
CPU持续集成另外保留原有depth/protocol两组；真实GPU、外部资产与训练不属于CI。

Geometry-Reliability Guided Sparse-View 3D Gaussian Splatting.

稀疏视角三维重建工程研究项目，基于固定版本 gsplat v1.5.3。
目前完成 fern 三路径对照，以及 flower/leaves 冻结18模型扩展（3-view、seed42/43/44、三路径各30k）；尚未加入可靠性模块。

最新两开发场景结果：统一深度相对RGB-only的测试PSNR配对增益，flower为 **+0.8426 ± 0.3462 dB**，
leaves为 **+0.1869 ± 0.2391 dB**；两场景等权+0.5147 dB。27个配对测试视角中9个至少一项指标退化，
明显缺口、拉丝与细枝伪影仍在。共19次尝试得到18个完整模型，1次环境中断保留；leaves44统一深度重启后重跑，
环境影响尚未隔离。工程验收通过，协议和研究结论有条件通过，不宣称稳定泛化或高质量重建。
上述25文件工程成果已提交并推送为2e1e679，原18模型矩阵和条件限制继续保留。
详见[正式训练验收](docs/phase2-flower-leaves-training.md)和[完整数值/退化](results/flower-leaves-depth-controls.json)。

补充的leaves seed44同次启动环境三路径复核已完成：统一深度相对RGB-only为PSNR +0.205986 dB、
SSIM +0.028098、LPIPS −0.000225；相对零权重，LPIPS均值反而变差+0.008457，且四张测试图全部退化。
本轮3个完整模型来自4次尝试，包含1次手动中断后经批准从头重跑；不替换原矩阵，不把重复差异归因于重启。
补充工程/配对协议通过，稳定感知收益未通过。复核证据已提交并推送为2754e1a，暂不进入可靠性模块。
见[环境复核报告](docs/leaves-seed44-postboot-controls.md)及[数值与全部退化](results/leaves-seed44-postboot-controls.json)。

[训练端D1误差与覆盖诊断](docs/leaves-train-error-coverage-d1.md)已完成一次冻结CPU离线执行，14项检查全部通过。
610个训练锚点观测按253个3D点分组留点，15折均可用；三图距离与残差相关为−0.0392/+0.0772/−0.0453，
当前证据不支持直接按锚点距离设计可靠性权重。仅44/324格具备至少5个误差观测，其余保持未知。
未读取测试资产，未训练、推理或渲染；不建议立即追加训练。9文件工程成果已提交为fbc61f2，核对并推送完成。
见[精简结果与全部折数据](results/leaves-train-error-coverage-d1.json)及[冻结前方案（历史）](docs/leaves-train-error-coverage-plan.md)。

[训练端D2多视图先验冲突诊断](docs/leaves-crossview-prior-d2.md)已完成一次冻结CPU执行，D2检查19项及D1回归14项均通过。
610个锚点中546个具备两源评分，其余64个保持评分未知；三图冲突与残差相关为−0.2380/+0.3794/−0.0303，
支持门槛通过、研究方向门槛未通过。停止当前评分路线，不扩场景、不进入可靠性模块或新训练。
9文件工程成果已提交并推送为9987bf4，提交内容和远程引用已核对。
见[完整统计与缺失结果](results/leaves-crossview-prior-d2.json)和[冻结前方案（历史）](docs/leaves-crossview-prior-plan.md)。

[简单鲁棒深度入口与CPU验收](docs/leaves-robust-depth-implementation.md)已完成：可选Cauchy损失，
65项相关CPU/回归检查通过，保留原L1函数与经结构核对的默认路径。尚未GPU预检、训练或证明质量收益。
[冻结方案](docs/leaves-robust-depth-plan.md)仍保留完整/半强度L1及首尾重复，共4次拟议训练；
[累计Diff与执行冻结审阅](docs/leaves-robust-depth-execution-review.md)已通过静态检查，四组命令、源码及评价资产已冻结。
[一次零更新GPU预检](docs/leaves-robust-depth-gpu-preflight.md)已通过：43项CPU检查、6组合成损失、3条光栅梯度链及9个训练图梯度探针通过，参数未更新。
[四次冻结30k对照与完整评价](docs/leaves-robust-depth-pilot.md)已完成，80项相关检查通过。
Cauchy三项测试均值均劣于三个对照，预定门槛三项均未通过；停止当前固定候选，不追加调参或进入可靠性模块。
全部逐图退化、首尾波动、真实资源及可恢复证据见[完整结果](results/leaves-robust-depth-pilot.json)，不宣称质量收益。

历史 fern 三种子测试场景均值如下（±为种子样本标准差；不与新场景混池）：

| 组别 | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| RGB-only主基线 | 14.7403 ± 0.1906 | 0.370706 ± 0.005169 | 0.461895 ± 0.010775 |
| 匹配零权重 | 14.2416 ± 0.3849 | 0.345609 ± 0.012857 | 0.478075 ± 0.001644 |
| 统一深度 | 15.5732 ± 0.4466 | 0.386655 ± 0.009941 | 0.435833 ± 0.011066 |

统一深度相对RGB-only的PSNR增益为+1.4338/+0.3352/+0.7295 dB，均值**+0.8328**、样本SD0.5565。
三个种子的场景均值均改善，但seed43/image016的PSNR下降0.6338 dB、image008的SSIM也退化。
RGB+ED零权重在三个seed场景均值上均弱于RGB-only，平均低0.4988 dB；两条路径不能视为已等价，原因未孤立确认。
保留该路径作深度项匹配消融，但不把相对它的+1.3316 dB当作对RGB-only的收益。
该轮随后固定设置扩flower/leaves开发场景并保留双对照，结果见上方；暂不进入可靠性模块，不宣称稳定跨场景收益。

| 历史 RGB 基线（去畸变协议） | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| 训练三视图 | 45.4400 | 0.998310 | 0.001151 |
| 测试三视图 | 15.0197 | 0.378023 | 0.449066 |


协议 `fern-undistorted-v1` 使用相同六个视角、504×378，统一去畸变并更新相机内参，
仅训练图像生成初始化点。公开相机来自数据集标定，不代表只用三张图恢复相机。

历史 RGB 基线最差测试图为 10.861 dB；统一深度的image016在seed43/44仍仅约11.309/11.320 dB，有严重伪影。
工程/协议验收通过，尚未证明稳定算法改进。去畸变前后更换了 GT 与初始化，不能作收益归因；
Phase 2 三条路径保持去畸变后的 GT 与初始化资产不变，同seed初始参数哈希一致。全部旧结果保留。

- [Phase 1：协议、真实指标、失败案例、资源与复现](docs/phase1-fern.md)
- [当前协议复现步骤与检查范围](docs/reproduce-fern.md)
- [首次提交候选与审阅清单（历史）](docs/initial-submission-review.md)
- [flower/leaves 累计提交候选与审阅结论](docs/phase2-flower-leaves-submission-review.md)
- [fern Phase 2 提交候选审阅（历史）](docs/phase2-submission-review.md)
- [flower/leaves 正式训练验收与条件限制](docs/phase2-flower-leaves-training.md)
- [flower/leaves 数据准备验收（训练前历史记录）](docs/phase2-flower-leaves-preparation.md)
- [Phase 2：统一深度对照、局限与复现](docs/phase2-uniform-depth.md)
- [Phase 2：三种子复核、基线波动与下一步](docs/phase2-seed-stability.md)
- [三种子指标](results/fern-3view-depth-seeds.json) · [冻结扩展配置](configs/fern-3view-depth-seeds.json)
- [Phase 2：RGB-only路径复核与最新判断](docs/phase2-rgb-path-check.md)
- [九模型三路径指标](results/fern-3view-rgb-path-check.json) · [路径复核配置](configs/fern-3view-rgb-path-check.json)
- [Phase 2 指标](results/fern-3view-uniform-depth.json) · [对照配置](configs/fern-3view-uniform-depth.json)
- [历史去畸变基线指标](results/fern-3view-undistorted.json) · [基线配置](configs/fern-3view-undistorted.json)
- [旧协议指标](results/fern-3view-seed42.json) · [诊断](results/fern-3view-diagnosis.json)
- [Phase 0：选型与环境](docs/phase0.md)
- [阶段授权与保护规则](AGENTS.md)

训练/评价入口为 `scripts/train_baseline.py` 和 `scripts/evaluate_baseline.py`。
主依赖见 `requirements-phase0.txt`；初始化使用独立 `requirements-colmap.txt` 环境。
数据、权重、原始日志、checkpoint/PLY/视频与历史源码副本均位于仓库外，不随 Git 发布。
尚未选择本项目许可证；第三方归属和适用许可证必须保留。
