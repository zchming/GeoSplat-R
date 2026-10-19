# GeoSplat-R

Geometry-Reliability Guided Sparse-View 3D Gaussian Splatting.

稀疏视角三维重建工程研究项目，基于固定版本 gsplat v1.5.3。
目前完成 fern 去畸变三路径对照（RGB-only、RGB+ED零权重、统一深度；3-view、seed42/43/44、各30k）；尚未加入可靠性模块。

Phase 2 三种子测试场景均值如下（±为种子样本标准差）：

| 组别 | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| RGB-only主基线 | 14.7403 ± 0.1906 | 0.370706 ± 0.005169 | 0.461895 ± 0.010775 |
| 匹配零权重 | 14.2416 ± 0.3849 | 0.345609 ± 0.012857 | 0.478075 ± 0.001644 |
| 统一深度 | 15.5732 ± 0.4466 | 0.386655 ± 0.009941 | 0.435833 ± 0.011066 |

统一深度相对RGB-only的PSNR增益为+1.4338/+0.3352/+0.7295 dB，均值**+0.8328**、样本SD0.5565。
三个种子的场景均值均改善，但seed43/image016的PSNR下降0.6338 dB、image008的SSIM也退化。
RGB+ED零权重在三个seed场景均值上均弱于RGB-only，平均低0.4988 dB；两条路径不能视为已等价，原因未孤立确认。
保留该路径作深度项匹配消融，但不把相对它的+1.3316 dB当作对RGB-only的收益。
下一步建议固定设置扩flower/leaves开发场景并保留双对照，暂不进入可靠性模块，不宣称稳定跨场景收益。

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
- [Phase 2 累计提交候选与审阅结论](docs/phase2-submission-review.md)
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
