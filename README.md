# GeoSplat-R

Geometry-Reliability Guided Sparse-View 3D Gaussian Splatting.

稀疏视角三维重建工程研究项目，基于固定版本 gsplat v1.5.3。
目前完成 fern 的去畸变 **Vanilla 3DGS 基线**（3-view、seed42、30k）；尚未加入深度或可靠性优化。

| 集合（新协议） | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
| 训练三视图 | 45.4400 | 0.998310 | 0.001151 |
| 测试三视图 | 15.0197 | 0.378023 | 0.449066 |


协议 `fern-undistorted-v1` 使用相同六个视角、504×378，统一去畸变并更新相机内参，
仅训练图像生成初始化点。公开相机来自数据集标定，不代表只用三张图恢复相机。

新视角仍严重过拟合，最差测试图为 10.861 dB。当前通过基线工程/协议验收，未取得算法改进成果。
GT 与初始化均已重建，新旧协议数值不能直接解释为性能提升；全部旧结果保留。

- [Phase 1：协议、真实指标、失败案例、资源与复现](docs/phase1-fern.md)
- [当前协议复现步骤与检查范围](docs/reproduce-fern.md)
- [首次提交候选与审阅清单](docs/initial-submission-review.md)
- [当前基线指标](results/fern-3view-undistorted.json) · [冻结配置](configs/fern-3view-undistorted.json)
- [旧协议指标](results/fern-3view-seed42.json) · [诊断](results/fern-3view-diagnosis.json)
- [Phase 0：选型与环境](docs/phase0.md)
- [阶段授权与保护规则](AGENTS.md)

训练/评价入口为 `scripts/train_baseline.py` 和 `scripts/evaluate_baseline.py`。
主依赖见 `requirements-phase0.txt`；初始化使用独立 `requirements-colmap.txt` 环境。
数据、权重、原始日志、checkpoint/PLY/视频与历史源码副本均位于仓库外，不随 Git 发布。
尚未选择本项目许可证；第三方归属和适用许可证必须保留。
