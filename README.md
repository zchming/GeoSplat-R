# GeoSplat-R

稀疏视角 3D Gaussian Splatting 工程：提供 FSGS 质量路径与统一 L1 快训选项。

## 使用

[操作说明](docs/usage.md) · [环境配置](configs/runtime.example.json) · [统一入口](scripts/run_geosplat.py)

| 路径 | 适用目标 | 配方 | 评价 |
|---|---|---|---|
| `--mode fsgs` | 新视角质量优先 | 原生10k循环，9,999次更新 | `fsgs-native` |
| `--mode l1` | 更短训练等待 | 统一深度L1，30,000次更新 | `standard` |

从已准备的 `train/`、`test/` 开始，输出检查点、逐图指标、误差与深度图、点云和新视角视频。环境、数据、权重和输出位置均显式指定，输出目录禁止覆盖。

## 已验证效果

horns、orchids、trex 三个独立场景中，FSGS 相对统一L1的场景等权 PSNR +4.456dB、SSIM +0.234、LPIPS -0.172，高斯参数存储减少55.5%—78.7%。训练耗时约625—659秒，L1为207—356秒。这是两套完整配方的质量与成本比较。

- [最终三场景及逐图结果](docs/fsgs-final-three-scenes.md)
- [fortress 独立验证](docs/fsgs-final-fortress.md)
- [新场景与多种子确认](docs/fsgs-generalization.md)
- [FSGS 适配与实现来源](docs/fsgs-reference.md)

## 复现与历史研究

已知相机、LLFF 3-view、504×378协议；初始化和先验仅使用训练图。模型输出深度不等同于真实几何精度。成熟方法集成的工程收益已得到上述证据支持。

历史统一深度基线存在逐视角退化；Cauchy、几何可靠性和margin候选的停止结论继续保留，避免重复尝试。相关代码、配置、指标和原始复现文档保持可查：

- [统一深度基线总结](docs/phase2-acceptance.md) · [fern复现](docs/reproduce-fern.md)
- [Cauchy完整对照](docs/leaves-robust-depth-pilot.md) · [margin独立确认](docs/courtyard-margin-confirmation.md)
- [环境与依赖](docs/phase0.md) · [CPU持续集成](docs/cpu-ci.md)

数据、权重、原始日志及模型产物保存在仓库外。第三方归属和许可见对应实现与 `docs/licenses/`；本项目尚未选择独立许可证。
