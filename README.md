# GeoSplat-R

面向已知相机的稀疏视角 3D Gaussian Splatting 工程，集成 **FSGS 质量路径**与**统一 L1 快训路径**，提供训练、模型转换、独立评价和可视化交付。

## 性能收益

在 horns、orchids、trex 三个独立场景上，FSGS 相对统一 L1 的场景等权 **PSNR +4.456 dB、SSIM +0.234、LPIPS −0.172**；全部19张测试图的三项指标均改善，高斯参数存储减少 **55.5%—78.7%**。

| 场景 | L1 PSNR ↑ | FSGS PSNR ↑ | ΔPSNR |
|---|---:|---:|---:|
| horns | 15.966 | 20.349 | +4.382 dB |
| orchids | 15.222 | 16.657 | +1.436 dB |
| trex | 13.949 | 21.499 | +7.550 dB |

[逐图指标、训练成本与补充验证](docs/results.md)

## 两条使用路径与训练成本

| 选项 | 适用目标 | 固定配方 | 上述三场景训练耗时 |
|---|---|---|---|
| `--mode fsgs` | 质量优先，模型更紧凑 | 原生10k循环，9,999次更新 | 625—659秒 |
| `--mode l1` | 更短训练等待 | 统一深度L1，30,000次更新 | 207—356秒 |

耗时为本机已保存的训练函数计时，不含进程启动；比较的是两套完整配方。FSGS 使用 `fsgs-native` 评价，L1 保留 `standard`。

[开始使用](docs/usage.md) · [环境配置模板](configs/runtime.example.json) · [统一入口](scripts/run_geosplat.py) · [实现说明](docs/implementation.md)

从已准备的 `train/`、`test/` 开始，显式指定环境、数据、权重和新输出目录；交付检查点、逐图指标、RGB／误差／深度图、点云及新视角视频。

## 复现范围

已验证协议为已知相机、LLFF 3-view、504×378；初始化和先验仅使用训练图，模型深度不作为真实几何精度证据。历史对照、负结果和适用范围见[实验结果](docs/results.md)。

[数据准备与复现](docs/usage.md) · [环境与依赖](docs/environment.md) · [CPU持续集成](.github/workflows/cpu-checks.yml)

数据、权重、原始日志和模型产物保存在仓库外。第三方归属与许可见对应实现及 `docs/licenses/`；本项目尚未选择独立许可证。
