# 实现说明

## 训练数据

已知相机的LLFF三视角输入统一去畸变至504×378，内参同步更新。初始化、归一化、观测锚点和深度对齐仅使用训练图；测试图由独立评价入口读取。输入清单固定图像、相机、点云的SHA256及初始点数，已有输出禁止覆盖。

统一L1路径使用Depth Anything V2 Small先验，由训练SfM锚点完成仿射逆深度对齐。优化采用30,000次更新，深度权重0.01，从第1,000步开始启用。实现见[depth.py](../src/geosplat_r/depth.py)与[训练入口](../scripts/train_baseline.py)。

## FSGS接入

沿用固定版本FSGS的优化、增密、虚拟视角和MiDaS DPT-Hybrid深度监督。原生10k循环实际执行9,999次optimizer更新。初始化沿用训练稀疏SfM点；上游要求的`3_views/dense/fused.ply`只是文件布局，未增加稠密MVS点或测试图。

[输入适配](../scripts/prepare_fsgs_scene.py)导出训练专用COLMAP布局、位姿与bounds；[训练适配](../scripts/run_fsgs_reference.py)从本地加载MiDaS，并修正真实及虚拟相机的像素采样约定。

## 渲染与模型转换

- 原生光栅器使用整数像素采样，gsplat使用像素索引+0.5。对于当前居中主点，相机投影额外平移-0.5像素，使输出射线对应。
- FSGS原生光栅器剔除相机空间中心Z≤0.2的高斯；`fsgs-native`使用等价的float32近裁规则，RGB、RGB+ED、深度图和视频共用此设置。L1保留`standard`的near=0.01。
- [检查点转换](../scripts/convert_fsgs_checkpoint.py)保留位置、log-scale、wxyz四元数、opacity logits和SH系数的数值及坐标系，仅展平opacity维度。转换不训练或重新拟合模型。

兼容渲染实现见[fsgs_render.py](../src/geosplat_r/fsgs_render.py)，统一交付入口见[run_geosplat.py](../scripts/run_geosplat.py)。

## 评价与适用范围

使用独立测试图报告PSNR、SSIM和LPIPS-Alex，保存每张图的预测、误差、深度与有效性掩码；训练函数用时、参数张量存储和预热后渲染计时分别报告。完整配方比较不作为单组件消融或等更新数加速结论。输出深度是模型的期望相机轴向Z，不作为真实几何精度测量。

第三方算法、预训练权重与CUDA扩展保留原归属及许可；固定版本和权重身份见[环境配置](environment.md)。
