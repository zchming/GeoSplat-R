# CPU 持续集成与主分支保护

## 检查范围

`CPU checks` 工作流在面向 main 的 PR、main 推送及手动触发时运行。
使用公开仓库的标准 Ubuntu CPU runner、Python 3.10.22 和独立的
`requirements-ci.txt`；不安装训练 requirements、gsplat 或 fused-ssim，
不下载数据集、模型权重或历史实验缓存。
第三方 Actions 固定到已核对的提交 SHA，令牌只有 contents:read，
checkout 不持久化凭据。任务超时 15 分钟，不使用付费 runner 配置。

深度组与协议组使用两个独立进程，避免 Torch 导入影响 D1/D2 的
无模型导入断言。合计运行 85 项：79 项便携检查通过，6 项外部证据
检查按原测试明确跳过。脚本要求跳过名单和测试数量符合固定清单；
意外跳过、测试缺失、失败、CUDA 初始化均使检查失败。
便携验收不等于此前本机使用外部证据完成的 80 项相关检查。

六项跳过包括开发数据准备的两项证据、D1、D2、fern seed 配对和
RGB 路径各一项证据；原因随 unittest 输出及 JSON 报告保留。
另外不收集需要 CUDA 基线/扩展的旧测试模块，以及
`test_robust_depth.DepthEntryAndWiring.test_existing_output_is_refused_before_cuda_or_input_reads`。
该单项会在导入时依赖 CUDA 扩展，不能用假模块替代后宣称验证了原实现。
CUDA 栅格器、真实资产及历史训练的复核继续由原有显式入口承担。

本地复现（独立 CPU 环境）：

```bash
python -m pip install -r requirements-ci.txt
python -W error::ResourceWarning scripts/check_cpu_ci.py --group depth --output /new/location/depth.json
python -W error::ResourceWarning scripts/check_cpu_ci.py --group protocol --output /new/location/protocol.json
```

两份输出路径必须尚不存在；父目录须已存在。执行前清除 GEOSPLAT_*
环境设置，避免本机历史证据或 GPU 开关混入便携检查。脚本屏蔽 CUDA
可见设备并拒绝 CUDA 初始化及训练扩展导入。完整 JSON 报告写入 runner
临时目录和 Actions summary，不提交本机日志或缓存。

## 主分支保护方案

先确认 PR 的真实 `CPU regression` 检查成功，再绑定该检查与实际
GitHub Actions App。main 必须通过 PR 合并，要求基于最新 main 的
CPU regression 成功并解决所有讨论；规则同样适用于管理员。
禁止强推和删除，保留 merge commit，不要求线性历史。

当前为个人仓库，不要求另一位人工审阅者批准（审批数量 0），避免
作者无法批准自己的 PR 而阻塞正常集成。技术审阅、Diff 检查和明确
授权仍须完成。此设置不等于已经获得独立同行审阅。

保护配置是 GitHub 设置，实际启用后应从 API 读回核对，不能仅凭
本文件宣称已启用。初次工作流在 PR 上通过后才能设为必需检查。

## 贡献与结论边界

本项增加可在干净 CPU runner 上执行的回归入口和 PR 集成约束。
冻结训练源码、配置、指标和所有负结果保持不变；不产生新的重建
质量证据，不改变 Cauchy 候选未通过收益门槛及暂不进入可靠性模块的结论。
