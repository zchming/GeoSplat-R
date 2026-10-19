# Phase 2 累计工程提交候选审阅

审阅日期：2026-10-19。基准为49f0522e45e59790ae44d0f4a2e296cdd3c4a5ec，
工作分支codex/phase2-uniform-depth。本页对应当前累计候选，不替代首次工程提交的历史审阅页。

## 结论与候选

**提交前工程审查通过，建议形成一个工程提交。** 尚未执行暂存、提交或推送。
建议标题：`feat: add train-only depth controls and paired fern evaluation`

候选解决了训练专用Small逆深度先验、SfM逆Z对齐、显式统一深度损失及匹配零权重消融的可复现问题，
并加入三种子统计、同源码RGB-only主参照和逐图退化披露。按工程能力组织为一个提交，
不为单次训练、调试或结果变化分别创建提交。不包含flower/leaves、可靠性或新方法扩展。

可用提交说明：

> Add frozen train-only inverse-depth priors, uniform regularization and paired seed/path validation for fern.
> Preserve RGB-only as the main reference and RGB+ED zero-weight as the matched loss control.
> Record the three-seed +0.8328 dB mean gain over RGB-only, local regressions and unresolved path differences.
> Validate all 35 checks with local evidence; asset-free mode runs 22 and explicitly skips 13.

## 完整范围

共27个文件：6个已有文件修改、21个新增文件。下面清单以基准提交为界，包含此前未提交的全部Phase 2改动。
本次审阅仅补充本页、授权记录和README文字/链接，没有更改冻结训练、深度、评价或统计源码。

| 文件 | 动作 |
|---|---|
| `AGENTS.md` | 修改 |
| `README.md` | 修改 |
| `configs/fern-3view-depth-seeds.json` | 新增 |
| `configs/fern-3view-rgb-path-check.json` | 新增 |
| `configs/fern-3view-uniform-depth.json` | 新增 |
| `docs/phase2-rgb-path-check.md` | 新增 |
| `docs/phase2-seed-stability.md` | 新增 |
| `docs/phase2-submission-review.md` | 新增 |
| `docs/phase2-uniform-depth.md` | 新增 |
| `docs/reproduce-fern.md` | 修改 |
| `results/fern-3view-depth-seeds.json` | 新增 |
| `results/fern-3view-rgb-path-check.json` | 新增 |
| `results/fern-3view-uniform-depth.json` | 新增 |
| `scripts/evaluate_baseline.py` | 修改 |
| `scripts/export_train_anchors.py` | 新增 |
| `scripts/prepare_depth_priors.py` | 新增 |
| `scripts/summarize_depth_seeds.py` | 新增 |
| `scripts/summarize_rgb_paths.py` | 新增 |
| `scripts/train_baseline.py` | 修改 |
| `src/geosplat_r/baseline.py` | 修改 |
| `src/geosplat_r/depth.py` | 新增 |
| `src/geosplat_r/isolation.py` | 新增 |
| `src/geosplat_r/rgb_path_summary.py` | 新增 |
| `src/geosplat_r/seed_summary.py` | 新增 |
| `tests/test_depth.py` | 新增 |
| `tests/test_rgb_paths.py` | 新增 |
| `tests/test_seed_summary.py` | 新增 |

四个新增脚本分别导出训练锚点、生成对齐先验、汇总六模型配对、汇总九模型三路径；
三个新增配置记录首次对照和两次扩展的冻结设置。三个结果JSON分别保留当时seed42、三种子深度对照、
九模型三路径的结论、来源和退化；它们记录不同验收阶段，保留历史解释，不删除/压缩旧证据。
README和最新路径报告给出当前更保守的主结论，历史报告标明对应阶段。

## 实现与隔离审阅

- 先验仅使用三张训练图和训练SfM观测，对齐目标是相机轴向逆Z；无测试先验、置信度加权或鲁棒损失。
- 模型权重和源码版本有固定哈希；源、权重、数据和输出通过显式位置传入；无在线下载兜底。
- 所有相关输出入口均拒绝覆盖已有目录。读取审计是可审计的误用检查，不能称作安全沙箱；公开相机来自全数据标定，不能称只用三图恢复相机。
- 深度为有效像素平均归一化L1，alpha门槛不传梯度，空有效集返回有限零损失；零权重分支不加入深度反向项。
- 评价改动仅为显示标签和标签元数据；渲染、GT、clamp、掩码和指标公式保持原口径。
- 汇总先计算种子场景均值再配对，报告样本SD并保留每个逐图退化；检查视图、输入、源码、初值、预算、检查点与GT来源。

## 结果边界与不阻止工程提交的限制

当前三种子测试PSNR均值为RGB-only14.7403、RGB+ED零权重14.2416、统一深度15.5732 dB。
统一相对RGB-only的配对差均值+0.8328、样本SD0.5565 dB；相对匹配零权重为+1.3316 dB，不能混用。
seed43/image016的PSNR退化0.633831 dB，seed43/image008的SSIM退化0.001523，均保留。

RGB+ED零权重三个种子场景均值均低于RGB-only，不能认定路径质量等价。
ED通道、先验缓存/诊断及同seed非确定性未独立分离，不能称已经证明或修复内核缺陷。
单fern场景、每个seed/路径一次、公开标定相机、没有独立密集几何真值；
本提交不构成跨场景稳定收益、新算法贡献或高质量重建完成的研究验收。
多时钟数值不一致仍未定位，原值全部保留，不宣称算法加速。

读取审计不拦截所有本地原生库IO；来源核验不等于不可信输入安全验证。
新增入口明确限定当前fern协议，不声称支持其他场景；后续推广须单独改造和验收。
当前工程提交价值是可复现对照、保守统计和可追溯证据，上述研究限制不阻止如实记录它们。

## 本轮实际检查

- 完整资产模式：35项执行，35项通过，0失败、0跳过，包含CUDA小场景前向/梯度检查。
- 无资产模式：22项通过、13项跳过、0失败；未用跳过结果冒充全部实验来源检查通过。
- 六个相关入口帮助和已有输出拒绝覆盖检查通过；仅调用帮助及在输出目录已存在时提前拒绝，未进行训练/推理。
- 训练、深度、评价、配置及统计源码与实验前冻结哈希一致；旧六个模型的检查点未变。
- 三份发布结果与原始汇总及对应评价逐项核对；每个模型的评价检查点与训练检查点哈希一致。
- 新增三组完整可视化的PNG/数组/PLY/48帧视频沿用前一步验证；本轮未重新渲染或重训。
- 全候选JSON/Python语法、Markdown本地链接、私有绝对路径/密钥特征、非文本/大生成物和补丁空白检查通过。
- 完整Diff在独立临时目录基于49f0522进行试应用，并将结果逐文件与候选哈希比较；不修改当前索引。

13项跳过按原因分为：未请求CUDA检查1项、缺少先验/配对资产2项、相机审计1项、
上游参照1项、原始数据/划分2项、原始训练1项、六/九模型清单2项、去畸变训练1项、
去畸变输入2项。完整资产模式显式提供这些依赖，全部执行通过。

检查日志、完整working.diff、逐文件SHA256清单和审阅验证结果保存在项目外
`~/.local/share/geosplat-r/phase2-submission-review-20261019T043514Z/`，不将机器专用路径清单、原始日志或模型纳入Git。
数据、先验、下载权重、上游源码、训练/评价产物和历史失败材料继续保存在原位置。

## 提交与推送建议

建议用户审阅本清单和完整Diff后批准上述一个工程提交；本轮未提交、未推送，HEAD仍为49f0522，索引为空。
若之后获准提交，先再次核对候选文件哈希与索引范围，再执行提交；提交后核对内容并汇报。
推送需独立确认，可在本地提交内容复核通过后建议推送当前codex分支，不修改main或已发布历史。
当前不启动flower/leaves下载或训练，也不进入可靠性模块。

累计贡献为：训练专用深度对照、三种子配对、RGB-only主参照与完整负结果记录，
使收益表述从单一较弱对照的+1.3316 dB收敛为对RGB-only的+0.8328 dB，同时保留两者的适用范围。
