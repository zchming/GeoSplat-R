# flower/leaves 累计工程提交候选审阅

审阅日期：2026-10-19。基准为42e297eb3f468fb53a7b5a7818face9a0298a8d7，
分支codex/phase2-uniform-depth。本页覆盖此后数据规划、准备和正式矩阵的全部累计改动。
旧fern提交审阅仍由[原报告](phase2-submission-review.md)解释，不改写其历史结论。

## 结论与候选提交

**提交前工程审查通过，建议形成一个工程提交。** 本轮没有暂存、提交或推送。
建议标题：`feat: add frozen flower and leaves depth comparisons`

三种子对照原先限定fern输入；本候选增加显式训练manifest，使flower/leaves复用相同训练数学，
并拒绝错误划分、输入哈希漂移及输出覆盖。保留旧fern入口，补齐两场景18模型的真实指标、
逐图退化、资源与中断恢复说明。适配、检查和证据形成一个完整工程单位，不按训练次数拆提交。

建议提交说明：

```text
Add explicit train-only manifests and audited preparation for flower/leaves
while preserving the fern entry points and frozen training/evaluation math.
Support the approved four/five-view summaries and retain every regression.

Record the 18-model development comparison, the interrupted attempt and full
retry, measured resources, and the unresolved post-reboot leaves44 limitation.
Validate 45 checks with local assets; asset-free mode passes 28 and skips 17.
```

## 完整文件清单

共25个文件：11个修改、14个新增；相对42e297e，包含全部未跟踪候选文件。
完整补丁与逐文件大小/SHA256保存在本机项目外审阅包，未把原始日志或大产物加入候选。

| 文件 | 动作 | 用途 |
|---|---|---|
| `AGENTS.md` | 修改 | 记录各步明确授权和当前审阅边界 |
| `README.md` | 修改 | 最新结果、限制和报告导航 |
| `configs/flower-leaves-phase2-plan.json` | 新增 | 原冻结21成员清单与18模型方案 |
| `docs/phase2-flower-leaves-plan.md` | 新增 | 保留下载前核实及提案 |
| `docs/phase2-flower-leaves-preparation.md` | 新增 | 数据准备与原审计失败记录 |
| `docs/phase2-flower-leaves-training.md` | 新增 | 正式矩阵、资源、退化和条件验收 |
| `docs/phase2-flower-leaves-submission-review.md` | 新增 | 本次提交候选和实际审查 |
| `docs/reproduce-fern.md` | 修改 | 保留旧入口并说明45项检查依赖 |
| `results/flower-leaves-preparation.json` | 新增 | 精简准备来源、先验与资源 |
| `results/flower-leaves-depth-controls.json` | 新增 | 18模型数值、来源和全部退化 |
| `scripts/check_scene_input.py` | 新增 | 不优化、不渲染的训练输入GPU预检 |
| `scripts/fetch_llff_subset.py` | 新增 | 固定清单Range获取，拒绝全包兜底 |
| `scripts/undistort_llff_subset.py` | 新增 | 场景独立标定、原规则去畸变 |
| `scripts/validate_llff_preparation.py` | 新增 | 划分、来源、训练几何及manifest审计 |
| `scripts/export_train_anchors.py` | 修改 | 可选显式场景输入，保留fern默认约束 |
| `scripts/prepare_depth_priors.py` | 修改 | 可选场景输入，不改先验数学 |
| `scripts/train_baseline.py` | 修改 | 增加可选--scene-manifest参数 |
| `scripts/summarize_depth_seeds.py` | 修改 | 验证新场景六模型来源及测试列表 |
| `scripts/summarize_rgb_paths.py` | 修改 | 验证新场景九模型和RGB-only主参照 |
| `src/geosplat_r/baseline.py` | 修改 | 验证输入并记录来源，不改训练循环 |
| `src/geosplat_r/scene_input.py` | 新增 | 批准划分、哈希和点多重集约束 |
| `src/geosplat_r/rgb_path_summary.py` | 修改 | 显式4/5视角列表，保留默认fern统计 |
| `src/geosplat_r/seed_summary.py` | 修改 | 配对统计的显式场景视角验证 |
| `tests/test_scene_input.py` | 新增 | Range/划分/几何/数学兼容与真实准备检查 |
| `tests/test_development_training.py` | 新增 | 真实18模型、等权汇总和退化检查 |

本轮审阅只补充本页、README导航和授权记录；没有改冻结实现、实验配置、数据或既有结果文件。

## 源码与证据审阅

- 新训练输入仅限flower/leaves各自批准的三张训练图；manifest必须位于显式train目录，
  schema不接受test字段，图像/K/点云由哈希核对。训练实际读取清单不含测试资产。
- 旧fern不传--scene-manifest时仍走原保护分支。训练循环、render/load_views与42e297e的AST一致；
  depth.py及evaluate_baseline.py未改。新适配文件哈希与旧版不同，正式矩阵有单独冻结，未冒用历史哈希。
- 点云来源审计按float32 XYZ与RGB多重集核对并保留重复数；修正COLMAP序列化的迭代顺序差异，
  没有改点、降低锚点门槛或用测试观测补齐训练几何。
- 两场景六/九模型汇总核对输入、源码、初值、预算、GT、指标版本和checkpoint；
  跨场景先算场景均值再等权，保留逐图负差，没有按5/4张测试图混池。
- 21成员清单仍为原SHA256；24个实现文件、121个输入/证据文件及原/恢复清单哈希一致。
  发布JSON的场景统计、逐图指标、资源、全部退化与原始汇总逐项匹配，18个模型及metrics哈希核验通过。
- 11份已有fern配置/结果逐字节等于HEAD；旧输入、模型、失败和入口继续保留。

## 本轮实际检查

| 检查 | 实际结果 |
|---|---|
| 完整资产模式，含CUDA小场景前向/梯度检查 | 45通过、0失败、0跳过 |
| 无显式资产及CUDA开关 | 28通过、0失败、17跳过 |
| 两场景×六/九模型汇总CLI | 4次通过；九模型输出与原冻结分析一致 |
| 上述4个汇总输出拒绝覆盖 | 4次正确拒绝，原输出未变 |
| 发布数值、初始化/checkpoint/metrics、冻结来源 | 通过 |
| 全候选Python/JSON语法、Markdown本地链接、补丁空白 | 通过 |
| 私有绝对路径、常见密钥特征、非文本/大产物检查 | 通过；这不是完整安全审计 |
| 完整Diff基于42e297e试应用及逐文件SHA256核对 | 通过；未修改真实索引 |

17项跳过逐项原因保存在无资产日志：CUDA未显式启用1项；先验/原配对2项；相机审计/上游参照2项；
原数据/划分2项；原训练1项；fern六/九模型清单2项；去畸变训练/数据3项；
开发场景准备2项；开发场景训练/汇总2项。完整模式明确提供相应资产后全部执行。

准备阶段的10个入口帮助/拒绝覆盖检查保留，相关入口源码与其冻结记录一致；
本轮没有通过启动训练或重新渲染评价来重复这些历史检查。
前一步135组视角产物、846张PNG、270个数组、18个PLY和18个48帧视频检查沿用其产物审计记录，
本轮核对模型/指标哈希，没有重新生成这些大产物。

审阅工具首次按错误目录拼接flower深度清单，在命令执行前路径断言失败；
改用RGB清单的显式depth_manifest引用后完成4次检查。错误与修正另存review-tool-path-error.json，
属于本轮检查编排错误，没有改实验输入、冻结源码或历史输出，也没有隐去失败。

## 不阻止工程提交的研究与复现限制

统一深度减RGB-only的PSNR配对均值±样本SD：flower +0.8426±0.3462，leaves +0.1869±0.2391 dB。
两场景等权+0.5147 dB；27个配对测试图中9个至少一项退化，全部保留。
flower缺口/拉丝、leaves细枝破碎和遮挡伪影仍明显，训练拟合高不能代替新视角质量。

共19次尝试获得18个完整30k模型；原leaves44统一深度在环境重启时中断，按原设置从头重跑，
旧目录、最后27k状态和缺失checkpoint的事实保留。前17模型重启前、末模型重启后，
驱动前后是否相同未获证实；leaves44的环境影响未隔离。因此协议和研究仍是有条件通过，
本次提交不能视为放行可靠性模块或确认稳定跨场景收益。

两场景是开发集、每路径/seed一次、公开相机可能来自全视角SfM、没有独立密集几何真值；
不声称新颖算法、相机恢复精度、最终保留场景泛化或高质量项目完成。
读取审计不是拦截原生库IO的安全沙箱，固定来源校验也不等于不可信数据安全认证。
本次入口限定批准场景，不声称任意场景通用；未验证全新机器重建环境及逐位复现。
多时钟不一致仍未定位，不把耗时差异当作算法加速。整卡显存为200ms采样，不能保证捕获瞬时峰值。

仓库保存可复用输入/训练/评价/单场景统计入口和明确协议；完整运行清单、恢复编排、
双场景画图与发布脚本在外部证据包，尚无仓库内的一键端到端跨机器复现实验入口。
本候选不把外部机器路径、原始日志、数据、权重、缓存、vendor源码、checkpoint、PLY、视频或大图纳入Git。
保持方案/准备/正式训练三份历史报告，它们对应不同授权状态，不合并或删减旧证据。

## 提交与下一步建议

建议审阅完整Diff和清单后，批准上述一个工程提交。暂不建议直接推送；
获得单独提交授权后，先复核候选哈希及索引范围，提交后再核对内容并给出推送判断。
如果需要加强leaves稳定性结论，最小实验仍是另获授权后补做重启后seed44的RGB-only与零权重控制；
本次未执行，不调系数、不进入可靠性模块，不增加场景。

审阅包位于 `~/.local/share/geosplat-r/phase2-flower-leaves-submission-review-20261019T124618Z/`，
包含complete-working.diff、candidate-files.json、checks.json、证据核验和实际测试日志；
原训练/准备/恢复证据保留在各自原目录。HEAD仍为42e297e，真实索引为空。

累计贡献是在fern九模型证据上补齐两开发场景的严格输入适配、18模型固定对照、
逐视角负结果与资源/中断审计；提交价值是可复核工程和保守经验结果，尚非算法新颖性证明。
