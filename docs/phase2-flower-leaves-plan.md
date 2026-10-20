# Phase 2：flower/leaves 数据核实与冻结实验提案

本页保留下载前的提案与当时授权状态；获批后的实际数据准备验收见 [准备报告](phase2-flower-leaves-preparation.md)。

核查日期：2026-10-19（Asia/Shanghai）。本轮仅授权推送42e297e、核实文件/体积并整理方案。
未下载任何数据集成员载荷，未准备数据、提取特征、三角化、推理或训练；未进入可靠性模块。
配置见 [冻结提案](../configs/flower-leaves-phase2-plan.json)。本页与配置尚未提交或推送。

## 推送核对

已将42e297eb3f468fb53a7b5a7818face9a0298a8d7推送到
[GitHub codex/phase2-uniform-depth](https://github.com/zchming/GeoSplat-R/tree/codex/phase2-uniform-depth)。
远端逐项核对：该分支为42e297e；Phase 1分支仍为49f0522；main仍为9ffc97b。
未创建PR、合并、强推或改写历史。

## 数据来源与精确范围

此前370,385,516-byte的NeRF示例包只含LLFF fern，不能用它获取flower/leaves。
[NeRF官方README](https://github.com/bmild/nerf#running-code)把完整数据指向Google Drive；
本轮该Drive文件预览请求返回401，没有获取其载荷，未核实官方完整包体积。
UCSD同目录下猜测的nerf_llff_data.zip返回404，不能当可用来源。

可用候选是[NerfBaselines维护的数据镜像](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/tree/6a638951a007c8b47eb8a0301d48b04d18e955cf/llff)，
它的[LLFF入口](https://github.com/nerfbaselines/nerfbaselines/blob/main/nerfbaselines/datasets/llff.py)实际引用该镜像。
固定镜像revision为 `6a638951a007c8b47eb8a0301d48b04d18e955cf`。
这是维护者镜像，不宣称与官方Drive归档逐字节相同。公开目录没有验证出独立的数据使用许可证，
代码许可证不能代替数据条款；后续只作本地研究，不将数据再分发进Git。

| 完整场景包（仅查元数据，不建议全量下载） | 精确 bytes | 发布方LFS SHA256（非本地全包验证） |
|---|---:|---|
| flower.zip | 188798089 | `b7a3ab4f340aea531edd1c5e93ff53c9da2f7d21a987cf3d87279006ee4eac56` |
| leaves.zip | 180271107 | `b4ceba46cdbfa86b7799c8757be9388e76a695682bb038eb0d19e2ba3203b4b7` |

本轮只查询文件API、22字节ZIP结束记录及精确中央目录区间。Range返回206且Content-Range/长度匹配，
两个ZIP的元数据响应体合计20,613 bytes；没有读取任何成员本地头或图像、标定、特征、点云压缩载荷。
其他网页/JSON清单查询流量另计。原始目录索引和请求范围保存在仓库外，可核对本轮读取边界。

按冻结视图规则，未来只需要以下21个成员；不取未用RGB、images_4、全分辨率图、数据库、points3D或nbv文件：

| 成员 | 压缩载荷 bytes | 解压内容 bytes |
|---|---:|---:|
| `flower/images_8/image000.png` | 359617 | 359586 |
| `flower/images_8/image001.png` | 355215 | 355194 |
| `flower/images_8/image008.png` | 360599 | 360590 |
| `flower/images_8/image016.png` | 358429 | 358426 |
| `flower/images_8/image017.png` | 354969 | 354972 |
| `flower/images_8/image024.png` | 353732 | 353739 |
| `flower/images_8/image032.png` | 352223 | 352238 |
| `flower/images_8/image033.png` | 349709 | 349698 |
| `flower/poses_bounds.npy` | 3900 | 4704 |
| `flower/sparse/0/cameras.bin` | 44 | 64 |
| `flower/sparse/0/images.bin` | 3080492 | 7539786 |
| `leaves/images_8/image000.png` | 449984 | 450027 |
| `leaves/images_8/image001.png` | 446291 | 446280 |
| `leaves/images_8/image008.png` | 446273 | 446274 |
| `leaves/images_8/image012.png` | 442626 | 442637 |
| `leaves/images_8/image016.png` | 441450 | 441486 |
| `leaves/images_8/image024.png` | 436173 | 436215 |
| `leaves/images_8/image025.png` | 434206 | 434283 |
| `leaves/poses_bounds.npy` | 3039 | 3616 |
| `leaves/sparse/0/cameras.bin` | 43 | 64 |
| `leaves/sparse/0/images.bin` | 2020031 | 5988154 |

| 合计 | 压缩载荷 bytes | 解压内容 bytes |
|---|---:|---:|
| flower（11个成员） | 5928929 | 10388997 |
| leaves（10个成员） | 5120116 | 9089036 |
| 总计（21个成员） | **11049045** | **19478033** |

建议批准的未来数据载荷为11,049,045 bytes（11.049045 MB，约10.54 MiB），
不是369,069,196-byte的两个全包。范围请求还可能需要21个30-byte成员头（630 bytes）及ZIP索引；
HTTP/TLS头、重定向和重试不是载荷体积，不能把载荷和真实网络计费字节混为一谈。
入口必须验证206、Content-Range和实际长度，拒绝忽略Range时回退全包；每个成员检查压缩方式、原始大小、CRC32，
获得批准并读取后才计算内容SHA256。配置中的完整包LFS哈希是发布方元数据，不是部分读取已验证全包。

poses_bounds仅使用前15个相机/H/W/f字段，不将由完整点云生成的near/far用于训练。
images.bin仅用于核对原名/排序/公开相机外参，解析时跳过所有特征观测，不落盘保存该原始二进制，
不得用其观测或全视角points3D作训练锚点。raw bytes包含此元数据的原始尺寸，不等于计划持久化大小。
权重Small与AlexNet均复用现有已校验文件，新增权重下载0 bytes；每个新场景必须重新用其训练图生成先验。

## 视图与数据协议冻结

仅枚举images_8下直接PNG成员并按文件名稳定排序，排除任何嵌套目录；
flower归档含.ipynb_checkpoints中的额外PNG，不能递归纳入视图清单。
测试为零基i%8==0；训练从剩余列表按np.rint(linspace(0,M-1,3))选取，保持ties-to-even规则。
这沿用Phase 0规则，未看图像内容、渲染或测试指标选视角。

| 场景 | 全部视图 | 训练索引 | 测试索引 | 不获取的剩余RGB |
|---|---:|---|---|---:|
| flower | 34 | 1、17、33 | 0、8、16、24、32 | 26 |
| leaves | 26 | 1、12、25 | 0、8、16、24 | 19 |

分辨率采用原生images_8，预期504×378但本轮未读取PNG内容，尚不能声称实际尺寸已通过。
相机模型、实际内参和畸变系数同样需下载获批后解析核验，不能套用fern的K或畸变。
SIMPLE_RADIAL路径保持alpha=0、主点居中、focal guard=1.001、INTER_LINEAR uint8与无padding检查；
若模型不同、实际尺寸不符或无padding检查失败，保留事实并停在协议复核，不偷偷改裁剪/缩放/保护系数。
不读取图像内容或指标来调整去畸变参数。

只从三张训练图提取SIFT与穷举匹配、在固定公开相机下三角化；归一化仅由训练相机中心决定。
仍是公开标定相机协议，不宣称三图端到端恢复相机。每个场景固定一套训练点和先验，所有路径/种子复用。
至少4个初始化点只是现有三角化底线；每张训练图还必须有至少20个有效深度锚点、正斜率和足够正目标。
失败就记录，不扩大训练视图、不借测试点补齐、不改用随机初始化或更大模型。

## 冻结训练与评价提案

未来正式矩阵为2场景×3种子（42/43/44）×3路径，共18次30k训练，当前未授权执行。

| 路径 | 训练渲染 | 训练专用先验/诊断 | 深度权重 |
|---|---|---|---:|
| RGB-only主基线 | RGB | 无 | 0 |
| RGB+ED零权重控制 | RGB+ED | 有 | 0 |
| 统一深度 | RGB+ED | 有 | 0.01，零基step1000开始固定 |

保持float32、batch1、SH3、每1000步增加SH、DefaultStrategy、classic/packed、原优化器/调度，
RGB损失0.8L1+0.2(1-fusedSSIM valid)，关闭pose/外观/曝光优化。
Small float32/输入518、训练锚点无权OLS逆Z对齐、深度有效掩码、归一化、损失均沿用fern冻结数学定义，
不增加鲁棒项或可靠性权重，不按场景调lambda或启用步数。
同scene/seed三路径的初始参数哈希必须相同；不把不同seed的随机初值误要求相同。

本地12 GB GPU顺序训练；优先保持GPU利用率和速度，保留测量的1 GiB余量，不以填满显存作为速度目标。
不为利用率改变batch、精度、分辨率或更新次数；完整记录GPU利用率、Torch与整卡峰值、四种时钟。
先验/对齐准备、评价、下载成本分开；不把计时域差异包装成加速结论。

执行顺序在配置中固定：flower先42/43/44，每个seed按RGB-only、零权重、统一；然后leaves同序。
所有18次完成后再进行新测试渲染和指标评价，不根据中途测试结果改变后续设置。
失败/OOM保留并停止依赖工作，不改协议继续、不静默排除失败、不另选seed或视图补位。
各输出均在仓库外用新目录保存，训练和评价分进程，训练不接收测试目录。

评价继续RGB+ED统一口径、sRGB/clamp、无mask/crop、PSNR/原SSIM/LPIPS-Alex、相同权重与独立指标检查。
每个scene先算各seed测试视图均值，再算配对差与样本SD；主对照uniform−RGB-only，
次对照uniform−RGB+ED零权重，以及零权重−RGB-only。
flower和leaves视图数不同，跨场景汇总对两个场景均值等权，不把9张图混为独立样本或偏重flower。
每场景/每seed/每图的退化和失败均保留；不挑较弱对照、不做n=3的显著性/等价性或独立几何精度结论。
fern仅作为已反复使用的开发证据，不混入两个新场景的主估计；最终五个保留场景不下载、不评价。

## 入口差距与放行条件

当前不是直接换路径即可执行的多场景工程：

- fetch_fern和audit_fern固定fern前缀/数量或相机映射；undistort_fern固定视图、504×378及协议名。
- baseline和prepare_depth_priors固定训练名字，后者还固定数组尺寸/协议名。
- export_train_anchors固定488点，六/九模型汇总固定fern的训练/测试名字。
- camera.pinhole_remap要求SIMPLE_RADIAL；本轮尚不知道两场景实际相机模型。

本轮只识别差距，不实现适配。后续须以显式场景/manifest适配输入及汇总，保留旧fern入口和历史证据，
不得把删除断言当成泛化支持。训练/损失/评价数值核心保持不变，并以旧fern解析回归与合成检查验证。
适配会改变实现文件哈希，所以正式训练前必须重新冻结新源码及数据哈希，不能谎称沿用旧文件哈希。
三个路径的新运行须全部使用同一冻结新实现，不把fern旧源码的结果作为严格同源码新增场景样本。

依次满足：

1. 明确批准以上21个成员及11,049,045-byte压缩载荷，之后才可获取数据。
2. 获批后核对CRC/SHA、尺寸/相机/映射，完成最小输入适配及训练测试隔离检查。
3. 独立构建各场景训练点/锚点/先验，所有几何与先验门槛通过；失败场景保留并报告，不直接训练。
4. 固定派生数据、先验、新实现来源哈希和18个命令，检查通过后取得正式训练授权。

## 判断与下一步

**本轮推送和文件/体积/方案核实通过；数据准备与训练放行尚未通过，也尚未执行。**
已经核实精确成员名、目录中的场景视图数/切分和压缩/原始大小，不能把未读取的尺寸、畸变、
相机映射、初始化点数或锚点拟合质量写成已验证事实。

建议下一步只批准上述精确范围下载及最小多场景输入适配/数据准备验收；
通过后再放行18次正式训练，而不是直接把当前fern脚本用于新场景。
这保持原先“先跨开发场景验证、不增加算法复杂度”的判断。
本轮新增方案文件/授权记录未提交未推送，不再修改刚推送的工程提交。

本轮证据在项目外 `~/.local/share/geosplat-r/phase2-development-plan-20261019T072600Z/`，
包含镜像元数据、ZIP目录原始字节、HTTP区间记录、逐场景选择和推送核对；这些只是元数据，不含数据集成员。
累计成果是在已有可复现深度对照上，补齐发布备份、两开发场景的最小获取清单和预先固定的三路径验证矩阵，
没有新增性能结果或可靠性能力。
