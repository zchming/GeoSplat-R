# FSGS 新场景与多种子确认

**固定 FSGS 配方完成 room 新场景对照及 leaves 三种子确认，全部预选均值判据通过。**

| 场景 | seed | 统一 L1 PSNR | FSGS→gsplat PSNR | ΔPSNR | FSGS SSIM | FSGS LPIPS-Alex |
|---|---:|---:|---:|---:|---:|---:|
| leaves | 42 | 12.219966 | 18.054225 | +5.834259 | 0.612122 | 0.156757 |
| leaves | 43 | 12.220644 | 17.838058 | +5.617414 | 0.606572 | 0.161784 |
| leaves | 44 | 12.298660 | 17.368537 | +5.069877 | 0.566035 | 0.171683 |
| room | 44 | 15.396070 | 20.769613 | +5.373543 | 0.804912 | 0.170618 |

leaves 的三种子测试 PSNR 为 17.753607 ± 0.350558 dB（样本标准差），对照为 12.246424 ± 0.045240。本表共 54/54 个逐图/指标比较改善；其中42项来自本轮新增运行，12项为保留的 leaves seed44 结果。所有逐图变化见[完整结果](../results/fsgs-generalization.json)。

room 的统一 L1 与 FSGS 均在本阶段运行，使用相同的3张训练图、6张测试图、相机和128个训练 SfM 点。room 在读取 RGB 前固定，采用41张源图中每8张一张作测试、剩余图均匀取3张作训练；没有更换场景、调整分辨率或按结果选视角。下载采用 ZIP range，选定成员与局部头部传输3,102,350字节（ZIP索引另计），包含9张 RGB 和公共相机/位姿元数据；未下载完整数据集或全场景点云。

leaves seed42/43 与历史同协议、同种子的统一 L1 对照比较；seed44 继承此前的 FSGS 模型及同次运行 L1 首尾重复均值。本轮没有重训 seed44。表内标准差描述这三个种子，不作为置信区间。

两条路径比较完整训练配方：统一 L1 为30,000次更新，FSGS 原生10k循环为9,999次更新。room 的 Small 先验仍按原训练锚点做仿射逆深度对齐；FSGS 保持既有 MiDaS、虚拟视角、增密和优化日程。没有根据测试指标调参，全部训练完成后才独立评价。

## 实测成本与交付

| 本轮训练 | 进程墙时 / 秒 | 整卡采样峰值 / MiB |
|---|---:|---:|
| room-L1-seed44-training | 177.214 | 1532 |
| leaves-FSGS-seed42-training | 669.783 | 3988 |
| leaves-FSGS-seed43-training | 677.358 | 4042 |
| room-FSGS-seed44-training | 619.408 | 3325 |

新增阶段 GPU 工作进程记账 2193.638/3600 秒；继承 6256.843 秒，累计 8450.482 秒。CPU 准备及下载另计；这条连续阶段账本不代表项目全部历史成本。原生训练、转换、独立评价和显存采样的完整记录保存在仓库外的 `fsgs-generalization-20261023T080252Z`，旧运行及失败记录均保留。

每个新 FSGS 检查点均先由原生渲染器评价，再无数值改动地转换并交给现有 gsplat 评价器；两套逐图结果和差异完整保留。运行成本按实际配方报告，不推算等更新数加速。room 的新视角视频、RGB/误差/深度图、点图、两套检查点及首张固定测试图对比均已保存。

质量优先的稀疏视图任务建议采用 FSGS 完整配方。room 的训练进程耗时从177.214秒增至619.408秒；最终高斯数从169,635降至48,822，相同评价器的 RGB+ED 渲染均值从1.431降至1.182 ms/图。这里报告本次采样值，不将单次计时外推为通用加速比例。原生与 gsplat 渲染器在本表全部模型上的最大逐图 PSNR 差为0.000920 dB。

## 复现入口

固定输入清单见[room 输入配置](../configs/fsgs-room-input.json)，运行矩阵、种子、预算及判据见[阶段配置](../configs/fsgs-generalization.json)。以下 CPU 准备命令均须使用新输出目录；prepare_fern.py 和 export_train_anchors.py 使用已有 pycolmap 环境，其余使用已有训练环境：

```bash
python scripts/fetch_llff_scene.py --plan configs/fsgs-room-input.json --output "$NEW_RAW"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); json.dump({"scenes":[p["scene"]]},open(sys.argv[2],"x"))' configs/fsgs-room-input.json "$SCENE_SELECTION"
python scripts/undistort_llff_subset.py --source "$NEW_RAW" --plan "$SCENE_SELECTION" --scene room --output "$NEW_UNDISTORTED"
python scripts/prepare_fern.py --source "$NEW_UNDISTORTED" --intrinsics "$NEW_UNDISTORTED/intrinsics.json" --output "$NEW_PREPARED"
python scripts/freeze_scene_input.py --data "$NEW_PREPARED/train" --scene room --output "$NEW_PREPARED/train/input-manifest.json"
python scripts/export_train_anchors.py --data "$NEW_PREPARED/train" --scene-manifest "$NEW_PREPARED/train/input-manifest.json" --output "$NEW_ANCHORS"
```

统一 L1 的 Small 先验继续使用已有显式本地源码和权重，在训练环境执行：

```bash
python scripts/prepare_depth_priors.py --data "$NEW_PREPARED/train" \
  --scene-manifest "$NEW_PREPARED/train/input-manifest.json" --anchors "$NEW_ANCHORS" \
  --source "$SMALL_SOURCE" --source-manifest "$SMALL_SOURCE_MANIFEST" \
  --weights "$SMALL_WEIGHTS" --output "$NEW_PRIORS"
```

`SMALL_SOURCE_MANIFEST` 指向已有 source-selection.json（包含源码文件 SHA256）。FSGS 直接使用下面的训练输入转换入口，不消费 Small 先验：

```bash
python scripts/prepare_fsgs_scene.py --train-dir "$NEW_PREPARED/train" --output "$NEW_FSGS_DATA"
```

沿用[已有训练、模型转换和评价入口](fsgs-reference.md#接入与复现)。room 的 train/input-manifest.json 固定记录图像、相机、点的 SHA256 和点数；Small 先验从已导出的训练观测锚点生成，初始点保持128个。新增场景名称只扩展显式白名单，原 flower/leaves 下载入口的固定计划校验保持不变。

累计工程成果覆盖 FSGS 原生训练、像素采样约定修正、无数值改动的 gsplat 模型导入、四场景质量对照及 leaves 三种子确认。图像质量收益由固定测试图支持；深度图仍为模型输出的相机轴向 Z，不作为真实几何精度证明。更广泛的最终 benchmark 尚未执行。实验执行期间没有自动提交或推送；发布决定与实验记录分开。新增清单生成命令在独立 CPU 副本上生成了与实际 room 训练输入一致的8字段清单，不重跑训练。
