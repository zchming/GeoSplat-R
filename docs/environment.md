# 环境与依赖

把 [runtime.example.json](../configs/runtime.example.json) 复制到仓库外，填入本机位置。路径可为绝对路径，也可相对配置文件；虚拟环境的 `bin/python` 路径会原样保留。入口使用标准库，配置中的 Python 才负责模型计算。

| 配置键 | 使用范围 |
|---|---|
| `python`、`cuda_home`、`metric_weights` | 两条路径及独立评价 |
| `fsgs_python`、`fsgs_source`、`midas_source`、`midas_weights` | FSGS 训练 |
| `gsplat_source` | L1 训练 |
| `colmap_python`、`small_source`、`small_source_manifest`、`small_weights` | L1 尚无对齐先验时；复用先验可省略 |

已验证环境是 Linux/WSL、Python 3.10.22、Torch 2.4.1+cu124、CUDA Toolkit 12.4.1、gsplat 1.5.3；COLMAP 使用独立 Python 3.12.3 / pycolmap 4.2.1。基础依赖见 [requirements-phase0.txt](../requirements-phase0.txt)、[requirements-colmap.txt](../requirements-colmap.txt)。当前入口使用本地 GPU 0，整卡采样上限 11,500 MiB；依赖、CUDA 扩展和权重须提前准备，运行时不安装或下载。

固定源码与权重：

| 资产 | 身份 |
|---|---|
| gsplat 源码 | `937e29912570c372bed6747a5c9bf85fed877bae` |
| FSGS 源码 | `a536a64c5b366b1088be64eeadf9e791ca26897c`，应用下述兼容补丁 |
| MiDaS | `v3_1`，`dpt_hybrid_384.pt`；SHA256 `501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853` |
| Depth Anything V2 Small | 源码 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`；权重 SHA256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` |
| 指标权重目录 | 包含 `hub/checkpoints/alexnet-owt-7be5be79.pth`；SHA256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |

`small_source_manifest` 是已有的 `source-selection.json`，记录 Small 源码文件散列。FSGS 不读取 Small 先验。

已有 FSGS 环境可直接复用。重建环境时，保持基础 Torch 栈，单独安装 `timm==0.9.12`、`plyfile==1.1.3`、`scipy==1.15.3`、`tqdm==4.70.1`，并在同一 FSGS 环境中编译两个 CUDA 扩展。下面是准备新源码的命令，变量指向已设置的位置：

```bash
git clone --recursive https://github.com/VITA-Group/FSGS.git "$NEW_FSGS_SOURCE"
git -C "$NEW_FSGS_SOURCE" checkout a536a64c5b366b1088be64eeadf9e791ca26897c
git -C "$NEW_FSGS_SOURCE" submodule update --init --recursive
patch -d "$NEW_FSGS_SOURCE" -p1 < "$PROJECT/patches/fsgs-cuda12.patch"
CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST=8.9 "$FSGS_PYTHON" -m pip install --no-build-isolation \
  "$NEW_FSGS_SOURCE/submodules/diff-gaussian-rasterization-confidence" \
  "$NEW_FSGS_SOURCE/submodules/simple-knn"
```

`8.9` 对应本次使用的 RTX 4070 SUPER；其他显卡需使用对应架构。补丁只补 `<cstdint>` 并移除两个未使用的 Open3D 导入，与已验证运行源码一致。像素采样修正和本地 MiDaS 加载由项目训练适配器完成。[FSGS](https://github.com/VITA-Group/FSGS/tree/a536a64c5b366b1088be64eeadf9e791ca26897c)、MiDaS 及其 CUDA 子模块保留各自许可，使用及分发须遵循原许可。
