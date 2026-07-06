# YOLOv5-Lite

本项目旨在通过剪枝、蒸馏、轻量化网络设计等技术，对 YOLOv5 进行模型压缩，实现在保持高精度的同时大幅降低模型大小和计算量。

## ✨ 新增特性

### 1. 32 种注意力机制集成
项目内置集成了主流的 32 种经典与前沿注意力模块，支持在 YAML 配置文件中**直接引用**：
- **无参/特征保护**：`SimAM`, `TripletAttention`, `SpatialGroupEnhance (SGE)`
- **通道/空间注意力**：`SEAttention`, `CBAMBlock`, `BAMBlock`, `EfficientChannelAttention (ECA)`, `CoordAtt`, `EMA`, `GAM_Attention`, `ELA`, `MHSA`, `ParNetAttention`, `ParallelPolarizedSelfAttention`, `ShuffleAttention`, `S2Attention`, `SKAttention`, `DoubleAttention`, `CoTAttention`, `LSKblock`, `LSKA`, `MLCA`, `CPCA`, `CAA`
- **高阶/多模态/Transformer注意力（需额外依赖）**：`GlobalContext`, `EffectiveSEModule`, `GatherExcite`, `DAttention`, `BiLevelRoutingAttention`, `EfficientAttention`, `MobileViTAttention`

### 2. 缺失核心轻量化组件实现
- **`C3_DSConv`**：使用深度可分离卷积（Depthwise Separable Convolution）重构的 C3 瓶颈层，显著降低计算量。
- **`CARAFE`**：内容感知特征重组上采样算子，相较于传统的 Nearest/Bilinear 插值，能更自适应地重构高分辨率特征。
- **`Decoupled_Detect`**：解耦检测头。将分类和回归分支解耦，加快模型收敛并提升检测精度。

---

## 🚀 快速开始

### 🔧 安装依赖

```bash
# 克隆仓库
git clone https://github.com/ppogg/YOLOv5-Lite.git
cd YOLOv5-Lite

# 安装基础及注意力依赖
pip install -r requirements.txt
```

### 🛠️ 配置文件示例 (`yolov5n-improve.yaml`)

你可以通过组合新添加的注意力模块和上采样算子，设计更强大的轻量化网络：

```yaml
# YOLOv5-Lite 改进模型配置示例
backbone:
  [[-1, 1, Conv, [64, 6, 2, 2]],
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 3, C3_DSConv, [128]],    # 使用深度可分离卷积的 C3
   [-1, 1, Conv, [256, 3, 2]],
   [-1, 6, C3_DSConv, [256]],
   [-1, 1, Conv, [512, 3, 2]],
   [-1, 9, C3_DSConv, [512]],
   [-1, 1, SimAM, []],           # 引入无参注意力机制
   [-1, 1, SPPF, [1024, 5]],
  ]

head:
  [[-1, 1, Conv, [256, 1, 1]],
   [-1, 1, CARAFE, [2]],         # 使用 CARAFE 上采样算子
   [[-1, 6], 1, Concat, [1]],
   [-1, 3, C3, [256, False]],
   ...
   [[16, 19, 22], 1, Decoupled_Detect, [nc, anchors]], # 使用解耦检测头
  ]
```

## 📄 许可证

本项目基于 AGPL-3.0 许可证开源，详见 [LICENSE](LICENSE) 文件。