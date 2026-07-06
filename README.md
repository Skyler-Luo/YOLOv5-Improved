# YOLOv5-Improved

基于 [YOLOv5](https://github.com/ultralytics/yolov5) 的模型改进与压缩研究框架。通过集成多种注意力机制、轻量化模块设计与结构化剪枝技术，在保持高检测精度的同时显著降低模型参数量与计算量。

## ✨ 特性总览

### 1. 多种注意力机制即插即用

项目内置多种主流注意力模块，均已在 `parse_model` 中注册，支持在 YAML 配置中一行引用：

| 类别 | 模块 |
|------|------|
| **无参注意力** | `SimAM`, `TripletAttention`, `SpatialGroupEnhance` |
| **通道/空间注意力** | `SEAttention`, `CBAMBlock`, `BAMBlock`, `ECA`, `CoordAtt`, `EMA`, `GAM_Attention`, `ELA`, `MHSA`, `ParNetAttention`, `ParallelPolarizedSelfAttention`, `ShuffleAttention`, `S2Attention`, `SKAttention`, `DoubleAttention`, `CoTAttention`, `LSKblock`, `LSKA`, `MLCA`, `CPCA`, `CAA` |
| **Transformer** | `GlobalContext`, `EffectiveSEModule`, `GatherExcite`, `DAttention`, `BiLevelRoutingAttention`, `EfficientAttention`, `MobileViTAttention` |

### 2. 轻量化核心组件

| 模块 | 说明 |
|------|------|
| `C3Ghost` | Ghost 瓶颈 C3 模块，通过 cheap operations 减少冗余特征 |
| `C3_DSConv` | 深度可分离卷积 C3 模块，大幅降低计算量 |
| `GSConvns` / `VoVGSCSP` | Slim-Neck 轻量化 Neck 模块 |
| `CARAFE` | 内容感知特征重组上采样算子，替代传统插值 |
| `Decoupled_Detect` | 解耦检测头，分离分类/回归分支，加速收敛 |

### 3. 模型剪枝与压缩

基于 `torch_pruning` 的结构化剪枝流程，支持：
- 灵敏度分析（`prune/sensitivity.py`）
- 依赖图自动构建（`prune/dependency.py`）
- 多梯度剪枝率对比实验

## 🚀 快速开始

### 🔧 环境安装

```bash
# 克隆仓库
git https://github.com/Skyler-Luo/YOLOv5-Improved.git
cd YOLOv5-Improved

# 安装依赖
pip install -r requirements.txt
```

### 📁 数据集准备

将数据集放在项目根目录下，目录结构示例：

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## 🏋️ 训练

### Baseline 训练（标准 YOLOv5n）

```bash
python train.py --weights '' --cfg models/yolov5n.yaml \
    --data dataset/data.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --cache --name yolov5n \
    --batch-size 64 --workers 8 --epochs 200
```

### 轻量化模型训练（C3Ghost + GSConv Neck）

```bash
python train.py --weights '' --cfg models/yolov5n-light.yaml \
    --data dataset/data.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --cache --name yolov5n_light \
    --batch-size 64 --workers 8 --epochs 200
```

## ✂️ 模型剪枝

基于训练好的权重进行结构化剪枝与微调：

```bash
# 轻度剪枝（rate=0.06）
python compress.py --model yolov5n --dataset VOC \
    --data dataset/data.yaml --batch 64 --epochs 100 \
    --weights runs/train/yolov5n_light/weights/best.pt \
    --workers 8 --initial_rate 0.06 --initial_thres 6. --topk 0.8 \
    --exp --cache --name yolov5n_light_prune

# 中度剪枝（rate=0.2）
python compress.py --model yolov5n --dataset VOC \
    --data dataset/data.yaml --batch 64 --epochs 100 \
    --weights runs/train/yolov5n_light/weights/best.pt \
    --workers 8 --initial_rate 0.2 --initial_thres 10. --topk 0.8 \
    --exp --cache --name yolov5n_light_prune2

# 重度剪枝（rate=0.4）
python compress.py --model yolov5n --dataset VOC \
    --data dataset/data.yaml --batch 64 --epochs 100 \
    --weights runs/train/yolov5n_light/weights/best.pt \
    --workers 8 --initial_rate 0.4 --initial_thres 20. --topk 0.8 \
    --exp --cache --name yolov5n_light_prune3
```

## 📊 测试评估

```bash
# Baseline 模型
python val.py --data dataset/data.yaml \
    --weights runs/train/yolov5n/weights/best.pt \
    --task test --name yolov5n --exist-ok --device 0

# 轻量化模型
python val.py --data dataset/data.yaml \
    --weights runs/train/yolov5n_light/weights/best.pt \
    --task test --name yolov5n_light --exist-ok --device 0

# 剪枝后模型
python val.py --data dataset/data.yaml \
    --weights runs/train/yolov5n_light_prune/weights/best.pt \
    --task test --name yolov5n_light_prune --exist-ok --device 0
```

### 推理检测

```bash
python detect.py --weights runs/train/yolov5n_light/weights/best.pt \
    --source data/images --img 640 --device 0
```

### 模型导出

```bash
python export.py --weights runs/train/yolov5n_light/weights/best.pt \
    --include onnx --img 640
```

## 🛠️ 自定义模型配置

通过 YAML 文件灵活组合改进模块，示例（`models/yolov5n-improve.yaml`）：

```yaml
backbone:
  [[-1, 1, Conv, [64, 6, 2, 2]],
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 3, C3_DSConv, [128]],      # 深度可分离卷积 C3
   [-1, 1, Conv, [256, 3, 2]],
   [-1, 6, C3_DSConv, [256]],
   [-1, 1, Conv, [512, 3, 2]],
   [-1, 9, C3_DSConv, [512]],
   [-1, 1, SimAM, []],             # 无参注意力机制
   [-1, 1, SPPF, [1024, 5]],
  ]

head:
  [[-1, 1, Conv, [256, 1, 1]],
   [-1, 1, CARAFE, [2]],           # 内容感知上采样
   [[-1, 6], 1, Concat, [1]],
   [-1, 3, C3, [256, False]],
   ...
   [[16, 19, 22], 1, Decoupled_Detect, [nc, anchors]],  # 解耦检测头
  ]
```

## 📂 项目结构

```
YOLOv5-Improved/
├── train.py                # 训练入口
├── val.py                  # 验证/测试入口
├── detect.py               # 推理检测入口
├── export.py               # 模型导出（ONNX/TensorRT 等）
├── compress.py             # 剪枝压缩入口
├── requirements.txt        # 依赖列表
│
├── models/
│   ├── common.py           # 通用模块（Conv, C3, C3Ghost, GSConv, CARAFE 等）
│   ├── yolo.py             # 模型构建与解析（Detect, Decoupled_Detect, parse_model）
│   ├── attention/          # 多种注意力机制
│   ├── backbones/          # 多种可替换骨干网络
│   ├── gfpn/               # GFPN Neck 模块
│   ├── goldyolo/           # GoldYOLO Neck 模块
│   ├── aux_head/           # 辅助训练头
│   ├── yolov5n.yaml        # Baseline 配置
│   ├── yolov5n-light.yaml  # 轻量化配置（C3Ghost + GSConv Neck）
│   └── yolov5n-improve.yaml# 改进配置（C3_DSConv + SimAM + CARAFE + 解耦头）
│
├── utils/                  # 工具模块（数据加载、损失函数、指标计算等）
├── prune/                  # 剪枝模块（灵敏度分析、依赖图构建）
├── tools/
│   ├── eval/               # 评估工具（热力图、FPS、误差分析、曲线对比等）
│   └── data/               # 数据处理工具（格式转换、数据集切分、数据增强）
│
├── runs/                   # 训练/验证输出
└── logs/                   # 训练日志
```

## 📄 许可证

本项目基于 AGPL-3.0 许可证开源，详见 [LICENSE](LICENSE) 文件。