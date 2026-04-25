# EuroSAT MLP 图像分类实验

本仓库是我在深度学习课程作业中完成的 EuroSAT 遥感图像分类实验。作业要求不使用 PyTorch、TensorFlow、JAX 等深度学习框架，因此我主要使用 NumPy 从较底层实现了多层感知机、自动微分、反向传播、优化器、训练流程和测试评估流程。

这个项目的目标不是追求一个很复杂的深度网络，而是通过一个相对完整的 MLP 分类器，把前向传播、损失函数、反向传播、参数更新、正则化、学习率衰减、模型保存和实验分析这些基本环节真正串起来。代码和报告中的实验结果都基于 EuroSAT RGB 数据集完成。

## 1. 实验内容概述

本次实验完成了以下几部分工作：

- 使用 EuroSAT RGB 数据集进行 10 类遥感场景分类。
- 使用 NumPy 实现全连接神经网络，不依赖深度学习框架。
- 自己实现简单的自动微分机制，并通过计算图完成反向传播。
- 实现线性层、ReLU、Sigmoid、Tanh、Dropout、Softmax Cross Entropy 等模块。
- 使用 SGD with Momentum 进行优化，并加入学习率衰减和 L2 正则化。
- 将数据集划分为训练集、验证集和测试集，使用验证集选择最佳模型。
- 在测试集上输出 Accuracy、Confusion Matrix 和错分样例。
- 对第一层权重进行可视化，观察 MLP 从原始图像中学习到的低层模式。
- 进行网格超参数搜索，对比学习率、隐藏层维度、激活函数和正则化强度的影响。

## 2. 仓库结构

```text
.
├── train.py                         # 训练入口，保存 best/last checkpoint 和训练曲线
├── evaluate.py                      # 测试集评估入口，输出测试指标、混淆矩阵和错例图
├── visualize.py                     # 第一层权重可视化入口
├── search.py                        # 超参数搜索入口
├── configs/
│   ├── grid_search.json             # 网格搜索配置
│   └── random_search.json           # 随机搜索配置
├── src/
│   ├── autograd.py                  # 自动微分和计算图
│   ├── nn.py                        # 网络层、激活函数、损失函数等
│   ├── optim.py                     # SGD、Momentum、学习率衰减、L2 正则
│   ├── data.py                      # EuroSAT 数据读取、划分、增强和 batch 生成
│   ├── trainer.py                   # 训练循环、验证、早停和模型保存
│   ├── metrics.py                   # Accuracy、混淆矩阵等指标
│   ├── checkpoint.py                # 权重保存与加载
│   ├── searcher.py                  # 超参数搜索流程
│   ├── visualization.py             # 训练曲线、混淆矩阵、权重和错例可视化
│   └── utils.py                     # 随机种子、JSON/CSV 保存等工具函数
├── output/train_run_final/
│   ├── best/weights.npz             # 验证集表现最好的模型权重
│   ├── best/meta.json               # best checkpoint 的模型和训练信息
│   ├── last/weights.npz             # 最后一轮模型权重
│   └── last/meta.json               # last checkpoint 的模型和训练信息
├── baogao/
│   ├── main.tex                     # 作业报告 LaTeX 源文件
│   ├── figures/                     # 报告中使用的实验图片
│   └── data/                        # 报告中引用的实验数据
├── tools/make_dummy_dataset.py      # 小规模假数据集生成脚本，用于冒烟测试
├── requirements.txt
└── README.md
```

说明：完整训练过程产生的中间输出比较多，`.gitignore` 默认忽略 `output/`，但我已经把最终 `best` 和 `last` 两份权重文件单独提交到仓库中，便于复现实验结果。

## 3. 环境配置

我实验时使用的是 Python 3.x 环境，主要依赖如下：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中只包含：

```text
numpy
Pillow
matplotlib
```

本项目不需要安装 PyTorch、TensorFlow 或 JAX。

## 4. 数据集准备

代码默认读取 EuroSAT RGB 原始图片目录。我的本地数据集路径为：

```text
/home/zhangzhiwei/deeplearning/homework1/EuroSAT_RGB
```

目录结构需要保持为按类别分文件夹的形式：

```text
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

如果在其他机器上运行，只需要把命令中的 `--data-root` 改成自己的 EuroSAT RGB 数据集路径即可。

## 5. 复现实验命令

下面这些命令是我最终实验使用的主要流程。建议在仓库根目录下运行。

### 5.1 训练模型

```bash
python train.py \
  --data-root /home/zhangzhiwei/deeplearning/homework1/EuroSAT_RGB \
  --output-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_run_final \
  --hidden-dims 512 \
  --activation relu \
  --dropout 0.2 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --lr-decay 0.98 \
  --weight-decay 0.0005 \
  --max-grad-norm 5.0 \
  --early-stopping-patience 6 \
  --batch-size 128 \
  --epochs 30 \
  --seed 42
```

训练结束后会保存：

```text
output/train_run_final/best/weights.npz
output/train_run_final/last/weights.npz
output/train_run_final/history.csv
output/train_run_final/loss_curve.png
output/train_run_final/accuracy_curve.png
output/train_run_final/training_summary.json
```

### 5.2 测试集评估

```bash
python evaluate.py \
  --data-root /home/zhangzhiwei/deeplearning/homework1/EuroSAT_RGB \
  --checkpoint-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_run_final/best \
  --split-file /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_run_final/split.json \
  --output-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/eval_run_final
```

评估结果会保存到：

```text
output/eval_run_final/test_metrics.json
output/eval_run_final/confusion_matrix.png
output/eval_run_final/misclassified_examples.png
output/eval_run_final/predictions.csv
```

### 5.3 第一层权重可视化

```bash
python visualize.py \
  --checkpoint-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_run_final/best \
  --output-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/vis_run_final \
  --max-filters 64 \
  --cols 8
```

输出文件为：

```text
output/vis_run_final/first_layer_weights.png
```

### 5.4 超参数搜索

```bash
python search.py \
  --data-root /home/zhangzhiwei/deeplearning/homework1/EuroSAT_RGB \
  --output-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/search_grid_final \
  --search-config /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/configs/grid_search.json \
  --seed 42
```

搜索结果主要保存在：

```text
output/search_grid_final/search_results.csv
output/search_grid_final/best_trial.json
```

## 6. 最终实验结果

最终模型使用的主要配置如下：

```text
输入维度: 64 x 64 x 3 = 12288
隐藏层: 512
激活函数: ReLU
Dropout: 0.2
优化器: SGD with Momentum
初始学习率: 0.01
Momentum: 0.9
学习率衰减: 0.98
L2 正则系数: 0.0005
Batch size: 128
随机种子: 42
数据划分: train 70%, val 15%, test 15%
```

主要结果如下：

```text
最佳验证集准确率: 67.01%
最佳 epoch: 20
测试集准确率: 66.67%
测试集样本数: 4050
测试集 loss: 0.9516
```

每个类别在测试集上的准确率如下：

| 类别 | Accuracy |
| --- | ---: |
| AnnualCrop | 55.33% |
| Forest | 81.56% |
| HerbaceousVegetation | 68.44% |
| Highway | 35.47% |
| Industrial | 85.60% |
| Pasture | 82.67% |
| PermanentCrop | 52.80% |
| Residential | 78.22% |
| River | 53.60% |
| SeaLake | 71.78% |

从结果上看，Industrial、Pasture、Forest 和 Residential 的识别效果相对较好；Highway、PermanentCrop、River 等类别更容易和外观相近的类别混淆。这也符合我在错分样例和混淆矩阵中观察到的现象：只使用 MLP 对图像拉平成向量后分类，空间结构信息保留得不充分，因此对于纹理和布局相近的遥感场景会比较吃力。

## 7. 超参数搜索结果

我使用 `configs/grid_search.json` 做了网格搜索，主要比较了学习率、隐藏层维度、激活函数和 L2 正则强度。搜索范围为：

```text
learning_rate: 0.1, 0.05, 0.01
hidden_dims: [256], [512]
activation: relu, tanh
weight_decay: 0.0, 0.0005
```

搜索中表现最好的组合为：

```text
hidden_dims: [512]
activation: relu
learning_rate: 0.01
weight_decay: 0.0
best_val_accuracy: 65.43%
best_epoch: 19
```

最终正式训练时，我在这个结果的基础上保留了 `hidden_dims=512`、`activation=relu`、`learning_rate=0.01`，并使用 `weight_decay=0.0005` 做轻微正则化，主要是为了在最终训练中增强一点泛化约束。

## 8. 报告文件

作业报告已经整理在 `baogao/` 目录中：

```text
baogao/main.tex
baogao/figures/
baogao/data/
```

可以将整个 `baogao/` 文件夹上传到 Overleaf，然后编译 `main.tex`。报告中使用的训练曲线、混淆矩阵、错分样例和第一层权重可视化图片都已经放在 `baogao/figures/` 中。

## 9. 已提交的模型权重

仓库中已经包含最终训练的两份权重：

```text
output/train_run_final/best/weights.npz
output/train_run_final/last/weights.npz
```

其中 `best` 是验证集准确率最高时保存的模型，通常用于测试集评估和报告结果；`last` 是训练停止时最后一轮模型，主要用于对照和备份。

## 10. 简单自检命令

如果只想确认脚本入口是否可用，可以运行：

```bash
python train.py --help
python evaluate.py --help
python visualize.py --help
python search.py --help
```

如果没有真实数据集，也可以先用 `tools/make_dummy_dataset.py` 生成一个很小的假数据集，检查训练、评估和可视化流程是否能跑通。

## 11. 我的实验理解

这次实验对我来说比较重要的一点是，MLP 本身结构并不复杂，但是把训练系统完整写出来并不只是搭几层线性层。数据读取、归一化、随机划分、mini-batch、前向传播、loss 计算、反向传播、优化器更新、学习率衰减、早停、checkpoint 保存、评估和可视化都需要互相配合。

从最终结果看，MLP 在 EuroSAT 上可以学到一定的类别区分能力，但准确率明显受到模型结构限制。遥感图像中很多类别的差异依赖局部纹理和空间布局，而 MLP 会直接把图像展平成一维向量，天然缺少卷积网络那种局部感受野和平移不变性。因此我认为这个结果是合理的：它能作为理解反向传播和训练流程的基础模型，但如果追求更高精度，更适合使用 CNN 或其他能够建模空间结构的网络。
