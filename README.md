# EuroSAT MLP 作业代码（NumPy + 自定义自动微分）

这是一个按作业要求实现的从零开始版本：

- **不使用 PyTorch / TensorFlow / JAX**；
- 使用 **NumPy** 做矩阵运算；
- 手写 **自动微分与反向传播**；
- 支持 **SGD、学习率衰减、交叉熵、L2 正则**；
- 支持 **训练 / 验证 / 测试划分、最佳模型保存、超参数搜索、混淆矩阵、权重可视化、错例分析**。

> 按作业口径，默认推荐使用 **三层 MLP：输入层 -> 1 个隐藏层 -> 输出层**。  
> 代码里 `--hidden-dims` 也支持写成 `512,256` 这种多隐藏层形式，方便你自己做额外实验；但如果你想和题目保持最一致，直接用单个隐藏层即可，比如 `--hidden-dims 512`。

---

## 1. 环境依赖

建议 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

本项目依赖非常少：

- numpy
- pillow
- matplotlib

---

## 2. 数据集目录结构

你的数据集根目录应该类似这样：

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

每个类别文件夹里直接放图片即可。

---

## 3. 训练

### 3.1 最常用训练命令

```bash
python train.py \
  --data-root /home/zhangzhiwei/deeplearning/homework1/EuroSAT_RGB \
  --output-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_run_1 \
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

### 3.2 如果你想额外试多隐藏层

```bash
python train.py \
  --data-root /home/zhangzhiwei/deeplearning/homework1/EuroSAT_RGB \
  --output-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_deeper \
  --hidden-dims 512,256 \
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

### 3.3 训练输出内容

训练结束后，`output-dir` 下会有：

```text
outputs/train_run_1/
├── accuracy_curve.png
├── loss_curve.png
├── lr_curve.png
├── history.csv
├── history.json
├── experiment_config.json
├── split.json
├── training_summary.json
├── best/
│   ├── meta.json
│   └── weights.npz
└── last/
    ├── meta.json
    └── weights.npz
```

其中：

- `best/`：验证集准确率最高时保存的模型；
- `split.json`：训练/验证/测试划分，后续评估时要复用它；
- `loss_curve.png`：训练集/验证集 loss 曲线；
- `accuracy_curve.png`：准确率曲线；
- `history.csv`：每个 epoch 的日志。

---

## 4. 测试集评估

使用训练好的最优模型在测试集上评估：

```bash
python evaluate.py \
  --data-root /home/zhangzhiwei/deeplearning/homework1/EuroSAT_RGB \
  --checkpoint-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_run_1/best \
  --split-file /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/train_run_1/split.json \
  --output-dir /home/zhangzhiwei/deeplearning/homework1/eurosat_mlp_hw/output/eval_run_1
```

评估输出：

```text
outputs/eval_run_1/
├── confusion_matrix.png
├── misclassified_examples.png
├── predictions.csv
└── test_metrics.json
```

其中：

- `confusion_matrix.png`：混淆矩阵图；
- `misclassified_examples.png`：测试集错例图；
- `predictions.csv`：每张测试图像的真实类别/预测类别；
- `test_metrics.json`：测试集 accuracy、每类 accuracy、混淆矩阵等。

---

## 5. 第一层权重可视化

```bash
python visualize.py \
  --checkpoint-dir outputs/train_run_1/best \
  --output-dir outputs/vis_run_1 \
  --max-filters 64 \
  --cols 8
```

输出文件：

```text
outputs/vis_run_1/
└── first_layer_weights.png
```

这个图就是你写报告时“第一层隐藏层权重恢复成图像后”的可视化结果。

---

## 6. 超参数搜索

### 6.1 网格搜索

```bash
python search.py \
  --data-root /path/to/EuroSAT_RGB \
  --output-dir outputs/search_grid \
  --search-config configs/grid_search.json \
  --seed 42
```

### 6.2 随机搜索

```bash
python search.py \
  --data-root /path/to/EuroSAT_RGB \
  --output-dir outputs/search_random \
  --search-config configs/random_search.json \
  --seed 42
```

### 6.3 搜索结果

搜索目录里会保存：

```text
outputs/search_grid/
├── best_trial.json
├── dataset_info.json
├── search_results.csv
├── search_results.json
├── split.json
└── trial_xxx_.../
```

`search_results.csv` 里会记录不同超参数组合的验证集性能，适合直接拿去写实验对比。

---

## 7. 主要参数说明

### train.py

- `--hidden-dims`：隐藏层大小，例：`512` 或 `512,256`
- `--activation`：`relu` / `sigmoid` / `tanh`
- `--dropout`：隐藏层 dropout 概率（仅训练时生效）
- `--learning-rate`：初始学习率
- `--momentum`：SGD 动量系数
- `--lr-decay`：每个 epoch 后乘上的衰减系数
- `--weight-decay`：L2 正则系数
- `--max-grad-norm`：全局梯度裁剪阈值，防止梯度爆炸
- `--disable-train-augmentation`：关闭训练集数据增强（默认开启翻转+90度旋转）
- `--early-stopping-patience`：验证集准确率无提升时提前停止训练
- `--batch-size`：批大小
- `--epochs`：训练轮数
- `--cache-images`：将图像缓存到内存，训练更快，但更吃 RAM

### search.py

- `--search-config`：搜索配置 JSON
- `--max-trials`：只跑前 N 个 trial，适合调试

### evaluate.py

- `--max-error-examples`：错例分析图里最多显示多少张

---

## 8. Linux 上怎么跑

假设你的工程目录叫 `eurosat_mlp_hw`，数据集在 `/data/EuroSAT_RGB`：

```bash
cd eurosat_mlp_hw
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

然后训练：

```bash
python train.py \
  --data-root /data/EuroSAT_RGB \
  --output-dir outputs/train_relu_512 \
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

训练完评估：

```bash
python evaluate.py \
  --data-root /data/EuroSAT_RGB \
  --checkpoint-dir outputs/train_relu_512/best \
  --split-file outputs/train_relu_512/split.json \
  --output-dir outputs/eval_relu_512
```

再做权重可视化：

```bash
python visualize.py \
  --checkpoint-dir outputs/train_relu_512/best \
  --output-dir outputs/vis_relu_512
```

---

## 9. GPU 说明

这份实现是 **纯 NumPy**，默认走 **CPU**。

也就是说：

- 你有 GPU 也没关系；
- 这份代码本身 **不依赖 GPU**；
- 不需要装 CUDA，也不需要装 PyTorch。

这样做的好处是和作业要求最一致，环境也最稳。  
如果你的 CPU 还可以，EuroSAT 这个作业是能跑起来的。为了提速，你可以：

- 打开 `--cache-images`
- 先用较小 epoch 做超参数搜索
- 搜到较优参数后再正式训练更久一点

---

## 10. 代码结构

```text
.
├── train.py
├── search.py
├── evaluate.py
├── visualize.py
├── configs/
│   ├── grid_search.json
│   └── random_search.json
├── src/
│   ├── autograd.py
│   ├── checkpoint.py
│   ├── data.py
│   ├── metrics.py
│   ├── nn.py
│   ├── optim.py
│   ├── searcher.py
│   ├── trainer.py
│   ├── utils.py
│   └── visualization.py
└── tools/
    └── make_dummy_dataset.py
```

模块对应作业要求：

- 数据加载与预处理：`src/data.py`
- 模型定义：`src/nn.py`
- 自动微分与反向传播：`src/autograd.py`
- 训练循环：`src/trainer.py`
- 测试评估：`evaluate.py` + `src/metrics.py`
- 超参数查找：`search.py` + `src/searcher.py`

---

## 11. 一个最小自测方法（可选）

如果你只是想先验证环境能不能跑通，可以生成一个假数据集：

```bash
python tools/make_dummy_dataset.py --output-dir ./dummy_eurosat
```

然后：

```bash
python train.py --data-root ./dummy_eurosat --output-dir ./outputs/dummy_run --epochs 2 --hidden-dims 64
```

这只是用于检查代码通不通，不代表真实实验结果。

---

## 12. 交作业建议

你最后整理报告时，建议至少把这些内容放进去：

1. 模型结构与自动微分实现思路；
2. 数据划分与预处理方式；
3. 训练/验证 loss 曲线；
4. 验证集 accuracy 曲线；
5. 超参数搜索表格；
6. 测试集 accuracy 与 confusion matrix；
7. 第一层权重可视化；
8. 错例分析；
9. GitHub Repo 链接；
10. 模型权重下载链接。
