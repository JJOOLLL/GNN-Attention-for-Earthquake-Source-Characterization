# 图神经网络地震震源表征中的注意力作用

本项目研究注意力在台站间消息传递和图级读出阶段的作用，预测经纬度、深度及震级，包含南加州和阿拉斯加实验。

## 六种结构

| 名称 | 台站间消息传递 | 图级读出 |
|---|---|---|
| `transformer_attention` | Graph Transformer | 注意力池化 |
| `transformer_max` | Graph Transformer | 最大池化 |
| `no_message_attention` | 无 | 注意力池化 |
| `no_message_max` | 无 | 最大池化 |
| `gcn_attention` | GCN | 注意力池化 |
| `gcn_max` | GCN | 最大池化 |

名称直接说明结构，不代表精度排名。旧编号只保留在[对应表](experiments/model_names.csv)。**本包不含 TEAM/STGNN 源码及运行入口，但保留论文中的对比结果与实验参数。**

## 文件在哪里

```text
models/              六种模型的实现
experiments/         分地区训练笔记本、配置和结果检查
data_preparation/    南加州下载脚本与数据准备说明
results/             论文指标、预测、固定划分与台站选择
run.py               单个实验入口，默认不训练
```

## 怎么运行

原实验使用 Python 3.12.8、PyTorch 2.5.1。先安装适合本机的 PyTorch CPU/CUDA 版本，再执行：

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gnn-paper --display-name "GNN paper"
python experiments/verify_results.py
python run.py --model transformer_max --region California
```

最后一条只显示参数，不启动训练。取得原始处理后数据后，如需训练：

```bash
python run.py --model transformer_max --region California --data-root /path/to/data --kernel gnn-paper --execute
```

Windows 路径也支持。数据根目录下应有 `data_DA/` 和 `data_ANCHORAGE_DA/`，详见[数据准备](data_preparation/README.md)。手动运行 notebook 时设置 `GNN_DATA_ROOT`。权重默认保存至 `save/地区/模型名/`，可用 `GNN_SAVE_ROOT` 改根目录；不会覆盖已有权重。原代码将增强波形放入内存，完整训练可能需要数十 GB 主存。

## 实验约定

随机种子为42，训练/验证/测试互斥，按最低验证损失选模。南加州三集合事件数为2026/253/254，阿拉斯加为1764/220/222。

阿拉斯加 `no_message_attention` 最终结果采用一次预先指定的训练稳定性修复，初始学习率1e-4；其他受控配置为1e-3，第二阶段均为2e-5。不能把这一对比中的全部改善归因于池化结构。

`results/metrics.csv` 保存64行最终指标，含两个外部模型的对比结果；六种自有配置的逐事件预测一并提供。位置MAE/MSE单位为km/km²，震级不换算；标准差是事件间差异，不是多种子不确定性；R²分母使用真实值均值。

`experiments/cross_region.ipynb` 只保留六种受控模型。跨区采用目标区域归一化尺度与完整可用目录。已有阿拉斯加 `no_message_attention` 跨区结果仍对应修复前权重，在 `experiments/weights.json` 中单独标明，不与最终区内结果混用。

## 尚未包含的材料

原始波形、最佳权重文件及权重下载地址尚未提供；阿拉斯加原始下载/预处理流程仍待补。权重哈希已整理。重新下载的目录可能不同于原始快照，因此训练入口检查目录文件哈希和固定划分；尚未在干净环境从头完成完整复现。

原MIT文本保留。不包含TEAM/STGNN实现，也不因移除它们而宣称所有基础代码均为原创；复用部分应保留适用的来源与署名。
