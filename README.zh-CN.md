# 注意力启发的地震震源表征图构建

[English](README.md)

论文 **Attention-Informed Graph Construction for Earthquake Source Characterization** 的配套代码。论文已投稿，尚未发表。

本研究比较多台站图神经网络的消息传递和读出方式，并根据注意力分析中的远距离台站交互规律，采用最大距离中位数准则替换地理 kNN 选邻居，无需预先知道震源位置。

## 六种配置

| 运行名称 | 消息传递 | 读出 |
|---|---|---|
| `no_message_max` | 无 | 最大池化（GNN 基线） |
| `no_message_attention` | 无 | 注意力池化 |
| `gcn_max` | GCN | 最大池化 |
| `gcn_attention` | GCN | 注意力池化 |
| `transformer_max` | Graph Transformer | 最大池化 |
| `transformer_attention` | Graph Transformer | 注意力池化 |

各模型预测纬度、经度、深度和震级。GT 使用全连接图；GCN 支持原 kNN 和最大距离中位数建图。框架基于 van den Ende & Ampuero (2020)，来源见 [REFERENCES.md](REFERENCES.md)。

## 文件用途

- `models/`：六种模型。
- `graphs/`：最大距离中位数精确求解及 GCN 接入，见[建图说明](graphs/README.md)。
- `training/`：最终 GCN 训练器；`source_cells/` 是其实际使用的数据准备和评估代码。
- `experiments/`：其他四种配置的区域训练 notebook、跨区评估及设置。
- `data_preparation/`：两地区下载预处理；`reference_data.json` 合并保存输入哈希、训练/验证/测试划分和测试台站选择。

## 运行

原环境为 Python 3.12.8、PyTorch 2.5.1、PyG 2.6.1。先安装适合设备的 PyTorch，再安装依赖：

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name gnn-paper --display-name "Python (gnn-paper)"
```

默认仅展示配置，不启动训练：

```bash
python run.py --model transformer_max --region California
python run.py --model gcn_max --region Alaska --graph median
```

按[数据准备说明](data_preparation/README.md)放置输入后，显式添加 `--execute` 训练：

```bash
python run.py --model transformer_max --region California --data-root /path/to/data --execute
python run.py --model gcn_max --region Alaska --graph knn --data-root /path/to/data --execute
python run.py --model gcn_max --region Alaska --graph median --data-root /path/to/data --execute
```

GCN 两种池化直接共用上述入口，不另设重复 notebook。`--save-root` 指定输出位置，GCN 续跑使用 `--resume`；其他参数见 `python run.py --help`。完整 GCN 训练需要 CUDA 和超过 40 GiB 的当前可用系统内存。

GCN 保留最终先 SiLU、后 ReLU 的训练设置，验证集选择最终 ReLU 权重；GT 和无消息传递配置保留原设置。详细参数见 `experiments/settings.json`。跨区 notebook 只评估 GT＋注意力池化，原权重路径及校验值见 `experiments/weights.json`。

## 范围与许可

本精简包不附逐事件预测、历史结果或复核报告；也不包含原始波形和训练权重。STGNN、TEAM-LM 第三方模型源码不分发，其实验设置保存在 `experiments/baseline_settings.json`。

保留原 [MIT 许可证](LICENSE)，第三方依赖遵循各自许可。使用本工作时请引用上述论文；发表后更新正式出版信息。
