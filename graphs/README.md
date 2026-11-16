# Maximum-median-distance graphs / 最大距离中位数建图

## Definition / 定义

For a target station i, choose k neighbours from the other valid stations. A candidate set S includes i and those neighbours. Its score is the median of all distances d(u,v), for distinct u,v in S. Select the highest-scoring set. An even number of distances uses the mean of the central two values. No source coordinates or source labels are used.

对目标台站 i，从其他有效台站中选 k 个邻居。候选集合 S 包括目标自身和邻居，以 S 内所有不同台站对距离的中位数评分，选择评分最大的集合。距离数量为偶数时取中间两项均值。建图不使用震源坐标或标签。

## Files / 文件

- `exact_gpu.py`: chunked exhaustive subset search on CPU/CUDA, with a small CPU oracle. / CPU或CUDA分块穷举，含小规模对照实现。
- `graph_adapter.py`: coordinate conversion, geometry caching, and GCN edge replacement. / 坐标换算、几何缓存及GCN替换边的接入。

Each subset is scored once and can update every target it contains. Scores use float32; equal scores choose the smallest colex subset rank after the adapter's geometry ordering. The solver retains a minimum-distance output internally, but the training CLI exposes only `knn` and `median`.

每个集合只评分一次，可更新其所有成员的候选最优集合。评分为 float32；并列时在适配器的几何排序后，按最小 colex 子集序号确定选择。求解器内部仍计算最小距离评分，但训练入口只开放 `knn` 和 `median`。

## GCN protocol / GCN 接入

- k=5; the target plus five neighbours gives six stations and 15 distinct distances. Fewer valid stations use all remaining stations. / 目标加5邻居组成6站集合，共15个距离；不足时使用全部其他有效站。
- kNN keeps the original Euclidean distance in normalized input coordinates. Median selection converts coordinates back to latitude/longitude and uses Haversine distances. / kNN保留原归一化输入坐标欧氏距离；中位数方案逆归一化为经纬度后使用球面距离。
- The adapter uses the coordinates actually supplied to the network, including augmented coordinates; padding is excluded. Reverse edges are added as in the original GCN protocol, without deduplicating repeated edges. / 适配器读取实际输入含增强后的坐标，排除填充站；按原GCN流程补反向边，不额外去重。
- Cached graphs do not change the input station set, model parameters, or readout. / 缓存及替换边不改变输入台站集合、模型参数或读出方式。

Run the Alaska experiment through `run.py --model gcn_max --region Alaska --graph median`; append `--execute` only when ready to train. The attention-readout variant uses `gcn_attention`.

阿拉斯加命令为 `run.py --model gcn_max --region Alaska --graph median`；准备好训练时再加 `--execute`。注意力池化版本改用 `gcn_attention`。

## STGNN integration / STGNN 接入边界

The reported STGNN experiment uses k=8 for geographical selection: nine stations give 36 distances. It replaces only the geographical neighbour branch; the eight feature-similarity neighbours and the original model/training settings remain unchanged. Its directed-edge convention is retained rather than importing GCN symmetrization. STGNN model/runner code is not bundled. Use the generic exact solver in a separately obtained STGNN implementation; baseline settings are in `experiments/baseline_settings.json`.

STGNN地理分支选8个邻居，9站共36个距离；只替换该分支，保留8个特征相似性邻居及原模型/训练设置。保留STGNN原有有向边约定，不照搬GCN补反向边。这里未打包STGNN模型及训练器；可在自行取得的实现中调用通用精确求解器。基线设置见 `experiments/baseline_settings.json`。

## Small self-check / 小规模自检

```bash
python graphs/exact_gpu.py --device cpu
python graphs/graph_adapter.py --device cpu
```

These optional PyTorch checks use small synthetic inputs and do not train. Full exact search grows combinatorially; chunking bounds memory, not total work.

上述命令需PyTorch，仅用小型合成输入核验，不训练。精确搜索随组合数增长；分块限制内存，不消除总计算量。
