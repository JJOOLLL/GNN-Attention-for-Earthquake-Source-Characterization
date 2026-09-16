# 数据准备 / Data preparation

## 已确认的输入格式 / Verified input contract

```text
GNN_DATA_ROOT/
  data_DA/                         # California
  data_ANCHORAGE_DA/               # Alaska
    catalogue.csv
    stations.csv
    catalogue_station_lookup_final.pickle
    waveforms_proc_broadband/<event_id>.npy
```

两个地区均需相同布局。`catalogue.csv` 至少包含 `lat, lon, depth, mag`，深度单位为米；训练按原始行索引识别事件，因此不能重排行。`stations.csv` 包含 `code, lat, lon`。lookup 中每个事件的台站顺序必须与波形第一个维度和台站坐标顺序一致。每个波形数组形状为 `[stations, time, 3]`。只加载可信来源的 pickle 文件，不要加载陌生人提供的 pickle。

Both regional folders use this layout. Catalogue depth is in metres. The original row order defines event IDs and must not change. Waveform station order must match the lookup and selected station coordinates. Arrays have shape `[stations, time, 3]`. Only load pickle files from trusted sources.

`data_preparation/input_hashes.json` 保存论文实际使用的三个目录文件 SHA-256；`results/splits/` 保存精确划分。训练会检查这些信息。重新请求 FDSN 数据可能得到不同的目录顺序、可用波形或修订参数，不能保证仅靠同一个随机种子重现原划分。

The provenance file records SHA-256 hashes for the exact catalogue, station table, and lookup used in the paper. Exact split IDs are included. A fresh FDSN query may return a different catalogue or different available waveforms; a matching seed alone does not establish reproduction.

## 南加州脚本 / California workflow

`download_california.ipynb` 来自原公开仓库提交 `6149eece34ab7cc1af0752870283726a1abec605`。保留原始科学处理：CI 台网，纬度32–36，经度-120至-116，时间2000-01-01至2020-01-01，最小震级查询值3.0；BH* 通道，事件前60秒至后180秒下载窗；去趋势、去仪器响应、0.1–8 Hz零相位带通、振幅乘1e5，插值为覆盖事件前40秒至后160秒的4096个点。

该插值网格包含两端点，名义间隔为200/4095秒，不能直接声称采样率是精确20 Hz。三分量沿用原始 ObsPy stream 顺序，尚未证明所有波形均为固定的 Z/N/E 顺序；本次未擅自重排历史输入。训练/测试再裁出2048点；测试起点为820。

The included California notebook retains the original query and signal-processing steps. It interpolates 4096 points over 200 seconds, including both endpoints; the grid is not exactly 20 Hz. Component order follows the original ObsPy stream order and has not been established as a universal Z/N/E ordering. No historical input is silently reordered. Evaluation uses a 2048-sample crop starting at index 820.

包装修改仅包括：使用 `GNN_DATA_ROOT`；默认 `ALLOW_DOWNLOAD=False`；已有非空数据目录拒绝继续，而不是删除；缓存目录文件写入区域数据目录；清空 notebook 历史输出。下载未在本次执行，网络服务的当前可用性尚未验证。重新下载获得的文件若与论文快照不同，训练入口会明确报错；应先核对快照，不能关闭检查后仍声称复现原结果。

Packaging changes are limited to portable paths, an explicit download opt-in, refusal to overwrite a non-empty data directory, region-local metadata paths, and removal of old outputs. The download was not executed in this packaging task, and current service availability was not tested. Hash mismatches must be resolved before claiming reproduction of the paper results.

## 阿拉斯加待补项 / Alaska gap

已找到最终训练配置、实际数据文件哈希、三集合编号、台站选择及测试预测，但在本次检查的定位项目和旧定位目录中未找到阿拉斯加原始下载脚本。不能把南加州脚本仅修改边界后称为“论文实际使用的阿拉斯加流程”。

The final Alaska model settings, input hashes, split IDs, station selections, and predictions are available. The original Alaska download/preprocessing notebook was not located in the inspected project directories. Changing only the California query bounds would not establish that it is the workflow actually used in the paper.

需作者提供原脚本或可公开获取的处理后数据快照及使用条件。原始波形和准备好的数据下载地址均未包含在本包内。

The original script or an accessible processed-data snapshot and its usage conditions must be supplied. Raw waveforms and a prepared-data download URL are not included.
