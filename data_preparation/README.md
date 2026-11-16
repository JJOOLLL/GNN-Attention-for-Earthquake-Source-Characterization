# 数据准备 / Data preparation

## 输入 / Input layout

```text
GNN_DATA_ROOT/
  data_DA/                         # Southern California
  data_ANCHORAGE_DA/               # Alaska
```

每个地区目录包含以下文件 / Each regional folder contains:

```text
catalogue.csv
stations.csv
catalogue_station_lookup_final.pickle
waveforms_proc_broadband/<event_id>.npy
```

目录包含 `lat, lon, depth, mag`，深度为米；事件编号沿用原始行索引，不可重排行。台站表包含 `code, lat, lon`。lookup 台站顺序必须与波形及坐标顺序一致。波形形状为 `[stations, time, 3]`。只加载可信来源的 pickle 数据及权重。

Catalogue fields include `lat, lon, depth, mag`; depth is in metres, and original row indices identify events. Station fields include `code, lat, lon`. Lookup station order must match waveform and coordinate order. Arrays have shape `[stations, time, 3]`. Load pickle data and checkpoints only from trusted sources.

`reference_data.json` 按地区合并保存三个原输入索引文件的 SHA-256、训练/验证/测试事件编号及测试台站选择，保留原顺序。训练入口读取这些配置核对输入，不依赖归档预测结果。原始和处理后波形的大文件未打包。

`reference_data.json` combines the original input hashes, event splits, and test-station selection by region, preserving their order. Training reads this configuration without depending on archived predictions. Large waveform files are not bundled.

## 下载流程 / Download workflows

| Notebook | Network | Period | Latitude | Longitude |
|---|---|---|---|---|
| `download_california.ipynb` | CI | 2000-01-01 – 2020-01-01 | 32–36° N | 120–116° W |
| `download_alaska.ipynb` | AK | 2015-01-01 – 2025-01-01 | 59–63° N | 153–147° W |

两者保留原流程的数值处理：最小震级查询值 3.0；下载发震前 60 秒至后 180 秒 BH* 波形；去趋势、去仪器响应、0.1–8 Hz 零相位带通；振幅乘 `1e5`，在发震前 40 秒至后 160 秒插值为 4096 点。网格含两端点，间隔为 `200/4095` 秒。模型裁取 2048 点，测试起点为 820。

Both retain the original numerical processing: minimum-magnitude query 3.0; BH* waveforms from −60 to +180 s; detrending, response removal, zero-phase 0.1–8 Hz filtering, amplitude scaling by `1e5`, and interpolation to 4096 points from −40 to +160 s. Both endpoints are included, giving an interval of `200/4095` s. Models use a 2048-sample crop; testing starts at index 820.

三分量保留原 ObsPy Stream 顺序；未为打包而重排历史输入。

Component order follows the original ObsPy Stream order; packaging does not reorder historical inputs.

## 运行 / Running

从仓库启动 Jupyter，设置 `GNN_DATA_ROOT`，打开对应 notebook。确认输出为新目录后，将 `ALLOW_DOWNLOAD` 改为 `True` 并顺序运行。非空目录会被拒绝，不删除已有文件。地图预览默认关闭，开启需安装 ObsPy 支持的地图后端。

Start Jupyter from the repository and set `GNN_DATA_ROOT`. Open the regional notebook, check the new destination, set `ALLOW_DOWNLOAD=True`, and run in order. Non-empty destinations are rejected. Optional map previews require an ObsPy-supported mapping backend.

下载可能耗时数小时。FDSN 目录和可用记录会更新，新下载不保证原事件顺序或相同哈希；差异应先核对，再判断是否复现了原结果。本次发布整理未执行下载。

Downloads may take hours. FDSN catalogues and record availability change, so fresh downloads may not match archived input hashes. Resolve differences before claiming reproduction. Downloads were not executed during packaging.

数据来源 / Sources: [CI](https://doi.org/10.7914/SN/CI), [AK](https://doi.org/10.7914/SN/AK), [FDSN services](https://service.iris.edu/).
