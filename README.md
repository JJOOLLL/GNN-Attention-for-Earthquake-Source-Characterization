# Attention-Informed Graph Construction for Earthquake Source Characterization

[中文说明](README.zh-CN.md)

Code accompanying **Attention-Informed Graph Construction for Earthquake Source Characterization**. The manuscript has been submitted and is not yet published.

We compare message passing and readout in multistation graph neural networks. Long-range interaction patterns identified through attention analysis guide maximum-median-distance neighbour selection, replacing geographical kNN selection without prior source locations.

## Six configurations

| CLI name | Message passing | Readout |
|---|---|---|
| `no_message_max` | None | Global max pooling (GNN baseline) |
| `no_message_attention` | None | Global attention pooling |
| `gcn_max` | GCN | Global max pooling |
| `gcn_attention` | GCN | Global attention pooling |
| `transformer_max` | Graph Transformer | Global max pooling |
| `transformer_attention` | Graph Transformer | Global attention pooling |

Models predict latitude, longitude, depth, and magnitude. GT uses a complete graph; GCN supports kNN and maximum-median graphs. The framework builds on van den Ende & Ampuero (2020); see [REFERENCES.md](REFERENCES.md).

## Layout

- `models/`: six model configurations.
- `graphs/`: exact maximum-median selection and GCN integration; see the [graph guide](graphs/README.md).
- `training/`: final GCN runner; `source_cells/` contains the preparation and evaluation code it executes.
- `experiments/`: regional notebooks for the other four configurations, cross-region evaluation, and settings.
- `data_preparation/`: both download workflows; `reference_data.json` combines input hashes, fixed event splits, and test-station selection.

## Running

The experiment environment used Python 3.12.8, PyTorch 2.5.1, and PyG 2.6.1. Install the appropriate PyTorch build for your device, then:

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name gnn-paper --display-name "Python (gnn-paper)"
```

Configuration preview does not start training:

```bash
python run.py --model transformer_max --region California
python run.py --model gcn_max --region Alaska --graph median
```

Prepare inputs using the [data guide](data_preparation/README.md), then explicitly add `--execute`:

```bash
python run.py --model transformer_max --region California --data-root /path/to/data --execute
python run.py --model gcn_max --region Alaska --graph knn --data-root /path/to/data --execute
python run.py --model gcn_max --region Alaska --graph median --data-root /path/to/data --execute
```

Both GCN readouts use this shared entry point, without duplicate notebook wrappers. Set `--save-root` to choose outputs, or `--resume` to resume a compatible GCN run. See `python run.py --help`. Full GCN training requires CUDA and more than 40 GiB of available system RAM.

GCN retains the final SiLU-to-ReLU schedule, with validation loss selecting the final ReLU checkpoint. GT and no-message configurations retain their original settings. Details are in `experiments/settings.json`. Cross-region evaluation uses GT with attention pooling only; original checkpoint paths and hashes are in `experiments/weights.json`.

## Scope and license

This slim source package does not bundle event-level predictions, archived results, verification reports, raw waveforms, or trained weights. STGNN and TEAM-LM implementations are not redistributed; their experiment settings remain in `experiments/baseline_settings.json`.

The existing [MIT License](LICENSE) is retained. Dependencies retain their own licenses. Please cite the manuscript above when using this work; formal publication details will be added when available.
