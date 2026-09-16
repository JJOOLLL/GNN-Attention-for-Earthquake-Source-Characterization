# Attention in GNN-based Earthquake Source Characterization

[中文说明](README.zh-CN.md)

This repository compares attention in inter-station message passing and graph-level readout for earthquake location and magnitude estimation in Southern California and Alaska.

## Models

| Name | Message passing | Readout |
|---|---|---|
| `transformer_attention` | Graph Transformer | Attention pooling |
| `transformer_max` | Graph Transformer | Max pooling |
| `no_message_attention` | None | Attention pooling |
| `no_message_max` | None | Max pooling |
| `gcn_attention` | GCN | Attention pooling |
| `gcn_max` | GCN | Max pooling |

These are the six controlled configurations implemented in this study. The names describe structure, not performance ranking. TEAM-LM and STGNN source code and runners are excluded; their paper comparison results and experimental settings are retained separately. Legacy experiment IDs appear only in [the mapping table](experiments/model_names.csv).

## Layout

```text
models/              Six model implementations
experiments/         Regional training notebooks, settings and checks
data_preparation/    California download notebook and data instructions
results/             Paper metrics, predictions, splits and station selections
run.py               Single-experiment entry point; no training by default
```

## Quick start

The original environment used Python 3.12.8 and PyTorch 2.5.1. Install a suitable PyTorch CPU/CUDA build, then:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gnn-paper --display-name "GNN paper"
python experiments/verify_results.py
python run.py --model transformer_max --region California
```

The last command displays settings only. To train after obtaining the original processed data:

```bash
python run.py --model transformer_max --region California --data-root /path/to/data --kernel gnn-paper --execute
```

The data root contains `data_DA/` and `data_ANCHORAGE_DA/`; see [data instructions](data_preparation/README.md). For manual notebook execution, set `GNN_DATA_ROOT`. Checkpoints default to `save/<region>/<model>/`; `GNN_SAVE_ROOT` can override the root. Existing checkpoints are protected. Full training can require tens of GB of RAM because the original dataset implementation materializes augmented waveforms.

## Experimental scope

Seed 42; disjoint train/validation/test sets; the best checkpoint is selected by validation loss. California has 2026/253/254 events in these sets, and Alaska has 1764/220/222. Final Alaska `no_message_attention` uses initial learning rate 1e-4 after one predefined stability retry; all other controlled configurations use 1e-3. The second-stage rate is 2e-5. The learning-rate difference is disclosed and prevents attributing its full improvement solely to pooling.

`results/metrics.csv` contains the final 64 metric rows, including the two external comparisons. Predictions for the six included models are provided. Location MAE/MSE use km/km²; magnitude is unscaled; standard deviations are across events, not across seeds. R² uses the true-target mean.

`experiments/cross_region.ipynb` includes only the six controlled models. Its archived results use target-region normalization and the full available target catalogue. The archived Alaska `no_message_attention` transfer run used the original pre-retry checkpoint, recorded separately in `experiments/weights.json`; it must not be confused with final within-region results.

## Availability and limitations

Model weights and raw waveforms are not bundled; checkpoint download URLs and the original Alaska data-preparation workflow remain to be supplied. Expected checkpoint hashes are in `experiments/weights.json`. A fresh FDSN download is not guaranteed to match the original catalogue snapshot, so training checks input hashes and split order. The provided environment has not been validated by clean-environment end-to-end retraining.

The existing MIT license is retained. This package does not include TEAM/STGNN implementations or claim that all reused base code was originally authored here; preserve applicable attribution for reused components.
