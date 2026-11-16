"""Frozen source cells and reference checks; no training or RNG calls here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('gcn_max', 'gcn_attention')
REGIONS = ('California', 'Alaska')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(path):
    """Git line-ending conversion must not invalidate unchanged source code."""
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()


def verify_sources(model, region):
    if model not in MODELS or region not in REGIONS:
        raise ValueError('Unsupported GCN experiment')
    key = f'{region}/{model}'
    manifest = read(ROOT/'training'/'source_manifest.json')[key]
    path = ROOT/'training'/'source_cells'/f'{model}_{region}.json'
    model_path = ROOT/'models'/f'{model}.py'
    if text_sha(path) != manifest['source_cells_text_sha256']:
        raise ValueError('Frozen preparation/evaluation source cells have changed')
    if text_sha(model_path) != manifest['model_text_sha256']:
        raise ValueError('Frozen corrected-coordinate GCN model has changed')
    cells = {int(i): value for i,value in read(path).items()}
    for i, code in cells.items():
        compile(code, f'{model}_{region}:cell{i}', 'exec')
    return cells, manifest


def load_sources(model, region, graph_scheme):
    from experiments.paths import get_data_root, check_input_hashes, get_data_reference
    if graph_scheme not in ('knn', 'median'):
        raise ValueError('Graph must be knn or median')
    cells, manifest = verify_sources(model, region)
    key = f'{region}/{model}'
    cfg = read(ROOT/'experiments'/'settings.json')[key]
    directory = get_data_root()/('data_DA' if region=='California' else 'data_ANCHORAGE_DA')
    check_input_hashes(region, directory)
    if not (directory/'waveforms_proc_broadband').is_dir():
        raise FileNotFoundError(directory/'waveforms_proc_broadband')
    input_hashes = get_data_reference(region)['input_hashes']
    reference = read(ROOT/'training'/'reference_initialization.json')[key]
    meta = dict(model=model, region=region, seed=42, initialization_seed=42,
        data_and_training_seed=42, graph_scheme=graph_scheme,
        message_passing=cfg['graph_definitions'][graph_scheme],
        readout='max' if model=='gcn_max' else 'attention',
        model_sha256=sha(ROOT/'models'/f'{model}.py'),
        source_cells_sha256=sha(ROOT/'training'/'source_cells'/f'{model}_{region}.json'),
        original_notebook_code_sha256=manifest['original_notebook_code_sha256'],
        frozen_training_runner_sha256=manifest['frozen_training_runner_sha256'],
        input_hashes=input_hashes,
        training_settings=dict(initial_lr=.001, second_stage_lr=.00002, num_epochs=50,
            no_improve_epoch=50 if model=='gcn_max' else 70,
            batch_size=32, loss_weights=[1,1,.3,.1]),
        **reference)
    if graph_scheme == 'median':
        cells[2] += ('\nfrom graphs.graph_adapter import make_graph_class\n'
                     f'GraphNet = make_graph_class(GraphNet, {region!r}, "median", cache_root=GRAPH_CACHE_ROOT)\n')
        meta.update(graph_adapter_sha256=sha(ROOT/'graphs'/'graph_adapter.py'),
                    exact_solver_sha256=sha(ROOT/'graphs'/'exact_gpu.py'))
    return cells, meta


def check_splits(split_ids, region):
    from experiments.paths import get_data_reference
    for name, ids in split_ids.items():
        ref_name = 'validation' if name == 'val' else name
        expected = get_data_reference(region)['splits'][ref_name]
        if sorted(ids) != sorted(expected):
            raise RuntimeError(f'{name} event split differs from the original paper evaluation')


def check_test_sampling(ns, region):
    import numpy as np
    from experiments.paths import get_data_reference
    dataset = ns['test_dataset']
    reference = get_data_reference(region)['test_station_selection']
    ids = dataset.cat[dataset.event_inds, 0].astype(int).tolist()
    if ids != [row['event_id'] for row in reference]:
        raise RuntimeError('Test-event order differs from the original paper evaluation')
    stations = ns['stations']
    for i, row in enumerate(reference):
        codes = ns['lookup'][row['event_id']]
        expected = stations[['lat', 'lon']][stations['code'].isin(codes)].values[row['station_indices']]
        n = len(expected)
        if not np.array_equal(dataset.station_coords[i, :n, 0, :2], expected):
            raise RuntimeError(f"Selected stations differ for test event {row['event_id']}")
        if not (np.all(dataset.weights[i, :n] == 1) and np.all(dataset.weights[i, n:] == 0)):
            raise RuntimeError(f"Station mask differs for test event {row['event_id']}")
    return {'test_events': len(ids), 'event_order_matches': True,
            'station_selection_and_masks_match': True}
