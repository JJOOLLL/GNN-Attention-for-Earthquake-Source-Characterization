"""Exact, source-independent station-neighbour selection for frozen GCNs.

Only the edge_index reaching the existing GCNConv is replaced. Original
parameters, activations, data augmentation, pooling and forward computations
remain untouched. Both objectives score the target together with its selected
neighbours. This module never reads waveforms, labels, or source coordinates.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOLVER_PATH = ROOT / "exact_gpu.py"
_spec = importlib.util.spec_from_file_location("dispersion_exact_solver", SOLVER_PATH)
_solver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_solver)
solve_exact_gpu = _solver.solve_exact_gpu
SOLVER_SHA256 = hashlib.sha256(SOLVER_PATH.read_bytes()).hexdigest()

REGIONS = {
    "California": (32.0, 4.0, -120.0, 4.0),
    "Alaska": (59.0, 4.0, -153.0, 6.0),
}
SCHEMES = ("maxmin", "median")
FORMAT_VERSION = 1


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalized_to_geographic_km(coords_float32, region):
    """Input [N,2] normalized model coordinates, including any augmentation."""
    coords = np.asarray(coords_float32, dtype=np.float32).astype(np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Expected finite normalized coordinates [N,2]")
    lat_min, lat_span, lon_min, lon_span = REGIONS[region]
    lat = (coords[:, 0] / 2 + .5) * lat_span + lat_min
    lon = (coords[:, 1] / 2 + .5) * lon_span + lon_min
    p = np.deg2rad(np.column_stack((lat, lon)))
    d = p[:, None, :] - p[None, :, :]
    a = np.sin(d[..., 0] / 2) ** 2
    a += np.cos(p[:, None, 0]) * np.cos(p[None, :, 0]) * np.sin(d[..., 1] / 2) ** 2
    D = (2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)
    np.fill_diagonal(D, 0)
    if not np.array_equal(D, D.T):
        raise AssertionError("Geographic distance matrix is not symmetric")
    return D


def _payload_digest(record):
    h = hashlib.sha256()
    for name in ("valid_nodes", "valid_coordinates", "min_sets", "min_scores",
                 "median_sets", "median_scores"):
        value = np.ascontiguousarray(record[name])
        h.update(name.encode("ascii"))
        h.update(str((value.dtype.str, value.shape)).encode("ascii"))
        h.update(value.tobytes())
    return h.hexdigest()


class ExactGraphBuilder:
    """RNG-free memoized builder; a cache entry stores both exact objectives."""

    def __init__(self, region, scheme, cache_root=None, k=5, device="cuda"):
        if region not in REGIONS or scheme not in SCHEMES or k != 5:
            raise ValueError("Expected a supported region, maxmin/median, and k=5")
        self.region, self.scheme, self.k = region, scheme, k
        self.device = torch.device(device)
        self.cache_root = None if cache_root is None else Path(cache_root).resolve()
        self._memory = {}
        self._edge_memory = {}
        self.stats = dict(memory_hits=0, disk_hits=0, solved=0, solver_seconds=0.0,
                          forward_batches=0, precomputed_events=0)
        self.contract = dict(format_version=FORMAT_VERSION, region=region, k=k,
            solver_sha256=SOLVER_SHA256, objective_pair="minimum_and_median_all_pairs",
            coordinates="actual_model_input_float32_inverse_region_normalization",
            distance="haversine_km_float64_then_float32", target_included=True,
            tie_break="lexicographic_coordinate_order_then_original_slot_then_colex_rank",
            edge_order="target_to_selected_then_reverse_without_deduplication")
        self._contract_text = json.dumps(self.contract, sort_keys=True, separators=(",", ":"))

    def _event_input(self, coords, weights):
        coords = np.asarray(_as_numpy(coords), dtype=np.float32)
        weights = np.asarray(_as_numpy(weights), dtype=np.float32)
        if coords.ndim == 3 and coords.shape[1] == 1:
            coords = coords[:, 0, :]
        if coords.ndim != 2 or coords.shape[1] < 2 or weights.shape != (len(coords),):
            raise ValueError("Event coordinates must be [N,1,C] or [N,C], weights [N]")
        if len(coords) > 50 or not np.isfinite(weights).all():
            raise ValueError("Expected at most 50 input slots and finite masks")
        valid = np.flatnonzero(weights > 0).astype(np.int64)
        points = np.ascontiguousarray(coords[valid, :2], dtype=np.float32)
        # +0 and -0 are the same geometry; canonicalize their cache bytes.
        points[points == 0] = 0
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite coordinates at a valid station")
        h = hashlib.sha256(self._contract_text.encode("ascii"))
        h.update(np.asarray([len(coords)], dtype=np.int64).tobytes())
        h.update(valid.tobytes())
        h.update(points.tobytes())
        return h.hexdigest(), valid, points

    def _validate_record(self, record, key, valid, points):
        if record["key"] != key or record["contract"] != self._contract_text:
            raise ValueError("Graph cache provenance mismatch")
        if not np.array_equal(record["valid_nodes"], valid) or not np.array_equal(record["valid_coordinates"], points):
            raise ValueError("Graph cache station geometry or valid-slot mismatch")
        if record["digest"] != _payload_digest(record):
            raise ValueError("Graph cache payload digest mismatch")
        n, m = len(valid), min(len(valid), self.k + 1)
        D = normalized_to_geographic_km(points, self.region)
        for objective in ("min", "median"):
            sets = record[objective + "_sets"]
            scores = record[objective + "_scores"]
            if sets.shape != (n, m) or scores.shape != (n,):
                raise ValueError("Malformed graph cache dimensions")
            if sets.dtype != np.int64 or scores.dtype != np.float32:
                raise ValueError("Malformed graph cache dtypes")
            if not np.isin(sets, valid).all() or not np.isfinite(scores).all():
                raise ValueError("Invalid station index or score in graph cache")
            if any(target not in row or len(set(row.tolist())) != m for target, row in zip(valid, sets)):
                raise ValueError("Cached subset omits its target or repeats a station")
            if m <= 1:
                expected = np.zeros(n, dtype=np.float32)
            else:
                local_sets = np.searchsorted(valid, sets)
                a, b = np.triu_indices(m, 1)
                pair_values = D[local_sets[:, a], local_sets[:, b]]
                if objective == "min":
                    expected = pair_values.min(axis=1)
                else:
                    # Match the solver's two half-sums for even pair counts.
                    ordered = np.sort(pair_values, axis=1)
                    middle = ordered.shape[1] // 2
                    expected = ordered[:, middle] if ordered.shape[1] % 2 else (
                        ordered[:, middle - 1] * np.float32(.5) + ordered[:, middle] * np.float32(.5))
            if not np.array_equal(expected, scores):
                raise ValueError("Cached scores do not match selected station distances")

    def _disk_path(self, key):
        if self.cache_root is None:
            return None
        return self.cache_root / self.region / key[:2] / (key + ".npz")

    def _read_disk(self, path, key, valid, points):
        try:
            with np.load(path, allow_pickle=False) as saved:
                record = {name: saved[name].copy() for name in (
                    "valid_nodes", "valid_coordinates", "min_sets", "min_scores", "median_sets", "median_scores")}
                record.update(key=str(saved["key"].item()), contract=str(saved["contract"].item()),
                              digest=str(saved["digest"].item()))
            self._validate_record(record, key, valid, points)
            return record
        except Exception as exc:
            raise RuntimeError("Invalid exact graph cache, refusing silent reuse: " + str(path)) from exc

    def _write_disk(self, path, record):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            other = self._read_disk(path, record["key"], record["valid_nodes"], record["valid_coordinates"])
            if other["digest"] != record["digest"]:
                raise RuntimeError("Different exact graph data already occupy cache path")
            return
        temp = path.with_name(path.name + f".{os.getpid()}.{time.monotonic_ns()}.tmp")
        try:
            with temp.open("xb") as stream:
                np.savez_compressed(stream, **record)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    @torch.no_grad()
    def get_event(self, coords, weights, device=None):
        """Return exact full subsets in ORIGINAL slot indices, rows by target slot."""
        key, valid, points = self._event_input(coords, weights)
        if key in self._memory:
            self.stats["memory_hits"] += 1
            return self._memory[key]
        path = self._disk_path(key)
        if path is not None and path.is_file():
            record = self._read_disk(path, key, valid, points)
            self.stats["disk_hits"] += 1
        else:
            # Solve in canonical geometric order. Slot number breaks exact
            # duplicate-coordinate ties, without drawing a random value.
            order = np.lexsort((valid, points[:, 1], points[:, 0]))
            ordered_points = points[order]
            ordered_slots = valid[order]
            D = normalized_to_geographic_km(ordered_points, self.region)
            solve_device = self.device if device is None else torch.device(device)
            solved = solve_exact_gpu(torch.as_tensor(D, device=solve_device), k=self.k, chunk_size=131072)
            back_to_valid_order = np.argsort(ordered_slots)
            record = dict(key=key, contract=self._contract_text,
                          valid_nodes=valid.copy(), valid_coordinates=points.copy())
            for objective in ("min", "median"):
                selected = solved[objective + "_sets"].cpu().numpy()
                record[objective + "_sets"] = np.ascontiguousarray(ordered_slots[selected][back_to_valid_order], dtype=np.int64)
                record[objective + "_scores"] = np.ascontiguousarray(solved[objective + "_scores"].cpu().numpy()[back_to_valid_order], dtype=np.float32)
            record["digest"] = _payload_digest(record)
            self._validate_record(record, key, valid, points)
            self._write_disk(path, record)
            self.stats["solved"] += 1
            self.stats["solver_seconds"] += solved["seconds"]
        self._memory[key] = record
        return record

    def _event_edges(self, record):
        cache_key = (record["key"], self.scheme)
        if cache_key not in self._edge_memory:
            objective = "min" if self.scheme == "maxmin" else "median"
            valid, sets = record["valid_nodes"], record[objective + "_sets"]
            n, m = sets.shape
            k = max(m - 1, 0)
            if k == 0:
                edges = np.empty((2, 0), dtype=np.int64)
            else:
                neighbours = np.stack([np.sort(row[row != target]) for target, row in zip(valid, sets)])
                directed = np.stack([np.repeat(valid, k), neighbours.reshape(-1)])
                edges = np.concatenate([directed, directed[::-1]], axis=1)
            self._edge_memory[cache_key] = edges
        return self._edge_memory[cache_key]

    @torch.no_grad()
    def get_batch(self, coords, weights, device=None):
        """Build [2,E], preserving original per-event forward+reverse ordering."""
        device = coords.device if device is None and isinstance(coords, torch.Tensor) else (self.device if device is None else torch.device(device))
        c = np.asarray(_as_numpy(coords), dtype=np.float32)
        w = np.asarray(_as_numpy(weights), dtype=np.float32)
        if c.ndim not in (3, 4) or w.ndim != 2 or c.shape[:2] != w.shape:
            raise ValueError("Expected batched station coordinates and [B,N] mask")
        batch_size, n_slots = w.shape
        parts = []
        for index in range(batch_size):
            record = self.get_event(c[index], w[index], device=device)
            parts.append(self._event_edges(record) + index * n_slots)
        all_edges = np.concatenate(parts, axis=1) if parts else np.empty((2, 0), dtype=np.int64)
        self.stats["forward_batches"] += 1
        return torch.as_tensor(all_edges, dtype=torch.long, device=device)

    def precompute_dataset(self, dataset, label, progress_callback=None):
        """Access only stored coordinate/mask arrays, never __getitem__ or labels."""
        if hasattr(dataset, "enhanced_coords"):
            coords, weights = dataset.enhanced_coords, dataset.enhanced_weights
        elif hasattr(dataset, "station_coords"):
            coords, weights = dataset.station_coords, dataset.weights
        elif hasattr(dataset, "coords"):
            coords, weights = dataset.coords, dataset.weights
        else:
            raise TypeError("Dataset has no recognized cached station coordinates")
        if len(coords) != len(weights):
            raise ValueError("Coordinate and mask cache lengths differ")
        started = time.perf_counter()
        before = self.stats.copy()
        total = len(coords)
        for i in range(total):
            self.get_event(coords[i], weights[i])
            self.stats["precomputed_events"] += 1
            if progress_callback is not None and (i == 0 or (i + 1) % 100 == 0 or i + 1 == total):
                progress_callback(dict(label=label, processed=i + 1, total=total,
                                       elapsed_seconds=time.perf_counter() - started, stats=self.stats.copy()))
        return dict(label=label, events=total, elapsed_seconds=time.perf_counter() - started,
                    delta={k: self.stats[k] - before[k] for k in self.stats},
                    stats=self.stats.copy(), cache_root=None if self.cache_root is None else str(self.cache_root))


def make_graph_class(BaseGraphNet, region, scheme, cache_root=None):
    """Subclass without adding parameters or buffers; state_dict stays identical."""
    if region not in REGIONS or scheme not in SCHEMES:
        raise ValueError("Unsupported region or graph scheme")

    class ExactDispersionGraphNet(BaseGraphNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if getattr(self, "K_neighbors", None) != 5:
                raise ValueError("This experiment preserves exactly K_neighbors=5")
            self._graph_builder = ExactGraphBuilder(region, scheme, cache_root=cache_root)
            self._dispersion_context = None
            self._dispersion_hook = self.gcn_conv.register_forward_pre_hook(self._replace_graph_edges)

        def _replace_graph_edges(self, _module, inputs):
            if self._dispersion_context is None or len(inputs) < 2:
                raise RuntimeError("Missing exact graph forward context")
            coords, weights = self._dispersion_context
            edges = self._graph_builder.get_batch(coords, weights, device=inputs[0].device)
            return (inputs[0], edges, *inputs[2:])

        def forward(self, waveforms, coords, weights):
            previous = self._dispersion_context
            self._dispersion_context = (coords, weights)
            try:
                return super().forward(waveforms, coords, weights)
            finally:
                self._dispersion_context = previous

    ExactDispersionGraphNet.__name__ = BaseGraphNet.__name__
    return ExactDispersionGraphNet


def self_test(device="cpu"):
    """Small deterministic mask, target, edge and disk-cache checks; no training."""
    import itertools
    import random
    import tempfile

    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    before_cuda = [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    cases = 0
    coords = np.array([[-.8, -.2], [-.3, .9], [.1, -.4], [.6, .4], [.6, .4], [.8, -.8], [-.6, -.7], [.3, .6]], dtype=np.float32)
    for region in REGIONS:
        for scheme in SCHEMES:
            for n in (0, 1, 4, 8):
                mask = np.zeros(10, dtype=np.float32)
                slots = np.array([0, 2, 3, 4, 5, 6, 8, 9])[:n]
                mask[slots] = 1
                padded = np.zeros((10, 1, 3), dtype=np.float32)
                padded[slots, 0, :2] = coords[:n]
                builder = ExactGraphBuilder(region, scheme, device=device)
                record = builder.get_event(padded, mask)
                D = normalized_to_geographic_km(coords[:n], region)
                m = min(6, n)
                for target in range(n):
                    alternatives = [c for c in itertools.combinations(range(n), m) if target in c]
                    for objective in ("min", "median"):
                        best = 0.0
                        for subset in alternatives:
                            pairs = [D[i, j] for i, j in itertools.combinations(subset, 2)]
                            value = (min(pairs) if objective == "min" else np.median(np.asarray(pairs, dtype=np.float32))) if pairs else 0.0
                            best = max(best, value)
                        if not np.isclose(record[objective + "_scores"][target], best, rtol=2e-7, atol=1e-5):
                            raise AssertionError("Adapter objective differs from independent tiny brute force")
                edge = builder.get_batch(padded[None], mask[None], device=device).cpu().numpy()
                count = n * max(m - 1, 0)
                assert edge.shape == (2, 2 * count)
                assert np.array_equal(edge[:, count:], edge[::-1, :count])
                assert np.array_equal(edge[0, :count], np.repeat(slots, max(m - 1, 0)))
                assert np.isin(edge, slots).all()
                # Changing invalid-slot geometry must not create a new graph.
                padded[mask == 0] = 123
                assert builder.get_event(padded, mask)["key"] == record["key"]
                cases += 1
    with tempfile.TemporaryDirectory() as cache:
        padded = np.zeros((8, 1, 3), dtype=np.float32)
        padded[:, 0, :2] = coords
        mask = np.ones(8, dtype=np.float32)
        a = ExactGraphBuilder("California", "maxmin", cache, device=device)
        saved = a.get_event(padded, mask)
        b = ExactGraphBuilder("California", "median", cache, device=device)
        restored = b.get_event(padded, mask)
        assert saved["digest"] == restored["digest"]
        assert b.stats["disk_hits"] == 1 and b.stats["solved"] == 0
        mask[3] = 0
        assert b.get_event(padded, mask)["key"] != saved["key"]
        changed = padded.copy()
        changed[0, 0, 0] += .01
        assert b.get_event(changed, mask)["key"] != restored["key"]
        # A corrupted payload must be rejected rather than silently trusted.
        path = b._disk_path(restored["key"])
        path.write_bytes(b"invalid-cache")
        try:
            ExactGraphBuilder("California", "median", cache, device=device).get_event(padded, np.ones(8, dtype=np.float32))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Corrupt cache was accepted")
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and np.array_equal(before_numpy[1], after_numpy[1]) and before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch, torch.random.get_rng_state())
    if before_cuda:
        assert all(torch.equal(x, y) for x, y in zip(before_cuda, torch.cuda.get_rng_state_all()))
    return dict(passed=True, cases=cases, device=str(device), training_rng_unchanged=True,
                target_inclusion=True, mask_exclusion=True, duplicate_coordinates=True, disk_cache_verified=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.use_deterministic_algorithms(True)
    print(json.dumps(self_test(args.device)), flush=True)
