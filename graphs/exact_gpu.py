"""Exact, shared-subset search for distance-based station neighborhoods.

This module does not train a model or modify any project data.  For every
target station i, search all subsets containing i and k other stations.  Both
objectives use *all* pairwise distances within that (k+1)-station subset:
maximum minimum distance, and maximum pairwise-distance median.

All (k+1)-subsets are evaluated once; their scores update every member target.
Subsets are generated in bounded chunks by combinadic/colex unranking, never
by torch.combinations (and never by an N**(k+1) Cartesian product).

The score calculation is float32.  Non-negative float32 score bits and the
inverted subset rank are packed into a positive int64 for one exact max
reduction per objective.  Equal scores therefore select the smallest colex
rank, independently of reduction order.  The rank packing supports fewer than
2**32 subsets, including all N<=50, k<=5 cases requested for this benchmark.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any

import torch


_RANK_MASK = (1 << 32) - 1


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tables(n: int, m: int, device: torch.device) -> list[torch.Tensor]:
    """C(x,j), x=0..n-1, for j=1..m; duplicate zeros are intentional."""
    return [
        torch.tensor(
            [math.comb(x, j) if x >= j else 0 for x in range(n)],
            dtype=torch.int64,
            device=device,
        )
        for j in range(1, m + 1)
    ]


def unrank_colex(ranks: torch.Tensor, tables: list[torch.Tensor]) -> torch.Tensor:
    """Unrank increasing subsets using rank=sum_j C(subset[j-1],j)."""
    residual = ranks.clone()
    m = len(tables)
    out = torch.empty((ranks.numel(), m), dtype=torch.int64, device=ranks.device)
    for j in range(m, 0, -1):
        table = tables[j - 1]
        index = torch.searchsorted(table, residual, right=True) - 1
        out[:, j - 1] = index
        residual = residual - table[index]
    return out


def _pack_scores(scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    # abs() canonicalizes a possible negative zero.  Distances are nonnegative.
    bits = scores.abs().contiguous().view(torch.int32).to(torch.int64)
    return (bits << 32) | (_RANK_MASK - ranks)


def _unpack_best(keys: torch.Tensor, tables: list[torch.Tensor]):
    bits = (keys >> 32).to(torch.int32)
    scores = bits.contiguous().view(torch.float32)
    ranks = _RANK_MASK - (keys & _RANK_MASK)
    sets = unrank_colex(ranks, tables)
    return scores, sets, ranks


@torch.no_grad()
def solve_exact_gpu(
    D: torch.Tensor,
    k: int = 5,
    chunk_size: int = 131072,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Return exact best objective values and containing subsets for all targets.

    D must be a finite, nonnegative symmetric N-by-N distance matrix on CPU or
    CUDA.  Distances to the diagonal are ignored.  k counts *other* stations;
    the effective value is min(k, N-1).  Returned sets include their target.
    For zero/one station (or k=0), both scores are defined as zero because the
    set contains no distinct station pair.  For an even pair count, median is
    the arithmetic mean of the two central ordered distances.

    ``seconds`` includes validation, chunk generation, score reductions and
    winning-subset reconstruction, after device synchronization.  It excludes
    the caller's distance-matrix construction and any graph symmetrization.
    No gradients are recorded.  Inputs are not modified.
    """
    if not isinstance(D, torch.Tensor):
        raise TypeError("D must be a torch.Tensor")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must have square shape [N,N]")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    n = D.shape[0]
    if n > 50:
        raise ValueError("This bounded benchmark is limited to N<=50")
    device = D.device
    _sync(device)
    started = time.perf_counter()
    D = D.to(dtype=torch.float32).contiguous()
    if validate and n:
        if not bool(torch.isfinite(D).all()):
            raise ValueError("D must contain only finite distances")
        if bool((D < 0).any()):
            raise ValueError("D must be nonnegative")
        if not torch.allclose(D, D.T, rtol=1e-6, atol=1e-6):
            raise ValueError("D must be symmetric")

    effective_k = min(k, max(n - 1, 0))
    m = min(effective_k + 1, n)
    if n == 0 or m <= 1:
        sets = torch.arange(n, dtype=torch.int64, device=device).reshape(n, m)
        scores = torch.zeros(n, dtype=torch.float32, device=device)
        _sync(device)
        return {
            "min_scores": scores.clone(),
            "median_scores": scores.clone(),
            "min_sets": sets.clone(),
            "median_sets": sets.clone(),
            "min_ranks": torch.arange(n, dtype=torch.int64, device=device),
            "median_ranks": torch.arange(n, dtype=torch.int64, device=device),
            "num_subsets": n,
            "effective_k": effective_k,
            "chunk_count": int(n > 0),
            "seconds": time.perf_counter() - started,
            "strategy": "exact_shared_colex_packed",
        }

    count = math.comb(n, m)
    if count > _RANK_MASK:
        raise ValueError("More than 2**32-1 subsets; not supported by packed ranks")
    tables = _tables(n, m, device)
    pair_i, pair_j = torch.triu_indices(m, m, offset=1, device=device)
    pair_count = pair_i.numel()
    min_keys = torch.full((n,), -1, dtype=torch.int64, device=device)
    median_keys = torch.full((n,), -1, dtype=torch.int64, device=device)

    for begin in range(0, count, chunk_size):
        ranks = torch.arange(begin, min(begin + chunk_size, count), device=device)
        subsets = unrank_colex(ranks, tables)
        pair_distances = D[subsets[:, pair_i], subsets[:, pair_j]]
        ordered = pair_distances.sort(dim=1).values
        minimum = ordered[:, 0]
        middle = pair_count // 2
        if pair_count % 2:
            median = ordered[:, middle]
        else:
            # Halve before summing to avoid overflow for otherwise finite inputs.
            median = ordered[:, middle - 1] * 0.5 + ordered[:, middle] * 0.5

        flat_members = subsets.reshape(-1)
        min_values = _pack_scores(minimum, ranks).repeat_interleave(m)
        median_values = _pack_scores(median, ranks).repeat_interleave(m)
        min_keys.scatter_reduce_(
            0, flat_members, min_values, reduce="amax", include_self=True
        )
        median_keys.scatter_reduce_(
            0, flat_members, median_values, reduce="amax", include_self=True
        )

    min_scores, min_sets, min_ranks = _unpack_best(min_keys, tables)
    median_scores, median_sets, median_ranks = _unpack_best(median_keys, tables)
    _sync(device)
    return {
        "min_scores": min_scores,
        "median_scores": median_scores,
        "min_sets": min_sets,
        "median_sets": median_sets,
        "min_ranks": min_ranks,
        "median_ranks": median_ranks,
        "num_subsets": count,
        "effective_k": effective_k,
        "chunk_count": math.ceil(count / chunk_size),
        "seconds": time.perf_counter() - started,
        "strategy": "exact_shared_colex_packed",
    }


def _brute_reference(D: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Small CPU oracle: Python enumeration and ordinary sorting, no packing."""
    D = D.detach().to("cpu", dtype=torch.float32)
    n = D.shape[0]
    m = min(k + 1, n)
    if n == 0 or m <= 1:
        return solve_exact_gpu(D, k=k)
    best = {name: [-float("inf")] * n for name in ("min", "median")}
    ranks = {name: [math.inf] * n for name in best}
    winners = {name: [None] * n for name in best}
    for subset in itertools.combinations(range(n), m):
        values = sorted(float(D[i, j]) for i, j in itertools.combinations(subset, 2))
        h = len(values) // 2
        med = values[h] if len(values) % 2 else values[h - 1] * 0.5 + values[h] * 0.5
        # Match the specified float32 objective, including even-size median.
        med = float(torch.tensor(med, dtype=torch.float32))
        scores = {"min": values[0], "median": med}
        rank = sum(math.comb(x, j) for j, x in enumerate(subset, 1))
        for name, value in scores.items():
            for target in subset:
                if value > best[name][target] or (
                    value == best[name][target] and rank < ranks[name][target]
                ):
                    best[name][target] = value
                    ranks[name][target] = rank
                    winners[name][target] = subset
    return {
        **{name + "_scores": torch.tensor(best[name]) for name in best},
        **{name + "_sets": torch.tensor(winners[name]) for name in best},
    }


def self_test(device: str = "cpu") -> dict[str, Any]:
    """Check subset coverage, exact scores, self inclusion and deterministic ties."""
    cases = 0
    generator = torch.Generator().manual_seed(74831)
    for n in (0, 1, 2, 4, 7, 9):
        xy = torch.rand((n, 2), generator=generator, dtype=torch.float32)
        D = torch.cdist(xy, xy)
        for k in sorted({0, 1, 2, 3, 5, max(n - 1, 0)}):
            expected = _brute_reference(D, k)
            got = solve_exact_gpu(D.to(device), k, chunk_size=13)
            for name in ("min", "median"):
                torch.testing.assert_close(
                    got[name + "_scores"].cpu(), expected[name + "_scores"],
                    rtol=0, atol=0,
                )
                assert torch.equal(got[name + "_sets"].cpu(), expected[name + "_sets"])
                for target, subset in enumerate(got[name + "_sets"].cpu().tolist()):
                    assert target in subset
                    assert len(subset) == len(set(subset))
            cases += 1

    for n in (6, 8):
        D = torch.ones((n, n), dtype=torch.float32)
        D.fill_diagonal_(0)
        expected = _brute_reference(D, 5)
        for chunk in (1, 7, 256):
            got = solve_exact_gpu(D.to(device), 5, chunk_size=chunk)
            for name in ("min", "median"):
                assert torch.equal(got[name + "_sets"].cpu(), expected[name + "_sets"])
            cases += 1
    return {"passed": True, "cases": cases, "device": device}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(self_test(args.device)), flush=True)
