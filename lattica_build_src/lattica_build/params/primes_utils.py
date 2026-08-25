"""See `params/README.md` for usage details."""

import json
import os
from typing import Counter, Sequence

import torch


def unflatten_rows_cols(a: Sequence[int], num_cols_per_row: Sequence[int]) -> Sequence[tuple[int]]:
    res = []
    idx = 0
    for col_len in num_cols_per_row:
        res.append(list(a[idx:idx + col_len]))
        idx += col_len
    return res

def get_primes_from_precisions_list(full_q_list_precision):
    """
        Create a tensor full_q_list with the same shape as full_q_list_precision,
        where each entry is a unique 'good prime' drawn from the JSON pool that
        matches its bit-length (precision value). Raises if any bucket runs out.
        """
    precisions = []
    # Verify precision is decreasing in each column
    for row in full_q_list_precision:
        row = torch.tensor(row)
        diffs = row[:-1] - row[1:]
        if not (diffs > 0).all():
            raise ValueError("Precision values must be strictly decreasing in each column.")
        relative_precisions_needed = diffs.tolist() + [int(row[-1])]
        precisions += relative_precisions_needed

    # Load pools: {"8": [...], "9": [...], ...}
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/good_primes.json", "r", encoding="utf-8") as f:
        pools_raw = json.load(f)

    # Normalize keys to int and copy lists so we can pop from them
    pools: dict[int, list[int]] = {int(k): list(v) for k, v in pools_raw.items()}

    # Validate precisions exist in pools
    needed_counts = Counter(int(p) for p in precisions)
    unknown = [p for p in needed_counts if p not in pools]
    if unknown:
        raise ValueError(f"No primes available for precisions (bits): {sorted(set(unknown))}")

    # Ensure enough primes per bit bucket
    shortages = {p: (needed_counts[p], len(pools[p])) for p in needed_counts if needed_counts[p] > len(pools[p])}
    if shortages:
        msg = ", ".join(f"{p}-bit need {need} but have {have}" for p, (need, have) in shortages.items())
        raise RuntimeError(f"Insufficient primes: {msg}")

    # Draw primes (unique globally because we pop from each pool)
    assigned = []
    for p in precisions:
        prime = pools[p].pop()  # take one and remove it
        assigned.append(prime)

    # Pack back to same shape
    factors_per_row = unflatten_rows_cols(
        assigned, tuple(len(r) for r in full_q_list_precision))

    return factors_per_row

