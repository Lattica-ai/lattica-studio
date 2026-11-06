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

# The kernels keep every modulus below 2**62, so the primes of a 62-bit row come from just below their powers of two
# (good_primes_below.json) rather than just above them (good_primes.json), and the row's modulus stays under it.
WORD_LIMIT_BITS = 62


def get_primes_from_precisions_list(full_q_list_precision, root_order=1):
    """
        Create a tensor full_q_list with the same shape as full_q_list_precision,
        where each entry is a unique 'good prime' drawn from the JSON pool that
        matches its bit-length (precision value). Raises if any bucket runs out.
        Every prime drawn is 1 modulo root_order, so that it has a primitive root_order-th root of unity (the pools'
        primes have those of the ring dimensions they serve; a conjugate-invariant ring of dimension N needs 4N).
        """
    precisions = []   # (bits, below): a prime of about 2**bits, from below 2**bits for a row at the word limit
    # Verify precision is decreasing in each column
    for row in full_q_list_precision:
        row = torch.tensor(row)
        diffs = row[:-1] - row[1:]
        if not (diffs > 0).all():
            raise ValueError("Precision values must be strictly decreasing in each column.")
        relative_precisions_needed = diffs.tolist() + [int(row[-1])]
        below = int(row[0]) == WORD_LIMIT_BITS
        precisions += [(int(p), below) for p in relative_precisions_needed]

    # Load pools: {"8": [...], "9": [...], ...}, keyed (bits, below); copy the lists so we can pop from them
    pools: dict[tuple[int, bool], list[int]] = {}
    for name, below in (("good_primes.json", False), ("good_primes_below.json", True)):
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/{name}", "r", encoding="utf-8") as f:
            pools.update({(int(k), below): [p for p in v if (p - 1) % root_order == 0]
                          for k, v in json.load(f).items()})

    # Validate precisions exist in pools
    needed_counts = Counter(precisions)
    unknown = [p for p in needed_counts if p not in pools]
    if unknown:
        raise ValueError(f"No primes available for precisions (bits, below 2**bits): {sorted(unknown)}")

    # Ensure enough primes per bit bucket
    shortages = {p: (needed_counts[p], len(pools[p])) for p in needed_counts if needed_counts[p] > len(pools[p])}
    if shortages:
        msg = ", ".join(f"{bits}-bit{' (below)' if below else ''} need {need} but have {have}"
                        for (bits, below), (need, have) in shortages.items())
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

