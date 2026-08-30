# Slot Operators

This folder contains operations that act on ciphertext slot structure and slot-wise aggregations.

## At a glance

| Operator | Operational intent |
| --- | --- |
| `HomExpand` | Assumes an input pattern of length `k` repeated `internal_n // k` times. Produces `k` ciphertext outputs, each carrying the same value across all slots. |
| `HomRotateSum` | Applies configured rotations and either sums rotated views or returns stacked rotated outputs, depending on operator configuration. |
| `HomRunningSum` | Performs staged cumulative aggregation over consecutive `k` elements, simultaneously across all `internal_n // k` groups (columns) induced by slot layout. |
| `HomSumSlots` | Performs slot-axis reduction over consecutive `k` elements, simultaneously across all `internal_n // k` groups (columns). |

## What `k` means

`k` is the workload span in slots:

- In `HomExpand`, `k` is how many expanded outputs are produced per input slot
  pattern (conceptually, you can think of this as creating `k` copies/positions).
- In `HomSumSlots`, `k` is how many slot elements are reduced into a sum.
- In `HomRunningSum`, `k` is how many slot elements are aggregated in the
  running-sum range.

Because slot operations use rotations, the same op family can span a range of
execution styles: from highly parallel aggregation to staged trees. In staged
binary-style schedules, this is typically completed in `log2(k)` stages.

## Level/scale notes

- `HomExpand` and `HomRunningSum` consume levels because stages include masked
  multiplications in addition to rotations.
- `HomSumSlots` is rotation/addition-only in this model and does not consume
  levels.
- `HomExpand` and `HomRunningSum` compute consumed levels exactly from the
  selected stage count and `stages_per_level`.
- `stages_per_level` means "how many stages are packed into one consumed
  level". Consumed levels are `stage_count // stages_per_level`.
- Effective level drops are applied via modswitch tracing utilities.

### Stage model (exact)

- Stage count (tree depth) is selected in one of two ways:
  - Default schedule: exactly `log2(k)` stages.
  - User schedule: exactly `len(stage_sizes)` stages, with the contract
    `prod(stage_sizes) == k`.
- In user schedules, each entry in `stage_sizes` should be a power of 2
  (typically `>= 2`).
- `stages_per_level` controls level consumption after stage count is fixed.
  Consumed levels are `stage_count // stages_per_level`.
- `stages_per_level` is required to divide `stage_count` exactly.

Contract:

- Enforce `prod(stage_sizes) == k` and exact divisibility of `stage_count` by
  `stages_per_level` in your pipeline configuration.

### Numeric example

Given:
- `k = 64`
- `stage_sizes = [4, 2, 2, 4]`
- `stages_per_level = 2`

Then:
- `prod(stage_sizes) = 4 * 2 * 2 * 4 = 64 = k` (valid schedule)
- `stage_count = len(stage_sizes) = 4`
- `consumed_levels = 4 // 2 = 2`

If `stages_per_level = 3`, this setup is invalid because `3` does not divide
`stage_count=4`.

If `stages_per_level` is not set, the same setup consumes exactly `4` levels
(one level per stage).

