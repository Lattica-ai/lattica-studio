# Slot Operators

This folder contains operations that act on ciphertext slot structure and slot-wise aggregations.

## At a glance

| Operator | Operational intent |
| --- | --- |
| `HomConj` | Conjugates every slot with one key switch. Adding the input back (`x + HomConj()(x)`) gives `2 * Re(x)`, e.g. to get a real result from complex-packed inputs. |
| `HomExpand` | Assumes an input pattern of length `k` repeated `internal_n // k` times. Produces `k` ciphertext outputs, each carrying the same value across all slots. |
| `HomRotateSum` | Applies configured rotations and either sums rotated views or returns stacked rotated outputs, depending on operator configuration. |
| `HomRunningSum` | Performs staged cumulative aggregation over consecutive `k` elements, simultaneously across all `internal_n // k` groups (columns) induced by slot layout. |
| `HomSumSlots` | Performs slot-axis reduction over consecutive `k` elements, simultaneously across all `internal_n // k` groups (columns). |

## What `k` means

### Plaintext execution

`forward_clear` computes on the input's slot layout, as the ciphertext holds it,
and returns what decryption reads back:

- The slot axis of a value is laid out in rings of `internal_n` slots. A slot
  axis of at most `n_slots` values (its period) is zero-padded to `n_slots`, and
  the period is repeated over the ring. A longer slot axis spans whole rings,
  the last one zero-padded.
- The operator acts on those slots, padding and repeats included, so padding
  can mix into the result (a rotation brings in zeros after the last value, a
  sum adds the padded slots).
- Decryption reads back the slots that hold the values, dropping the padding
  and the repeats, at the input's positions on the slot axis.

Clear execution of a traced graph (`HomomorphicPipeline.forward_clear` with
`hom_params`) gives each slot operator its input's traced value (slot axis
`n_axis`, period `n_slots`) and the ring size `internal_n`. Outside a traced
graph, the input's last axis is the slot axis and is one whole ring, of its
length rounded up to a power of 2.

Per operator:

- `HomExpand` puts value `i` of the zero-padded period in every slot of the
  ring of output `i`: the values go on a new axis inserted at `expand_axis`
  (in the input's axes; -1 inserts before the last one), and the slot axis has
  `internal_n` slots. The slot axis must fit in its period. An `expand_axis`
  outside `[-(rank + 1), rank]` raises `ValueError`.
- `HomRunningSum` along `cumsum_axis`, when it is not the slot axis, is a
  cumulative sum. Along the slot axis, each ring is `k` rows of
  `internal_n / k` columns, and the sum runs down the rows of all blocks (row
  `r` of every block, then row `r + 1`); the blocks are the rings, or the
  entries of `blocks_axis_external`. `k` must be a power of 2 dividing
  `internal_n`.
- `HomSumSlots` adds, in each period of `n_slots` slots, the `k` slots
  `n_slots / k` apart: slot `j` gets `x[j] + x[j + n_slots / k] + ...`, and the
  output's period is `n_slots / k`. `k` (by default `n_slots`) must be a power
  of 2 dividing `n_slots`.
- `HomRotateSum` rotates every ring and sums or stacks the requested views.
  Sum mode always includes the original input once; stack mode includes it
  only when `add_identity_rotation` is set.

### Homomorphic execution

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
- `HomConj` is a single key switch and does not consume levels.
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
