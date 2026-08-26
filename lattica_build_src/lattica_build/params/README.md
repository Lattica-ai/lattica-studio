# Params

`lattica-build` uses `HomParams` to define how your pipeline is represented in encrypted space.

Use this page to answer practical questions:
- how much level budget do I have,
- where can budget be spent,
- how do I keep scales compatible,
- which parameter knobs matter for my workload.

## What this package controls

The `params` package provides:
- the `HomParams` schema,
- level/scale tracing helpers,
- shape tracing helpers,
- prime selection for your configured modulus structure.

In practice, `HomParams` is the build-time contract between your graph and backend execution.

## `HomParams` at a glance

Main fields you will tune:

| Field | Why it matters |
| --- | --- |
| `n` | Ring size; `internal_n = n // 2` is the derived physical slot count, and controls packing capacity and cost envelope. |
| `n_slots` | Optional sub-ring slot count for the primary input (power of two). `None` means pack on the full ring (`internal_n`). |
| `full_q_list_precision` | Level-budget blueprint (rows/cols structure). |
| `pt_scale` | Default plaintext scale for the primary encrypted input. |
| `decomposition_type` | Key-switch decomposition mode (`BV` or `HYBRID`). |
| `num_special_primes` | Required for `HYBRID`; must be `0` for `BV`. |
| `bv_gadget_bits` | Gadget base bits for `BV`. |
| `err_std` | Encryption noise standard deviation. |
| `sk_hw` | Secret-key hamming weight (also used when bootstrapping params are created). |
| `num_init_rows` | Bootstrap-oriented starting-budget cap for the primary input: initialize fewer active rows up front when the input is expected to be bootstrapped later. |
| `bootstrapping_variant` | Bootstrapping flavor used when bootstrap is present. |

Fields populated during serialization:
- `boot_params`,
- `ring_switch_params`,
- `mod_chain`.

These are derived based on your graph (for example, when bootstrap or ring-switch ops appear).

## Level budget: rows/cols model

Unlike a single scalar "levels left" model, this library tracks level budget as an active subset of rows and columns.

This means levels are not treated as identical slots in a flat stack. Different rows can
carry different precision profiles, so you can spend budget with finer control over both
depth and precision at each stage of the graph.

Each `HomValue` carries:
- `active_rows`,
- `active_cols`,
- `pt_scale`.

`full_q_list_precision` defines the initial budget layout. Ops consume budget by dropping rows/cols.

If you want a simplified, near-uniform model, use repeated rows such as:

```python
full_q_list_precision = ((60, 30),) * N
```

for your chosen stage count `N`.

### Why this matters

Rows/cols let you plan budget spending with finer control:
- major steps via rows,
- finer granularity via cols inside a row.

This is especially useful when different parts of the graph have different level pressure.

### Designing custom `full_q_list_precision`

`full_q_list_precision` is a tuple of rows, and each row is a tuple of precision bit-sizes.

Example:

```python
full_q_list_precision = (
    (61, 35),
    (59, 33),
    (57, 31),
    (55,),
)
```

Design guidance:
- keep values in each row strictly decreasing,
- keep choices within available prime buckets,
- use multi-column rows when you want finer drop control,

Prime buckets are loaded from `good_primes.json`, so unsupported precision values will fail at build time.

## Modswitch controls in operators

Many ops expose:
- `with_modswitch`,
- `rows_budget`,
- and sometimes an op-level scale-up knob (often named `pt_scale`).

How to think about them:

- `with_modswitch=True`: op may spend budget (drop row/col) in its optional modswitch step.
- `with_modswitch=False`: skip that drop in the optional step.
- `rows_budget`: restrict where drops are allowed (placement control, not amount control).

This lets you keep budget spending predictable in larger graphs.

## Scale control and compatibility

Level and scale are related but separate concerns:
- level asks whether ciphertext budget is still available,
- scale asks whether numeric magnitudes are aligned for stable arithmetic.

### Add/Sub vs Mul expectations

For binary arithmetic, keep this mental model:

- `add` / `sub`:
  - if inputs are at the same active rows/cols level, scales should match,
  - if levels differ, alignment to a common lower level (including scale alignment) is handled automatically.

- `mul`:
  - inputs are aligned to a common lower level,
  - scales are not matched before multiplication; output scale is the product of input scales,
  - when `with_modswitch=True`, the multiplication path may then scale back down by a prime during the optional drop step.

### Ops that scale up constants

Some ops multiply by constants, and scaling behavior depends on how constants are represented.

Packed constants are always scaled.
For non-packed constants, scale-up is applied when constants are non-integer.

When an op-level `pt_scale` is provided, it gives you manual control over local scale-up.
Use this to correct or align scales across branches, but avoid values that are too small, or precision can degrade.

### `with_modswitch=True` and net scale preservation

When `with_modswitch=True` and an op does not provide an explicit scale-up value,
the default behavior is that the same factor is used for scale-up and subsequent drop.

Practical consequence: that optional step can preserve net `pt_scale` while still spending one budget unit.

## Decomposition configuration

`decomposition_type` chooses key-switch decomposition mode:

- `BV`:
  - `num_special_primes` must be `0`,
  - uses `bv_gadget_bits`.

- `HYBRID`:
  - requires `num_special_primes > 0`,
  - derives `g_base_bits` from special-prime count.

Pick decomposition settings with your backend performance profile in mind.

## Slot packing: `internal_n` vs `n_slots`

Two distinct quantities describe slot capacity:

- `internal_n` (read-only property, `n // 2`): the number of slots in a
  ciphertext. It is fixed by the ring size and is what shape tracing uses to
  locate the packed axis (`n_axis`).
- `n_slots` (optional field, default `None`): the sub-ring the input is
  packed on. Must be a power of two. When it is smaller than `internal_n`, the
  data is repeated `internal_n // n_slots` times across the ring.

Leave `n_slots` unset to pack on the full ring. Set it when your vector is shorter than `internal_n`.

`HomParams.n_slots` applies to the primary encrypted input. Additional encrypted
inputs can be packed on their own sub-rings via `HomomorphicPipeline.custom_n_slots`
(see [`base_classes/README.md`](../base_classes/README.md)); inputs not listed there
fall back to `HomParams.n_slots`.

## Bootstrapping and ring-switch interaction

You do not usually set `boot_params`/`ring_switch_params` directly.
During pipeline serialization:
- presence of bootstrap or ring-switch ops can trigger internal derived params,
- the active mod chain is built from your `HomParams` plus those derived sections.

For users, the key point is to size initial budget and scale assumptions to the graph you actually build.

## Slot-stage level behavior

For staged slot ops (`HomRunningSum`, `HomExpand`, `HomSumSlots`):
- `k` is the work size (for example: `HomExpand` expands one ciphertext to `k` outputs,
  while `HomRunningSum` and `HomSumSlots` reduce/sum `k` elements),
- the schedule is implemented as a binary-tree-style staged structure,
- stage depth is `log2(k)` by default or `len(stage_sizes)` for custom schedule,
- custom schedule should satisfy `prod(stage_sizes) == k`,
- `HomRunningSum` and `HomExpand` consume levels because each stage multiplies by a mask,
- for those two ops, `stages_per_level` sets how many stages are packed into one consumed level,
- for those two ops, `stages_per_level` should divide the selected stage count.

`HomSumSlots` follows the staged structure but does not consume levels.

See the [slot-operator guide](../operators/slots/README.md) for a worked
example.

## Practical workflow

1. Start with a conservative `full_q_list_precision` and baseline `pt_scale`.
2. Build and serialize your graph.
3. If level budget fails, adjust row/col layout and `rows_budget` placement.
4. If scale compatibility fails (especially add/sub joins), adjust op-level scale controls.
5. Re-run until both level and scale traces are stable for your target graph.
