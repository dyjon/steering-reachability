# Steering reachability — twelve runs

Follow-up to *Steered LLM Activations are Non-Surjective* (Mishra et al., 2026).
Qwen2.5-0.5B-Instruct, residual stream, single position, next-token KL.
torch 2.10.0+cu128, transformers 5.0.0, datasets 5.0.0.

**The question.** Thm 4.2 is about exact collision. Vulnerability is about behaviour, and
missing the reachable set by a hair and by a mile are both probability zero. So: at a given
activation-space distance, how much does behaviour actually differ?

**The construction.** Take two real prompts, subtract their layer-*j* last-position activations,
and add that difference back onto the first prompt's run. Everything upstream is identical to a
steering run, so the two differ only in the vector added at one position — and this one lands on
a point some prompt actually produces. Call it the **graft**. Compare steering against it at
matched magnitude.

Each run below exists because the previous one had a flaw, usually one named in its own writeup
before it was measured. Every run from the fifth on was pushed and fetched through the Kaggle
API with the executed file verified by hash against the commit it claims to be.

---

## The runs

| # | | Question | Answer |
|---|---|---|---|
| 1 | [sensitivity](results/steering-sensitivity.md) | Does steering beat a random direction of the same norm? | 1.23×, flat. **Baseline was a straw man** — an isotropic direction in 896 dims points nowhere the data goes |
| 2 | [subspace](results/steering-sensitivity.md) | Against a direction that points somewhere? | Steering is the **quietest** tested: 0.4337 vs 0.7289 |
| 3 | [graft](results/steering-graft.md) | Versus a step to a reachable point at the same distance? | Steering fits ‖d‖^1.98, the graft ^2.65. Kills run 2's geometric model |
| 4 | [graft-wide](results/steering-graft-wide.md) | At the magnitude steering actually uses? | 5.4× at one activation norm. Run 3 replicated bit-identically |
| 5 | [layers](results/steering-layers.md) | Property of the model or of layer 12? | Holds at 6/12/18: 6.0×, 5.5×, 7.5×. Position confound real (28–38°) and behaviourally worth <0.5 units |
| 6 | [refusal](results/steering-refusal.md) | Does it hold for refusal, not sentiment? | 5.9× and 6.5×. Two near-orthogonal directions, both exponent ≈2 |
| 7 | [instruct](results/steering-instruct.md) | Where refusal is live? | **Inverts.** 0.24 against ~7 |
| 8 | [fraction](results/steering-fraction.md) | Is that caused by a half-refusable prompt set? | No. Ratio flat at 0.32/0.26/0.26/0.25 while subspace energy fell 97.7%→69.0% |
| 9 | [overlap](results/steering-overlap.md) | Same displacement range in both domains? | Two curves, constant 3.9× offset at layer 18 |
| 10 | [format](results/steering-format.md) | Content, or chat template? | Format 2.78×, content 1.58×. Templating identical text lifts refusal's subspace alignment 8.8%→49.9% |
| 11 | [kind](results/steering-kind.md) | Does donor kind change the answer? | **0.73× to 4.87×** on the same prompts at the same displacement. Retracts 9 and 10 |
| 12 | [graded](results/steering-graded.md) | Was run 11 fitting noise? | No. And the graft **falls** as displacement rises: +16% distance, −38% KL, monotone |

## What stands

**Graft KL is not a function of displacement.** Run 12 shows it directly: five ordered points
with distance rising and the effect falling. Two donors at ‖d‖ 5.39 and 5.38 give graft KL
1.4502 and 1.0160. Any ladder fitted against distance alone mixes conditions differing by 1.4×
to 2.3× at identical distance. This is the one I would most want checked by someone else.

**Chat templating flips the ordering at layer 12.** Raw text: a reachable step moves behaviour
more than refusal, all five conditions, 2.26 to 3.97. Templated: the other way, all six
conditions, 0.11 to 0.91. Both content types, all three donor kinds, no exceptions. No account
of why.

**The graft's growth exponent exceeds refusal's in 11 of 12 conditions**, mean 2.35 against 1.71
at layer 12. So the gap between them narrows as displacement grows.

**Scoped to raw wikitext and standing as measurements:** steering is quieter than a random
on-manifold direction at matched norm; steering stays quadratic (2.03 twice) while the graft
leaves that regime; a reachable displacement at one position is small, 0.40 activation norms for
a three-quarter context swap against steering's 1.0; grafting one position recovers 10–26% of
the KL of the whole prompt change.

## What was retracted

| | killed by |
|---|---|
| The 1.23× isotropic result | run 2 |
| A geometric model predicting steering's KL to 6% and 15% | run 3 |
| A reading rule written into run 3's own docstring | run 4 |
| **The 5.4×** | run 11 |
| Run 9's constant 3.9× domain offset | run 11 |
| Run 10's 2.78× format and 1.58× content split | run 11 |
| "Not refusal-specific", from a single overlapping point | run 9 |
| Treating graft KL as one-dimensional in ‖d‖ | run 12 |
| Two graded cells with negative fitted exponents | run 12, self |

Nine against three. Most came from a control the previous writeup had already flagged as
missing, which is the order that makes a retraction cheap.

## Limits, carried throughout

One model, one layer family, one position, next-token only. Thm 4.3 is about trajectory
divergence at *i+1* and is untouched. No ablation was run and nothing here tests that the
refusal direction actually controls refusal behaviour — "refusal direction" is a claim about how
the vector was built. The refusal vector uses 96 AdvBench/Alpaca pairs; only AdvBench's `goal`
column is read. Three of twelve cells in run 11 span under 2× in displacement and their
exponents are weakly determined.

## Running any of it

Each script is standalone and self-documenting; the module docstring states what the run is for,
what it fixes from the previous one, and how to read the output.

```
pip install torch transformers datasets
python src/steering_graded.py
```

A T4 is enough. The longest run is about four minutes.
