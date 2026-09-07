# Across depth, and what the position confound was worth

Run of `src/steering_layers.py` at commit `95e813f`, 7 September 2026. Layers 6, 12 and 18 of
24, same 128 passages, same seed, same ladder. torch 2.10.0+cu128, transformers 5.0.0,
datasets 5.0.0.

First run pushed and fetched entirely through the Kaggle API. The file that executed hashes to
`52606e72...0676`, byte-identical to the committed script.

---

## The anchor held

Layer 12's `graft`, `steer_lo`, `subspace` and `full` columns came back **bit-identical** to
[`steering-graft-wide.md`](steering-graft-wide.md) at `4ccdfbc` — all six rows, all six
columns, and the fitted exponents (2.70 / 1.92 / 2.32) match as well. The refactor that added
the layer loop and the second steering construction changed nothing it should not have, so
differences at layers 6 and 18 are real.

## The gap holds across depth

| layer | median norm | graft | steer_lo | **graft / steer_lo** |
|---|---|---|---|---|
| 6 | 13.1 | 2.41 | 0.40 | **6.0×** |
| 12 | 16.0 | 3.20 | 0.59 | **5.5×** |
| 18 | 34.7 | 8.19 | 1.10 | **7.5×** |

All at a displacement of exactly one activation norm, interpolated between the two bracketing
rows. The ratio runs 5.5 to 7.5 with no sign of collapsing, and layer 18 is the widest rather
than the narrowest.

**So the gap is a property of the model, not of layer 12.** Note the ratio is the scale-free
quantity here and the absolute KLs are not comparable across depth: the residual norm grows
2.2× from layer 12 to 18, so "one activation norm" means something different at each.

### Steering stays quadratic at every depth

Log-log slopes over all rows:

| | layer 6 | layer 12 | layer 18 |
|---|---|---|---|
| graft | 2.62 | 2.70 | 2.33 |
| steer_lo | 2.12 | 1.92 | 1.94 |
| **steer_hi** | **2.07** | **1.98** | **2.06** |
| subspace | 2.13 | 2.32 | 2.13 |

The position-matched steering vector sits at 2.07, 1.98, 2.06 — mean **2.04**, and closer to
exactly quadratic than the original construction at every depth. The graft averages 2.55 and is
above 2 everywhere. The finding that steering stays in its Fisher regime while reachable
displacements leave it is robust to both depth and the choice of steering vector.

## The position confound: real, and it did not matter

The steering vector had been built from sentences three to five tokens long, read at sequence
positions 2 to 4, then applied at position 95. `steer_hi` rebuilds it from the same sentences
with neutral filler prepended so each ends at position 95, with the filler shared between the
positive and negative arms so it cancels in the difference of means.

| layer | cos(lo, hi) | angle | shared variance | graft/steer_lo | graft/steer_hi |
|---|---|---|---|---|---|
| 6 | +0.887 | 28° | 79% | 6.0× | 5.8× |
| 12 | +0.837 | 33° | 70% | 5.5× | 5.1× |
| 18 | +0.786 | 38° | 62% | 7.5× | 7.2× |

**Geometrically the confound is real.** The two constructions sit 28 to 38 degrees apart and
share only 62 to 79 percent of their variance. They are not the same vector, and the alignment
degrades with depth, which is what you would expect if deeper layers encode more
position-dependent structure.

**Behaviourally it was worth almost nothing.** Every ratio moves by less than half a unit and
no conclusion changes. The limitation declared in three writeups was correctly identified and
turns out not to have been load-bearing.

That is the useful form of the answer. "We flagged it and it did not matter" is only worth
saying because the alternative was live until it was measured.

## Structure is stable across depth

| layer | top-32 explains | d in sub at k = all |
|---|---|---|
| 6 | 61.1% | 42.0% |
| 12 | 61.7% | 42.8% |
| 18 | 60.2% | 40.8% |

Both quantities barely move. The gap between "how much variance the principal subspace holds"
and "how much of an unrelated-pair displacement lands in it" is a stable structural fact about
this model rather than an accident of one depth.

## Limits

- **Small-k rows at layer 6 are noise.** KLs of 0.0029 and 0.0013 with error bars a third of
  the mean. Only the top two or three rows at each layer carry the fits.
- Slopes here are whole-range OLS. `steering-graft-wide.md` showed that averages across real
  curvature and understates the graft at the top; the same caveat applies to every number in
  the exponent table above.
- The layer-18 interpolation is bracketed tightly (0.82 and 1.03 norms), so it is the most
  reliable of the three; layer 6 brackets 0.78 and 1.21 and is the loosest.
- Still one model, one position, still sentiment rather than refusal, still nothing touching
  Thm 4.3.

## Next

The two structural questions are answered, so what remains is the one that carries the safety
argument: **a refusal vector on the instruct model.** Sentiment has stood in for it through
five runs and has no vulnerability semantics at all.

If the ratio survives that substitution, the threshold table in
[`steering-graft-wide.md`](steering-graft-wide.md) becomes a statement about the thing the
paper is actually cautioning about.
