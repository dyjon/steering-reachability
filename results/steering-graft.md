# Steering against a reachable displacement, at matched norm

Run of `src/steering_graft.py` at commit `b8d3c9f`, 2 September 2026. Qwen2.5-0.5B-Instruct,
layer 12 of 24, last token position, 128 wikitext passages of exactly 96 tokens. 3m 20s on a
Kaggle T4.

Follows [`steering-sensitivity.md`](steering-sensitivity.md), which established that steering
is quieter than a random on-manifold direction at matched magnitude but left the KL axis
uncalibrated.

---

## Design

Swap the first *k* tokens of a passage for another passage's, holding length and the final
token fixed. The variant is a real prompt, so its layer-12 last-position activation is a
genuine element of `Im(F)`.

`d = h_last(variant) − h_last(base)`. **Graft** runs the *original* prompt with `h_last += d`.
Everything upstream of the hook is identical to a steering run, so the two differ only in the
vector added at one position, and the graft lands on a point prompting reaches. Steering and a
per-prompt random top-32 direction then fire at exactly ‖d‖.

`full` is the variant run end to end, no hook. Not a control — it moves every position — but
it is the only interpretable unit on the KL axis.

## Result

Median activation norm 16.0. Top-32 subspace holds 61.7% of activation variance. Steering
carries 12.7% of its energy there, against a 3.6% isotropic null.

| k | ‖d‖ | d in sub | graft | steering | subspace | full |
|---|---|---|---|---|---|---|
| 0 | 0.00 | — | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 8 | 1.77 | 9.5% | 0.0086 ± 0.0013 | 0.0071 ± 0.0020 | 0.0064 ± 0.0009 | 0.0819 ± 0.0342 |
| 24 | 2.88 | 10.1% | 0.0345 ± 0.0124 | 0.0281 ± 0.0133 | 0.0325 ± 0.0158 | 0.2429 ± 0.0599 |
| 48 | 4.45 | 10.8% | 0.1343 ± 0.0517 | 0.0564 ± 0.0230 | 0.0466 ± 0.0089 | 0.5161 ± 0.0881 |
| 72 | 6.37 | 12.4% | 0.2561 ± 0.0741 | 0.0900 ± 0.0250 | 0.1525 ± 0.0369 | 1.0426 ± 0.1327 |

± is standard error **over prompts**, which the previous script did not report at all. The
`k = 0` row is the sanity check and reads 0.0000 across.

### Steering is the only direction that stays quadratic

Fitting `KL ∝ ‖d‖^p` between the first and last rows:

| | p |
|---|---|
| steering | **1.98** |
| full | 1.99 |
| subspace | 2.48 |
| graft | **2.65** |

At the smallest displacement all three matched-norm columns are within error of each other
(0.0086 / 0.0071 / 0.0064), so the local curvature `vᵀFv` is roughly direction-independent.
Everything that separates them is higher-order, and steering has almost none of it: p = 1.98
is a clean Fisher regime across a 3.6× range of ‖d‖. Displacements toward reachable points
leave that regime.

The consequence is that graft/steering grows with distance: 1.21, 1.23, 2.38, **2.85**.

### The geometry does not explain it

`steering-sensitivity.md` closed with an additive energy-split model predicting steering's KL
from the fraction of its energy inside the principal subspace, to within 6% and 13% and with
no free parameters. **That model is dead here.**

Steering sits at 12.7% subspace energy; the k = 48 graft sits at 10.8%. Nearly identical
geometry. Their KLs differ by 2.4×, and in the wrong direction — steering has *more* subspace
energy and is *quieter*, where the previous run found subspace directions louder. Two
directions matched on the only geometric quantity we measured behave completely differently.

One thing did replicate. At the largest displacement subspace/steering = 1.69, against 1.68
from the previous run's independent prompt set. At smaller displacements the ratio is
noise-dominated (0.90, 1.16, 0.83) and should not be read.

### Reachable displacements are small

Expressed as fractions of the median activation norm, ‖d‖ runs 0.111 → 0.398. Swapping
**three quarters** of the context moves the last-position activation by 0.4 activation norms.
Steering at the δ = 1 used in the previous run moves it by 1.0.

So steering operates at roughly 2.5× the single-position displacement that a heavy context
swap produces. Within this variant family, at least: same length, same final token, same
corpus. A wider family would reach further.

### Where the behavioural difference lives

graft as a fraction of full: 10.5%, 14.2%, 26.0%, 24.6%. Grafting the last-position activation
alone recovers a quarter of the KL of the whole prompt change at large *k*, a tenth at small
*k*. The rest arrives through attention to the other positions.

## What this says about the paper, and a correction

The script's own docstring said `steering < graft` would mean a near-miss costs less than its
L2 implies, so Thm 4.2 has little behavioural bite. **That reading was wrong and the
conclusion runs the other way.**

The relevant direction is the one separating a steered point from its nearest reachable
neighbour, and the best proxy available for "directions among reachable points" is the graft
direction. Those are the *loudest* per unit norm, and they get louder faster than quadratic.
A given L2 miss along such a direction therefore costs **more** behaviourally than a generic
displacement of that size, not less. That supports the paper's caution rather than weakening
it.

Stated honestly: this is an inference, not a measurement. The nearest reachable point is never
located, so the residual's direction is never observed.

## Calibration, which was the point

Steering at δ = 1 produced 0.4337 nats in the previous run. Here two prompts sharing only their
last 24 of 96 tokens differ by 1.0426 nats, and two differing in just their first 8 tokens
differ by 0.0819.

So steering at operating strength sits between prompts that share half their context and
prompts that share a quarter of it. Not a small perturbation, and not a catastrophic one.

Cross-run consistency check: steering at ‖d‖ = 6.37 here gives 0.0900. Scaling quadratically to
15.3 predicts 0.519; the previous run measured 0.4337 on a different prompt set. Slightly
sub-quadratic, consistent with saturation, and close enough to trust both runs.

## Limits

- Sentiment steering has no vulnerability semantics. The paper uses refusal and persona
  vectors.
- One layer, one model, one steering vector, no error bar on steering's *direction* (only on
  its per-prompt KL).
- The variant family is narrow by construction, which caps ‖d‖ at 0.4 activation norms.
- Single position. Thm 4.3 is about trajectory divergence at i+1 and is untouched.
- Extrapolating graft's p = 2.65 out to ‖d‖ = 15.3 gives 2.61 nats against steering's measured
  0.4337, a 6× gap. That is extrapolation well beyond the data, and prompts may not be able to
  displace one position that far at all. Do not quote it.

## Next

Widen the variant family until ‖d‖ reaches 1.0 activation norms — different corpus, different
final token, different length — and check whether graft's exponent holds or bends. That
decides whether the 6× is real or an artefact of extrapolating from a narrow family.
