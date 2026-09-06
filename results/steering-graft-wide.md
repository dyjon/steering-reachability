# Steering at its own operating magnitude

Run of `src/steering_graft_wide.py` at commit `4ccdfbc`, 6 September 2026. Same model, layer,
prompt set and seed as [`steering-graft.md`](steering-graft.md); the only change is two extra
rows on the ladder. 229.7 s on a Kaggle T4 x2.

**torch 2.10.0+cu128, transformers 5.0.0, datasets 5.0.0.** The previous run's versions were
unrecoverable, so these are the first recorded ones.

---

## Both checks passed

**Replication.** Rows k = 0 through 72 came back **bit-identical** to `b8d3c9f`, every digit of
every mean and every standard error. The extension changed nothing it should not have.

**The falsifiable prediction held.** The graft writeup explained `d in sub` sitting near 10% by
arguing that related prompts cancel their shared structure. That predicted the fraction must
climb once variants stop being related. It does:

| k | 8 | 24 | 48 | 72 | 95 | all |
|---|---|---|---|---|---|---|
| d in sub | 9.5% | 10.1% | 10.8% | 12.4% | **27.8%** | **42.8%** |

A 4.5× climb. The explanation survives.

It lands at 42.8% rather than the 61.7% the top-32 holds, and that gap is itself informative:
the basis is estimated from 128 samples in 896 dimensions, so the sample covariance has rank at
most 127 and **61.7% is an in-sample figure**. 42.8% is the honest out-of-sample number. Any
future use of "explained variance" here should say which one it means.

## Result

Median activation norm 16.0, so `/norm` is displacement in activation norms. Steering at
δ = 1 sits at ‖d‖ = 16.0, which the last two rows now bracket.

| k | ‖d‖ | /norm | d in sub | graft | steering | subspace | full |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | — | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 8 | 1.77 | 0.11 | 9.5% | 0.0086 ± 0.0013 | 0.0071 ± 0.0020 | 0.0064 ± 0.0009 | 0.0819 ± 0.0342 |
| 24 | 2.88 | 0.18 | 10.1% | 0.0345 ± 0.0124 | 0.0281 ± 0.0133 | 0.0325 ± 0.0158 | 0.2429 ± 0.0599 |
| 48 | 4.45 | 0.28 | 10.8% | 0.1343 ± 0.0517 | 0.0564 ± 0.0230 | 0.0466 ± 0.0089 | 0.5161 ± 0.0881 |
| 72 | 6.37 | 0.40 | 12.4% | 0.2561 ± 0.0741 | 0.0900 ± 0.0250 | 0.1525 ± 0.0369 | 1.0426 ± 0.1327 |
| 95 | 14.19 | 0.89 | 27.8% | 1.6824 ± 0.1842 | 0.4577 ± 0.0900 | 0.9198 ± 0.1419 | 5.0815 ± 0.2512 |
| all | 18.76 | 1.17 | 42.8% | 7.2912 ± 0.3614 | 0.8068 ± 0.1053 | 1.7675 ± 0.1890 | 10.4783 ± 0.3979 |

`k = 95` shares only the final token with the base; `k = all` is a different passage entirely.
Letting the final token move was what unlocked the range: it had been holding the single
largest contributor to the last-position residual constant.

### The comparison this run existed to make

Interpolating between the two bracketing rows to ‖d‖ = 16.0:

| | at ‖d‖ = 16.0 |
|---|---|
| steering | **0.58** nats |
| graft | **3.16** nats |
| ratio | **5.4×** |

**At steering's own operating magnitude, moving the same distance to a reachable point changes
behaviour 5.4× more than steering does.** That is now a measurement between two measured rows
rather than an extrapolation.

The graft writeup extrapolated from k ≤ 72 and predicted 2.61 nats and about 6×, with a note
saying not to quote it. Flagging it was right; the number also happened to hold.

### Steering is quadratic all the way out

Local exponents between consecutive rows, which the single OLS fit hides:

| segment | graft | steering | subspace | full |
|---|---|---|---|---|
| 8 → 24 | 2.85 | 2.83 | 3.34 | 2.23 |
| 24 → 48 | 3.12 | 1.60 | 0.83 | 1.73 |
| 48 → 72 | 1.80 | 1.30 | 3.31 | 1.96 |
| 72 → 95 | 2.35 | **2.03** | 2.24 | 1.98 |
| 95 → all | **5.25** | **2.03** | 2.34 | 2.59 |

The first three segments are noise-dominated, with KLs in the hundredths and error bars up to
40% of the mean. The last two are where the numbers are large enough to fit. There:

- **steering reads 2.03 and 2.03.** Exactly quadratic, twice, over a 2.9× span of displacement.
  It stays in its Fisher regime out past a full activation norm.
- **graft goes 2.35 then 5.25.** It leaves that regime and accelerates hard.

The whole-range OLS numbers (graft 2.70, steering 1.92, subspace 2.32, full 2.01) average
across that curvature and understate what graft does at the top. Quote the local exponents.

### Grafting one position eventually dominates

graft as a fraction of full: 10.5%, 14.2%, 26.0%, 24.6%, 33.1%, **69.6%**. When the
last-position activation comes from an unrelated passage it accounts for 70% of the behavioural
difference of the whole prompt change, up from a tenth when the passages are near-identical.

## What this does to the question

The ratio graft/steering runs **1.21, 1.23, 2.38, 2.85, 3.68, 9.04**.

At small displacement direction barely matters. At 0.11 activation norms the three matched-norm
columns are 0.0086, 0.0071, 0.0064 — within error of each other, and all negligible in absolute
terms. At 1.17 norms they are 7.29, 0.81, 1.77 and direction is almost the whole story.

**So the reachability question reduces entirely to how close a prompt can get.** That is part
(a), the SipIt-style search that was set aside as intractable. This run turns it into a
threshold:

| prompt gets within | behavioural difference |
|---|---|
| ~0.1 activation norms | KL ≈ 0.01. Negligible. Steering **is** behaviourally reproducible |
| ~0.4 norms | KL ≈ 0.26. Small but real |
| ~0.9 norms | KL ≈ 1.7. Not reproducible |

That is the deliverable. The bridge question is no longer "is behaviour continuous enough" but
"measure ε and read it off this curve." Whoever runs the inversion has a number to hit.

## What it says about the paper

Reachable-type directions are the loudest per unit norm and they accelerate, so an L2 miss
along one costs more behaviourally than a generic displacement of that size. That supports the
caution in the paper.

But the same table says the caution is only load-bearing if ε is large. If prompt search closes
to within a tenth of an activation norm, formal non-surjectivity has no behavioural consequence
at all. Both halves are in the numbers and neither should be quoted without the other.

Still an inference in one respect: the nearest reachable point is never located, so the
direction of the actual residual is never observed. The graft direction is a proxy for it.

## Limits

Carried forward unchanged from [`steering-graft.md`](steering-graft.md): the position confound
in the steering vector, sentiment having no vulnerability semantics, one layer, one model, one
steering vector, one position, and Thm 4.3 untouched.

New here:

- The three smallest segments cannot support an exponent. Their error bars reach 40% of the
  mean. Only the 72 → 95 and 95 → all segments are fit-worthy.
- The interpolation to ‖d‖ = 16.0 uses the local exponent between the two bracketing rows.
  It is an interpolation, not an extrapolation, but it is still a two-point fit.
- `k = all` grafts an activation from an unrelated passage onto this passage's prefix. The
  resulting state is a real element of `Im(F)` at that position but an odd one in context, and
  some of the 7.29 nats is that incoherence rather than distance as such.

## Next

The threshold table is the useful artefact and it is worth hardening before October:

1. **More prompts.** 128 gives error bars of 11% on the k = 95 graft. 512 would quarter that
   and make the local exponents fittable further down the ladder.
2. **A second layer.** Everything so far is layer 12 of 24. If the 5.4× holds at layer 6 and 18
   it is a property of the model rather than of one depth.
3. **A refusal vector**, where the vulnerability semantics actually live.
