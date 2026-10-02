# The three controls: one claim made, two retired, and a common cause

Run of `src/steering_controls.py` at commit `ef7fbdc`, 1 October 2026. Executed file hashes
`afedf5a9...b86d`, byte-identical to the commit. Layers 12 and 18, 128 prompts per arm, 95%
bootstrap intervals over prompts (2,000 resamples) on every ratio and exponent.

**Anchor exact.** Part 2's raw arm returned graft 0.0086, 0.0345, 0.1343, 0.2561 at layer 12,
matching `4ccdfbc` to the digit.

This run exists because the [1 October audit](../README.md#reopened-by-audit-1-october) found
that two standing claims did not hold up against the code and the strongest remaining one had no
error bars.

---

## Part 1 — the kind effect, now with intervals. Statable.

Cross-kind over own-kind graft KL, at matched distance, paired over the prompts where both
selections landed within 15% of the target.

| layer | base | target | cross/own | 95% CI |
|---|---|---|---|---|
| 12 | passage | 0.4 | 0.67 | [0.60, 0.74] |
| 12 | instruction | 0.4 | 0.58 | [0.49, 0.71] |
| 12 | instruction | 0.5 | **0.51** | [0.42, 0.62] |
| 18 | passage | 0.4 | 1.60 | [1.43, 1.78] |
| 18 | passage | 0.5 | **1.28** | [1.19, 1.39] |
| 18 | passage | 0.7 | 1.77 | [1.63, 1.94] |
| 18 | instruction | 0.4 | 1.02 | [0.86, 1.22] |
| 18 | instruction | 0.5 | 0.74 | [0.68, 0.81] |
| 18 | instruction | 0.7 | **0.58** | [0.54, 0.63] |

**Eight of nine intervals exclude 1.** The three cells run 13 reported without error bars —
0.51, 1.27, 0.58 — come back as 0.51, 1.28 and 0.58, all three with intervals clear of 1. The
1.27 I said should not be quoted turns out to be the tightest of them.

The layer-18 passage-arm reversal is significant at every target.

## Part 2 — the templating flip. Retired; it was the vector's origin.

Two refusal vectors from the same 96 AdvBench/Alpaca pairs: one built from chat-templated
prompts as in every earlier run, one built from the same prompts untemplated. Donors built
identically in both arms — the same passages with their first *k* tokens swapped, raw in one arm
and inside a chat turn in the other.

**The two vectors are nearly orthogonal**: cos +0.125 at layer 12, +0.088 at layer 18.

**Each is louder in the format it was built from** (refusal KL at k = 72):

| | native | foreign | ratio |
|---|---|---|---|
| L12 raw arm | 0.1763 | 0.0887 | 2.0× |
| L12 chat arm | 0.2424 | 0.0615 | **3.9×** |
| L18 raw arm | 0.2346 | 0.1651 | 1.4× |
| L18 chat arm | 0.5183 | 0.4006 | 1.3× |

**Holding the vector fixed, the template stops mattering at layer 12.** Graft over refusal with
the raw-built vector at k = 72: 1.45 [0.77, 2.27] raw, 1.36 [1.06, 1.79] templated. No
detectable difference. With the templated-built vector, the original claim: 2.89 → 0.35.

So the flip at layer 12 came from building the refusal vector in templated space and applying
it to raw text. **The claim is retired.** At layer 18 a residual template effect survives with
the raw-built vector — 2.45 → 1.08, about 2.3× — but it never crosses 1.

One honest qualification: built at the last token of an untemplated instruction, before any
assistant turn exists, the raw vector may encode *harmful content* rather than *refusal*. Nothing
here ablates either vector, so neither is shown to control refusal. What is shown is that the
difference-of-means direction for the same prompt pairs depends heavily on format, and that the
choice moves behaviour by up to 3.9×.

## Part 3 — exponents along single rays. The superquadratic graft is retired.

Each graft difference scaled along its own ray, α·d for α ∈ {0.25, 0.5, 0.75, 1}:

| | graft | refusal (tmpl) | refusal (raw) |
|---|---|---|---|
| L12 raw | 2.16 [1.75, 2.69] | 2.26 [2.17, 2.35] | 2.10 [1.76, 2.48] |
| L12 chat | 2.04 [1.97, 2.13] | 2.06 [2.03, 2.10] | 2.04 [2.00, 2.08] |
| L18 raw | 2.21 [1.95, 2.48] | 2.04 [1.99, 2.09] | 2.14 [2.06, 2.24] |
| L18 chat | 2.08 [2.03, 2.13] | 1.93 [1.90, 1.95] | 2.04 [1.98, 2.09] |

**Along a single direction, the reachable step is as quadratic as steering is.** The exponents of
2.65 and 5.25 reported for the graft ladder in runs 3 and 4 came from the donor — and therefore
the direction — changing between rungs, not from curvature. That claim is retired.

The rays here reach 0.15 to 0.40 activation norms; steering's quadratic behaviour out to about
one norm stands from runs 4 and 5.

## The common cause

Put the three parts next to the retraction table and one pattern accounts for nearly all of it.

| what looked like an effect of distance | what it was |
|---|---|
| The 5.4× | a mixture of donor kinds — directions |
| "Reachable steps grow faster than quadratic" | the direction changing between rungs |
| The 3.9× domain gap, the 2.78×/1.58× split | donor kind — direction |
| The graft falling as distance rises (run 12) | donor kind — direction |
| The templating flip | the refusal vector's origin — its direction |

**Along any single direction, behavioural change grows as distance squared. Every effect in this
line that looked like an effect of distance was an effect of direction.**

That is what a local quadratic form predicts — next-token KL ≈ ½ dᵀFd for the Fisher matrix
*F* — with *F* strongly anisotropic. The data are *consistent* with that picture; *F* itself has
not been measured.

## What it says about the paper

Thm 4.2 is stated in terms of collision, and the natural follow-up question has been posed in
terms of distance: how close can a prompt get to a steered state? These results say L2 distance
is the wrong yardstick. A steered state and its nearest reachable neighbour can be equally far
apart and differ by a factor of two or more in behaviour depending on the direction between
them. The quantity that governs behavioural reproducibility is the Fisher-weighted residual, not
its length.

## Next

The confirmatory experiment follows directly: estimate *F* at the base state (Fisher-vector
products make dᵀFd cheap without forming *F*) and test whether ½dᵀFd predicts the measured KL
across every kind, domain and vector in this repo. If it does, the whole line collapses into one
equation and one plot. Then a second model.

## Limits

Carried throughout: one model, one position, next-token only, Thm 4.3 untouched, no ablation.
New: the ray exponents rest on four points each; the raw-built vector's semantics are uncertain
as described above; part 1 reports only cells where both kinds were covered, which excludes the
sub-0.4-norm region by construction.
