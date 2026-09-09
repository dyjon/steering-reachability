# The set composition was not the confound. The displacement range was.

Run of `src/steering_fraction.py` at commit `bf36ef0`, 9 September 2026. Executed file hashes
`482e2b96...30a2`, byte-identical to the commit. Four instruction arms at harmful fraction
f ∈ {0.50, 0.20, 0.05, 0.00}, plus wikitext, layers 12 and 18, 128 bases against a disjoint
512-prompt donor pool.

---

## The prediction was falsified, cleanly

Written into the docstring before the run: if run 7's inversion was an artefact of building the
ladder set half refusable, then as f falls refusal stops being the principal axis, its subspace
energy collapses toward the wikitext ~9%, and graft/refusal climbs back toward the wikitext ~7.

| f | refusal E_sub, L12 | graft/refusal @0.3, L12 | refusal E_sub, L18 | graft/refusal @0.5, L18 |
|---|---|---|---|---|
| 0.50 | 97.7% | 0.32 | 98.3% | 1.40 |
| 0.20 | 97.7% | 0.26 | 98.9% | 1.20 |
| 0.05 | 94.4% | 0.26 | 95.7% | 1.10 |
| 0.00 | **69.0%** | **0.25** | **61.6%** | **1.13** |

**The manipulation worked and the ratio did not move.** Refusal's subspace energy falls 29
points at layer 12 and 37 at layer 18, so f genuinely changed the set's geometry. graft/refusal
sat at 0.32, 0.26, 0.26, 0.25.

At **f = 0.00** there is not one refusable prompt in the base set or the pool, refusal is no
longer close to the principal component, and refusal is exactly as loud as it was at f = 0.50.
Set composition does not cause the inversion. That confound is dead.

It also answers the sub-question f = 0.00 existed for: the effect is about the **chat format**,
not about refusal being behaviourally live. There is nothing to refuse in that arm.

## But there was a second confound, and it is mine

The two domains barely occupy the same displacement range:

| | span, ‖d‖ / activation norm |
|---|---|
| instructions, L12 | 0.18 – 0.75 |
| instructions, L18 | 0.27 – 0.80 |
| wikitext, L12 | 0.81 – 1.40 |
| wikitext, L18 | 0.74 – 1.43 |

They overlap in a sliver near 0.8. **The 0.25× is measured at a displacement where wikitext has
no data at all**, and the wikitext ~7 is measured where the instruction ladder cannot reach.
Decoupling the donor pool widened both ranges but did not make them meet.

At the one displacement where both domains have measurements — about 0.82 norms, layer 18:

| | graft/refusal | graft/sentiment |
|---|---|---|
| instructions | 2.42 | 2.55 |
| wikitext | 4.95 | 6.06 |
| domain effect | **2.04×** | **2.38×** |

**About 2×, not the ~18× the headline comparison implies. And the same size for sentiment,**
so at that depth it is not refusal-specific. Layer 12's overlap is worse — instructions top out
at 0.75 against wikitext's 0.81 floor — and extrapolating the instruction curve the short
distance to 0.81 gives roughly 0.69 against 4.65, a 6.7× domain effect. So the effect is real
and layer-dependent, somewhere between 2× and 7×, rather than the order of magnitude run 7
suggested.

## What is actually going on: the ratio depends on displacement

The thing both earlier writeups missed is that graft/refusal is not a constant.

| instructions f=0.00, L12 | | wikitext, L12 | |
|---|---|---|---|
| 0.18 norms | 0.26× | 0.81 norms | 4.65× |
| 0.23 norms | 0.21× | 0.92 norms | 5.25× |
| 0.29 norms | 0.25× | 1.03 norms | 6.75× |
| 0.35 norms | 0.21× | 1.11 norms | 7.10× |
| 0.75 norms | 0.62× | 1.40 norms | 6.67× |

It rises with displacement in both domains and the domains sit on different stretches of the
same rising curve. Whether the curves would coincide given a common range is **not determined by
this run**. What is determined: within the instruction domain the ratio is genuinely below 1 at
small displacement, meaning refusal there produces more KL than a step to a reachable
activation of the same size.

## Standing after four runs on this question

**Established.** graft/refusal inverts in the instruction domain at small displacement, is
independent of the harmful fraction, and survives the f = 0.00 control. Within that domain,
refusal is louder than a matched reachable step.

**Not established.** That this is a domain *difference* of the size run 7 implied. At matched
displacement it is 2× to 7×, and at layer 18 sentiment shows the same, so the refusal-specific
reading does not hold at that depth.

**Retracted from run 7.** The framing that the domains differ by an order of magnitude. That
compared instructions at 0.3 norms against wikitext at 0.8, which is not a comparison.

## The fix

Force a common displacement range rather than hoping for one:

1. **Interpolate the ladder.** Instead of taking whole donors, graft `α · d` for α < 1 toward a
   donor. That lands off `Im(F)` and so is no longer a reachable step, which is the whole point
   of the construction, so it will not do. The honest alternative is to build a donor pool with
   deliberately distant members — instructions from a different distribution (other languages,
   code, very long or very short) — to push the instruction ladder past 1.0 norms.
2. **Pull the wikitext floor down** with a far larger pool, so the nearest neighbour lands
   nearer than 0.74 norms. A 4096-prompt pool would roughly halve it.

Either fixes the overlap. Both together give a clean domain comparison across 0.2 to 1.4 norms,
and that is the run that settles it.

## Limits

Carried forward: one model, one position, next-token only, Thm 4.3 untouched, no ablation and
no test that the refusal direction controls refusal behaviour.

New: the sentiment column is chat-templated and comparable to runs 7 and 8 only, not to run 6.
And the f = 0.00 arm shows the effect is about instruction format rather than live refusal,
which means "refusal direction" is doing less work in the story than its name suggests.
