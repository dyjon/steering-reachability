# No single number survives

Run of `src/steering_kind.py` at commit `74eff12`, 9 September 2026. Executed file hashes
`d2a815eb...6c17`, byte-identical to the commit. Six cells per layer: two chat-templated arms ×
three donor kinds, four magnitudes each, layers 12 and 18.

**This lands on the third pre-registered branch, and it retracts the headline numbers of runs 9
and 10.**

---

## The two-way table

Exponent of the graft curve, and graft/refusal at 0.4 activation norms.

### Layer 12

| | self | own | xdom |
|---|---|---|---|
| wiki-chat | p=2.41, 0.40× | p=2.70, 0.91× | p=2.17, 0.55× |
| instr-chat | p=2.09, 0.24× | p=2.40, 0.37× | p=2.33, 0.11× |

### Layer 18

| | self | own | xdom |
|---|---|---|---|
| wiki-chat | p=1.95, 0.82× | p=2.04, 1.07× | p=1.72, 1.40× |
| instr-chat | p=2.16, 1.12× | p=2.54, 0.95× | p=1.76, 0.72× |

## The kind effect is not an offset. It changes sign.

| | layer 12 | layer 18 |
|---|---|---|
| own/xdom, wiki-chat | 1.61× | **0.65×** |
| own/xdom, instr-chat | 2.42× | 1.14× |
| own/self, wiki-chat | 2.06× | 1.18× |
| own/self, instr-chat | 1.10× | **0.77×** |

Run 10 found cross-domain donors quieter than own-domain at layer 12 and read it as a
multiplicative offset. At layer 18 the sign flips for wiki-chat: cross-domain donors are
*louder*. A quantity that reverses between two depths of the same model is not an offset.

## The domain effect depends entirely on which kind you measure it with

| kind | layer 12 | layer 18 |
|---|---|---|
| self | 1.68× | 0.73× |
| own | 2.45× | **1.13×** |
| xdom | 4.87× | 1.95× |

**The "domain effect" ranges from 0.73× to 4.87× across this table**, on the same model, the same
prompts and the same reference displacement, varying only by which donor kind is used to
measure it.

Run 10 reported a content effect of 1.58× at layer 18. The `own` cell here gives 1.13×, `self`
gives 0.73× — the other direction — and `xdom` gives 1.95×. Run 10's figure was the average of
whatever kind mixture its ladder happened to contain.

**So the 3.9× of run 9, and the 2.78× and 1.58× of run 10, are retracted as quantities.** They
describe a sampling of kinds, not a property of the model.

## Exponents differ, but part of that is my fitting

Graft exponents span 1.72 to 2.70, a 1.57× spread. Before reading that as real, three of twelve
ladders cover less than a factor of two in displacement:

| cell | span | p |
|---|---|---|
| L12 instr-chat xdom | **1.31×** | 2.33 |
| L18 instr-chat xdom | **1.52×** | 1.76 |
| L18 wiki-chat xdom | **1.83×** | 1.72 |

All three are `xdom`, and for a structural reason: ranks 0 through 64 of a *cross-domain* pool
sit at nearly the same distance, because everything in the other domain is far away and roughly
equidistant. So the cross-domain ladder barely moves, and its exponent is weakly determined.

That is a design fault, and it means the exponent spread is partly real and partly fitting
noise. It does not rescue the offsets above, which are read at a fixed reference and change sign
regardless.

## What does survive

**1. Templating flips the ordering at layer 12, robustly.** graft/refusal across all six
templated cells: **0.11, 0.24, 0.37, 0.40, 0.55, 0.91** — every one below 1. Raw wikitext at the
same layer in run 10: 2.26, 2.78, 3.23, 3.62, 3.97 — every one above 1. Both content types,
all three kinds, no exceptions. In a templated prompt at layer 12, refusal moves behaviour more
than a step to a reachable activation of the same size does.

This is the format effect, qualitatively, and it is the one thing in this line that has not
moved under any control.

**2. The graft's exponent exceeds refusal's in 11 of 12 cells.** Mean 2.35 against 1.71 at layer
12, 2.03 against 1.63 at layer 18. The one exception is layer 12 instr-chat `self`. So the gap
between them narrows as displacement grows, which is the same shape as the original
quadratic-steering finding and the most durable quantitative pattern here.

## Where this leaves the line

**Retracted:** every cross-domain ratio quoted in runs 9 and 10 as a number.

**Standing:** the qualitative flip at layer 12 under templating, and the graft-exponent-exceeds-
refusal-exponent ordering.

**The methodological lesson,** which is the real output of this run: a ratio between two curves
is only a number if both curves are sampled the same way. Every ladder in this line quietly
chose a kind mixture, and the ratios moved with it. Reporting a single "domain effect" was never
sound, and it took a two-way design to see that.

## Limits

Carried forward: one model, one position, next-token only, Thm 4.3 untouched, no ablation and no
test that the refusal direction controls refusal behaviour.

New: the `xdom` ladders have too little dynamic range to fit an exponent, by construction rather
than by accident. Fixing it needs a cross-domain pool that spans distance — a graded mixture
rather than a foreign one — which is a different experiment. And `self` is the only kind whose
construction differs between arms, so its column is the least comparable of the three.
