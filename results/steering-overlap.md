# Two curves, and the offset is refusal-specific

Run of `src/steering_overlap.py` at commit `a733a21`, 9 September 2026. Executed file hashes
`f3c77909...16f8`, byte-identical to the commit. Hybrid ladders spanning a common range of
displacement in both domains, layers 12 and 18, 128 bases each.

**Anchor exact.** Wikitext `swap8`, `swap24` and `swap72` graft returned 0.0086, 0.0345 and
0.2561 at layer 12, matching `4ccdfbc` to the digit. The hybrid construction did not disturb
what it reused.

---

## The answer: two curves

Domain offset is the wikitext ratio divided by the instruction ratio at the same fraction of
each domain's own median activation norm.

### Layer 18

| fraction | graft/refusal wiki | instr | **offset** | graft/sentiment wiki | instr | offset |
|---|---|---|---|---|---|---|
| 0.2 | 3.48 | 0.92 | **3.79** | 3.69 | 2.71 | 1.36 |
| 0.3 | 3.48 | 0.76 | **4.56** | 3.84 | 2.31 | 1.66 |
| 0.4 | 3.51 | 0.90 | **3.90** | 4.02 | 2.85 | 1.41 |
| 0.5 | 3.58 | 0.87 | **4.12** | 4.23 | 2.72 | 1.56 |
| 0.6 | 3.64 | 0.95 | **3.82** | 4.42 | 2.90 | 1.52 |

**The refusal offset is a constant ~3.9×** across a threefold span of displacement, varying by
under 10%. It is not a displacement artefact. The two domains lie on two parallel curves.

**And the offset is refusal-specific.** Sentiment shows one too, but at ~1.5×, so refusal's is
about 2.6 times larger. Both directions get relatively louder in the instruction domain;
refusal gains far more.

### Layer 12

| fraction | graft/refusal offset | graft/sentiment offset |
|---|---|---|
| 0.2 | 8.78 | 1.17 |
| 0.3 | 10.91 | 1.21 |
| 0.4 | 19.26 | 2.00 |
| 0.5 | 20.34 | 2.30 |
| 0.6 | 12.04 | 1.67 |

Same sign, much larger, and not constant. The raggedness has a specific cause, below.

## This corrects run 8

[`steering-fraction.md`](steering-fraction.md) concluded the effect was **not** refusal-specific,
on the grounds that its single overlapping displacement gave 2.04× for refusal and 2.38× for
sentiment. That point was at 0.82 norms, the extreme end of both ladders where interpolation was
least reliable, and its instruction arm was the f = 0.50 base set rather than this one. With
real overlap across 0.2 to 0.6, refusal and sentiment separate clearly. **Withdraw the
not-refusal-specific reading.**

## The mechanism is refusal gaining, not the graft going quiet

At matched small displacement, both columns are louder in the instruction domain:

| | graft | refusal |
|---|---|---|
| layer 12, ~0.10 norms | **6.6×** louder | **21.2×** louder |
| layer 18, ~0.12 norms | **9.7×** louder | **16.8×** louder |

So the instruction domain is more behaviourally sensitive across the board, which makes sense at
the assistant-generation position where the model is about to commit to a response. Refusal
simply gains far more than a reachable step does, and that difference is the offset.

## The `passage` condition is off-curve, and that matters

At layer 12 the instruction graft rises 0.3043 → 0.3718 going from 0.29 to 0.46 norms, then
jumps to 3.15 by 0.75. The 0.46 point is the `passage` donor, a wikitext passage put through the
chat template. Its graft KL is far too low for its displacement.

That is why layer 12's offsets are ragged: the interpolation at 0.4 and 0.5 runs through it.

The finding underneath is worth more than the nuisance: **graft KL is not a function of ‖d‖ even
within one domain.** Moving an instruction's activation toward a pasted-passage activation is
behaviourally quiet for its size, while moving it toward another instruction is loud. The kind
of reachable displacement matters, not only its magnitude. Every earlier writeup treated the
graft curve as one-dimensional in ‖d‖. It is not.

## Where the line now stands

**Settled.** There is a real domain effect. At layer 18 it is a constant ~3.9× on graft/refusal
across 0.2 to 0.6 norms, and it is refusal-specific, with sentiment showing ~1.5×. In the
instruction domain graft/refusal sits near 0.9, meaning refusal produces roughly as much
behavioural change as a step to a reachable activation of the same size, where on wikitext it
produces a third to a quarter as much.

**Withdrawn.** Run 7's order-of-magnitude framing (inflated, and measured off-overlap). Run 8's
not-refusal-specific reading (one unreliable point).

**Still open.** Layer 12's offset is large but unstable, and the one-dimensional-in-‖d‖
assumption is now known to be false, which means the curves themselves need re-examining with
displacement *kind* as a second variable.

## Limits

Carried forward: one model, one position, next-token only, Thm 4.3 untouched, no ablation and no
test that the refusal direction controls refusal behaviour.

New: the two arms use different donor constructions by design, so a residual construction effect
cannot be fully separated from a domain effect — the `passage` anomaly is direct evidence that
construction matters. The cleanest follow-up holds the construction fixed and varies only the
base domain, which is exactly what the rank ladder did and what could not reach a common range.
That tension is not yet resolved.
