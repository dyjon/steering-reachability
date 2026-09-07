# Refusal, the direction the paper is actually about

Run of `src/steering_refusal.py` at commit `b9cd9fd`, 7 September 2026. Same ladder, same 128
passages, same seed, same layers as [`steering-layers.md`](steering-layers.md); one column
added. Executed file hashes `a8f519f2...77df`, byte-identical to the commit.

---

## Anchors

Both held. At layer 12, `graft`, `subspace` and `full` came back bit-identical to `95e813f`,
and the sentiment column matched that run's `steer_hi` exactly — at layers 6 and 18 as well,
not just 12. Two independent checks, both clean, so the added column is the only new thing.

## The gap survives the substitution

| layer | graft | sentiment | refusal | g / sentiment | g / refusal |
|---|---|---|---|---|---|
| 6 | 2.41 | 0.42 | 0.19 | 5.8× | *12.4×* |
| 12 | 3.20 | 0.63 | 0.54 | 5.1× | **5.9×** |
| 18 | 8.19 | 1.14 | 1.26 | 7.2× | **6.5×** |

At one activation norm. Refusal behaves like sentiment at the depths where refusal exists:
5.9× and 6.5× against sentiment's 5.1× and 7.2×.

**Do not quote the 12.4×.** Refusal at layer 6 produces 0.19 nats and carries 2.5% of its
energy in the principal subspace, *below* the 3.6% isotropic null. Refusal is a mid-to-late
computation and is barely formed a quarter of the way in, so that ratio is large because the
denominator is nearly nothing, not because the graft is loud.

### Two near-orthogonal directions, both exactly quadratic

| | layer 6 | layer 12 | layer 18 | mean |
|---|---|---|---|---|
| refusal | 1.93 | 2.01 | 2.00 | **1.98** |
| sentiment | 2.07 | 1.98 | 2.06 | 2.04 |
| graft | 2.62 | 2.70 | 2.33 | 2.55 |

And they are not the same direction dressed differently. `cos(sentiment, refusal)` reads
**−0.029, −0.303, −0.121** — near-orthogonal, mildly anti-aligned. Two behavioural directions
sharing almost no geometry both sit at an exponent of 2, while the graft sits well above it at
every depth.

That is the strongest form the Fisher-regime finding has taken. It is no longer a fact about
one hand-built sentiment vector.

### Refusal is not an on-manifold direction

Energy in the top-32 subspace: **2.5%, 5.6%, 4.2%**, against a 3.6% isotropic null. Refusal is
at or below chance for the principal subspace at every depth, where sentiment sits at 7.7%,
15.7% and 11.1%.

So refusal is behaviourally potent, quadratic, and geometrically indistinguishable from a
random direction by the only structural measure being taken. That is one more nail in the
energy-split model retracted in [`steering-graft.md`](steering-graft.md).

## The diagnostic is a weak pass, and I am not going to dress it up

Refusal is not live on encyclopaedia text, so the ladder above could be measuring a direction
that is simply inert in that domain. The diagnostic fires each vector at one activation norm on
six held-out instructions and on wikitext:

| layer | vector | instructions | wikitext | ratio |
|---|---|---|---|---|
| 6 | sentiment | 0.4987 | 0.6263 | 0.80 |
| 6 | refusal | 0.2973 | 0.2228 | **1.33** |
| 12 | sentiment | 1.3941 | 0.9614 | 1.45 |
| 12 | refusal | 1.3166 | 0.6968 | **1.89** |
| 18 | sentiment | 1.4611 | 1.3644 | 1.07 |
| 18 | refusal | 2.3250 | 1.5924 | **1.46** |

Refusal is more instruction-specific than sentiment at **every** layer, and the direction of the
effect is consistent. That is the pass.

But I wrote that a behaviourally specific vector "should move instructions considerably more
than encyclopaedia text," and 1.33 to 1.89 is not considerably more. It is a factor under two.
Judged against the bar set before the run, this is a weak result, and the design has three
flaws that make it weaker still:

- **n = 6 held-out instructions** against 32 wikitext prompts, with no error bars computed. The
  differences here may not survive a proper interval.
- **Mismatched lengths.** The instruction prompts run through the chat template and are short;
  the wikitext prompts are exactly 96 tokens. Activation statistics differ with sequence
  length, so the two arms are not on equal footing.
- **The magnitude is normalised to the wrong domain.** "One activation norm" uses the median
  norm of the *wikitext* activations, then applies that same absolute magnitude to instruction
  activations whose own norm was never measured. If instruction activations are systematically
  larger or smaller, the instruction arm is being perturbed by a different relative amount.
  This is the serious one.

**What this does and does not license.** The ladder result stands on its own: it is the same
construction with two anchors intact, and it says the gap does not depend on which behavioural
direction is used. What is *not* yet earned is the sentence I wanted to write, that the result
now speaks directly to what the paper cautions about. The refusal vector was measured acting on
encyclopaedia text, and the evidence that it is doing something refusal-specific there is thin.

## Fixing it

The diagnostic should be its own experiment, not a coda:

1. Build the ladder on **instruction prompts** rather than wikitext, so refusal is live in the
   base distribution and the graft displaces between reachable instruction states.
2. Measure the instruction-domain activation norm and normalise to it.
3. Enough held-out instructions for real error bars, 128 rather than 6.

That is a clean single-variable change from this run, and it is the version that would let the
threshold table be quoted as a statement about refusal.

## Limits

Carried forward: one model, one position, next-token only, Thm 4.3 untouched, whole-range OLS
slopes averaging across real curvature.

New: the refusal vector rests on 12 matched pairs, which is small for a difference-of-means
construction; published refusal directions use hundreds. And nothing here tests whether the
direction actually controls refusal behaviour, only that adding it moves the next-token
distribution. That was deliberate — no ablation was run — but it means "refusal direction" is a
claim about how the vector was built, not a demonstration of what it does.
