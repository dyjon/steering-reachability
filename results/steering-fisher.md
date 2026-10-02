# The Fisher form explains the direction effects — at short range

Run of `src/steering_fisher.py` at commit `ec74c19`, 1 October 2026. Executed file hashes
`e577c12c...40be`, byte-identical to the commit. Layers 12 and 18, 128 prompts per arm, 95%
bootstrap intervals on every ratio.

**Calibration passed**: a graft scaled to 5% of its length returned measured/predicted of 0.99,
1.00, 1.00 and 1.01 across the four arms, so the Fisher computation is correct. **Anchor exact**:
raw arm, layer 12, measured graft 0.0086, 0.0345, 0.1343, 0.2561.

---

## What was tested

Run 14 ended on a synthesis: every apparent effect of distance in this line was an effect of
direction, which is what a local quadratic form predicts — KL ≈ ½ dᵀFd, with F the Fisher
information of the next-token distribution with respect to the activation. This run measures
that form directly and asks whether it reproduces the ratios run 14 measured.

dᵀFd needs neither F nor gradients. For a categorical output it equals the variance, under the
base distribution, of the change in logits along d — taken here by central difference from two
forward passes at ±1% of d.

## The result, scored strictly

Run 14 measured 41 ratios. For each, the Fisher prediction is compared with the measurement.

| family | n | right side of 1 | within 20% | within 35% |
|---|---|---|---|---|
| refusal vector, templated-built over raw-built | 16 | **16** | 11 | 14 |
| graft over refusal | 16 | **16** | 9 | 13 |
| cross-kind over own-kind, at matched distance | 9 | 7 | 4 | 6 |
| **all** | 41 | **39** | 24 | 33 |

A lenient score — 38 of 41 with overlapping intervals — is also printed by the script, but
overlap is easy when intervals are wide, so the table above is the one to read.

### The vector-provenance effect is fully explained

The most striking agreement is where run 14 found the largest effect. In chat-templated prompts at
layer 12, the templated-built refusal vector moves behaviour 3.88× more than the raw-built one;
the Fisher form predicts **3.85×**, and the same at every k (3.90 vs 3.74, 3.86 vs 3.69, 3.94 vs
3.75). The graft-over-refusal ratio in the same arm: measured 0.45, 0.43, 0.34, 0.35, predicted
0.43, 0.43, 0.38, 0.36.

So the provenance effect that retired the templating flip is not a mystery. The templated-built
vector simply lies along high-Fisher directions of templated activations, and one forward-pass
statistic says so.

### The kind effect is only partly explained

Seven of nine cross/own ratios fall on the right side of 1. One real failure: at layer 12 on
passage bases at 0.4 norms, measured **0.67** against predicted **1.82** — opposite sides. The
cause is visible in the magnitudes: the own-kind displacement there is 2.3× larger than the
quadratic form predicts, while the cross-kind one is not. (The other off-side cell, layer 18
instruction bases at 0.4, has a measured ratio of 1.02 [0.85, 1.23], indistinguishable from 1.)

## Where the quadratic form stops working

| distance | measured / predicted |
|---|---|
| 0.08–0.40 norms (swap arms) | 0.71–1.55, nearly all within 0.8–1.3 |
| 0.4 norms (matched cells) | median 1.48, range 0.85–2.30 |
| 0.5 norms | median 1.61 |
| 0.7 norms | median **3.35**, range 2.14–3.42 |

Out to about 0.4 activation norms the local form predicts magnitudes to within roughly 20–30%.
Beyond that it **underpredicts**, increasingly — by about 3× at 0.7 norms. Ratios survive the
drift where both arms of a comparison drift together (layer 18 instruction bases at 0.7: own 3.35,
cross 3.42, ratio predicted 0.57 against measured 0.58), and fail where they do not.

That locates the kind effect's partial miss: the kind comparisons are the only ones made beyond
0.4 norms, which is where higher-order terms take over.

## The synthesis, revised

**Run 14:** every apparent effect of distance was an effect of direction, consistent with a local
Fisher form.

**Now, tested:** within about 0.4 activation norms, the local Fisher form ½dᵀFd reproduces
behavioural change in magnitude and in every vector and provenance ratio measured, from one
forward-pass statistic. Beyond 0.4 norms it underpredicts by up to about 3×, and destination-kind
effects there are only partly captured.

## What it says about the paper

The question Thm 4.2 leaves open is whether a prompt that *nearly* reaches a steered state
reproduces its behaviour. A near-miss is, by definition, a small residual — which is the regime
where the Fisher form is accurate. So for near-misses, the governing quantity is the Fisher
length of the residual, not its Euclidean length, and this run shows it can be computed cheaply.

What this does *not* show is how small a residual realistic prompt search achieves. That is
SipIt's question, and the answer decides whether near-misses live inside the regime measured here.

## One number not to quote

The script also reports a correlation of 0.994 between log measured and log predicted KL across
all conditions. It is real but uninformative: the conditions span more than four orders of
magnitude in KL, so almost any monotone predictor would score highly. The ratio comparisons above
are the evidence.

## Limits

One model, one position, next-token only, no ablation. The Fisher form is local by construction
and is evaluated at the base state only. The kind comparisons sit almost entirely beyond 0.4
norms, so this run cannot say whether the Fisher form would explain them at shorter range — only
that it does not fully explain them where they were measured.
