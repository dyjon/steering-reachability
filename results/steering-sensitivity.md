# Behavioural sensitivity of the steering direction

Checkpoint, 2 September 2026. Second research line (steering reachability), independent of
the LoRA+ work in `01-horizon.md` / `02-ratio-ladders.md` / `03-width.md`.

Everything below is measured, not projected. Two runs, both complete.

---

## The question

Mishra et al., *Steered LLM Activations are Non-Surjective*, prove that a steered activation
almost surely does not **exactly** equal any prompt-reachable activation (Thm 4.2:
`P(r̃ᵢ = r'ₖ) = 0`, with `Im(F) = {F(r<i, sᵢ, Θ) | s ∈ V^≤K}`), and caution against reading
steering success as evidence of prompt-based vulnerability.

Vulnerability is about behaviour, not equality. So:

> If a steered activation is formally unreachable but close to something reachable, and
> output is continuous in the activation, is it still behaviourally reproducible?

Split, because the direct version is intractable:

- **(a)** how close can a prompt get in activation space? — needs SipIt-style search, skipped
- **(b)** at that distance, how much does behaviour move? — cheap, measured here

Everything below is (b).

## Setup, common to both runs

Qwen2.5-0.5B-Instruct (smallest of the three models in the paper), fp32. Residual stream at
layer 12 of 24, final token position only, perturbed by a forward hook. Prompts are wikitext-103
passages. Metric is mean `KL(base ‖ perturbed)` over the next-token distribution. Magnitudes
are δ × the median activation norm at that layer, which is **15.3**. Hidden dim 896.

`δ = 0` is the sanity row and reads exactly `0.0000` in both runs.

---

## Run 1 — steering vs isotropic random

`src/steering_sensitivity.py`, 200 prompts, 4 random draws per magnitude.

| δ | ‖v‖ | steering KL | random KL | ratio |
|---|---|---|---|---|
| 0.00 | 0.0 | 0.0000 | 0.0000 | — |
| 0.05 → 1.00 | 0.8 → 15.3 | ≈ 0.42 δ² | ≈ 0.35 δ² | **≈ 1.23, flat** |
| 2.00 | 30.6 | — | — | 0.80 |

Flat ratio across a 20× magnitude range. At δ=2 the ratio inverts to 0.80: random keeps
accelerating (0.54 δ²) while steering saturates.

**This result does not support its own conclusion.** Activations occupy a low-dimensional
manifold inside 896 dimensions, so an isotropic Gaussian direction is almost entirely
orthogonal to anywhere the data goes. "Steering beats random by 23%" may only mean "steering
beats a direction pointing nowhere." Same failure mode as the first LoRA ratio ladder, which
looked like a large effect until the control showed it was the stability wall.

## Run 2 — four matched-magnitude baselines

`src/steering_subspace.py`, 256 prompts, 5 draws per random baseline, top-32 principal
subspace of the natural activations.

Geometry, before any KL:

| | |
|---|---|
| top-32 subspace holds | **60.4%** of activation variance |
| steering energy in that subspace | **19.4%** |
| isotropic null (k/D = 32/896) | 3.6% |

So steering is 5.4× enriched on-manifold — and still leaves 80.6% of its energy in the
complement.

| δ | steering | subspace | complement | natdiff |
|---|---|---|---|---|
| 0.00 | 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 0.25 | 0.0234 | 0.0307 ± 0.0037 | 0.0199 ± 0.0012 | 0.0304 ± 0.0072 |
| 1.00 | 0.4337 | 0.7289 ± 0.1570 | 0.4437 ± 0.1105 | 1.8458 ± 1.4325 |

`subspace` = random inside the top-32 · `complement` = random in the orthogonal complement,
deliberately off-manifold · `natdiff` = unit-normalised difference of two natural activations.
± is the spread over draws, not over prompts.

### What it says

**Steering is the quietest direction in the table, not the loudest.** At δ=1 it is 0.43
against subspace's 0.73 — about 4 SE apart (0.157/√5 = 0.070). The ordering holds at δ=0.25
(0.0234 vs 0.0307, ~4.3 SE), so it is not a saturation artefact. The 1.23 of Run 1 is dead:
against a baseline that can compete, steering scores **below** 1.

**The geometry predicts the behaviour with nothing left over.** Weight the two measured KLs by
steering's energy split (19.4% subspace / 80.6% complement), no free parameters:

| δ | predicted | observed | error |
|---|---|---|---|
| 0.25 | 0.0220 | 0.0234 | 6% |
| 1.00 | 0.499 | 0.434 | 13% |

Steering has no behavioural signature beyond where its energy sits. Any direction with 19% of
its mass in the principal subspace does roughly what it does.

**Steering is the least curved direction available.** δ=0.25 → δ=1 is 4×, so pure quadratic
KL would give 16×. Observed: steering 18.5×, subspace 23.7×, complement 22.3×, natdiff 60.7×.
Steering stays closest to its own Fisher regime.

---

## What is still missing

**There is no scale.** Nothing here says whether 0.43 nats is a lot. The gate asks whether
behaviour is sensitive at the steering magnitude, and that question has no answer without a
reference drawn from behaviour people already accept as equivalent.

`natdiff` is not that reference. Rescaling `hᵢ − hₖ` to a fixed magnitude produces a
*direction*, not a reachable *point* — which is why its spread is 78% of its mean.

## Designed next experiment — graft calibration

Build the reference out of activations that real prompts actually produce, and compare at
matched displacement.

1. Take a wikitext passage truncated to a fixed 96 tokens. Swap its first *k* tokens for
   another passage's, keeping length and the final token fixed. The variant is a real prompt,
   so its layer-12 last-position activation is a genuine element of `Im(F)`.
2. `d = h_last(variant) − h_last(base)`. Graft: run the *original* prompt with `h_last += d`.
   Everything upstream is identical, so this differs from steering only in the vector added at
   one position — a perfectly matched intervention that lands on a reachable point.
3. Fire steering, and a random top-32 direction, at exactly ‖d‖.
4. Sweep *k* ∈ {8, 24, 48, 72} to sweep ‖d‖. The result is a curve, not a point.

Read it as: **at equal L2, does steering move behaviour more or less than a step to a
reachable activation?**

| | |
|---|---|
| steering < reachable step | a near-miss costs less than its L2 implies; formal non-surjectivity has little behavioural bite |
| steering > reachable step | steering reaches a sensitive direction prompting does not; the paper's caution carries through to behaviour |

Also fix two things Run 2 got wrong: report SE **over prompts** (per-row KL is already
computed and then thrown away, and steering currently has no error bar at all), and give each
prompt its own random draw instead of averaging 5 global draws.

## Known limits, declared

- **Sentiment steering has no vulnerability semantics.** The paper uses refusal and persona
  vectors. A refusal vector on the instruct model is the follow-up if the calibration opens
  the gate.
- **Single position.** Thm 4.3 is about trajectory divergence at i+1; a next-token measurement
  does not touch it.
- One layer, one model, one steering vector. No error bar on the steering column.
- The energy-split model is crude and additive, fitted against baselines with wide spreads.

## Resuming cold

Repo `interp-prep`. Run 2 is commit `1206f986836dc28c365cb490f6e3dc0b3e2b2dc5`. Kaggle, T4 ×2
(**not** P100 — sm_60 has no compiled kernels and fails only at the first real matmul):

```python
import urllib.request, sys
sha = "1206f986836dc28c365cb490f6e3dc0b3e2b2dc5"
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/dyjon/interp-prep/{sha}/src/steering_subspace.py",
    "steering_subspace.py")
sys.modules.pop("steering_subspace", None)
import steering_subspace as m
m.main()
```

`sys.modules.pop` matters — the session caches the previous script under a stale name.
Check the `δ = 0.00` row reads `0.0000` across before trusting anything else.

`src/steering_corpus_scale.py` is in the repo and **superseded before it ever ran**: growing a
reference corpus does not approach `Im(F)`, so nearest-neighbour distance to one cannot bear
on a theorem about exact collision.

## Where the collaboration sits

Aayush Mishra (first author, 5th-year in Anqi Liu's group) has seen the plan, not these
numbers. Last message: "Sounds good! Looking forward to see what you find." Meeting in October,
results to him before then. Anqi Liu is on the thread and is the primary advisor target.
