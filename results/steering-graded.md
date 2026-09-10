# The graft runs backwards

Run of `src/steering_graded.py` at commit `cf115ee`, 9 September 2026. Executed file hashes
`725edd9a...6844`, byte-identical to the commit.

**Anchor exact.** wiki-chat `own` returned 0.0696, 0.0864, 0.1339 and 1.4502 at layer 12,
matching run 11 to the digit.

**The fix I designed failed on one arm, and the failure is the finding.**

---

## The graded family did not get range where it was most needed

| cell | graded span | vs the `xdom` it replaced |
|---|---|---|
| L12 wiki-chat | 2.56× | better than 2.00× |
| L18 wiki-chat | 1.63× | **worse** than 1.83× |
| L12 instr-chat | **1.16×** | **worse** than 1.31× |
| L18 instr-chat | **1.07×** | **worse** than 1.52× |

The reason is structural and I should have seen it. For an *instruction* base, every donor
containing any passage at all is already far away, so varying how much passage it contains
barely moves the distance. The family interpolates in prompt space, but prompt-space
interpolation is not activation-space interpolation, and the map between them is extremely
non-linear at the domain boundary.

So the two-arm test this run was built for is not available. Only the wikitext arm is readable.

## The graft decreases as displacement increases

In the instruction arm the graded family produced this:

| α | ‖d‖ | graft |
|---|---|---|
| 1.00 (pure instruction) | 5.82 | 0.6016 |
| 0.75 | 6.06 | 0.5688 |
| 0.50 | 6.36 | 0.5668 |
| 0.25 | 6.44 | 0.5615 |
| 0.00 (pure passage) | 6.76 | **0.3718** |

**Displacement rises 16%, the graft falls 38%.** Monotone in both, in opposite directions. The
fitted exponent is −2.72, and −3.61 at layer 18.

A negative exponent is not a curve, so `KL@0.4` in those two cells is meaningless — the 8.6934
reported for instr-chat at layer 18 is an extrapolation off a 1.07× span and should be
discarded, along with the 4.90× and 0.19× it produces in the summary table.

**But the negative slope is the strongest result in this line so far.** Monotone-decreasing
behavioural change against monotone-increasing distance is *qualitatively impossible* if graft
KL were a function of ‖d‖ alone. Runs 10 and 11 inferred that from offsets and sign changes.
This shows it directly, in a single ordered column.

What the graft tracks is how **instruction-like** the donor is, not how far away it is. Moving
an instruction's activation toward a pure passage takes you further and does less.

## The layer reversal reproduces

Cross-domain against own-domain graft KL at matched 0.4 norms, wikitext arm:

| | own | cross | cross/own |
|---|---|---|---|
| layer 12 | 1.4360 | 0.9222 | **0.64** |
| layer 18 | 1.1616 | 1.5088 | **1.30** |

Layer 12 has cross-domain quieter, matching run 10's `passage` anomaly. Layer 18 has it louder.

Expressed as run 11 expressed it, `own/cross` reads **1.47** and **0.84** here against run 11's
**1.61** and **0.65** — same sign, same order, on a different cross-domain construction.

**So run 11's sign flip is not purely a fitting artefact.** It survives replacing a foreign pool
with a graded one. That was the question this run existed to answer, and for the wikitext arm it
answers it: the reversal is real.

## Standing

**Strengthened.** Run 11's conclusion that no single number describes the kind effect. The
reversal reproduces under a different construction, and the negative slope makes the
one-dimensional model untenable rather than merely ill-fitting.

**Discarded from this run.** Both instr-chat graded cells and everything computed from them:
the −2.72 and −3.61 exponents, the 8.6934, the 4.90× and the 0.19×.

**Still missing.** A cross-domain family with real range on an instruction base. Prompt-space
interpolation cannot deliver it, so the next attempt has to grade in a way that tracks
activation distance rather than surface composition — for instance by selecting donors at
target distances from a large mixed pool, rather than constructing them.

## Limits

Carried forward: one model, one position, next-token only, Thm 4.3 untouched, no ablation and no
test that the refusal direction controls refusal behaviour.

New: three of four graded cells span under 2×, so only the layer-12 wikitext graded fit is
sound. The layer-18 wikitext cell, on which half the reversal claim rests, spans 1.63× and
should be treated as suggestive rather than measured.
