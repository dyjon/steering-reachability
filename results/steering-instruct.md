# The gap inverts where refusal is live

Run of `src/steering_instruct.py` at commit `b2a36dd`, 9 September 2026. Executed file hashes
`8010b52e...0a6b`, byte-identical to the commit. Qwen2.5-0.5B-Instruct, layers 12 and 18,
128 prompts per domain, AdvBench + Alpaca, 96 disjoint pairs building the refusal vector.

**This run reverses the headline of [`steering-refusal.md`](steering-refusal.md).**

---

## graft / refusal, per rank

| rank | wikitext L12 | **instructions L12** | wikitext L18 | **instructions L18** |
|---|---|---|---|---|
| 1 | 6.88 | **0.24** | 4.96 | **0.75** |
| 4 | 7.01 | **0.25** | 5.50 | **0.93** |
| 16 | 7.13 | **0.27** | 5.91 | **0.91** |
| 64 | 6.58 | **0.48** | 5.43 | **1.50** |
| 127 | 7.01 | **0.95** | 5.97 | **2.04** |

On wikitext the ratio is flat at about 7 and 5.5, holding across a 1.5x span of displacement.
**In the instruction domain it inverts.** At the nearest-neighbour rank, refusal produces four
times the KL of a reachable step of the same size at layer 12, and a third more at layer 18.

Every previous run concluded that steering directions are behaviourally quiet. In the domain
where refusal actually operates, refusal is the **loudest** direction tested at small
displacement.

## The confound, which is serious

Refusal's energy in the top-32 principal subspace, by domain:

| | wikitext | instructions |
|---|---|---|
| subspace holds | 61.7% of variance | **90.7%** |
| refusal energy there | 8.8% | **97.7%** |
| sentiment energy there | 8.0% | 36.2% |

Layer 18 is the same picture: 98.3% for refusal against an 89.9% subspace.

The ladder set was built **half refusable by construction**. That makes the harmful/harmless
split the dominant axis of variation in the set, and the refusal vector points almost exactly
along it. Refusal being loud is then close to tautological: it is the principal component of
the very set the graft displacements are drawn from.

So the inversion is real as measured, and the design cannot separate *"refusal is loud in the
instruction domain"* from *"refusal is loud in a set I built to vary along refusal."* A
realistic instruction mix is mostly benign. That version is untested.

## Three things this run got wrong

**The headline number came back `nan`, and that is a design flaw rather than a crash.** The
interpolation to one activation norm has nothing to stand on, because the instruction ladder
never reaches one norm:

| | reach, max ‖d‖ / norm |
|---|---|
| wikitext L12 / L18 | 1.37 / 1.39 |
| instructions L12 / L18 | **0.67 / 0.78** |

Templated instructions cluster far more tightly than wikitext passages, so even the farthest of
128 sits two thirds of a norm away. The two ladders barely overlap in magnitude at all, which
means the cross-domain comparison at matched displacement is mostly unavailable. Reporting at a
common *fraction* both domains reach, around 0.5 norms, would have been the right call.

**I changed the sentiment vector without flagging it.** Run 6 built it position-matched, placing
the sentences at the end of 96 tokens of filler. This run builds it through the chat template.
Different construction, different vector, so the sentiment column is **not comparable across
runs**, which is why graft/sentiment reads 13.7x here against 5.1x before at layer 12. My error.

**The bridge holds only for refusal.** Refusal uses the same chat-template construction in both
runs, and it bridges: 7.0x and 5.6x here against 5.9x and 6.5x from the token-swap ladder. That
is close enough to say the rank ladder measures the same thing. Sentiment cannot be used as a
bridge because of the construction change above.

## What this does to the story

The wikitext result stands and is now well replicated across two ladder constructions. What
does not survive is generalising it to refusal-relevant behaviour.

The threshold table in [`steering-graft-wide.md`](steering-graft-wide.md) is a statement about
encyclopaedia text. In the instruction domain the curve has a different shape, a much shorter
reach, and at small displacement the ordering is reversed.

**Consequence for the Mishra thread.** The email sent on 7 September quotes 5.4x, which is a
sentiment-on-wikitext number and was accurate for what it described. It should not be read as
bearing on refusal-based vulnerability. If the inversion survives the fix below, that needs
saying explicitly before October rather than being discovered in the meeting.

## The fix, and it is a real experiment

1. **Vary the harmful fraction of the ladder set** across 50%, 20%, 5%. If refusal stays loud as
   it stops being the principal axis, the domain effect is real. If its loudness tracks the
   fraction, it was an artifact of set construction. This is the experiment that decides it.
2. **Report at a common fraction of norm** both domains reach, not at 1.0.
3. **Hold the sentiment construction fixed** across runs, or drop the column.
4. Extend the instruction ladder's reach, probably by drawing donors from a wider instruction
   pool rather than the 128 in the ladder set.

## Limits

Carried forward: one model, one position, next-token only, Thm 4.3 untouched, no ablation and
no test that the refusal direction actually controls refusal behaviour.

New: the instruction ladder spans 0.18 to 0.78 norms while the wikitext ladder spans 0.78 to
1.39, so the two overlap at a single point and the domains are compared over largely disjoint
ranges of displacement.
