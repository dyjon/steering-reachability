# It was mostly the chat template

Run of `src/steering_format.py` at commit `b2ab834`, 9 September 2026. Executed file hashes
`0be049f2...8b64`, byte-identical to the commit. Three base arms, layers 12 and 18, 128 bases
against 512-prompt pools.

**Anchors hold.** `wiki-raw` `swap8` graft returned 0.0086 at layer 12, and `own-far` returned
10.9153 at 1.40 norms, matching `bf36ef0`'s 512-pool figure exactly.

---

## The ~3.9× decomposes, and the template is the larger half

Layer 18. Format effect is `wiki-raw / wiki-chat`, the same passages with and without the chat
template. Content effect is `wiki-chat / instr-chat`, the same template over different content.

| fraction | format | content | product |
|---|---|---|---|
| 0.2 | 3.59 | 0.82 | 2.94 |
| 0.3 | 2.96 | 1.42 | 4.20 |
| 0.4 | 2.46 | 1.83 | 4.50 |
| 0.5 | 2.33 | 2.02 | 4.71 |
| 0.6 | 2.55 | 1.83 | 4.67 |
| **mean** | **2.78×** | **1.58×** | **4.40×** |

The product is consistent with the ~3.9× [`steering-overlap.md`](steering-overlap.md) measured
between `wiki-raw` and `instr-chat`, which is the reassuring part: this run reproduces run 9's
number and then splits it.

**Roughly two thirds of the effect is the chat template and one third is the content.** Putting
the *same wikitext passages* inside a user turn moves graft/refusal from 3.45 to 1.48 at half a
norm, without changing a word of the content.

Layer 12 gives format 3.2–5.0× and content 2.1–7.5×, both larger and the content term unstable,
so the clean decomposition is the layer-18 one.

## The mechanism is visible in the geometry

| arm | top-32 holds | refusal energy there |
|---|---|---|
| wiki-raw | 61.7% | **8.8%** |
| wiki-chat | 82.4% | **49.9%** |
| instr-chat | 89.1% | 69.0% |

**Templating the same passages lifts refusal's subspace energy from 8.8% to 49.9%.** The
template compresses the activation set into a lower-dimensional region — top-32 variance goes
61.7% → 82.4% on identical text — and the refusal direction aligns with that region.

So the story is not "refusal is special in the instruction domain." It is closer to: the chat
template imposes a low-dimensional structure at the generation position, refusal lies largely
inside it, and directions inside it move behaviour disproportionately.

## Displacement kind matters at matched magnitude

Run 9's `passage` anomaly was real and it reproduces in both directions.

The cleanest evidence is an almost perfectly matched pair in `wiki-chat` at layer 12:

| kind | ‖d‖ | /norm | graft | refusal |
|---|---|---|---|---|
| own-far | 5.39 | 0.40 | **1.4502** | 1.5861 |
| xdom-near | 5.38 | 0.40 | **1.0160** | 1.5689 |

Same base set, same displacement to two decimal places, **graft differs by 1.43×**. Refusal is
identical across the pair, as it must be for a fixed direction at fixed magnitude, which
confirms the difference belongs to the graft and therefore to the donor kind.

The same holds going the other way. In `instr-chat` at layer 12 the own-domain curve runs
0.1014 at 0.18 norms to 3.1539 at 0.75, an exponent of 2.41, which predicts 0.78 at 0.42 norms.
The observed cross-domain graft there is 0.3406, **2.3× quieter**.

**Cross-domain displacements are quiet for their size, in both directions.** Graft KL is a
function of magnitude *and* kind, and every ladder before run 9 was read one dimension short.

## What this does to the line

**Reframed, not destroyed.** There is a real effect and run 9's magnitude reproduces. But
"instruction domain" was doing about a third of the work its name claimed; the chat template
does the rest. Every earlier statement contrasting "wikitext" with "instructions" is really
contrasting raw text with templated text, plus a smaller content term.

**The refusal-specific claim survives at layer 18 but needs restating.** Refusal aligns with
the structure the template creates, and that is why it is loud there — a geometric fact about
the template, not a semantic fact about refusal.

**The ladder framework needs kind as an axis.** Any future curve fitted against ‖d‖ alone will
mix cross-domain and own-domain donors that differ by 1.4× to 2.3× at identical magnitude.

## Limits

Carried forward: one model, one position, next-token only, Thm 4.3 untouched, no ablation and no
test that the refusal direction controls refusal behaviour.

New: `self` variants use different prefixes in the two templated arms — instruction openers
against passage openers — so that one kind is not construction-matched across arms, though the
four others are. And the content contrast compares wikitext passages against Alpaca
instructions, which differ in length as well as kind; the templated lengths are closer than the
raw ones but not equal.
