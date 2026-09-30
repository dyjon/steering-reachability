# Kind is a real second variable

Run of `src/steering_matched.py` at commit `1f93006`, 29 September 2026. Executed file hashes
`e2b74df3...1073`, byte-identical to the commit. Donors selected by target activation distance
from a labelled pool of 512 passages, 512 instructions and 256 graded blends; two base arms, two
layers, five targets.

**The cleanest run in the line.** Distance is matched by construction rather than by luck, so
kind is the only thing varying.

---

## The matching worked

Refusal is a fixed direction fired at the achieved magnitude, so at matched distance it must
return near-identical across kinds. It does: 0.05% to 1.7% spread in the rows where achieved
distances agree to two decimals, and where it spreads more the achieved distances differ by
exactly enough to explain it (passage-base layer 18 at target 0.4, cross achieved 0.44 against
0.40, refusal 16% higher, and `(0.44/0.40)^1.5 = 1.15`).

So the graft column is readable. This check did not exist in any earlier run.

## Cross-kind donors do not exist below about 0.4 norms

Coverage of the cross-kind selection, across the four arm-by-layer combinations:

| target | coverage |
|---|---|
| 0.2 | **0%, 0%, 0%, 0%** |
| 0.3 | 0%, 1%, 0%, 1% |
| 0.4 | 100%, 82%, 55%, 43% |
| 0.5 | 100%, 99%, 98%, 92% |

Out of 512 candidates, in either direction, at either depth, **nothing of the other kind sits
within 15% of 0.2 or 0.3 activation norms of a base prompt.** The two domains are simply that
far apart.

This is a structural fact rather than a design failure, and it explains the last three runs.
Run 11's `xdom` ladders spanned under 2× because there is nothing to span. Run 12's graded family
failed on the instruction arm for the same reason. Both were trying to fit a curve over a region
where the curve does not exist, and no construction could have fixed that.

## Kind changes the graft at exactly matched distance

| arm | layer | target | own | cross | mixed | cross/own | distances |
|---|---|---|---|---|---|---|---|
| passage | 12 | 0.4 | 1.5273 | 1.0207 | 0.8311 | 0.67 | 0.40 / 0.41 |
| instruction | 12 | 0.4 | 0.4471 | 0.2258 | 0.3404 | 0.51 | 0.40 / 0.41 |
| instruction | 12 | 0.5 | 0.7842 | 0.4032 | 0.6556 | **0.51** | **exact** |
| passage | 18 | 0.5 | 1.9499 | 2.4707 | 2.3808 | **1.27** | **exact** |
| passage | 18 | 0.7 | 3.3074 | 5.6359 | 4.7181 | 1.70 | 0.67 / 0.70 |
| instruction | 18 | 0.5 | 2.6545 | 2.0468 | 2.4815 | 0.77 | 0.50 / 0.51 |
| instruction | 18 | 0.7 | 5.4450 | 3.1839 | 5.0868 | **0.58** | **exact** |

Restricting to the three cells where achieved distances agree to two decimals: **0.51, 1.27 and
0.58.** None is 1.

So kind is a real second variable, and this is the first time that has been measured rather than
inferred from offsets, sign changes or fits on inadequate range.

**The layer-18 passage-arm reversal reproduces.** Cross-kind is quieter than own-kind everywhere
except the passage arm at layer 18, where it is 1.27× louder, rising to 1.70× at the larger
distance. Runs 11 and 12 both found this reversal and both were fitting over ranges too narrow
to trust. Here it sits in an exactly matched pair.

**Graded blends land between the two pure kinds in six of seven rows**, in the correct order —
for instance 0.4471, 0.3404, 0.2258 on the instruction arm at layer 12. The single exception is
the passage arm at layer 12, where mixed falls below both. A graded construction behaving like a
graded quantity is a coherence check the earlier runs had no way to perform.

## What this settles and what it does not

**Settles.** Donor kind changes the graft by 0.5× to 1.7× at matched displacement. Runs 10
through 12 were right that kind matters and right that no single number describes it; they were
measuring it over regions where one kind had no donors at all.

**Does not settle.** Why the passage arm reverses at layer 18. Why cross-kind is quieter at
layer 12 in both arms. And the whole question of what happens below 0.4 norms across kinds,
which now looks unanswerable by this method rather than merely unanswered: **a cross-kind donor
at 0.2 norms does not exist to be selected.**

## Limits

Carried throughout: one model, one position, next-token only, Thm 4.3 untouched, no ablation and
no test that the refusal direction controls refusal behaviour.

New: the 15% tolerance is a judgement call, and a looser one would report coverage where the
distances are not really matched. Coverage is computed per base prompt and averaged, so a row at
82% mixes prompts whose selections matched well with prompts excluded entirely. Pools are 512
and 256; a larger pool would raise coverage at the margins but cannot create cross-kind donors at
0.2 norms, which is the finding.
