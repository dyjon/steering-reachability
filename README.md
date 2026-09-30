# Steering reachability — thirteen runs

A follow-up to *Steered LLM Activations are Non-Surjective* (Mishra et al., 2026).
Qwen2.5-0.5B-Instruct, fp32. Everything is measured on the residual stream at one layer, at the
final token position, on the next-token distribution only.

---

## The question

You can pause a language model part-way through, add a vector to its internal state, and let it
finish. It then behaves as though it had read something it never did. That is **steering**, and
it is how people turn traits like refusal up and down.

Mishra et al. point out that an attacker does not get to reach inside the model; they get a text
box. And they prove that a steered state is *almost surely not* a state any prompt could
produce — probability zero, not merely hard to find.

But that proof is about landing on a point **exactly**, and vulnerability is about **behaviour**.
Missing the set of reachable states by a hair and missing it by a mile are both probability zero.
So the question this repo asks is:

> At a given distance in activation space, how much does the model's behaviour actually differ?

## The construction, and why it exists

Measuring "how much did behaviour change" is easy. The hard part is what to compare against.

The first attempt compared steering to a random direction of the same length and found steering
23% more disruptive. That number is worthless: the state lives in 896 dimensions, and a random
direction in 896 dimensions points almost entirely into empty space the model never visits.
Beating it means beating a direction pointing nowhere.

So instead — the **graft**:

1. Take two real prompts. Run both, and record the internal state each produces at the chosen
   layer and position.
2. Subtract them. That difference is a step from one genuinely reachable state to another.
3. Re-run the **first** prompt, and at that one spot, add the difference.

Everything before the graft point is identical to an ordinary run. The only thing that changed
is one vector at one position — exactly what steering changes — except this time the destination
is somewhere a prompt actually goes. Now steering and the graft can be compared at the same
distance, and the comparison means something.

## Reading the numbers

| term | what it means |
|---|---|
| **KL**, in *nats* | How far the next-token probability distribution moved. 0 means identical, bigger means more disturbed. All the headline numbers are KL |
| **activation norm** | The distance unit. The state itself has a typical size; "0.4 norms" means moved by 40% of that. Makes distances comparable across layers, where raw sizes differ by 2× |
| **the graft** | KL produced by stepping to another *reachable* state |
| **steering** / **refusal** | KL produced by a steering vector, scaled to the *same* distance |
| **graft ÷ steering** | Above 1: the reachable step disturbs behaviour more. Below 1: steering does. This ratio is the spine of the whole line |
| **donor kind** | What sort of prompt the graft steps toward. **own** = the same sort as the starting prompt, **cross** = the other sort, **mixed** = a blend |
| **subspace energy** | How much of a direction lies inside the few dimensions the model's states actually spread through. A random direction scores 3.6%; more than that means the direction is aligned with where the data lives |

Two prompt domains appear throughout: **raw passages** (encyclopaedia text) and **chat-templated
instructions** (a request inside a user turn, which is the setting the safety argument concerns).

---

## The runs

Each run exists because the previous one had a flaw — usually one named in its own writeup
before it was measured. From the fifth on, every run was pushed and fetched through the Kaggle
API with the executed file verified by hash against the commit it claims to be.

| # | | The question it asked | What came back |
|---|---|---|---|
| 1 | [sensitivity](results/steering-sensitivity.md) | Does steering disturb behaviour more than a random direction of the same length? | Yes, by 23%, across a twentyfold range of distance. **The baseline was a straw man** and the result was abandoned |
| 2 | [subspace](results/steering-sensitivity.md) | And against a direction that points somewhere the model's states actually live? | Steering is the **quietest** direction tested: 0.43 nats against 0.73 |
| 3 | [graft](results/steering-graft.md) | Against a step to a genuinely reachable state at the same distance? | Steering's disturbance grows as distance^1.98, the reachable step's as ^2.65. Kills run 2's geometric explanation |
| 4 | [graft-wide](results/steering-graft-wide.md) | What happens at the distance steering actually operates at? | A reachable step disturbs behaviour **5.4× more**. Run 3's rows reproduced bit-identically |
| 5 | [layers](results/steering-layers.md) | Is this a fact about the model, or just about layer 12? | Holds at layers 6, 12 and 18 — 6.0×, 5.5×, 7.5×. A flaw flagged in three earlier writeups turns out to be worth under half a unit |
| 6 | [refusal](results/steering-refusal.md) | Does it hold for refusal, the direction the safety argument is about, not just sentiment? | Yes, 5.9× and 6.5×. Two directions sharing almost no geometry both grow as distance² |
| 7 | [instruct](results/steering-instruct.md) | What happens in chat-templated prompts, where refusal is actually live? | **It inverts.** 0.24 against roughly 7 — refusal now disturbs behaviour *more* than a reachable step |
| 8 | [fraction](results/steering-fraction.md) | Was that caused by building the prompt set half-refusable? | No. The ratio held at 0.32, 0.26, 0.26, 0.25 while the geometry it should depend on moved 29 points |
| 9 | [overlap](results/steering-overlap.md) | Are the two domains even being compared at the same distances? | They were not. Forced to a common range: two parallel curves, a constant 3.9× apart |
| 10 | [format](results/steering-format.md) | Is that difference about the content, or just the chat formatting? | Mostly the formatting — 2.78× against 1.58×. Wrapping *identical* text in a chat turn lifts refusal's subspace energy from 8.8% to 49.9% |
| 11 | [kind](results/steering-kind.md) | Does it matter what sort of prompt the graft steps toward? | Enormously. The same "domain effect" reads anywhere from **0.73× to 4.87×** depending only on donor kind. **Retracts runs 9 and 10** |
| 12 | [graded](results/steering-graded.md) | Was run 11 just fitting noise on too little range? | No. And the disturbance **falls as the distance rises**: +16% distance, −38% KL, monotone. Distance is not the variable |
| 13 | [matched](results/steering-matched.md) | Select donors already sitting at a target distance, rather than building them and hoping | Kind changes the graft **0.51× to 1.27×** at exactly matched distance. And **no cross-kind donor exists below 0.4 norms at all** |

---

## What stands

**Disturbance is not a function of distance alone.**
Run 12 shows it in a single ordered column: as the graft's destination moves further away, the
behavioural change gets *smaller*, monotonically, across five points. That is impossible if
distance were what mattered. Run 13 then measures the second variable directly, by selecting
donors that already sit at a chosen distance instead of constructing them, and finds the graft
changing by 0.51×, 1.27× and 0.58× purely by swapping the *kind* of destination — in the three
cells where the achieved distances agree to two decimals.

*Why it matters:* any experiment that fits behavioural change against activation distance alone
is silently mixing conditions that differ by up to 2× at the same distance. This is the finding
I would most want someone else to check.

**No cross-kind destination exists below about 0.4 activation norms.**
Searching 512 candidates, in either direction, at either depth, nothing of the other prompt sort
sits within 15% of 0.2 or 0.3 norms of a starting prompt. The two domains are simply that far
apart in activation space.

*Why it matters:* it explains two failed runs. Runs 11 and 12 could not get measurement range for
cross-kind comparisons because there is no range to get — they were fitting a curve over a region
where the curve does not exist, and no cleverer construction would have helped.

**Chat templating reverses which is louder, at layer 12.**
In raw passages, a reachable step disturbs behaviour more than refusal does, at ratios of 2.26 to
3.97. In chat-templated prompts the reverse, in all six cells — two content types by three donor
kinds — at 0.11 to 0.91.

*One caveat on the evidence:* the six templated figures are independent cells, while the five raw
ones are a single condition read at five distances. So the templated side is much better
supported than the raw side. And there is no account of *why* the template does this.

**Steering stays in the gentle regime; reachable steps leave it.**
Near any minimum, disturbance should grow as the square of the distance. Steering does — exponent
1.98, then 2.03 twice more, out past a full activation norm. Reachable steps grow faster, and the
graft's exponent exceeds refusal's in 11 of 12 conditions, so the gap between them narrows as the
distance grows.

**Measured on raw passages, and true there:** steering is quieter than a random direction drawn
from where the data actually lives; a reachable step at one position is small, 0.40 norms even
when three quarters of the context is swapped, against steering's 1.0; and grafting a single
position reproduces 10–26% of the behavioural change of swapping the whole prompt.

## What was retracted

| claim | killed by |
|---|---|
| Steering beats a random direction by 23% | run 2 — the baseline pointed nowhere |
| A geometric model predicting steering's disturbance to within 6% | run 3 — two directions matched on that geometry behaved 2.4× apart |
| A reading rule written into run 3's own docstring | run 4 — the inference ran backwards |
| **The 5.4×** | run 11 — it described a donor mixture, not the model |
| Run 9's constant 3.9× gap between domains | run 11 |
| Run 10's 2.78× formatting and 1.58× content split | run 11 |
| "The effect is not refusal-specific" | run 9 — it rested on a single unreliable point |
| Treating disturbance as one-dimensional in distance | run 12 |
| Two cells whose fitted exponents came out negative | run 12, self-caught |

Nine retractions against four standing claims. Most came from a control that a previous writeup
had already flagged as missing, which is the order that makes a retraction cheap — the flaw was
named first and measured later.

## Limits, throughout

One model. One position. Next-token distribution only, so Thm 4.3, which concerns trajectory
divergence at the following position, is untouched. No ablation was ever run: nothing here tests
that the refusal direction actually *controls* refusal behaviour, so "refusal direction" is a
claim about how the vector was built, not about what it does. The refusal vector uses 96
AdvBench/Alpaca pairs, and only AdvBench's `goal` column is ever read. Three of twelve cells in
run 11 span under 2× in distance and their exponents are weakly determined.

## Running any of it

Each script is standalone. The module docstring says what the run is for, what it fixes from the
previous one, and how to read the output.

```
pip install torch transformers datasets
python src/steering_matched.py
```

A T4 is enough. The longest run is about four minutes.
