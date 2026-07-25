"""Steering-vector reachability on GPT-2 small.

Khashabi et al., "Steered LLM Activations are Non-Surjective", argue that activation
steering pushes the residual stream off the manifold of states reachable from ordinary
prompts, so mechanistic conclusions drawn from steering may describe internal states
the model would never produce on its own.

This is a small-scale check of that claim, and equally a check of how hard the claim is
to test. The honest difficulty is that "no prompt produces this activation" is a
statement about all possible prompts, and all we can do is compare against a finite
corpus.

Method.
  1. Build a steering vector as a difference in means between two prompt sets.
  2. Collect natural residual-stream activations at one layer over a corpus.
  3. Add alpha * v to held-out natural activations.
  4. Ask how far each steered activation sits from the nearest natural one, and compare
     against how far natural activations sit from each other.
  5. Repeat with the steered activation rescaled to its original norm, since steering
     changes the norm and that alone moves you away from the corpus.

Run: python src/steering_reachability.py
"""
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer
from transformer_lens import utilities as utils

torch.set_grad_enabled(False)

LAYER = 6
# Steering strength as a fraction of the median natural activation norm. Absolute
# coefficients are meaningless here: residual-stream norms are ~100 at this layer, so a
# unit-norm steering vector does nothing regardless of how it is scaled in the abstract.
NORM_FRACTIONS = [0.1, 0.25, 0.5, 1.0, 2.0]

POSITIVE = [
    "I love this", "This is wonderful", "What a delight", "I adore it",
    "Absolutely fantastic", "This makes me happy", "A joy to use", "I am so pleased",
]
NEGATIVE = [
    "I hate this", "This is terrible", "What a disaster", "I despise it",
    "Absolutely awful", "This makes me angry", "A pain to use", "I am so annoyed",
]

CORPUS = [
    "The weather today is unusually mild for this time of year",
    "She opened the letter and read it twice before speaking",
    "Scientists have discovered a new species of deep sea fish",
    "The train was delayed by an hour due to signal problems",
    "He learned to play the piano when he was seven years old",
    "The library closes at six on weekdays and four on Sundays",
    "Their argument continued long after the guests had left",
    "A small bakery opened on the corner of the main street",
    "The report concluded that further study would be needed",
    "She walked home through the park despite the cold wind",
    "The company announced record profits for the third quarter",
    "Rain fell steadily against the windows throughout the night",
    "He forgot his umbrella and arrived completely soaked",
    "The museum exhibit featured artifacts from ancient Egypt",
    "They agreed to meet again the following Tuesday afternoon",
    "The recipe calls for three eggs and a cup of flour",
    "Traffic on the highway moved slowly because of the accident",
    "Her presentation lasted twenty minutes and went very well",
    "The old bridge was finally replaced after many complaints",
    "Students gathered in the courtyard before the ceremony began",
    "The film received mixed reviews from most major critics",
    "A power outage affected several neighbourhoods last evening",
    "He collected stamps as a child and still has the album",
    "The garden needed weeding after the long summer holiday",
    "Negotiations broke down over the question of working hours",
    "The doctor recommended rest and plenty of fluids",
    "Birds returned to the lake earlier than usual this spring",
    "The manuscript had been sitting in the archive for decades",
    "She repaired the bicycle herself using a borrowed toolkit",
    "The conference will be held in a different city next year",
    "Snow covered the fields and made the roads impassable",
    "He wrote three drafts before he was satisfied with it",
]


def collect_activations(model, prompts, layer, batch_size=8):
    """Residual-stream activations at `layer`, flattened over positions, BOS dropped."""
    name = utils.get_act_name("resid_post", layer)
    out = []
    for i in range(0, len(prompts), batch_size):
        tokens = model.to_tokens(prompts[i : i + batch_size])
        _, cache = model.run_with_cache(tokens, names_filter=name)
        resid = cache[name][:, 1:, :]  # drop BOS
        out.append(resid.reshape(-1, resid.shape[-1]))
    return torch.cat(out, dim=0)


def nearest_distances(query, reference, exclude_self=False):
    """L2 distance from each query vector to its nearest reference vector."""
    d = torch.cdist(query, reference)
    if exclude_self:
        d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values


def main():
    model = HookedTransformer.from_pretrained("gpt2")

    # steering vector: difference in means at the final position
    pos = collect_activations(model, POSITIVE, LAYER)
    neg = collect_activations(model, NEGATIVE, LAYER)
    steering = pos.mean(0) - neg.mean(0)
    steering = steering / steering.norm()

    natural = collect_activations(model, CORPUS, LAYER)
    print(f"{natural.shape[0]} natural activations at layer {LAYER}, d={natural.shape[1]}")

    # split so steered vectors are compared against activations they didn't come from
    split = natural.shape[0] // 2
    reference, held_out = natural[:split], natural[split:]

    baseline = nearest_distances(held_out, reference)
    natural_norm = natural.norm(dim=1)
    print(f"natural nearest-neighbour distance: median {baseline.median():.2f}")
    print(f"natural norm: median {natural_norm.median():.2f}\n")

    median_norm = natural_norm.median().item()
    rows = []
    for frac in NORM_FRACTIONS:
        alpha = frac * median_norm
        steered = held_out + alpha * steering
        raw = nearest_distances(steered, reference)

        # norm control: rescale each steered vector back to its pre-steering norm, so the
        # comparison isn't just measuring that steering made the vector longer
        rescaled = steered * (held_out.norm(dim=1, keepdim=True) / steered.norm(dim=1, keepdim=True))
        controlled = nearest_distances(rescaled, reference)

        rows.append((frac, raw.median().item(), controlled.median().item(),
                     steered.norm(dim=1).median().item()))
        print(f"strength={frac:4.2f}x norm  NN dist {raw.median():6.2f}  "
              f"norm-controlled {controlled.median():6.2f}  "
              f"norm {steered.norm(dim=1).median():6.2f}")

    b = baseline.median().item()
    print(f"\nbaseline (natural vs natural): NN dist {b:.2f}")
    print("\nHow much of the apparent departure survives the norm control:")
    for frac, raw, controlled, _ in rows:
        gained = raw - b
        kept = controlled - b
        pct = kept / gained if gained > 0 else float("nan")
        print(f"  strength={frac:4.2f}x: excess {gained:6.2f} -> {kept:6.2f} "
              f"({pct:5.1%} survives)")

    # plot
    xs = [r[0] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.axhline(b, color="grey", linestyle="--", label="natural vs natural")
    plt.plot(xs, [r[1] for r in rows], "o-", label="steered")
    plt.plot(xs, [r[2] for r in rows], "s-", label="steered, norm-controlled")
    plt.xlabel("steering strength (multiple of median activation norm)")
    plt.ylabel("distance to nearest natural activation")
    plt.title(f"Steering vs natural activations, GPT-2 small layer {LAYER}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_steering.png", dpi=150, bbox_inches="tight")
    print("\nSaved report_steering.png")


if __name__ == "__main__":
    main()
