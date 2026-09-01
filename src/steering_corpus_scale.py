"""Does the steering reachability gap survive corpus scaling, and what is it made of?

Follow-up to steering_reachability.py, rebuilt after feedback from Aayush Mishra, first
author of "Steered LLM Activations are Non-Surjective".

Two things change.

1. THE NORM ANALYSIS WAS FRAMED WRONG.

   The earlier script reported how much of the excess nearest-neighbour distance survived
   rescaling the steered vector back to its original norm, and treated the surviving
   fraction as the real effect. That is not right. The theorem is about collisions in
   activation space, so a norm difference is itself a sufficient reason for two
   activations not to collide. Norm is part of what is being claimed, not a nuisance
   variable to control away.

   So instead of controlling, decompose. For a steered activation and its nearest natural
   neighbour, the squared distance splits exactly:

       ||a - b||^2 = (||a|| - ||b||)^2 + 2||a|| ||b|| (1 - cos theta)
                     \_____________/    \_______________________/
                        norm part              angular part

   Report both. Neither is an artefact; the question is what the departure is made of.

2. THE CORPUS WAS TOO SMALL TO MEAN ANYTHING.

   32 hand-written sentences give a few hundred activations in 768 dimensions, where
   everything is far from everything. The earlier run found natural activations sitting a
   median of 69 apart when their norm was only 96, which leaves very little manifold to be
   off.

   The testable version of that objection: sweep the reference corpus over orders of
   magnitude and ask whether the gap between steered and natural nearest-neighbour
   distances persists.

     - gap persists as the corpus grows  ->  evidence the departure is real
     - gap closes                        ->  the original result was small-sample

   Either answer is worth having, and only the second is bad news for the earlier run.

Everything else is held identical to steering_reachability.py so the numbers stay
comparable: GPT-2 small, layer 6, difference-in-means steering vector from the same prompt
sets, strengths expressed as multiples of the median activation norm.

Run: python src/steering_corpus_scale.py
"""
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils

torch.set_grad_enabled(False)

LAYER = 6
NORM_FRACTIONS = [0.1, 0.25, 0.5, 1.0, 2.0]
CORPUS_SIZES = [200, 1_000, 5_000, 20_000, 50_000]
N_QUERY = 1_000
CHUNK = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# identical to steering_reachability.py so the vector is the same
POSITIVE = [
    "I love this", "This is wonderful", "What a delight", "I adore it",
    "Absolutely fantastic", "This makes me happy", "A joy to use", "I am so pleased",
]
NEGATIVE = [
    "I hate this", "This is terrible", "What a disaster", "I despise it",
    "Absolutely awful", "This makes me angry", "A pain to use", "I am so annoyed",
]


def activations(model, texts, batch_size=32):
    """Residual stream at LAYER, BOS dropped, flattened over positions."""
    name = utils.get_act_name("resid_post", LAYER)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        toks = model.to_tokens(batch)
        _, cache = model.run_with_cache(toks, names_filter=name)
        resid = cache[name][:, 1:, :]
        mask = toks[:, 1:] != model.tokenizer.pad_token_id
        out.append(resid[mask].to(DEVICE))
    return torch.cat(out, 0)


def nearest(query, reference):
    """For each query row, the distance to its nearest reference row, plus the norm and
    angular parts of that distance.

    Returns (dist, norm_part, ang_part), each of shape (len(query),). The parts satisfy
    dist^2 = norm_part^2 + ang_part^2 up to floating point.
    """
    d_all, n_all, a_all = [], [], []
    for i in range(0, query.shape[0], CHUNK):
        q = query[i:i + CHUNK]
        d = torch.cdist(q, reference)
        dist, idx = d.min(dim=1)
        nb = reference[idx]
        qn, bn = q.norm(dim=1), nb.norm(dim=1)
        cos = (q * nb).sum(1) / (qn * bn).clamp_min(1e-9)
        norm_sq = (qn - bn) ** 2
        ang_sq = (2 * qn * bn * (1 - cos)).clamp_min(0)
        d_all.append(dist)
        n_all.append(norm_sq.sqrt())
        a_all.append(ang_sq.sqrt())
    return torch.cat(d_all), torch.cat(n_all), torch.cat(a_all)


def main():
    print(f"device: {DEVICE}")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    steering = (activations(model, POSITIVE).mean(0)
                - activations(model, NEGATIVE).mean(0))
    steering = steering / steering.norm()

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:4_000]
    print(f"corpus texts: {len(texts)}")

    pool = activations(model, texts)
    median_norm = pool.norm(dim=1).median().item()
    print(f"pool activations: {pool.shape[0]}, median norm {median_norm:.1f}\n")

    perm = torch.randperm(pool.shape[0], device=DEVICE)
    query = pool[perm[:N_QUERY]]
    rest = pool[perm[N_QUERY:]]

    print(f"{'corpus':>8}{'baseline':>10}" +
          "".join(f"{'a=' + str(a):>24}" for a in NORM_FRACTIONS))
    print(f"{'':>8}{'':>10}" + "".join(f"{'dist (norm/ang)':>24}" for _ in NORM_FRACTIONS))
    print("-" * (18 + 24 * len(NORM_FRACTIONS)))

    for size in CORPUS_SIZES:
        if size > rest.shape[0]:
            print(f"{size:>8}  pool exhausted, stopping")
            break
        ref = rest[:size]
        base, _, _ = nearest(query, ref)
        row = f"{size:>8}{base.mean().item():>10.1f}"
        for a in NORM_FRACTIONS:
            steered = query + a * median_norm * steering
            d, n, g = nearest(steered, ref)
            row += f"{d.mean().item():>10.1f} ({n.mean().item():>4.0f}/{g.mean().item():>4.0f})"
        print(row, flush=True)

    print("\nRead the table by column, not by row.")
    print("  baseline falling with corpus size is expected: more neighbours, closer fit.")
    print("  the question is whether steered minus baseline also falls, or holds.")
    print("  norm/ang shows what the distance is made of, not what to discount.")


if __name__ == "__main__":
    main()
