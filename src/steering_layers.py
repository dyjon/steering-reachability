r"""Does the graft-versus-steering gap hold across depth, and does the position confound matter?

`steering_graft_wide.py` measured, at one activation norm, steering moving the next-token
distribution 0.58 nats against a reachable displacement's 3.16. That is one layer of one model.
Two things need settling before it means anything general.

DEPTH. Everything so far is layer 12 of 24. If the gap holds at 6 and 18 it is a property of
the model; if it is a mid-depth artefact, that is worth knowing before anyone quotes 5.4x.

THE POSITION CONFOUND. The steering vector has been a difference of means over sentences three
to five tokens long, read at sequence positions 2 to 4, and then applied at position 95 of a
96-token passage. Under RoPE those are different regimes, so some unknown fraction of "the
sentiment direction" has been a short-sequence direction all along. It has been declared as a
limitation in three writeups and never measured.

The fix is to build a second vector from the SAME sentences with neutral wikitext filler
prepended, so each sentiment sentence ends at position 95 and is read exactly where the vector
gets applied. The filler is shared between the positive and negative arms, so it cancels in the
difference of means.

**This adds a column rather than replacing one.** `steer_lo` is the original construction,
`steer_hi` the position-matched one. Replacing would have destroyed the replication anchor
below; running both measures how much the confound was ever worth. The cosine between them is
printed and is the cheapest decisive number here: near 1 and the confound never mattered, near
0 and three writeups were measuring something other than sentiment.

REPLICATION ANCHOR. N_PROMPTS, SEQ_LEN, SWAP_K, the corpus filter and the seed are unchanged,
and the generator is re-seeded at the start of each layer so the draw sequence per layer
matches the one `steering_graft_wide.py` made. **Layer 12's `graft`, `steer_lo`, `subspace` and
`full` columns must therefore come back bit-identical to commit 4ccdfbc.** Any drift there is a
bug in this refactor, not a finding about depth.

`full` is the variant prompt run end to end with no hook, which does not depend on which layer
would have been hooked, so it is computed once outside the layer loop.

Run: python src/steering_layers.py
"""
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [6, 12, 18]            # quarter, half, three-quarter depth of 24
N_PROMPTS = 128                 # unchanged, so layer 12 replicates 4ccdfbc
SEQ_LEN = 96
SWAP_K = [0, 8, 24, 48, 72, 95, 96]
TOP_K = 32
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSITIVE = [
    "I love this", "This is wonderful", "What a delight", "I adore it",
    "Absolutely fantastic", "This makes me happy", "A joy to use", "I am so pleased",
]
NEGATIVE = [
    "I hate this", "This is terrible", "What a disaster", "I despise it",
    "Absolutely awful", "This makes me angry", "A pain to use", "I am so annoyed",
]


def unpack(output):
    return output[0] if isinstance(output, tuple) else output


def repack(output, h):
    return (h,) + output[1:] if isinstance(output, tuple) else h


def add_at_last_position(vecs):
    def hook(module, args, output):
        h = unpack(output).clone()
        h[:, -1, :] += vecs
        return repack(output, h)
    return hook


def logprobs(model, layer, ids, delta=None):
    out = []
    for i in range(0, ids.shape[0], BATCH):
        chunk = ids[i:i + BATCH]
        handle = None
        if delta is not None:
            handle = layer.register_forward_hook(add_at_last_position(delta[i:i + BATCH]))
        try:
            logits = model(input_ids=chunk).logits[:, -1, :]
        finally:
            if handle is not None:
                handle.remove()
        out.append(F.log_softmax(logits.float(), dim=-1))
    return torch.cat(out, 0)


def hidden_at_last(model, layer, ids):
    grabbed = []
    handle = layer.register_forward_hook(
        lambda m, a, o: grabbed.append(unpack(o)[:, -1, :]))
    try:
        for i in range(0, ids.shape[0], BATCH):
            model(input_ids=ids[i:i + BATCH])
    finally:
        handle.remove()
    return torch.cat(grabbed, 0)


def hidden_from_texts(model, layer, tok, texts):
    """Short prompts, left-padded. Used for the original steering construction."""
    grabbed = []
    handle = layer.register_forward_hook(
        lambda m, a, o: grabbed.append(unpack(o)[:, -1, :]))
    try:
        enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
        model(**enc)
    finally:
        handle.remove()
    return torch.cat(grabbed, 0)


def kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def stat(v):
    m = v.mean().item()
    se = (v.std(unbiased=True) / (v.shape[0] ** 0.5)).item()
    return f"{m:>8.4f}+-{se:.4f}"


def energy_in(basis, v):
    u = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return ((u @ basis) ** 2).sum(1)


def loglog_slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pts)
    if n < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0:
        return float("nan")
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def at_unit_norm(mags, vals, median_norm):
    """Interpolate KL to a displacement of exactly one activation norm, using the local
    exponent between the two rows that bracket it. Returns nan if it is not bracketed."""
    lo = hi = None
    for i in range(len(mags) - 1):
        if mags[i] <= median_norm <= mags[i + 1]:
            lo, hi = i, i + 1
    if lo is None or vals[lo] <= 0 or vals[hi] <= 0:
        return float("nan")
    p = math.log(vals[hi] / vals[lo]) / math.log(mags[hi] / mags[lo])
    return vals[lo] * math.exp(p * math.log(median_norm / mags[lo]))


def collect_ids(tok, texts, n):
    ids = []
    for t in texts:
        e = tok(t, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(e) == SEQ_LEN:
            ids.append(e)
            if len(ids) == n:
                break
    if len(ids) < n:
        raise RuntimeError(f"only found {len(ids)} passages of {SEQ_LEN} tokens, need {n}")
    return torch.tensor(ids, device=DEVICE)


def sentiment_ids_at_end(tok, sentences, filler_ids):
    """Each sentence padded on the LEFT with neutral filler to exactly SEQ_LEN tokens, so the
    sentiment ends at position SEQ_LEN-1 -- where the steering vector is actually applied.
    Filler row i is shared between the positive and negative arms, so it cancels in the
    difference of means."""
    rows = []
    for i, s in enumerate(sentences):
        body = tok(s, add_special_tokens=False)["input_ids"]
        body = body[:SEQ_LEN]
        pad = filler_ids[i % filler_ids.shape[0], : SEQ_LEN - len(body)].tolist()
        rows.append(pad + body)
    return torch.tensor(rows, device=DEVICE)


def main():
    import datasets as _datasets
    import transformers as _transformers
    print(f"torch {torch.__version__}, transformers {_transformers.__version__}, "
          f"datasets {_datasets.__version__}")
    print(f"device: {DEVICE}, model: {MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()

    layers = model.model.layers
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 600]
    ids = collect_ids(tok, texts, 2 * N_PROMPTS)
    base_ids, donor_ids = ids[:N_PROMPTS], ids[N_PROMPTS:]
    print(f"{len(layers)} layers, hooking {LAYERS}, {N_PROMPTS} prompts of {SEQ_LEN} tokens")

    pos_ids = sentiment_ids_at_end(tok, POSITIVE, donor_ids)
    neg_ids = sentiment_ids_at_end(tok, NEGATIVE, donor_ids)

    # layer-independent: the variant prompt run end to end, no hook
    base_lp = logprobs(model, layers[0], base_ids)
    var_ids_by_k, full_kl_by_k = {}, {}
    for k in SWAP_K:
        v = torch.cat([donor_ids[:, :k], base_ids[:, k:]], dim=1)
        var_ids_by_k[k] = v
        full_kl_by_k[k] = kl(base_lp, logprobs(model, layers[0], v))

    summary = []

    for L in LAYERS:
        layer = layers[L]
        print(f"\n{'=' * 96}\nLAYER {L}\n{'=' * 96}")

        s_lo = hidden_from_texts(model, layer, tok, POSITIVE).mean(0) \
            - hidden_from_texts(model, layer, tok, NEGATIVE).mean(0)
        s_lo = s_lo / s_lo.norm()

        s_hi = hidden_at_last(model, layer, pos_ids).mean(0) \
            - hidden_at_last(model, layer, neg_ids).mean(0)
        s_hi = s_hi / s_hi.norm()

        base_h = hidden_at_last(model, layer, base_ids)
        dim = base_h.shape[1]
        median_norm = base_h.norm(dim=1).median().item()

        hc = base_h - base_h.mean(0)
        _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
        basis = Vh[:TOP_K].T.contiguous()
        var = S ** 2
        cos = torch.dot(s_lo, s_hi).item()

        print(f"median activation norm {median_norm:.1f}, "
              f"top-{TOP_K} holds {(var[:TOP_K].sum() / var.sum()).item():.1%} of variance")
        print(f"steer_lo energy in subspace {energy_in(basis, s_lo[None, :]).item():.1%}, "
              f"steer_hi {energy_in(basis, s_hi[None, :]).item():.1%}, "
              f"isotropic null {TOP_K / dim:.1%}")
        print(f"cosine(steer_lo, steer_hi) = {cos:+.3f}"
              f"   <- near 1 means the position confound never mattered\n")

        gen = torch.Generator(device=DEVICE).manual_seed(SEED)   # re-seeded per layer

        print(f"{'k':>4}{'||d||':>8}{'/norm':>7}{'d_sub':>7}"
              f"{'graft':>17}{'steer_lo':>17}{'steer_hi':>17}{'subspace':>17}")
        print("-" * 104)

        mags, cols = [], {"graft": [], "steer_lo": [], "steer_hi": [], "subspace": []}

        for k in SWAP_K:
            var_h = hidden_at_last(model, layer, var_ids_by_k[k])
            d = var_h - base_h
            mag = d.norm(dim=1)

            c = torch.randn(N_PROMPTS, TOP_K, generator=gen, device=DEVICE)
            sub = c @ basis.T
            sub_d = sub / sub.norm(dim=1, keepdim=True) * mag[:, None]

            vals = {
                "graft": kl(base_lp, logprobs(model, layer, base_ids, d)),
                "steer_lo": kl(base_lp, logprobs(model, layer, base_ids,
                                                 s_lo[None, :] * mag[:, None])),
                "steer_hi": kl(base_lp, logprobs(model, layer, base_ids,
                                                 s_hi[None, :] * mag[:, None])),
                "subspace": kl(base_lp, logprobs(model, layer, base_ids, sub_d)),
            }

            m = mag.mean().item()
            mags.append(m)
            for name in cols:
                cols[name].append(vals[name].mean().item())

            share = energy_in(basis, d).mean().item() if k else float("nan")
            label = "all" if k == SEQ_LEN else str(k)
            print(f"{label:>4}{m:>8.2f}{m / median_norm:>7.2f}{share:>7.1%}"
                  + "".join(stat(vals[n]) for n in
                            ("graft", "steer_lo", "steer_hi", "subspace"))
                  + f"   full {full_kl_by_k[k].mean().item():>7.4f}", flush=True)

        print(f"\nloglog slope, rows with k > 0:")
        for name in ("graft", "steer_lo", "steer_hi", "subspace"):
            xs = [mags[i] for i, k in enumerate(SWAP_K) if k > 0]
            ys = [cols[name][i] for i, k in enumerate(SWAP_K) if k > 0]
            print(f"  {name:>9}  {loglog_slope(xs, ys):.2f}")

        g = at_unit_norm(mags, cols["graft"], median_norm)
        lo = at_unit_norm(mags, cols["steer_lo"], median_norm)
        hi = at_unit_norm(mags, cols["steer_hi"], median_norm)
        print(f"\nat one activation norm ({median_norm:.1f}):  "
              f"graft {g:.2f}   steer_lo {lo:.2f}   steer_hi {hi:.2f}")
        print(f"  graft/steer_lo = {g / lo:.1f}x    graft/steer_hi = {g / hi:.1f}x")
        summary.append((L, median_norm, cos, g, lo, hi))

    print(f"\n{'=' * 96}\nACROSS DEPTH\n{'=' * 96}")
    print(f"{'layer':>6}{'norm':>8}{'cos(lo,hi)':>12}"
          f"{'graft':>9}{'steer_lo':>10}{'steer_hi':>10}{'g/lo':>8}{'g/hi':>8}")
    for L, n, c, g, lo, hi in summary:
        print(f"{L:>6}{n:>8.1f}{c:>+12.3f}{g:>9.2f}{lo:>10.2f}{hi:>10.2f}"
              f"{g / lo:>8.1f}{g / hi:>8.1f}")

    print("\nCHECK: layer 12 graft, steer_lo and subspace must match 4ccdfbc bit-identically.")
    print("If the ratio holds at 6 and 18, the gap is a property of the model, not of depth 12.")
    print("If cos(lo, hi) is near 1, the position confound declared three times never mattered.")


if __name__ == "__main__":
    main()
