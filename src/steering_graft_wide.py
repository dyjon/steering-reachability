r"""Push the reachable displacement out to steering's operating magnitude.

`steering_graft.py` measured graft against steering at matched norm and found steering the
quieter direction, fitting `KL ~ |d|^1.98` against the graft's 2.65. But the variant family
capped ‖d‖ at 6.37, which is 0.40 activation norms, while steering at delta = 1 operates at
about 16. Comparing the two at steering's own magnitude meant extrapolating a power law a
factor of 2.4 beyond the data, which is not a measurement.

This widens the family without changing anything else. Two rows are appended to the ladder:

    k = 95    the variant shares only its FINAL token with the base
    k = 96    the variant is a different passage entirely, final token included

Holding the final token fixed is what was capping ‖d‖. That token's own embedding dominates
the residual stream at its position, so every earlier row was measuring displacements with the
largest single contributor held constant. Letting it move should carry ‖d‖ into the range
steering actually uses, turning the comparison into a measurement.

Both new variants are still real prompts, so their layer-j last-position activations are still
genuine elements of `Im(F)` and the graft still lands on a reachable point. Nothing about the
intervention changes.

TWO CHECKS COME FREE.

1. REPLICATION. `SWAP_K` is extended, not altered, and the conditions run in order, so the
   generator draws for k = 0, 8, 24, 48, 72 are the same draws in the same sequence as
   `steering_graft.py` made. The prompt set, the SVD basis, the steering vector and the base
   log-probabilities are all unchanged. **Those five rows must come back bit-identical.** If
   they do not, something in the environment moved and the previous numbers are suspect.

2. A FALSIFIABLE PREDICTION. The graft writeup explained `d in sub` sitting at 9.5-12.4%, far
   under the 61.7% of variance the top-32 holds, by arguing that `d` is the difference between
   two *related* prompts so the shared structure cancels, leaving a residual in low-variance
   directions. If that is right, `d in sub` must climb toward 61.7% as the variants stop being
   related, and the k = 96 row is a difference between two unrelated passages. **If `d in sub`
   stays near 10% there, the explanation in that writeup is wrong.**

Also prints library versions, because the run this follows used an unpinned pip install on
Kaggle's floating "Latest Container Image" and its versions are not recoverable. The
`unpack`/`repack` helpers exist only because a transformers bump changed Qwen2 decoder layers
from returning a tuple to returning a bare tensor, so the version is load-bearing.

Run: python src/steering_graft_wide.py
"""
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 128                 # held at 128 so the first five rows stay comparable
SEQ_LEN = 96
SWAP_K = [0, 8, 24, 48, 72, 95, 96]   # 95 shares only the last token, 96 shares nothing
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
    """Qwen2 decoder layers return a bare tensor; older versions return a tuple."""
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
    return f"{m:>9.4f}+-{se:.4f}"


def energy_in(basis, v):
    u = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return ((u @ basis) ** 2).sum(1)


def loglog_slope(xs, ys):
    """OLS slope of log y on log x. Returns nan if fewer than two usable points."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pts)
    if n < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den > 0 else float("nan")


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
    j = len(layers) // 2
    layer = layers[j]

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 600]
    ids = collect_ids(tok, texts, 2 * N_PROMPTS)
    base_ids, donor_ids = ids[:N_PROMPTS], ids[N_PROMPTS:]
    print(f"{len(layers)} layers, hooking layer {j}, "
          f"{N_PROMPTS} prompts of {SEQ_LEN} tokens")

    steering = hidden_from_texts(model, layer, tok, POSITIVE).mean(0) \
        - hidden_from_texts(model, layer, tok, NEGATIVE).mean(0)
    steering = steering / steering.norm()

    base_h = hidden_at_last(model, layer, base_ids)
    dim = base_h.shape[1]
    median_norm = base_h.norm(dim=1).median().item()
    print(f"dim {dim}, median activation norm {median_norm:.1f}")

    hc = base_h - base_h.mean(0)
    _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
    basis = Vh[:TOP_K].T.contiguous()
    var = S ** 2
    explained = (var[:TOP_K].sum() / var.sum()).item()
    print(f"top-{TOP_K} subspace holds {explained:.1%} of activation variance")
    print(f"steering energy in that subspace: "
          f"{energy_in(basis, steering[None, :]).item():.1%}"
          f"   (isotropic null is {TOP_K / dim:.1%})\n")

    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    base_lp = logprobs(model, layer, base_ids)

    print(f"{'k':>4}{'||d||':>8}{'/norm':>7}{'d in sub':>10}"
          f"{'graft':>16}{'steering':>16}{'subspace':>16}{'full':>16}")
    print("-" * 93)

    mags, cols = [], {"graft": [], "steering": [], "subspace": [], "full": []}

    for k in SWAP_K:
        var_ids = torch.cat([donor_ids[:, :k], base_ids[:, k:]], dim=1)
        var_h = hidden_at_last(model, layer, var_ids)

        d = var_h - base_h
        mag = d.norm(dim=1)

        steer_d = steering[None, :] * mag[:, None]
        c = torch.randn(N_PROMPTS, TOP_K, generator=gen, device=DEVICE)
        sub = c @ basis.T
        sub_d = sub / sub.norm(dim=1, keepdim=True) * mag[:, None]

        vals = {
            "graft": kl(base_lp, logprobs(model, layer, base_ids, d)),
            "steering": kl(base_lp, logprobs(model, layer, base_ids, steer_d)),
            "subspace": kl(base_lp, logprobs(model, layer, base_ids, sub_d)),
            "full": kl(base_lp, logprobs(model, layer, var_ids)),
        }

        mean_mag = mag.mean().item()
        mags.append(mean_mag)
        for name in cols:
            cols[name].append(vals[name].mean().item())

        share = energy_in(basis, d).mean().item() if k else float("nan")
        label = "all" if k == SEQ_LEN else str(k)
        print(f"{label:>4}{mean_mag:>8.2f}{mean_mag / median_norm:>7.2f}{share:>9.1%} "
              + "".join(stat(vals[n]) for n in ("graft", "steering", "subspace", "full")),
              flush=True)

    print(f"\nsteering at delta = 1 would sit at ||d|| = {median_norm:.1f}, "
          f"for reference against the column above.")

    print("\nloglog slope of KL against ||d||:")
    print(f"{'':>12}{'k<=72 only':>13}{'all rows':>11}")
    narrow = [i for i, k in enumerate(SWAP_K) if 0 < k <= 72]
    wide = [i for i, k in enumerate(SWAP_K) if k > 0]
    for name in ("graft", "steering", "subspace", "full"):
        a = loglog_slope([mags[i] for i in narrow], [cols[name][i] for i in narrow])
        b = loglog_slope([mags[i] for i in wide], [cols[name][i] for i in wide])
        print(f"{name:>12}{a:>13.2f}{b:>11.2f}")

    print("\nCHECK 1: rows k = 0..72 must match steering_graft.py at commit b8d3c9f exactly.")
    print("CHECK 2: d in sub must climb toward the explained-variance figure by k = all.")
    print("         If it stays near 10%, the tangent-vs-principal explanation is wrong.")


if __name__ == "__main__":
    main()
