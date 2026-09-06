r"""At equal displacement, does steering move behaviour more or less than a reachable step?

steering_subspace.py established that steering is the quietest direction tested: 0.4337 KL
against 0.7289 for a random direction inside the top-32 principal subspace at matched
magnitude, with an additive energy-split model predicting steering's KL to within 6% and 13%
and no free parameters.

What that leaves open is the scale. Nobody can say whether 0.43 nats is a lot, so the gate
question -- is behaviour sensitive at the magnitude steering operates at -- still has no
answer. `natdiff` in the previous script was not a reference either: rescaling h_i - h_k to a
fixed magnitude produces a direction, not a reachable point, which is why its spread came back
at 78% of its mean.

So build the reference out of activations that real prompts actually produce.

Take a passage truncated to exactly SEQ_LEN tokens. Swap its first k tokens for another
passage's, holding the length and the final token fixed. The variant is a real prompt, so its
layer-j last-position activation is a genuine element of Im(F) = {F(r<i, s_i, Theta)}.

    d = h_last(variant) - h_last(base)

GRAFT: run the *original* prompt with h_last += d. Everything upstream of the hook is
identical, so this differs from a steering run only in the vector added at one position, and
the point it lands on is one a prompt reaches. That is the matched control the whole question
needs, and it uses the same intervention machinery as steering rather than a different one.

Then fire steering, and a random top-k direction, at exactly ||d||, per prompt. Sweeping k
sweeps ||d||, so the output is a curve rather than a point.

Reading it:

    steering < graft
        a near-miss costs less behaviourally than its L2 implies. Formal non-surjectivity
        (Thm 4.2) has little behavioural bite, and the expensive prompt search is worth running.

    steering > graft
        steering reaches a direction prompting does not, at the same distance. The paper's
        caution carries through to behaviour without further work.

    steering ~= graft
        activation-space distance is the whole story and direction does not matter at this
        scale.

`full` is the variant run end to end with no hook. It is not a control -- it moves every
position, not just the last -- but it says how much behaviour differs between two prompts a
reader would treat as related, which is the only interpretable unit on the KL axis.

Two fixes to the previous script, both of which mattered:

  - error bars are over PROMPTS, not over draws, so the steering column finally has one
  - every prompt gets its own random draw instead of five global directions shared by all

k = 0 makes the variant identical to the base, so that row must read 0.0000 across.

Run: python src/steering_graft.py
"""
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 128
SEQ_LEN = 96                    # every sequence exactly this long, so no padding, no mask
SWAP_K = [0, 8, 24, 48, 72]     # leading tokens replaced; 0 is the sanity row
TOP_K = 32                      # principal subspace dimension, as in steering_subspace.py
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
    """Add a per-example vector (B, D) to the residual stream at the final position."""
    def hook(module, args, output):
        h = unpack(output).clone()
        h[:, -1, :] += vecs
        return repack(output, h)
    return hook


def logprobs(model, layer, ids, delta=None):
    """Next-token log probabilities, optionally grafting a per-example vector at the last
    position. `delta` is (N, D) aligned with `ids`, or None."""
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
    """For the short sentiment prompts, which need padding."""
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
    """mean and standard error over prompts."""
    m = v.mean().item()
    se = (v.std(unbiased=True) / (v.shape[0] ** 0.5)).item()
    return f"{m:>9.4f}+-{se:.4f}"


def energy_in(basis, v):
    """Fraction of each row of v (N, D) lying in span(basis), basis is (D, k) orthonormal."""
    u = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return ((u @ basis) ** 2).sum(1)


def collect_ids(tok, texts, n):
    """Sequences of exactly SEQ_LEN tokens, so batches need no padding or mask."""
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
    print(f"dim {dim}, median activation norm {base_h.norm(dim=1).median().item():.1f}")

    hc = base_h - base_h.mean(0)
    _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
    basis = Vh[:TOP_K].T.contiguous()                    # (D, k), orthonormal
    var = S ** 2
    print(f"top-{TOP_K} subspace holds {(var[:TOP_K].sum() / var.sum()).item():.1%} "
          f"of activation variance")
    print(f"steering energy in that subspace: "
          f"{energy_in(basis, steering[None, :]).item():.1%}"
          f"   (isotropic null is {TOP_K / dim:.1%})\n")

    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    base_lp = logprobs(model, layer, base_ids)

    print(f"{'k':>3}{'||d||':>8}{'d in sub':>10}"
          f"{'graft':>16}{'steering':>16}{'subspace':>16}{'full':>16}")
    print("-" * 85)

    for k in SWAP_K:
        var_ids = torch.cat([donor_ids[:, :k], base_ids[:, k:]], dim=1)
        var_h = hidden_at_last(model, layer, var_ids)

        d = var_h - base_h                               # (N, D), a reachable displacement
        mag = d.norm(dim=1)                              # (N,)

        # steering and a fresh per-prompt subspace direction, both at exactly ||d||
        steer_d = steering[None, :] * mag[:, None]
        c = torch.randn(N_PROMPTS, TOP_K, generator=gen, device=DEVICE)
        sub = c @ basis.T
        sub_d = sub / sub.norm(dim=1, keepdim=True) * mag[:, None]

        kl_graft = kl(base_lp, logprobs(model, layer, base_ids, d))
        kl_steer = kl(base_lp, logprobs(model, layer, base_ids, steer_d))
        kl_sub = kl(base_lp, logprobs(model, layer, base_ids, sub_d))
        kl_full = kl(base_lp, logprobs(model, layer, var_ids))

        share = energy_in(basis, d).mean().item() if k else float("nan")
        print(f"{k:>3}{mag.mean().item():>8.2f}{share:>9.1%} "
              f"{stat(kl_graft)}{stat(kl_steer)}{stat(kl_sub)}{stat(kl_full)}",
              flush=True)

    print("\nk = 0 must read 0.0000 across: the variant is the base prompt.")
    print("graft vs steering at the same ||d|| is the comparison that matters.")
    print("full moves every position, not just the last, so it is a reference not a control.")


if __name__ == "__main__":
    main()
