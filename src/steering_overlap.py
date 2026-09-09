r"""Do the two domains lie on one curve, or on two?

`steering_fraction.py` killed the set-composition confound: graft/refusal held at 0.32, 0.26,
0.26, 0.25 while the harmful fraction went 0.50 to 0.00 and refusal's subspace energy fell from
97.7% to 69.0%. But it exposed a second confound. The instruction ladder spanned 0.18 to 0.80
activation norms and the wikitext one 0.74 to 1.43, overlapping in a sliver, so the headline
0.25x was measured where wikitext has no data and the wikitext ~7 where instructions cannot
reach. At the single overlapping displacement the domain effect was 2.0x for refusal and 2.4x
for sentiment, not the order of magnitude the raw comparison implied.

The ratio turned out to rise with displacement in both domains. So the question is now sharp:

    is graft/refusal a function of displacement alone, with the two domains merely sampling
    different stretches of one curve, or are there two curves?

Answering it needs the domains to span a common range, which rank ladders alone cannot deliver
because instruction activations cluster tightly and wikitext ones do not. So this run stops
insisting on one construction and uses whichever reaches a given displacement. Every one of them
still lands on an activation some real prompt produces, which is the only property the graft
needs.

    WIKITEXT     swap the first 8 / 24 / 72 tokens with another passage's   (small)
                 nearest and farthest of a 4096 pool                        (large)

    INSTRUCTIONS same instruction behind a benign prefix                    (tiny)
                 nearest, rank-32 and farthest of a 512 pool                (small)
                 a wikitext passage put through the chat template           (large)

A passage pasted into a chat turn is a perfectly ordinary prompt, and its activation sits far
from a short instruction's, which is what lifts the instruction ladder's ceiling. The prefix
variants do the same at the other end.

ANCHOR. The wikitext base set and the first 128 of its pool are the same passages as
`steering_graft_wide.py` used, so the swap-8, swap-24 and swap-72 graft values must reproduce
`4ccdfbc` exactly: 0.0086, 0.0345 and 0.2561 at layer 12.

READING IT. Both domains are reported against ‖d‖ / median activation norm, then interpolated to
the common fractions 0.2, 0.4, 0.6, 0.8, 1.0. If graft/refusal agrees at matched fraction, there
is no domain effect and everything earlier was a displacement effect. If the instruction values
sit systematically below, the domain effect is real and this measures its size.

Instructions are Alpaca only. Run 8 established the harmful fraction does not matter, so
harmless-only is the cleanest base and AdvBench is needed solely for the refusal vector.

No ablation and no jailbreak. Only AdvBench's `goal` column is ever read.

Run: python src/steering_overlap.py
"""
import csv
import io
import math
import urllib.request

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [12, 18]
BASE_N = 128
WIKI_POOL = 4096
INS_POOL = 512
N_VEC = 96
SEQ_LEN = 96
SWAPS = [8, 24, 72]
REPORT_AT = [0.2, 0.4, 0.6, 0.8, 1.0]
TOP_K = 32
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
PREFIXES = ["Please ", "Could you ", "I would like you to ", "Kindly "]
POSITIVE = ["I love this", "This is wonderful", "What a delight", "I adore it",
            "Absolutely fantastic", "This makes me happy", "A joy to use", "I am so pleased"]
NEGATIVE = ["I hate this", "This is terrible", "What a disaster", "I despise it",
            "Absolutely awful", "This makes me angry", "A pain to use", "I am so annoyed"]


def unpack(o):
    return o[0] if isinstance(o, tuple) else o


def repack(o, h):
    return (h,) + o[1:] if isinstance(o, tuple) else h


def add_at_last(vecs):
    def hook(module, args, output):
        h = unpack(output).clone()
        h[:, -1, :] += vecs
        return repack(output, h)
    return hook


def logprobs(model, layer, ids, mask, delta=None):
    out = []
    for i in range(0, ids.shape[0], BATCH):
        handle = layer.register_forward_hook(add_at_last(delta[i:i + BATCH])) \
            if delta is not None else None
        try:
            lg = model(input_ids=ids[i:i + BATCH],
                       attention_mask=mask[i:i + BATCH]).logits[:, -1, :]
        finally:
            if handle is not None:
                handle.remove()
        out.append(F.log_softmax(lg.float(), dim=-1))
    return torch.cat(out, 0)


def hidden(model, layer, ids, mask):
    got = []
    h = layer.register_forward_hook(lambda m, a, o: got.append(unpack(o)[:, -1, :]))
    try:
        for i in range(0, ids.shape[0], BATCH):
            model(input_ids=ids[i:i + BATCH], attention_mask=mask[i:i + BATCH])
    finally:
        h.remove()
    return torch.cat(got, 0)


def kl(p, q):
    return (p.exp() * (p - q)).sum(-1)


def energy_in(basis, v):
    u = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return ((u @ basis) ** 2).sum(1)


def at_frac(fracs, vals, target):
    """Interpolate a value to a target fraction of activation norm, in log-log."""
    order = sorted(range(len(fracs)), key=lambda i: fracs[i])
    fs = [fracs[i] for i in order]
    vs = [vals[i] for i in order]
    for i in range(len(fs) - 1):
        if fs[i] <= target <= fs[i + 1] and vs[i] > 0 and vs[i + 1] > 0:
            p = math.log(vs[i + 1] / vs[i]) / math.log(fs[i + 1] / fs[i])
            return vs[i] * math.exp(p * math.log(target / fs[i]))
    return float("nan")


def chat(tok, prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    return e["input_ids"], e["attention_mask"]


def load_harmful():
    """AdvBench goals only; the `target` column of affirmative-response prefixes is never read."""
    try:
        d = load_dataset("walledai/AdvBench", split="train")
        return [r.strip() for r in d["prompt"]]
    except Exception:
        raw = urllib.request.urlopen(ADVBENCH_CSV, timeout=60).read().decode("utf-8")
        return [r["goal"].strip() for r in csv.DictReader(io.StringIO(raw)) if r.get("goal")]


def wiki_rows(tok, n):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    ids, texts = [], []
    for t in ds["text"]:
        t = t.strip()
        if len(t) <= 600:
            continue
        e = tok(t, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(e) == SEQ_LEN:
            ids.append(e)
            texts.append(tok.decode(e))
            if len(ids) == n:
                break
    return torch.tensor(ids, device=DEVICE), texts


def report(label, layer_no, base_h, base_lp, ids, mask, donors, vectors, model, layer, gen):
    mn = base_h.norm(dim=1).median().item()
    hc = base_h - base_h.mean(0)
    _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
    basis = Vh[:TOP_K].T.contiguous()
    var = S ** 2
    print(f"\n--- {label}, layer {layer_no} --- median norm {mn:.1f}, "
          f"top-{TOP_K} holds {(var[:TOP_K].sum() / var.sum()).item():.1%}")
    print("    vector energy: " + ", ".join(
        f"{k} {energy_in(basis, v[None]).item():.1%}" for k, v in vectors.items())
        + f"   (null {TOP_K / base_h.shape[1]:.1%})")

    names = ["graft"] + list(vectors.keys()) + ["subspace"]
    print(f"\n{'condition':>12}{'||d||':>8}{'/norm':>7}" + "".join(f"{n:>17}" for n in names))
    print("-" * (27 + 17 * len(names)))
    fracs, cols = [], {n: [] for n in names}
    for cname, d_h in donors:
        d = d_h - base_h
        mag = d.norm(dim=1)
        c = torch.randn(ids.shape[0], TOP_K, generator=gen, device=DEVICE)
        sub = c @ basis.T
        vals = {"graft": kl(base_lp, logprobs(model, layer, ids, mask, d))}
        for vn, v in vectors.items():
            vals[vn] = kl(base_lp, logprobs(model, layer, ids, mask, v[None] * mag[:, None]))
        vals["subspace"] = kl(base_lp, logprobs(model, layer, ids, mask,
                                                sub / sub.norm(dim=1, keepdim=True) * mag[:, None]))
        m = mag.mean().item()
        fracs.append(m / mn)
        for n in names:
            cols[n].append(vals[n].mean().item())
        print(f"{cname:>12}{m:>8.2f}{m / mn:>7.2f}"
              + "".join(f"{vals[n].mean().item():>8.4f}"
                        f"+-{(vals[n].std(unbiased=True) / (vals[n].shape[0] ** 0.5)).item():.4f}"
                        for n in names), flush=True)
    return fracs, cols


def main():
    import datasets as _d
    import transformers as _t
    print(f"torch {torch.__version__}, transformers {_t.__version__}, datasets {_d.__version__}")
    print(f"device: {DEVICE}, model: {MODEL}\n")

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()
    layers = model.model.layers

    harmful = load_harmful()
    d = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"].strip() for r in d
                if not r["input"].strip() and 20 < len(r["instruction"]) < 160]
    print(f"AdvBench {len(harmful)} harmful, Alpaca {len(harmless)} harmless")

    w_ids, w_texts = wiki_rows(tok, BASE_N + WIKI_POOL)
    wb_ids, wp_ids = w_ids[:BASE_N], w_ids[BASE_N:]
    wb_mask, wp_mask = torch.ones_like(wb_ids), torch.ones_like(wp_ids)
    print(f"wikitext: base {BASE_N}, pool {wp_ids.shape[0]}")

    ins_base = harmless[N_VEC:N_VEC + BASE_N]
    ins_pool = harmless[N_VEC + BASE_N:N_VEC + BASE_N + INS_POOL]
    ib, ibm = chat(tok, ins_base)
    ip, ipm = chat(tok, ins_pool)
    pre_texts = [PREFIXES[i % len(PREFIXES)] + t[0].lower() + t[1:] for i, t in enumerate(ins_base)]
    pb, pbm = chat(tok, pre_texts)
    wc, wcm = chat(tok, w_texts[BASE_N:BASE_N + BASE_N])
    print(f"instructions: base {len(ins_base)}, pool {len(ins_pool)}, "
          f"prefix variants {len(pre_texts)}, passage donors {BASE_N}\n")

    vh, vhm = chat(tok, harmful[:N_VEC])
    vs, vsm = chat(tok, harmless[:N_VEC])
    pi, pim = chat(tok, POSITIVE)
    ni, nim = chat(tok, NEGATIVE)

    out = {}
    for L in LAYERS:
        lay = layers[L]
        ref = hidden(model, lay, vh, vhm).mean(0) - hidden(model, lay, vs, vsm).mean(0)
        sen = hidden(model, lay, pi, pim).mean(0) - hidden(model, lay, ni, nim).mean(0)
        vectors = {"refusal": ref / ref.norm(), "sentiment": sen / sen.norm()}

        # wikitext arm
        wb_h = hidden(model, lay, wb_ids, wb_mask)
        wp_h = hidden(model, lay, wp_ids, wp_mask)
        wb_lp = logprobs(model, lay, wb_ids, wb_mask)
        wo = torch.cdist(wb_h, wp_h).argsort(dim=1)
        donors_w = []
        for k in SWAPS:
            v = torch.cat([wp_ids[:BASE_N, :k], wb_ids[:, k:]], dim=1)
            donors_w.append((f"swap{k}", hidden(model, lay, v, torch.ones_like(v))))
        donors_w.append(("near", wp_h[wo[:, 0]]))
        donors_w.append(("far", wp_h[wo[:, -1]]))
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        fw, cw = report("wikitext", L, wb_h, wb_lp, wb_ids, wb_mask, donors_w,
                        vectors, model, lay, gen)

        # instruction arm
        ib_h = hidden(model, lay, ib, ibm)
        ip_h = hidden(model, lay, ip, ipm)
        pb_h = hidden(model, lay, pb, pbm)
        wc_h = hidden(model, lay, wc, wcm)
        ib_lp = logprobs(model, lay, ib, ibm)
        io_ = torch.cdist(ib_h, ip_h).argsort(dim=1)
        donors_i = [("prefix", pb_h), ("near", ip_h[io_[:, 0]]),
                    ("mid", ip_h[io_[:, 32]]), ("far", ip_h[io_[:, -1]]),
                    ("passage", wc_h)]
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        fi, ci = report("instructions", L, ib_h, ib_lp, ib, ibm, donors_i,
                        vectors, model, lay, gen)
        out[L] = (fw, cw, fi, ci)

    print(f"\n{'=' * 96}\nONE CURVE OR TWO?  graft/refusal at matched fraction of activation norm\n{'=' * 96}")
    for L in LAYERS:
        fw, cw, fi, ci = out[L]
        print(f"\nlayer {L}")
        print(f"{'fraction':>10}{'wikitext':>12}{'instructions':>15}{'ratio':>10}")
        for t in REPORT_AT:
            gw = at_frac(fw, cw["graft"], t)
            rw = at_frac(fw, cw["refusal"], t)
            gi = at_frac(fi, ci["graft"], t)
            ri = at_frac(fi, ci["refusal"], t)
            a = gw / rw if rw == rw and rw > 0 else float("nan")
            b = gi / ri if ri == ri and ri > 0 else float("nan")
            f_ = a / b if a == a and b == b and b > 0 else float("nan")
            fmt = lambda x: f"{x:.2f}x" if x == x else "--"
            print(f"{t:>10.1f}{fmt(a):>12}{fmt(b):>15}{fmt(f_):>10}")

    print("\nANCHOR: wikitext swap8/24/72 graft must read 0.0086, 0.0345, 0.2561 at layer 12.")
    print("ONE CURVE: the two columns agree at matched fraction, and every earlier domain")
    print("           difference was a displacement difference.")
    print("TWO CURVES: instructions sit systematically below, and the last column is the")
    print("           size of the real domain effect.")


if __name__ == "__main__":
    main()
