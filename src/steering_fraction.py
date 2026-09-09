r"""Is refusal loud in the instruction domain, or loud in a set built to vary along refusal?

`steering_instruct.py` found graft/refusal inverting in the instruction domain: 0.24 at the
nearest rank against a flat ~7 on wikitext, meaning refusal produced four times the KL of a
reachable step of the same size. But the ladder set was half refusable by construction, which
makes the harmful/harmless split the dominant axis of variation. Refusal carried 97.7% of its
energy in a top-32 subspace holding 90.7% of the variance, so its loudness was close to
tautological.

This sweeps the one variable that separates the two readings.

    harmful fraction f in {0.50, 0.20, 0.05, 0.00}, in BOTH the base set and the donor pool

PREDICTIONS, WRITTEN BEFORE THE RUN.

  If the inversion was an artefact of set construction, then as f falls refusal stops being the
  principal axis: its subspace energy should collapse from ~98% toward the wikitext value of
  ~9%, and graft/refusal should climb back toward the wikitext ~7.

  If the domain effect is real, refusal stays loud even at f = 0.05, where 6 prompts in 128 are
  refusable and refusal cannot possibly be the principal component.

  f = 0.00 is the clean control: instruction-formatted, no refusal content anywhere. It
  separates "chat template" from "refusal is live". If refusal is still loud there, the effect
  is about the format rather than the behaviour.

TWO FIXES FROM THE LAST RUN.

The donor pool is now decoupled from the base set. Previously donors were the other 127 base
prompts, so the ladder's reach was whatever that set happened to span: wikitext covered 0.78 to
1.39 activation norms, instructions only 0.18 to 0.67, and the two overlapped at a single point.
A separate pool of 512, disjoint from the 128 bases, gives near neighbours that are nearer and
far ones that are farther, so both domains cover a usable range and can be read at a common
fraction of norm.

Reporting is therefore at fixed fractions (0.3, 0.5, 0.8) rather than at 1.0, which the
instruction ladder never reached.

The sentiment column is built chat-templated, the same way as run 7 and NOT the same way as
run 6. It is comparable to run 7 only. It is kept as a control for whether any behavioural
direction goes loud in instruction format or only refusal does.

No ablation and no jailbreak. Directions are added at matched magnitude and the next-token
distribution measured. Only AdvBench's `goal` column is read.

Run: python src/steering_fraction.py
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
POOL_N = 512
N_VEC = 96
SEQ_LEN = 96
FRACTIONS = [0.50, 0.20, 0.05, 0.00]
RANKS = [0, 4, 32, 128, 511]
REPORT_AT = [0.3, 0.5, 0.8]
TOP_K = 32
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
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


def at_frac(mags, vals, mn, frac):
    t = frac * mn
    for i in range(len(mags) - 1):
        if mags[i] <= t <= mags[i + 1] and vals[i] > 0 and vals[i + 1] > 0:
            p = math.log(vals[i + 1] / vals[i]) / math.log(mags[i + 1] / mags[i])
            return vals[i] * math.exp(p * math.log(t / mags[i]))
    return float("nan")


def chat(tok, prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    return e["input_ids"], e["attention_mask"]


def load_harmful():
    """AdvBench goals only. The companion `target` column of affirmative-response prefixes is
    for jailbreak optimisation and is deliberately never read."""
    try:
        d = load_dataset("walledai/AdvBench", split="train")
        rows = [r.strip() for r in d["prompt"]]
        print(f"AdvBench via HF: {len(rows)}")
        return rows
    except Exception as e:
        print(f"HF AdvBench unavailable ({type(e).__name__}), trying canonical CSV")
    raw = urllib.request.urlopen(ADVBENCH_CSV, timeout=60).read().decode("utf-8")
    rows = [r["goal"].strip() for r in csv.DictReader(io.StringIO(raw)) if r.get("goal")]
    print(f"AdvBench via CSV: {len(rows)}")
    return rows


def wikitext_ids(tok, n):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rows = []
    for t in ds["text"]:
        t = t.strip()
        if len(t) <= 600:
            continue
        e = tok(t, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(e) == SEQ_LEN:
            rows.append(e)
            if len(rows) == n:
                break
    ids = torch.tensor(rows, device=DEVICE)
    return ids, torch.ones_like(ids)


def ladder(model, layer, b_ids, b_mask, p_h, vectors, gen):
    """Rank ladder from base prompts onto a disjoint donor pool."""
    base_h = hidden(model, layer, b_ids, b_mask)
    dim = base_h.shape[1]
    mn = base_h.norm(dim=1).median().item()
    hc = base_h - base_h.mean(0)
    _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
    basis = Vh[:TOP_K].T.contiguous()
    var = S ** 2
    base_lp = logprobs(model, layer, b_ids, b_mask)
    order = torch.cdist(base_h, p_h).argsort(dim=1)

    names = ["graft"] + list(vectors.keys()) + ["subspace"]
    mags, cols, rows = [], {n: [] for n in names}, []
    for r in RANKS:
        d = p_h[order[:, r]] - base_h
        mag = d.norm(dim=1)
        c = torch.randn(b_ids.shape[0], TOP_K, generator=gen, device=DEVICE)
        sub = c @ basis.T
        vals = {"graft": kl(base_lp, logprobs(model, layer, b_ids, b_mask, d))}
        for vn, v in vectors.items():
            vals[vn] = kl(base_lp, logprobs(model, layer, b_ids, b_mask, v[None] * mag[:, None]))
        vals["subspace"] = kl(base_lp, logprobs(model, layer, b_ids, b_mask,
                                                sub / sub.norm(dim=1, keepdim=True) * mag[:, None]))
        m = mag.mean().item()
        mags.append(m)
        for n in names:
            cols[n].append(vals[n].mean().item())
        rows.append((r, m, m / mn, energy_in(basis, d).mean().item(),
                     {n: (vals[n].mean().item(),
                          (vals[n].std(unbiased=True) / (vals[n].shape[0] ** 0.5)).item())
                      for n in names}))
    stats = {"mn": mn, "explained": (var[:TOP_K].sum() / var.sum()).item(),
             "null": TOP_K / dim,
             "vec_energy": {vn: energy_in(basis, v[None]).item() for vn, v in vectors.items()}}
    return rows, mags, cols, stats, names


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
    print(f"Alpaca: {len(harmless)} harmless\n")

    vec_h, vec_s = harmful[:N_VEC], harmless[:N_VEC]
    pool_h, pool_s = harmful[N_VEC:], harmless[N_VEC:]
    need_h = int(round(max(FRACTIONS) * (BASE_N + POOL_N)))
    if len(pool_h) < need_h:
        raise RuntimeError(f"need {need_h} harmful beyond the vector set, have {len(pool_h)}")

    vh, vhm = chat(tok, vec_h)
    vs, vsm = chat(tok, vec_s)
    pi, pim = chat(tok, POSITIVE)
    ni, nim = chat(tok, NEGATIVE)

    vectors = {}
    for L in LAYERS:
        lay = layers[L]
        ref = hidden(model, lay, vh, vhm).mean(0) - hidden(model, lay, vs, vsm).mean(0)
        sen = hidden(model, lay, pi, pim).mean(0) - hidden(model, lay, ni, nim).mean(0)
        vectors[L] = {"refusal": ref / ref.norm(), "sentiment": sen / sen.norm()}
        print(f"layer {L}: cos(sentiment, refusal) = "
              f"{torch.dot(vectors[L]['sentiment'], vectors[L]['refusal']).item():+.3f}")

    arms = []
    for f in FRACTIONS:
        nb_h = int(round(f * BASE_N))
        np_h = int(round(f * POOL_N))
        base = pool_h[:nb_h] + pool_s[:BASE_N - nb_h]
        pool = pool_h[nb_h:nb_h + np_h] + pool_s[BASE_N - nb_h:BASE_N - nb_h + POOL_N - np_h]
        assert len(base) == BASE_N and len(pool) == POOL_N, (len(base), len(pool))
        b, bm = chat(tok, base)
        p, pm = chat(tok, pool)
        arms.append((f"instructions f={f:.2f}", b, bm, p, pm))

    wb, wbm = wikitext_ids(tok, BASE_N + POOL_N)
    arms.append(("wikitext (reference)", wb[:BASE_N], wbm[:BASE_N], wb[BASE_N:], wbm[BASE_N:]))

    summary = []
    for label, b, bm, p, pm in arms:
        print(f"\n{'#' * 108}\n{label}   base {b.shape[0]}x{b.shape[1]}, pool {p.shape[0]}\n{'#' * 108}")
        for L in LAYERS:
            p_h = hidden(model, layers[L], p, pm)
            gen = torch.Generator(device=DEVICE).manual_seed(SEED)
            rows, mags, cols, st, names = ladder(model, layers[L], b, bm, p_h, vectors[L], gen)
            print(f"\n--- layer {L} --- median norm {st['mn']:.1f}, "
                  f"top-{TOP_K} holds {st['explained']:.1%}, null {st['null']:.1%}")
            print("    vector energy in subspace: " + ", ".join(
                f"{k} {v:.1%}" for k, v in st["vec_energy"].items()))
            print(f"\n{'rank':>5}{'||d||':>8}{'/norm':>7}{'d_sub':>7}"
                  + "".join(f"{n:>17}" for n in names))
            print("-" * (27 + 17 * len(names)))
            for r, m, fr, ds, vals in rows:
                print(f"{r:>5}{m:>8.2f}{fr:>7.2f}{ds:>7.1%}"
                      + "".join(f"{vals[n][0]:>8.4f}+-{vals[n][1]:.4f}" for n in names), flush=True)
            line = {}
            for frac in REPORT_AT:
                g = at_frac(mags, cols["graft"], st["mn"], frac)
                rf = at_frac(mags, cols["refusal"], st["mn"], frac)
                line[frac] = (g, rf, g / rf if rf == rf and rf > 0 else float("nan"))
            print("  graft/refusal at " + ", ".join(
                f"{k}norms = " + (f"{v[2]:.2f}x" if v[2] == v[2] else "--")
                for k, v in line.items()))
            summary.append((label, L, st["vec_energy"]["refusal"], line))

    print(f"\n{'=' * 108}\nTHE SWEEP\n{'=' * 108}")
    print(f"{'arm':>26}{'layer':>7}{'refusal E_sub':>15}"
          + "".join(f"{'g/ref @' + str(k):>14}" for k in REPORT_AT))
    for label, L, e, line in summary:
        print(f"{label:>26}{L:>7}{e:>15.1%}"
              + "".join(f"{(f'{line[k][2]:.2f}x' if line[k][2] == line[k][2] else '--'):>14}"
                        for k in REPORT_AT))

    print("\nARTEFACT: refusal E_sub collapses toward the wikitext value as f falls, and")
    print("          graft/refusal climbs back toward the wikitext ratio.")
    print("REAL:     refusal stays loud at f = 0.05, where 6 of 128 prompts are refusable")
    print("          and refusal cannot be the principal component.")
    print("f = 0.00 separates the chat format from refusal being live at all.")


if __name__ == "__main__":
    main()
