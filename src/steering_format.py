r"""Was it the domain, or was it the chat template?

`steering_overlap.py` measured a constant ~3.9x graft/refusal offset between wikitext and
instructions at layer 18, refusal-specific against sentiment's ~1.5x. But its two arms differed
in two ways at once: the wikitext arm was **raw text** and the instruction arm was
**chat-templated**. So "domain" confounded the content with the format, and the single
templated-wikitext condition in that run, `passage`, was precisely the point that sat off the
curve.

That is a large enough crack to check before anything is sent.

THE FIX IS TO TEMPLATE EVERYTHING. Three base arms:

    wiki-raw     96-token passages, no template          the anchor back to earlier runs
    wiki-chat    the same passages inside a user turn    same format as instructions
    instr-chat   Alpaca instructions inside a user turn  as before

`wiki-chat` against `wiki-raw` isolates the **format**: same content, template or not.
`wiki-chat` against `instr-chat` isolates the **content**: same format, passage or instruction.

If wiki-chat lands near instr-chat's ~0.9, the effect was the template all along and the domain
framing in runs 7 to 9 is wrong. If it stays near wiki-raw's ~3.5, the effect is genuinely about
content and the ~3.9x stands.

DISPLACEMENT KIND IS THE SECOND VARIABLE, which run 9 showed is needed. Each templated arm gets
the same five donor kinds, applied identically in both directions:

    self      the same prompt behind a benign prefix
    own-near  nearest in its own domain's pool
    own-far   farthest in its own domain's pool
    xdom-near nearest in the *other* domain's pool
    xdom-far  farthest in the *other* domain's pool

If cross-domain donors sit off the curve in both directions, kind matters independently of
magnitude and every ladder so far has been read one dimension short.

ANCHOR. wiki-raw `swap8` graft must return 0.0086 at layer 12, as in `4ccdfbc` and `a733a21`.

No ablation and no jailbreak. Only AdvBench's `goal` column is read.

Run: python src/steering_format.py
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
REPORT_AT = [0.2, 0.3, 0.4, 0.5, 0.6]
TOP_K = 32
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
INSTR_PREFIXES = ["Please ", "Could you ", "I would like you to ", "Kindly "]
PASSAGE_PREFIXES = ["Please read the following. ", "Consider this text. ",
                    "Here is a passage. ", "Take a look at this. "]
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
    o = sorted(range(len(fracs)), key=lambda i: fracs[i])
    fs = [fracs[i] for i in o]
    vs = [vals[i] for i in o]
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
    """AdvBench goals only; the `target` column of affirmative prefixes is never read."""
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


def arm(name, model, layer, layer_no, ids, mask, donors, vectors, gen):
    base_h = hidden(model, layer, ids, mask)
    mn = base_h.norm(dim=1).median().item()
    hc = base_h - base_h.mean(0)
    _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
    basis = Vh[:TOP_K].T.contiguous()
    var = S ** 2
    base_lp = logprobs(model, layer, ids, mask)
    print(f"\n--- {name}, layer {layer_no} --- median norm {mn:.1f}, "
          f"top-{TOP_K} holds {(var[:TOP_K].sum() / var.sum()).item():.1%}")
    print("    vector energy: " + ", ".join(
        f"{k} {energy_in(basis, v[None]).item():.1%}" for k, v in vectors.items()))

    names = ["graft"] + list(vectors.keys()) + ["subspace"]
    print(f"\n{'kind':>10}{'||d||':>8}{'/norm':>7}" + "".join(f"{n:>17}" for n in names))
    print("-" * (25 + 17 * len(names)))
    fr, cols = [], {n: [] for n in names}
    for kname, d_h in donors:
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
        fr.append(m / mn)
        for n in names:
            cols[n].append(vals[n].mean().item())
        print(f"{kname:>10}{m:>8.2f}{m / mn:>7.2f}"
              + "".join(f"{vals[n].mean().item():>8.4f}"
                        f"+-{(vals[n].std(unbiased=True) / (vals[n].shape[0] ** 0.5)).item():.4f}"
                        for n in names), flush=True)
    return fr, cols


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
    print(f"AdvBench {len(harmful)}, Alpaca {len(harmless)}")

    w_ids, w_texts = wiki_rows(tok, BASE_N + POOL_N)
    wb_raw, wp_raw = w_ids[:BASE_N], w_ids[BASE_N:]
    wb_txt, wp_txt = w_texts[:BASE_N], w_texts[BASE_N:]
    ins_base = harmless[N_VEC:N_VEC + BASE_N]
    ins_pool = harmless[N_VEC + BASE_N:N_VEC + BASE_N + POOL_N]

    wc_b, wc_bm = chat(tok, wb_txt)
    wc_p, wc_pm = chat(tok, wp_txt)
    ic_b, ic_bm = chat(tok, ins_base)
    ic_p, ic_pm = chat(tok, ins_pool)
    wc_pre, wc_prem = chat(tok, [PASSAGE_PREFIXES[i % 4] + t for i, t in enumerate(wb_txt)])
    ic_pre, ic_prem = chat(tok, [INSTR_PREFIXES[i % 4] + t[0].lower() + t[1:]
                                 for i, t in enumerate(ins_base)])
    print(f"wiki-raw {BASE_N}/{wp_raw.shape[0]}, wiki-chat same texts, "
          f"instr-chat {len(ins_base)}/{len(ins_pool)}\n")

    vh, vhm = chat(tok, harmful[:N_VEC])
    vs, vsm = chat(tok, harmless[:N_VEC])
    pi, pim = chat(tok, POSITIVE)
    ni, nim = chat(tok, NEGATIVE)

    res = {}
    for L in LAYERS:
        lay = layers[L]
        ref = hidden(model, lay, vh, vhm).mean(0) - hidden(model, lay, vs, vsm).mean(0)
        sen = hidden(model, lay, pi, pim).mean(0) - hidden(model, lay, ni, nim).mean(0)
        vectors = {"refusal": ref / ref.norm(), "sentiment": sen / sen.norm()}

        wr_bh = hidden(model, lay, wb_raw, torch.ones_like(wb_raw))
        wr_ph = hidden(model, lay, wp_raw, torch.ones_like(wp_raw))
        wc_bh = hidden(model, lay, wc_b, wc_bm)
        wc_ph = hidden(model, lay, wc_p, wc_pm)
        ic_bh = hidden(model, lay, ic_b, ic_bm)
        ic_ph = hidden(model, lay, ic_p, ic_pm)
        wc_preh = hidden(model, lay, wc_pre, wc_prem)
        ic_preh = hidden(model, lay, ic_pre, ic_prem)

        sw = torch.cat([wp_raw[:BASE_N, :8], wb_raw[:, 8:]], dim=1)
        sw_h = hidden(model, lay, sw, torch.ones_like(sw))
        o_raw = torch.cdist(wr_bh, wr_ph).argsort(dim=1)
        o_wc = torch.cdist(wc_bh, wc_ph).argsort(dim=1)
        o_ic = torch.cdist(ic_bh, ic_ph).argsort(dim=1)
        o_wx = torch.cdist(wc_bh, ic_ph).argsort(dim=1)     # wiki-chat -> instruction pool
        o_ix = torch.cdist(ic_bh, wc_ph).argsort(dim=1)     # instr-chat -> passage pool

        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        fr_r, c_r = arm("wiki-raw", model, lay, L, wb_raw, torch.ones_like(wb_raw),
                        [("swap8", sw_h), ("own-near", wr_ph[o_raw[:, 0]]),
                         ("own-far", wr_ph[o_raw[:, -1]])], vectors, gen)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        fr_w, c_w = arm("wiki-chat", model, lay, L, wc_b, wc_bm,
                        [("self", wc_preh), ("own-near", wc_ph[o_wc[:, 0]]),
                         ("own-far", wc_ph[o_wc[:, -1]]), ("xdom-near", ic_ph[o_wx[:, 0]]),
                         ("xdom-far", ic_ph[o_wx[:, -1]])], vectors, gen)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        fr_i, c_i = arm("instr-chat", model, lay, L, ic_b, ic_bm,
                        [("self", ic_preh), ("own-near", ic_ph[o_ic[:, 0]]),
                         ("own-far", ic_ph[o_ic[:, -1]]), ("xdom-near", wc_ph[o_ix[:, 0]]),
                         ("xdom-far", wc_ph[o_ix[:, -1]])], vectors, gen)
        res[L] = {"wiki-raw": (fr_r, c_r), "wiki-chat": (fr_w, c_w), "instr-chat": (fr_i, c_i)}

    print(f"\n{'=' * 92}\nFORMAT OR CONTENT?  graft/refusal at matched fraction of own norm\n{'=' * 92}")
    for L in LAYERS:
        print(f"\nlayer {L}")
        print(f"{'frac':>6}{'wiki-raw':>11}{'wiki-chat':>12}{'instr-chat':>13}"
              f"{'  format effect':>16}{'  content effect':>17}")
        for t in REPORT_AT:
            vals = {}
            for k in ("wiki-raw", "wiki-chat", "instr-chat"):
                fr, c = res[L][k]
                g = at_frac(fr, c["graft"], t)
                r = at_frac(fr, c["refusal"], t)
                vals[k] = g / r if r == r and r > 0 and g == g else float("nan")
            fmt = lambda x: f"{x:.2f}" if x == x else "--"
            a, b, c_ = vals["wiki-raw"], vals["wiki-chat"], vals["instr-chat"]
            fe = a / b if a == a and b == b and b > 0 else float("nan")
            ce = b / c_ if b == b and c_ == c_ and c_ > 0 else float("nan")
            print(f"{t:>6.1f}{fmt(a):>11}{fmt(b):>12}{fmt(c_):>13}{fmt(fe):>16}{fmt(ce):>17}")

    print("\nANCHOR: wiki-raw swap8 graft must read 0.0086 at layer 12.")
    print("FORMAT: wiki-chat near instr-chat means the template did it, and the domain")
    print("        framing of runs 7 to 9 is wrong.")
    print("CONTENT: wiki-chat near wiki-raw means content did it and the ~3.9x stands.")
    print("KIND: xdom donors sitting off their arm's curve means magnitude alone is")
    print("      not enough to place a displacement, in either direction.")


if __name__ == "__main__":
    main()
