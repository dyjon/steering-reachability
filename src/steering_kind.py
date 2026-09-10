r"""Fit the graft curve with displacement kind as a second axis.

`steering_format.py` established two things that make every earlier one-dimensional fit unsafe.
Two thirds of what runs 7 to 9 called a domain effect was the chat template, and donor kind
changes graft KL by 1.4x to 2.3x at identical magnitude. The cleanest evidence for the second
was a matched pair in wiki-chat at layer 12: own-far at 5.39 and xdom-near at 5.38, the same
displacement to two decimals, graft 1.4502 against 1.0160, with refusal identical across the
pair as a fixed direction at fixed magnitude must be.

But that was one pair. Kind had one magnitude per level, so no exponent could be fitted for it
and the offset could not be separated from a difference in slope.

This gives every kind its own ladder.

    ARMS   (format held fixed, both chat-templated, since format is the larger term)
        wiki-chat    wikitext passages in a user turn
        instr-chat   Alpaca instructions in a user turn

    KINDS  (four magnitudes each)
        self   the same prompt behind a benign prefix, at four prefix lengths
        own    ranks 0, 8, 64, 511 of its own domain's pool
        xdom   ranks 0, 8, 64, 511 of the other domain's pool

Fitting log KL against log(‖d‖ / own median norm) separately per (arm, kind) gives an exponent
and an intercept for each of the six cells. Then:

    kind offset   compare cells within an arm at a common fraction
    arm offset    compare the same kind across arms

which is the two-way decomposition run 10 could only gesture at.

WHAT WOULD FALSIFY THE KIND EFFECT. If the six exponents agree and the intercepts agree, kind is
an artefact of where each ladder happened to sample and one dimension was enough all along. If
the intercepts separate while the exponents agree, kind is a clean multiplicative offset. If the
exponents themselves differ, kind is not an offset at all and the curves cross somewhere, which
would mean no single number describes it.

The `self` ladder uses four prefix lengths rather than a rank, so its magnitudes come from how
much text is prepended. It is the only kind whose construction differs between arms, since a
passage takes a reading preamble and an instruction takes a request opener.

No ablation and no jailbreak. Only AdvBench's `goal` column is read, and solely to build the
refusal vector.

Run: python src/steering_kind.py
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
RANKS = [0, 8, 64, 511]
REF_FRAC = 0.4
TOP_K = 32
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
INSTR_PRE = [
    "Please ",
    "When you have a moment, could you please ",
    "I would be very grateful if you could take the time to carefully ",
    "Before we begin I want to say that I appreciate your help with this, and when you are "
    "ready I would be very grateful if you could take the time to carefully and thoroughly ",
]
PASSAGE_PRE = [
    "Please read. ",
    "Here is a short passage I would like you to read. ",
    "I am going to give you a passage of text and I would like you to read it carefully. ",
    "Before we begin I want to say that I appreciate your help with this. I am going to give "
    "you a passage of text below and I would like you to read through it carefully and "
    "thoroughly before doing anything else. ",
]
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


def fit(fr, va):
    """OLS of log KL on log fraction. Returns (exponent, value predicted at REF_FRAC)."""
    pts = [(math.log(a), math.log(b)) for a, b in zip(fr, va) if a > 0 and b > 0]
    if len(pts) < 2:
        return float("nan"), float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0:
        return float("nan"), float("nan")
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
    intercept = my - slope * mx
    return slope, math.exp(intercept + slope * math.log(REF_FRAC))


def chat(tok, prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    return e["input_ids"], e["attention_mask"]


def load_harmful():
    try:
        d = load_dataset("walledai/AdvBench", split="train")
        return [r.strip() for r in d["prompt"]]
    except Exception:
        raw = urllib.request.urlopen(ADVBENCH_CSV, timeout=60).read().decode("utf-8")
        return [r["goal"].strip() for r in csv.DictReader(io.StringIO(raw)) if r.get("goal")]


def wiki_texts(tok, n):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    out = []
    for t in ds["text"]:
        t = t.strip()
        if len(t) <= 600:
            continue
        e = tok(t, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(e) == SEQ_LEN:
            out.append(tok.decode(e))
            if len(out) == n:
                break
    return out


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
    wt = wiki_texts(tok, BASE_N + POOL_N)
    print(f"AdvBench {len(harmful)}, Alpaca {len(harmless)}, wikitext {len(wt)}")

    arms = {}
    wb, wp = wt[:BASE_N], wt[BASE_N:]
    ib_, ip_ = harmless[N_VEC:N_VEC + BASE_N], harmless[N_VEC + BASE_N:N_VEC + BASE_N + POOL_N]
    arms["wiki-chat"] = (chat(tok, wb), chat(tok, wp),
                         [chat(tok, [p + t for t in wb]) for p in PASSAGE_PRE])
    arms["instr-chat"] = (chat(tok, ib_), chat(tok, ip_),
                          [chat(tok, [p + t[0].lower() + t[1:] for t in ib_]) for p in INSTR_PRE])
    print(f"wiki-chat base {BASE_N} pool {len(wp)}, instr-chat base {len(ib_)} pool {len(ip_)}\n")

    vh = chat(tok, harmful[:N_VEC])
    vs = chat(tok, harmless[:N_VEC])
    pi = chat(tok, POSITIVE)
    ni = chat(tok, NEGATIVE)

    results = {}
    for L in LAYERS:
        lay = layers[L]
        ref = hidden(model, lay, *vh).mean(0) - hidden(model, lay, *vs).mean(0)
        sen = hidden(model, lay, *pi).mean(0) - hidden(model, lay, *ni).mean(0)
        vectors = {"refusal": ref / ref.norm(), "sentiment": sen / sen.norm()}

        base_h = {a: hidden(model, lay, *arms[a][0]) for a in arms}
        pool_h = {a: hidden(model, lay, *arms[a][1]) for a in arms}
        self_h = {a: [hidden(model, lay, *e) for e in arms[a][2]] for a in arms}

        for a in arms:
            ids, mask = arms[a][0]
            bh = base_h[a]
            mn = bh.norm(dim=1).median().item()
            hc = bh - bh.mean(0)
            _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
            basis = Vh[:TOP_K].T.contiguous()
            base_lp = logprobs(model, lay, ids, mask)
            other = "instr-chat" if a == "wiki-chat" else "wiki-chat"
            o_own = torch.cdist(bh, pool_h[a]).argsort(dim=1)
            o_x = torch.cdist(bh, pool_h[other]).argsort(dim=1)

            ladders = {
                "self": [(f"pre{i}", h) for i, h in enumerate(self_h[a])],
                "own": [(f"r{r}", pool_h[a][o_own[:, r]]) for r in RANKS],
                "xdom": [(f"r{r}", pool_h[other][o_x[:, r]]) for r in RANKS],
            }
            print(f"\n{'=' * 96}\n{a}, layer {L}   median norm {mn:.1f}\n{'=' * 96}")
            gen = torch.Generator(device=DEVICE).manual_seed(SEED)
            for kind, steps in ladders.items():
                print(f"\n  {kind}")
                print(f"    {'step':>6}{'||d||':>8}{'/norm':>7}{'graft':>11}"
                      f"{'refusal':>11}{'sentiment':>11}")
                fr, gs, rs, ss = [], [], [], []
                for sname, dh in steps:
                    dd = dh - bh
                    mag = dd.norm(dim=1)
                    g = kl(base_lp, logprobs(model, lay, ids, mask, dd)).mean().item()
                    r_ = kl(base_lp, logprobs(model, lay, ids, mask,
                                              vectors["refusal"][None] * mag[:, None])).mean().item()
                    s_ = kl(base_lp, logprobs(model, lay, ids, mask,
                                              vectors["sentiment"][None] * mag[:, None])).mean().item()
                    m = mag.mean().item()
                    fr.append(m / mn); gs.append(g); rs.append(r_); ss.append(s_)
                    print(f"    {sname:>6}{m:>8.2f}{m / mn:>7.2f}{g:>11.4f}{r_:>11.4f}{s_:>11.4f}",
                          flush=True)
                pg, vg = fit(fr, gs)
                pr, vr = fit(fr, rs)
                ok = vr == vr and vr > 0 and vg == vg
                ratio = f"{vg / vr:.2f}x" if ok else "--"
                print(f"    fit: graft p={pg:.2f} KL@{REF_FRAC}={vg:.4f}   "
                      f"refusal p={pr:.2f} KL@{REF_FRAC}={vr:.4f}   "
                      f"graft/refusal@{REF_FRAC}={ratio}")
                results[(L, a, kind)] = (pg, vg, pr, vr)

    print(f"\n{'#' * 96}\nTWO-WAY TABLE  (exponent, and graft/refusal at {REF_FRAC} norms)\n{'#' * 96}")
    for L in LAYERS:
        print(f"\nlayer {L}")
        print(f"{'':>12}" + "".join(f"{k:>22}" for k in ("self", "own", "xdom")))
        for a in ("wiki-chat", "instr-chat"):
            row = f"{a:>12}"
            for kind in ("self", "own", "xdom"):
                pg, vg, pr, vr = results[(L, a, kind)]
                row += f"{f'p={pg:.2f}  {vg / vr:.2f}x':>22}" if vr > 0 else f"{'--':>22}"
            print(row)
        print("  kind offset within an arm = ratio of the graft KL@ref across kinds:")
        for a in ("wiki-chat", "instr-chat"):
            vs_ = {k: results[(L, a, k)][1] for k in ("self", "own", "xdom")}
            print(f"    {a:>12}  own/xdom = {vs_['own'] / vs_['xdom']:.2f}x, "
                  f"own/self = {vs_['own'] / vs_['self']:.2f}x")
        print("  arm offset within a kind = ratio of graft/refusal@ref across arms:")
        for kind in ("self", "own", "xdom"):
            w = results[(L, "wiki-chat", kind)]
            i = results[(L, "instr-chat", kind)]
            print(f"    {kind:>12}  {(w[1] / w[3]) / (i[1] / i[3]):.2f}x")

    print("\nEXPONENTS AGREE, INTERCEPTS SEPARATE -> kind is a clean multiplicative offset.")
    print("EXPONENTS DIFFER -> kind is not an offset, the curves cross, no single number.")
    print("BOTH AGREE -> kind was an artefact of sampling and one dimension sufficed.")


if __name__ == "__main__":
    main()
