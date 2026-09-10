r"""Give the cross-domain ladder room, and see whether the sign flip survives.

`steering_kind.py` found that the kind effect changes sign between depths — own/xdom reading
1.61x at layer 12 and 0.65x at layer 18 in wiki-chat — and concluded no single number describes
it. But it also recorded a design fault that could produce exactly that: three of twelve ladders
spanned under 2x in displacement, all of them `xdom`, and structurally so. Ranks 0 through 64 of
a **foreign** pool sit at nearly the same distance, because everything foreign is far and
roughly equidistant, so the cross-domain ladder barely moves and its exponent is fitted on
almost no range.

A sign flip read off two badly-determined fits is not a finding. This gives that family range.

THE GRADED FAMILY. Instead of a foreign pool, build donors that interpolate between the two
domains **in prompt space**:

    donor(alpha) = the pool passage truncated to (1 - alpha) of its tokens,
                   followed by the pool instruction when alpha > 0

    alpha = 0.00   a pure passage
    alpha = 0.25   three quarters of a passage, then an instruction
    alpha = 0.50   half a passage, then an instruction
    alpha = 0.75   a quarter of a passage, then an instruction
    alpha = 1.00   a pure instruction

Every one is an ordinary prompt someone could type, so each still lands in `Im(F)` and the graft
is unchanged. The family runs from passage-like to instruction-like continuously, which is the
range the foreign pool could not provide.

**The same donor set serves both arms, traversed in opposite directions.** For a passage base,
alpha = 0 is the near end and alpha = 1 the far end. For an instruction base it is the reverse.
That makes the cross-domain family construction-matched across arms for the first time, which
`self` never was.

`own` is carried unchanged as the reference kind and as an anchor: same pools and same seed as
run 11, so wiki-chat own must return 0.0696, 0.0864, 0.1339 and 1.4502 at layer 12.

WHAT IT DECIDES.

  If the graded family has real range and the kind effect still reverses between layers 12 and
  18, the reversal is a property of the model and run 11's conclusion stands on firmer ground.

  If the reversal disappears once the family can be fitted, it was an artefact of fitting
  exponents on a 1.3x span, and run 11 over-read its own noise.

No ablation and no jailbreak. Only AdvBench's `goal` column is read.

Run: python src/steering_graded.py
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
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OWN_RANKS = [0, 8, 64, 511]
REF_FRAC = 0.4
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


def fit(fr, va):
    pts = [(math.log(a), math.log(b)) for a, b in zip(fr, va) if a > 0 and b > 0]
    if len(pts) < 2:
        return float("nan"), float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0:
        return float("nan"), float("nan")
    p_ = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
    return p_, math.exp(my + p_ * (math.log(REF_FRAC) - mx))


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


def graded_texts(tok, passage_ids, instructions, alpha):
    """Passage truncated to (1-alpha) of its tokens, then the instruction when alpha > 0."""
    keep = int(round((1.0 - alpha) * SEQ_LEN))
    out = []
    for i, instr in enumerate(instructions):
        head = tok.decode(passage_ids[i, :keep]) if keep > 0 else ""
        out.append((head + " " + instr).strip() if alpha > 0 else head)
    return out


def run_arm(name, model, layer, L, ids, mask, kinds, vectors, gen):
    base_h = hidden(model, layer, ids, mask)
    mn = base_h.norm(dim=1).median().item()
    base_lp = logprobs(model, layer, ids, mask)
    print(f"\n{'=' * 92}\n{name}, layer {L}   median norm {mn:.1f}\n{'=' * 92}")
    out = {}
    for kname, steps in kinds.items():
        print(f"\n  {kname}")
        print(f"{'step':>10}{'||d||':>8}{'/norm':>7}{'graft':>11}{'refusal':>11}{'sentiment':>11}")
        fr, cols = [], {"graft": [], "refusal": [], "sentiment": []}
        for sname, d_h in steps:
            d = d_h - base_h
            mag = d.norm(dim=1)
            vals = {"graft": kl(base_lp, logprobs(model, layer, ids, mask, d))}
            for vn, v in vectors.items():
                vals[vn] = kl(base_lp, logprobs(model, layer, ids, mask, v[None] * mag[:, None]))
            m = mag.mean().item()
            fr.append(m / mn)
            for k in cols:
                cols[k].append(vals[k].mean().item())
            print(f"{sname:>10}{m:>8.2f}{m / mn:>7.2f}"
                  + "".join(f"{vals[k].mean().item():>11.4f}" for k in
                            ("graft", "refusal", "sentiment")), flush=True)
        pg, vg = fit(fr, cols["graft"])
        pr, vr = fit(fr, cols["refusal"])
        span = max(fr) / min(fr) if min(fr) > 0 else float("nan")
        print(f"    span {span:.2f}x   graft p={pg:.2f} KL@{REF_FRAC}={vg:.4f}   "
              f"refusal p={pr:.2f} KL@{REF_FRAC}={vr:.4f}   g/r={vg / vr:.2f}x")
        out[kname] = {"span": span, "pg": pg, "vg": vg, "pr": pr, "vr": vr, "gr": vg / vr}
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
    w_ids, w_texts = wiki_rows(tok, BASE_N + POOL_N)
    print(f"AdvBench {len(harmful)}, Alpaca {len(harmless)}, wikitext {w_ids.shape[0]}")

    wb_txt = w_texts[:BASE_N]
    wp_txt = w_texts[BASE_N:]
    wp_ids = w_ids[BASE_N:]
    ins_base = harmless[N_VEC:N_VEC + BASE_N]
    ins_pool = harmless[N_VEC + BASE_N:N_VEC + BASE_N + POOL_N]

    wc_b, wc_bm = chat(tok, wb_txt)
    wc_p, wc_pm = chat(tok, wp_txt)
    ic_b, ic_bm = chat(tok, ins_base)
    ic_p, ic_pm = chat(tok, ins_pool)

    graded = {}
    for a in ALPHAS:
        txt = graded_texts(tok, wp_ids[:BASE_N], ins_pool[:BASE_N], a)
        graded[a] = chat(tok, txt)
    print(f"graded family: {len(ALPHAS)} levels, {BASE_N} donors each\n")
    for a in (0.0, 0.5, 1.0):
        s = graded_texts(tok, wp_ids[:1], ins_pool[:1], a)[0]
        print(f"  alpha={a:.2f}: {s[:88]}{'...' if len(s) > 88 else ''}")

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

        wc_bh = hidden(model, lay, wc_b, wc_bm)
        wc_ph = hidden(model, lay, wc_p, wc_pm)
        ic_bh = hidden(model, lay, ic_b, ic_bm)
        ic_ph = hidden(model, lay, ic_p, ic_pm)
        g_h = {a: hidden(model, lay, *graded[a]) for a in ALPHAS}

        o_wc = torch.cdist(wc_bh, wc_ph).argsort(dim=1)
        o_ic = torch.cdist(ic_bh, ic_ph).argsort(dim=1)

        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        res[(L, "wiki-chat")] = run_arm(
            "wiki-chat", model, lay, L, wc_b, wc_bm,
            {"own": [(f"r{r}", wc_ph[o_wc[:, r]]) for r in OWN_RANKS],
             "graded": [(f"a{a:.2f}", g_h[a]) for a in ALPHAS]},
            vectors, gen)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        res[(L, "instr-chat")] = run_arm(
            "instr-chat", model, lay, L, ic_b, ic_bm,
            {"own": [(f"r{r}", ic_ph[o_ic[:, r]]) for r in OWN_RANKS],
             "graded": [(f"a{a:.2f}", g_h[a]) for a in reversed(ALPHAS)]},
            vectors, gen)

    print(f"\n{'#' * 92}\nDOES THE SIGN FLIP SURVIVE A FITTABLE CROSS-DOMAIN FAMILY?\n{'#' * 92}")
    print(f"{'layer':>6}{'arm':>13}{'own span':>10}{'grad span':>11}"
          f"{'own g/r':>10}{'grad g/r':>11}{'own/graded':>12}")
    for L in LAYERS:
        for a in ("wiki-chat", "instr-chat"):
            r = res[(L, a)]
            print(f"{L:>6}{a:>13}{r['own']['span']:>10.2f}{r['graded']['span']:>11.2f}"
                  f"{r['own']['gr']:>10.2f}{r['graded']['gr']:>11.2f}"
                  f"{r['own']['gr'] / r['graded']['gr']:>12.2f}")

    print("\nrun 11 gave own/xdom = 1.61x at layer 12 and 0.65x at layer 18 for wiki-chat,")
    print("on xdom spans of 2.00x and 1.83x. Compare the own/graded column above.")
    print("SURVIVES: the reversal is real and run 11's conclusion is safe.")
    print("VANISHES: run 11 fitted exponents on too little range and over-read its noise.")
    print("\nANCHOR: wiki-chat own must give 0.0696, 0.0864, 0.1339, 1.4502 at layer 12.")


if __name__ == "__main__":
    main()
