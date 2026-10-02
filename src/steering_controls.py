r"""Run 14: the three controls the 1 October audit asked for.

The audit (README, "Reopened by audit") found that two of the four standing claims did not hold
up against the code, and that the strongest remaining one had no error bars. This run addresses
all three, each in its own part, so each can fail independently.

PART 1 — ERROR BARS ON THE DISTANCE FINDING.
Run 13 selected donors already sitting at a target distance and found the graft changing by
about 2x with the kind of destination: cross/own 0.51, 1.27, 0.58. Runs 12 and 13 computed no
standard error anywhere. This repeats the own-versus-cross selection and reports every ratio
with a 95% bootstrap interval over prompts, computed on the prompts where *both* selections
matched, so each ratio is paired.

PART 2 — THE PROVENANCE CONTROL FOR THE TEMPLATING FLIP.
The flip says: at layer 12, refusal outweighs a reachable step in chat-templated prompts and the
reverse holds in raw text. But every refusal vector so far was built from chat-templated
prompts, so on raw text it was a direction from one distribution applied to another.

Here there are two refusal vectors, built from the same 96 AdvBench/Alpaca pairs:

    refusal_tmpl   built from chat-templated prompts   (as in every earlier run)
    refusal_raw    built from the same prompts, untemplated

and the donors are built **identically** in both arms — the same 96-token passages with their
first k tokens swapped, used raw in one arm and wrapped in a chat turn in the other. So the only
thing differing between the arms is the template, and the vector's provenance is varied
independently of it.

    flip survives with each vector in its own domain   ->  a property of the template
    flip disappears once provenance is matched          ->  it was provenance all along

PART 3 — EXPONENTS ALONG SINGLE RAYS.
Earlier ladders compared steering, one direction scaled, against grafts that took a different
donor at every rung. Here each graft difference d is scaled along its own ray, alpha * d for
alpha in {0.25, 0.5, 0.75, 1}, and refusal is scaled by the same factors. Points with alpha < 1
are off Im(F); that is fine, because this measures curvature along a ray, not reachability.

    graft ray exponent ~ 2   ->  "reachable steps leave the quadratic regime" was the direction
                                 change, not curvature
    graft ray exponent > 2   ->  graft directions really are more curved than steering

ANCHOR. Part 2's raw arm uses exactly the passages, swaps and layer of `steering_graft_wide.py`,
so its graft column at layer 12 must read 0.0086, 0.0345, 0.1343, 0.2561.

No ablation and no jailbreak. Only AdvBench's `goal` column is read.

Run: python src/steering_controls.py
"""
import csv
import io
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
SWAPS = [8, 24, 48, 72]
TARGETS = [0.4, 0.5, 0.7]
ALPHAS = [0.25, 0.5, 0.75, 1.0]
TOL = 0.15
N_BOOT = 2000
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")


def unpack(o):
    return o[0] if isinstance(o, tuple) else o


def repack(o, h):
    return (h,) + o[1:] if isinstance(o, tuple) else h


def add_at_last(v):
    def hook(m, a, o):
        h = unpack(o).clone()
        h[:, -1, :] += v
        return repack(o, h)
    return hook


def _mask(mask, i):
    return None if mask is None else mask[i:i + BATCH]


def logprobs(model, layer, ids, mask, delta=None):
    out = []
    for i in range(0, ids.shape[0], BATCH):
        hd = layer.register_forward_hook(add_at_last(delta[i:i + BATCH])) if delta is not None else None
        try:
            lg = model(input_ids=ids[i:i + BATCH], attention_mask=_mask(mask, i)).logits[:, -1, :]
        finally:
            if hd is not None:
                hd.remove()
        out.append(F.log_softmax(lg.float(), dim=-1))
    return torch.cat(out, 0)


def hidden(model, layer, ids, mask):
    got = []
    h = layer.register_forward_hook(lambda m, a, o: got.append(unpack(o)[:, -1, :]))
    try:
        for i in range(0, ids.shape[0], BATCH):
            model(input_ids=ids[i:i + BATCH], attention_mask=_mask(mask, i))
    finally:
        h.remove()
    return torch.cat(got, 0)


def kl(p, q):
    return (p.exp() * (p - q)).sum(-1)


def unit(v):
    return v / v.norm()


def chat(tok, prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True) for p in prompts]
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    return e["input_ids"], e["attention_mask"]


def raw_enc(tok, prompts):
    e = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    return e["input_ids"], e["attention_mask"]


def ms(v):
    return f"{v.mean().item():.4f}+-{(v.std(unbiased=True) / v.shape[0] ** 0.5).item():.4f}"


def boot_ratio(a, b, gen):
    """mean(a)/mean(b), paired over prompts, with a 95% bootstrap interval."""
    n = a.shape[0]
    idx = torch.randint(0, n, (N_BOOT, n), generator=gen)
    r = a[idx].mean(1) / b[idx].mean(1)
    lo, hi = torch.quantile(r, torch.tensor([0.025, 0.975]))
    return (a.mean() / b.mean()).item(), lo.item(), hi.item()


def boot_slope(Y, gen):
    """Slope of log mean-KL against log alpha. Y is (len(ALPHAS), n_prompts)."""
    lx = torch.log(torch.tensor(ALPHAS))
    lx = lx - lx.mean()

    def slope(m):
        ly = torch.log(m)
        return ((lx * (ly - ly.mean())).sum() / (lx ** 2).sum()).item()

    n = Y.shape[1]
    idx = torch.randint(0, n, (N_BOOT, n), generator=gen)
    s = torch.tensor([slope(Y[:, i].mean(1)) for i in idx])
    lo, hi = torch.quantile(s, torch.tensor([0.025, 0.975]))
    return slope(Y.mean(1)), lo.item(), hi.item()


def fmt_ci(x):
    return f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"


def load_harmful():
    try:
        return [r.strip() for r in load_dataset("walledai/AdvBench", split="train")["prompt"]]
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
    gen = torch.Generator().manual_seed(SEED)

    harmful = load_harmful()
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"].strip() for r in alp
                if not r["input"].strip() and 20 < len(r["instruction"]) < 160]
    w_ids, w_texts = wiki_rows(tok, BASE_N + POOL_N)
    wb_ids, wd_ids = w_ids[:BASE_N], w_ids[BASE_N:2 * BASE_N]   # base and swap donors, as runs 3-5
    print(f"AdvBench {len(harmful)}, Alpaca {len(harmless)}, wikitext {len(w_texts)}")

    swap_raw = {k: torch.cat([wd_ids[:, :k], wb_ids[:, k:]], dim=1) for k in SWAPS}
    swap_chat = {k: chat(tok, [tok.decode(r) for r in swap_raw[k]]) for k in SWAPS}
    wc_b = chat(tok, w_texts[:BASE_N])
    p_pool = chat(tok, w_texts[BASE_N:])
    ins_base = harmless[N_VEC:N_VEC + BASE_N]
    i_pool = chat(tok, harmless[N_VEC + BASE_N:N_VEC + BASE_N + POOL_N])
    ib = chat(tok, ins_base)

    vh_t, vs_t = chat(tok, harmful[:N_VEC]), chat(tok, harmless[:N_VEC])
    vh_r, vs_r = raw_enc(tok, harmful[:N_VEC]), raw_enc(tok, harmless[:N_VEC])

    summary = {}
    for L in LAYERS:
        lay = layers[L]
        r_t = unit(hidden(model, lay, *vh_t).mean(0) - hidden(model, lay, *vs_t).mean(0))
        r_r = unit(hidden(model, lay, *vh_r).mean(0) - hidden(model, lay, *vs_r).mean(0))
        print(f"\n{'#' * 96}\nLAYER {L}   cos(refusal_tmpl, refusal_raw) = {torch.dot(r_t, r_r).item():+.3f}\n{'#' * 96}")

        # ---------------- parts 2 and 3: identical donors, template on or off ----------------
        arms = (("raw", wb_ids, None, {k: (swap_raw[k], None) for k in SWAPS}),
                ("chat", wc_b[0], wc_b[1], swap_chat))
        for arm, b_ids, b_mask, variants in arms:
            base_h = hidden(model, lay, b_ids, b_mask)
            mn = base_h.norm(dim=1).median().item()
            base_lp = logprobs(model, lay, b_ids, b_mask)
            print(f"\n--- PART 2, {arm} arm, layer {L}, median norm {mn:.1f} ---")
            print(f"{'k':>4}{'/norm':>7}{'graft':>17}{'refusal_tmpl':>17}{'refusal_raw':>17}"
                  f"{'g/r_tmpl [95% CI]':>24}{'g/r_raw [95% CI]':>24}")
            ray = None
            for k in SWAPS:
                v_ids, v_mask = variants[k]
                d = hidden(model, lay, v_ids, v_mask) - base_h
                mag = d.norm(dim=1)
                g = kl(base_lp, logprobs(model, lay, b_ids, b_mask, d)).cpu()
                kt = kl(base_lp, logprobs(model, lay, b_ids, b_mask, r_t[None] * mag[:, None])).cpu()
                kr = kl(base_lp, logprobs(model, lay, b_ids, b_mask, r_r[None] * mag[:, None])).cpu()
                ct, cr = boot_ratio(g, kt, gen), boot_ratio(g, kr, gen)
                summary[(L, arm, k)] = (ct, cr)
                print(f"{k:>4}{mag.mean().item() / mn:>7.2f}{ms(g):>17}{ms(kt):>17}{ms(kr):>17}"
                      f"{fmt_ci(ct):>24}{fmt_ci(cr):>24}", flush=True)
                if k == SWAPS[-1]:
                    ray = (d, mag)

            d, mag = ray
            Yg, Yt, Yr = [], [], []
            for a in ALPHAS:
                Yg.append(kl(base_lp, logprobs(model, lay, b_ids, b_mask, a * d)).cpu())
                Yt.append(kl(base_lp, logprobs(model, lay, b_ids, b_mask, a * r_t[None] * mag[:, None])).cpu())
                Yr.append(kl(base_lp, logprobs(model, lay, b_ids, b_mask, a * r_r[None] * mag[:, None])).cpu())
            eg, et, er = (boot_slope(torch.stack(Y), gen) for Y in (Yg, Yt, Yr))
            summary[(L, arm, "ray")] = (eg, et, er)
            print(f"  PART 3, ray exponent along the k={SWAPS[-1]} direction:  "
                  f"graft {fmt_ci(eg)}   refusal_tmpl {fmt_ci(et)}   refusal_raw {fmt_ci(er)}")

        # ---------------- part 1: own vs cross at matched distance, with intervals ----------------
        p_h, i_h = hidden(model, lay, *p_pool), hidden(model, lay, *i_pool)
        for arm, (ids, mask), own_h, cross_h in (("passage-base", wc_b, p_h, i_h),
                                                 ("instruction-base", ib, i_h, p_h)):
            base_h = hidden(model, lay, ids, mask)
            mn = base_h.norm(dim=1).median().item()
            base_lp = logprobs(model, lay, ids, mask)
            print(f"\n--- PART 1, {arm}, layer {L}, median norm {mn:.1f} ---")
            print(f"{'target':>7}{'paired':>8}{'own':>17}{'cross':>17}{'cross/own [95% CI]':>26}")
            for t in TARGETS:
                want = t * mn
                sel = {}
                for name, ph in (("own", own_h), ("cross", cross_h)):
                    j = (torch.cdist(base_h, ph) - want).abs().argmin(dim=1)
                    dd = ph[j] - base_h
                    sel[name] = (dd, ((dd.norm(dim=1) - want).abs() / want) < TOL)
                both = (sel["own"][1] & sel["cross"][1]).cpu()
                if both.float().mean() < 0.3:
                    print(f"{t:>7.1f}{both.float().mean().item():>7.0%}   too few paired prompts, skipped")
                    continue
                go = kl(base_lp, logprobs(model, lay, ids, mask, sel["own"][0])).cpu()[both]
                gc = kl(base_lp, logprobs(model, lay, ids, mask, sel["cross"][0])).cpu()[both]
                c = boot_ratio(gc, go, gen)
                summary[(L, arm, t)] = c
                print(f"{t:>7.1f}{both.float().mean().item():>7.0%}{ms(go):>17}{ms(gc):>17}{fmt_ci(c):>26}",
                      flush=True)

    print(f"\n{'=' * 96}\nREADING IT\n{'=' * 96}")
    print("ANCHOR: PART 2 raw arm, layer 12, graft must read 0.0086, 0.0345, 0.1343, 0.2561.")
    print("PART 1: a cross/own interval that excludes 1 is a statable kind effect at that cell.")
    print("PART 2: compare raw-arm g/r_raw with chat-arm g/r_tmpl, each vector in its own domain.")
    print("        Opposite sides of 1 -> the flip is the template. Same side -> it was provenance.")
    print("PART 3: graft ray exponent near 2 -> earlier 'leaves the quadratic regime' was the")
    print("        direction changing between rungs. Clearly above 2 -> genuine curvature.")


if __name__ == "__main__":
    main()
