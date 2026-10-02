r"""Run 15: does the Fisher quadratic form explain every direction effect?

Run 14 ended on a synthesis. Along any single direction, next-token KL grows as distance squared;
yet at matched distance, the kind of destination changes the effect by up to 1.8x, and the
format a refusal vector was built from changes it by up to 3.9x. Every apparent effect of
distance in fourteen runs was an effect of direction.

That is exactly what a local quadratic form predicts:

    KL(p_h || p_{h+d})  ~  1/2 d^T F d

with F the Fisher information of the next-token distribution with respect to the activation,
and F strongly anisotropic. Run 14 only said the data were *consistent* with this. This run
tests it.

COMPUTING d^T F d WITHOUT F. For a categorical output with probabilities p and logits z(h),
F = J^T (diag(p) - p p^T) J where J = dz/dh. So along a direction d, with u = J d the change in
logits per unit step,

    d^T F d  =  sum_y p_y u_y^2 - (sum_y p_y u_y)^2  =  Var_{y ~ p}(u_y).

u is a directional derivative, taken here by central difference: two hooked forward passes at
+t*d and -t*d with t = 0.01. No Fisher matrix is formed and nothing is differentiated. The
prediction is one half of that variance, per prompt.

CALIBRATION, BEFORE ANYTHING ELSE. At a tiny displacement the quadratic approximation must be
nearly exact, so a graft scaled to 5% of its length has to return measured / predicted ~ 1. If
it does not, the Fisher computation is wrong and no other row can be read.

THE TEST. The overall scale is the weaker evidence; a quadratic form fits any single curve. What
matters is whether the prediction reproduces the *ratios* run 14 measured:

    cross / own              at matched distance       (run 14 part 1)
    refusal_tmpl / refusal_raw at the same magnitude   (run 14 part 2, provenance)
    graft / refusal_tmpl     the old headline ratio

    predicted ratios match measured   ->  direction effects are Fisher anisotropy; the
                                         synthesis holds and fourteen runs reduce to one form
    predicted ratios ~ 1, measured not ->  the effects are not local; the synthesis is wrong
    measured/predicted drifts from 1 ->  the quadratic regime breaks at these distances,
                                         whatever the ratios do

ANCHOR. The raw arm reuses run 14's passages, swaps and layer, so its measured graft at layer 12
must read 0.0086, 0.0345, 0.1343, 0.2561.

No ablation and no jailbreak. Only AdvBench's `goal` column is read.

Run: python src/steering_fisher.py
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
T_FD = 0.01            # central-difference step, as a fraction of d
CAL_ALPHA = 0.05       # calibration: graft scaled to 5% of its length
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


def batch_logits(model, layer, ids, mask, vec):
    """Final-position logits for one batch, with vec added at the last position (or none)."""
    hd = layer.register_forward_hook(add_at_last(vec)) if vec is not None else None
    try:
        return model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
    finally:
        if hd is not None:
            hd.remove()


def base_logprobs(model, layer, ids, mask):
    return torch.cat([F.log_softmax(batch_logits(model, layer, ids[i:i + BATCH], _mask(mask, i), None), -1)
                      for i in range(0, ids.shape[0], BATCH)])


def measure_and_predict(model, layer, ids, mask, base_lp, d):
    """Per prompt: measured KL(base || base+d), and predicted 1/2 d^T F d."""
    meas, pred = [], []
    for i in range(0, ids.shape[0], BATCH):
        sl, m = slice(i, i + BATCH), _mask(mask, i)
        blp = base_lp[sl]
        lp = F.log_softmax(batch_logits(model, layer, ids[sl], m, d[sl]), -1)
        meas.append((blp.exp() * (blp - lp)).sum(-1))
        zp = batch_logits(model, layer, ids[sl], m, T_FD * d[sl])
        zm = batch_logits(model, layer, ids[sl], m, -T_FD * d[sl])
        u = (zp - zm) / (2 * T_FD)
        p = blp.exp()
        mu = (p * u).sum(-1)
        pred.append(0.5 * ((p * u * u).sum(-1) - mu ** 2))
    return torch.cat(meas).cpu(), torch.cat(pred).cpu()


def hidden(model, layer, ids, mask):
    got = []
    h = layer.register_forward_hook(lambda m, a, o: got.append(unpack(o)[:, -1, :]))
    try:
        for i in range(0, ids.shape[0], BATCH):
            model(input_ids=ids[i:i + BATCH], attention_mask=_mask(mask, i))
    finally:
        h.remove()
    return torch.cat(got, 0)


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


def boot_ratio(a, b, gen):
    n = a.shape[0]
    idx = torch.randint(0, n, (N_BOOT, n), generator=gen)
    r = a[idx].mean(1) / b[idx].mean(1)
    lo, hi = torch.quantile(r, torch.tensor([0.025, 0.975]))
    return (a.mean() / b.mean()).item(), lo.item(), hi.item()


def ci(x):
    return f"{x[0]:.2f} [{x[1]:.2f},{x[2]:.2f}]"


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


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    return ((x * y).sum() / (x.norm() * y.norm())).item()


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
    wb_ids, wd_ids = w_ids[:BASE_N], w_ids[BASE_N:2 * BASE_N]

    swap_raw = {k: torch.cat([wd_ids[:, :k], wb_ids[:, k:]], dim=1) for k in SWAPS}
    swap_chat = {k: chat(tok, [tok.decode(r) for r in swap_raw[k]]) for k in SWAPS}
    wc_b = chat(tok, w_texts[:BASE_N])
    p_pool = chat(tok, w_texts[BASE_N:])
    i_pool = chat(tok, harmless[N_VEC + BASE_N:N_VEC + BASE_N + POOL_N])
    ib = chat(tok, harmless[N_VEC:N_VEC + BASE_N])
    vh_t, vs_t = chat(tok, harmful[:N_VEC]), chat(tok, harmless[:N_VEC])
    vh_r, vs_r = raw_enc(tok, harmful[:N_VEC]), raw_enc(tok, harmless[:N_VEC])

    all_meas, all_pred, comparisons = [], [], []

    def record(label, m, p):
        all_meas.append(m.mean().item())
        all_pred.append(p.mean().item())
        r = boot_ratio(m, p, gen)
        print(f"{label:>34}{m.mean().item():>10.4f}{p.mean().item():>10.4f}{ci(r):>22}", flush=True)

    def compare(label, ma, pa, mb, pb):
        mr, pr = boot_ratio(ma, mb, gen), boot_ratio(pa, pb, gen)
        agree = not (mr[2] < pr[1] or pr[2] < mr[1])
        comparisons.append((label, mr, pr, agree))

    hdr = f"{'condition':>34}{'measured':>10}{'pred':>10}{'meas/pred [95% CI]':>22}"
    for L in LAYERS:
        lay = layers[L]
        r_t = unit(hidden(model, lay, *vh_t).mean(0) - hidden(model, lay, *vs_t).mean(0))
        r_r = unit(hidden(model, lay, *vh_r).mean(0) - hidden(model, lay, *vs_r).mean(0))
        print(f"\n{'#' * 76}\nLAYER {L}\n{'#' * 76}")

        arms = (("raw", wb_ids, None, {k: (swap_raw[k], None) for k in SWAPS}),
                ("chat", wc_b[0], wc_b[1], swap_chat))
        for arm, b_ids, b_mask, variants in arms:
            base_h = hidden(model, lay, b_ids, b_mask)
            blp = base_logprobs(model, lay, b_ids, b_mask)
            print(f"\n--- {arm} arm, layer {L} ---\n{hdr}")
            for k in SWAPS:
                v_ids, v_mask = variants[k]
                d = hidden(model, lay, v_ids, v_mask) - base_h
                mag = d.norm(dim=1)
                if k == SWAPS[0]:
                    mc, pc = measure_and_predict(model, lay, b_ids, b_mask, blp, CAL_ALPHA * d)
                    record(f"CALIBRATION graft x{CAL_ALPHA} k={k}", mc, pc)
                mg, pg = measure_and_predict(model, lay, b_ids, b_mask, blp, d)
                mt, pt = measure_and_predict(model, lay, b_ids, b_mask, blp, r_t[None] * mag[:, None])
                mr, pr = measure_and_predict(model, lay, b_ids, b_mask, blp, r_r[None] * mag[:, None])
                record(f"graft k={k}", mg, pg)
                record(f"refusal_tmpl k={k}", mt, pt)
                record(f"refusal_raw k={k}", mr, pr)
                compare(f"L{L} {arm} k={k}  tmpl/raw vector", mt, pt, mr, pr)
                compare(f"L{L} {arm} k={k}  graft/refusal_tmpl", mg, pg, mt, pt)

        p_h, i_h = hidden(model, lay, *p_pool), hidden(model, lay, *i_pool)
        for arm, (ids, mask), own_h, cross_h in (("passage-base", wc_b, p_h, i_h),
                                                 ("instruction-base", ib, i_h, p_h)):
            base_h = hidden(model, lay, ids, mask)
            mn = base_h.norm(dim=1).median().item()
            blp = base_logprobs(model, lay, ids, mask)
            print(f"\n--- {arm}, layer {L} ---\n{hdr}")
            for t in TARGETS:
                want = t * mn
                sel = {}
                for name, ph in (("own", own_h), ("cross", cross_h)):
                    j = (torch.cdist(base_h, ph) - want).abs().argmin(dim=1)
                    dd = ph[j] - base_h
                    sel[name] = (dd, ((dd.norm(dim=1) - want).abs() / want) < TOL)
                both = (sel["own"][1] & sel["cross"][1]).cpu()
                if both.float().mean() < 0.3:
                    continue
                mo, po = measure_and_predict(model, lay, ids, mask, blp, sel["own"][0])
                mx, px = measure_and_predict(model, lay, ids, mask, blp, sel["cross"][0])
                mo, po, mx, px = mo[both], po[both], mx[both], px[both]
                record(f"own @{t}", mo, po)
                record(f"cross @{t}", mx, px)
                compare(f"L{L} {arm} @{t}  cross/own", mx, px, mo, po)

    print(f"\n{'=' * 92}\nTHE TEST: do predicted ratios reproduce measured ratios?\n{'=' * 92}")
    print(f"{'comparison':>40}{'measured [95% CI]':>22}{'predicted [95% CI]':>22}{'agree':>8}")
    for label, mr, pr, agree in comparisons:
        print(f"{label:>40}{ci(mr):>22}{ci(pr):>22}{'yes' if agree else 'NO':>8}")
    n_ok = sum(c[3] for c in comparisons)
    lm, lp = torch.log(torch.tensor(all_meas)), torch.log(torch.tensor(all_pred))
    mr_vals = torch.tensor([c[1][0] for c in comparisons])
    pr_vals = torch.tensor([c[2][0] for c in comparisons])
    print(f"\nratio comparisons with overlapping intervals: {n_ok} of {len(comparisons)}")
    print(f"correlation of log measured vs log predicted ratios: "
          f"{pearson(torch.log(mr_vals), torch.log(pr_vals)):.3f}")
    print(f"correlation of log measured vs log predicted KL, all conditions: {pearson(lm, lp):.3f}")

    print("\nCALIBRATION rows must read meas/pred ~ 1. If not, nothing else can be read.")
    print("ANCHOR: raw arm, layer 12, graft measured 0.0086, 0.0345, 0.1343, 0.2561.")


if __name__ == "__main__":
    main()
