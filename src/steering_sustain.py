r"""Run 16, stages E0 and E1: does a prompt that matches a steered model at one step keep matching
it over the following tokens?

Runs 1 to 15 were single-step: one intervention at the last token, next-token distribution only.
Theorem 4.5 of Mishra et al. says even an exact state collision at one position is lost at the
next. This run measures what that costs in behaviour, token by token. The plan is in the private
notes; this file implements its first two stages.

TERMS
    S   the base prompt with steering         B   the base prompt without steering
    Q   another prompt, no steering, compared against S
    collision donor   a prompt whose layer-L state at its last position defines
                      v* = h_donor - h_base, so S's state there equals the donor's exactly
    near miss         for an ordinary vector no prompt matches, the pool prompt whose state is
                      closest to the steered state
    schedules         once: last prompt token only.  onward: last prompt token and every later
                      token.  all: every token, prompt included (Definition 4.1)

E0, VERIFICATION. Nothing in E1 is read unless every check passes; the script stops otherwise.
    1. single layer   on CPU, fixed tokens: adding at earlier positions of layer L leaves layer L's
                      output at the last position unchanged, while layer L+1's changes
    2. schedules      the hook adds at exactly the intended positions
    3. collision      under every schedule, S's layer-L state at the last prompt token equals the
                      donor's (the single-layer claim in practice)
    4. one-pass       logits from the token-by-token generation loop equal one-pass teacher-forced
                      scoring of the same tokens
    5. chat indices   the last chat-prompt token is the assistant header
    6. hooks removed  no hook left on the layer
    7. anchor         run 15's raw arm, layer 12, graft KL 0.0086, 0.0345, 0.1343, 0.2561

E1, PILOT. Eight bases (four wikitext passages, four Alpaca instructions in the chat template),
layer 12, 20 generated tokens, greedy.
    donors      wikitext: the base with its first 24 tokens swapped (same ending). Alpaca: another
                instruction. Schedules onward and once.
    sentiment   positive-minus-negative vector at 0.5 and 1.0 activation norms, schedules all and
                once, against the nearest prompt in a 32-prompt same-format pool and the base with
                a fixed instruction asking for a positive tone.

    Test A   ||state_S - state_Q|| at layer L, as a fraction of ||v|| and in activation norms
    Test B   KL(S || Q) and KL(S || B) in nats at each token, all scored on the same text: S's own
             greedy tokens (onward for donors, all for sentiment)
    Test C   free runs: tokens until each condition first departs from S's text
    The starting KL (token 1) is reported separately: a layer-L collision does not make it zero,
    because the layers above still read different earlier positions.

    Read nothing as a finding at n = 8. The pilot decides whether the contrasts are measurable,
    whether 1.0 norms keeps the text coherent, and what the main run costs.

Harmless prompts only. No AdvBench, no ablation, no refusal vector. Raw per-prompt arrays are
saved to run16_e1.pt.

Run: python src/steering_sustain.py
"""
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
L = 12
T = 20
SWAP_K = 24
POOL = 32
SCALES = [0.5, 1.0]
CHECKPOINTS = [1, 5, 20]
SEQ_LEN = 96
BATCH = 16
N_BOOT = 2000
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANCHOR_SWAPS = [8, 24, 48, 72]
ANCHOR = [0.0086, 0.0345, 0.1343, 0.2561]

POSITIVE = ["I love this", "This is wonderful", "What a delight", "I adore it",
            "Absolutely fantastic", "This makes me happy", "A joy to use", "I am so pleased"]
NEGATIVE = ["I hate this", "This is terrible", "What a disaster", "I despise it",
            "Absolutely awful", "This makes me angry", "A pain to use", "I am so annoyed"]
TRAIT_PASSAGE = "The following passage is written in a warm, cheerful, positive tone.\n\n"
TRAIT_CHAT = "\n\nPlease answer in a warm, cheerful, positive tone."


def unpack(o):
    return o[0] if isinstance(o, tuple) else o


def repack(o, h):
    return (h,) + o[1:] if isinstance(o, tuple) else h


def positions(sched, p, n):
    if sched == "once":
        return [p - 1] if p - 1 < n else []
    if sched == "onward":
        return list(range(p - 1, n))
    if sched == "all":
        return list(range(n))
    return []


class Steer:
    """Adds vec to layer L's output at the scheduled positions; keeps the post-add states."""

    def __init__(self, vec, sched, p):
        self.vec, self.sched, self.p = vec, sched, p
        self.added, self.state = [], None

    def __call__(self, m, a, o):
        h = unpack(o)
        self.added = positions(self.sched, self.p, h.shape[1]) if self.vec is not None else []
        if self.added:
            h = h.clone()
            h[0, self.added] += self.vec
        self.state = h[0].detach()
        return repack(o, h)


def run(model, layer, ids, vec=None, sched=None, p=None):
    """One unbatched, uncached forward. Returns all logits, layer-L states and the hook."""
    hk = Steer(vec, sched, p)
    h = layer.register_forward_hook(hk)
    try:
        logits = model(input_ids=torch.tensor([ids], device=DEVICE)).logits[0].float()
    finally:
        h.remove()
    return logits, hk.state, hk


def generate(model, layer, prompt, vec, sched, eos, keep=False):
    """Greedy, no KV cache: every step re-runs the whole sequence with the same schedule."""
    # ponytail: O(T^2) recompute, fine at T <= 128 on 0.5B; add a KV cache only if E2 is too slow
    y, rows = [], []
    for _ in range(T):
        logits, _, _ = run(model, layer, prompt + y, vec, sched, len(prompt))
        if keep:
            rows.append(F.log_softmax(logits[-1], -1))
        y.append(int(logits[-1].argmax()))
        if y[-1] in eos:
            break
    return y, rows


def score(model, layer, prompt, y, vec=None, sched=None):
    """Teacher-forced: log-probs and layer-L states at the positions predicting y[0..]."""
    p = len(prompt)
    logits, state, _ = run(model, layer, prompt + y[:-1], vec, sched, p)
    sl = slice(p - 1, p - 1 + len(y))
    return F.log_softmax(logits[sl], -1), state[sl]


def kl(lp, lq):
    return (lp.exp() * (lp - lq)).sum(-1)


def padded(x):
    out = torch.full((T,), float("nan"))
    out[:len(x)] = torch.as_tensor(x, dtype=torch.float32).cpu()
    return out


def match_len(y, ref):
    n = 0
    for a, b in zip(y, ref):
        if a != b:
            break
        n += 1
    return n


def boot_mean(x, gen):
    x = x[~torch.isnan(x)]
    if x.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    idx = torch.randint(0, x.numel(), (N_BOOT, x.numel()), generator=gen)
    lo, hi = torch.quantile(x[idx].mean(1), torch.tensor([0.025, 0.975]))
    return x.mean().item(), lo.item(), hi.item()


def wiki_rows(tok, n):
    """Same selection as runs 3 to 15, so the first 256 rows are the anchor's."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    ids = []
    for t in ds["text"]:
        t = t.strip()
        if len(t) <= 600:
            continue
        e = tok(t, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(e) == SEQ_LEN:
            ids.append(e)
            if len(ids) == n:
                break
    return ids


def chat_ids(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def last_state(model, layer, ids):
    return run(model, layer, ids)[1][-1]


# ---------------------------------------------------------------------------------- E0 checks

def check_single_layer(ids, d_model):
    cpu = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    lay, up = cpu.model.layers[L], cpu.model.layers[L + 1]
    v = torch.randn(d_model, generator=torch.Generator().manual_seed(SEED))
    v = 16.0 * v / v.norm()
    x = torch.tensor([ids])

    def capture(add):
        got = {}

        def h_add(m, a, o):
            h = unpack(o)
            if add:
                h = h.clone()
                h[0, :-1] += v
            got["L"] = h[0, -1].clone()
            return repack(o, h)

        hs = [lay.register_forward_hook(h_add),
              up.register_forward_hook(lambda m, a, o: got.__setitem__("L1", unpack(o)[0, -1].clone()))]
        try:
            cpu(input_ids=x)
        finally:
            for h in hs:
                h.remove()
        return got

    a, b = capture(False), capture(True)
    dl, dl1 = (a["L"] - b["L"]).abs().max().item(), (a["L1"] - b["L1"]).abs().max().item()
    del cpu
    return dl < 1e-6 and dl1 > 1e-3, f"layer {L} max diff {dl:.2e} (want < 1e-6), layer {L + 1} {dl1:.2e} (want > 1e-3)"


def anchor_kl(model, layer, w_ids):
    base, donor = torch.tensor(w_ids[:128], device=DEVICE), torch.tensor(w_ids[128:256], device=DEVICE)

    def last_h(ids):
        got = []
        h = layer.register_forward_hook(lambda m, a, o: got.append(unpack(o)[:, -1, :]))
        try:
            for i in range(0, ids.shape[0], BATCH):
                model(input_ids=ids[i:i + BATCH])
        finally:
            h.remove()
        return torch.cat(got)

    def last_lp(ids, d=None):
        out = []
        for i in range(0, ids.shape[0], BATCH):
            h = None
            if d is not None:
                dd = d[i:i + BATCH]

                def add(m, a, o, dd=dd):
                    x = unpack(o).clone()
                    x[:, -1, :] += dd
                    return repack(o, x)
                h = layer.register_forward_hook(add)
            try:
                out.append(F.log_softmax(model(input_ids=ids[i:i + BATCH]).logits[:, -1, :].float(), -1))
            finally:
                if h is not None:
                    h.remove()
        return torch.cat(out)

    hb, blp = last_h(base), last_lp(base)
    got = []
    for k in ANCHOR_SWAPS:
        v_ids = torch.cat([donor[:, :k], base[:, k:]], dim=1)
        got.append(kl(blp, last_lp(base, last_h(v_ids) - hb)).mean().item())
    return got


def e0(model, tok, layer, w_ids, pairs, eos):
    checks = []
    t0 = time.perf_counter()
    ok, msg = check_single_layer(w_ids[0], model.config.hidden_size)
    checks.append(("1 single layer (CPU)", ok, msg))

    n, p = 10, 6
    want = {"once": [5], "onward": list(range(5, 10)), "all": list(range(10)), None: []}
    vec = torch.ones(model.config.hidden_size, device=DEVICE)
    got = {s: run(model, layer, w_ids[0][:n], vec, s, p)[2].added for s in want}
    checks.append(("2 schedule positions", got == want, str(got)))

    worst = 0.0
    for dom, base, donor, _ in (pairs[0], pairs[4]):
        hd, hb = last_state(model, layer, donor), last_state(model, layer, base)
        for s in ("once", "onward", "all"):
            st = run(model, layer, base, hd - hb, s, len(base))[1][-1]
            worst = max(worst, (st - hd).abs().max().item())
    checks.append(("3 collision at last prompt token, all schedules", worst < 1e-4,
                   f"max |state_S - state_donor| = {worst:.2e} (want < 1e-4)"))

    worst = 0.0
    for dom, base, donor, _ in (pairs[0], pairs[4]):
        v = last_state(model, layer, donor) - last_state(model, layer, base)
        for s in ("once", "onward", "all"):
            y, rows = generate(model, layer, base, v, s, eos, keep=True)
            lp, _ = score(model, layer, base, y, v, s)
            worst = max(worst, (torch.stack(rows) - lp).abs().max().item())
    checks.append(("4 generation loop == one-pass scoring", worst < 1e-3,
                   f"max |log-prob diff| = {worst:.2e} (want < 1e-3)"))

    tail = tok.decode(pairs[4][1][-3:])
    checks.append(("5 chat prompt ends at assistant header", "assistant" in tail, repr(tail)))
    checks.append(("6 hooks removed", len(layer._forward_hooks) == 0, f"{len(layer._forward_hooks)} left"))

    got = anchor_kl(model, layer, w_ids)
    ok = all(abs(g - a) < 1e-4 for g, a in zip(got, ANCHOR))
    checks.append(("7 run 15 anchor, raw arm, layer 12", ok,
                   " ".join(f"{g:.4f}" for g in got) + "  (want " + " ".join(f"{a:.4f}" for a in ANCHOR) + ")"))

    print(f"\n{'=' * 92}\nE0  VERIFICATION  ({time.perf_counter() - t0:.0f} s)\n{'=' * 92}")
    for name, ok, msg in checks:
        print(f"{'PASS' if ok else 'FAIL':<6}{name:<48}{msg}")
    return all(c[1] for c in checks)


# ---------------------------------------------------------------------------------- E1 pilot

def conditions_for(model, layer, base, y_ref, arms, eos):
    """Score every arm on the shared text y_ref; free-run every arm. arms: name -> (prompt, vec, sched)."""
    out = {}
    for name, (prompt, vec, sched) in arms.items():
        lp, st = score(model, layer, prompt, y_ref, vec, sched)
        y, _ = generate(model, layer, prompt, vec, sched, eos)
        out[name] = (lp, st, y)
    return out


def e1(model, tok, layer, pairs, pools, eos, gen):
    t0 = time.perf_counter()
    pool_h = {dom: torch.stack([last_state(model, layer, x) for x in pool]) for dom, pool in pools.items()}
    med = {dom: h.norm(dim=1).median().item() for dom, h in pool_h.items()}

    def raw_last(texts):
        return torch.stack([last_state(model, layer, tok(t)["input_ids"]) for t in texts]).mean(0)
    u = raw_last(POSITIVE) - raw_last(NEGATIVE)
    u = u / u.norm()

    res = {"donor": [], "sentiment": {s: [] for s in SCALES}, "median_norm": med}
    samples = []
    for i, (dom, base, donor, raw) in enumerate(pairs):
        hb = last_state(model, layer, base)
        vstar = last_state(model, layer, donor) - hb
        y_on, _ = generate(model, layer, base, vstar, "onward", eos)
        c = conditions_for(model, layer, base, y_on, {
            "S_on": (base, vstar, "onward"), "S_once": (base, vstar, "once"),
            "Q": (donor, None, None), "B": (base, None, None)}, eos)
        lp = {k: v[0] for k, v in c.items()}
        res["donor"].append({
            "domain": dom, "v_norms": vstar.norm().item() / med[dom], "len": len(y_on),
            "kl_on_Q": padded(kl(lp["S_on"], lp["Q"])), "kl_once_Q": padded(kl(lp["S_once"], lp["Q"])),
            "kl_on_B": padded(kl(lp["S_on"], lp["B"])), "kl_once_B": padded(kl(lp["S_once"], lp["B"])),
            "gap_on": padded((c["S_on"][1] - c["Q"][1]).norm(dim=1) / vstar.norm()),
            "match": {k: match_len(c[k][2], y_on) for k in ("S_once", "Q", "B")}})

        for s in SCALES:
            v = s * med[dom] * u
            j = int((pool_h[dom] - (hb + v)).norm(dim=1).argmin())
            near = pools[dom][j]
            instr = (tok(TRAIT_PASSAGE)["input_ids"] + base if dom == "wiki"
                     else chat_ids(tok, raw + TRAIT_CHAT))
            y_all, _ = generate(model, layer, base, v, "all", eos)
            c = conditions_for(model, layer, base, y_all, {
                "S_all": (base, v, "all"), "S_once": (base, v, "once"),
                "near": (near, None, None), "instr": (instr, None, None), "B": (base, None, None)}, eos)
            lp = {k: x[0] for k, x in c.items()}
            res["sentiment"][s].append({
                "domain": dom, "near_dist": (pool_h[dom][j] - hb - v).norm().item() / med[dom],
                "len": len(y_all),
                **{f"kl_{a}_{q}": padded(kl(lp[a], lp[q])) for a in ("S_all", "S_once")
                   for q in ("near", "instr", "B")},
                "gap_near": padded((c["S_all"][1] - c["near"][1]).norm(dim=1) / v.norm()),
                "gap_instr": padded((c["S_all"][1] - c["instr"][1]).norm(dim=1) / v.norm()),
                "match": {k: match_len(c[k][2], y_all) for k in ("S_once", "near", "instr", "B")}})
            if i in (0, 4):
                samples.append((dom, s, tok.decode(c["B"][2]), tok.decode(y_all)))

    report(res, gen, samples)
    print(f"\nE1 time {time.perf_counter() - t0:.0f} s")
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    torch.save(res, "run16_e1.pt")


def table(title, rows, keys, gen):
    print(f"\n{title}\n{'token':>6}{'n':>4}" + "".join(f"{k:>24}" for k in keys))
    for t in CHECKPOINTS:
        cells, n = [], 0
        for k in keys:
            x = torch.stack([r[k] for r in rows])[:, t - 1]
            n = int((~torch.isnan(x)).sum())
            m, lo, hi = boot_mean(x, gen)
            cells.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"{t:>6}{n:>4}" + "".join(f"{c:>24}" for c in cells))


def report(res, gen, samples):
    print(f"\n{'=' * 92}\nE1  PILOT  (n = 8; descriptive only)\n{'=' * 92}")
    print("median activation norm at layer", L, {k: round(v, 1) for k, v in res["median_norm"].items()})
    d = res["donor"]
    print(f"\nDONORS: ||v*|| in activation norms: " + ", ".join(f"{r['domain']} {r['v_norms']:.2f}" for r in d))
    table("Test B, donors (nats), scored on S_onward's text",
          d, ["kl_on_Q", "kl_once_Q", "kl_on_B", "kl_once_B"], gen)
    table("Test A, donors: ||state_S_onward - state_Q|| / ||v*||", d, ["gap_on"], gen)
    print("Test C, donors: tokens matching S_onward's free text (of " + str(T) + "): "
          + "; ".join(f"{k} {[r['match'][k] for r in d]}" for k in ("S_once", "Q", "B")))
    for s in SCALES:
        r = res["sentiment"][s]
        print(f"\nSENTIMENT at {s} activation norms. Nearest-pool distance in norms: "
              + ", ".join(f"{x['near_dist']:.2f}" for x in r))
        table(f"Test B, sentiment {s} (nats), scored on S_all's text", r,
              ["kl_S_all_near", "kl_S_all_instr", "kl_S_all_B"], gen)
        table(f"Test B, sentiment {s}, once-only arm", r,
              ["kl_S_once_near", "kl_S_once_instr", "kl_S_once_B"], gen)
        table(f"Test A, sentiment {s}: gap / ||v||", r, ["gap_near", "gap_instr"], gen)
        print("Test C: tokens matching S_all's free text: "
              + "; ".join(f"{k} {[x['match'][k] for x in r]}" for k in ("S_once", "near", "instr", "B")))
    print("\nSAMPLE TEXT (coherence check for the scale choice)")
    for dom, s, b, y in samples:
        print(f"\n[{dom}, {s} norms]\n  unsteered: {b!r}\n  steered:   {y!r}")


def main():
    import datasets as _d
    import transformers as _t
    print(f"torch {torch.__version__}, transformers {_t.__version__}, datasets {_d.__version__}")
    print(f"device: {DEVICE}, model: {MODEL}, layer {L}, T = {T}\n")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE).eval()
    layer = model.model.layers[L]
    gen = torch.Generator().manual_seed(SEED)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
           tok.convert_tokens_to_ids("<|endoftext|>")}

    w_ids = wiki_rows(tok, 256 + 8 + POOL)
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"].strip() for r in alp
                if not r["input"].strip() and 20 < len(r["instruction"]) < 160]

    wb, wsrc = w_ids[256:260], w_ids[260:264]
    pairs = [("wiki", b, s[:SWAP_K] + b[SWAP_K:], None) for b, s in zip(wb, wsrc)]
    pairs += [("alpaca", chat_ids(tok, harmless[1000 + i]), chat_ids(tok, harmless[1004 + i]),
               harmless[1000 + i]) for i in range(4)]
    pools = {"wiki": w_ids[264:264 + POOL],
             "alpaca": [chat_ids(tok, h) for h in harmless[1010:1010 + POOL]]}

    if not e0(model, tok, layer, w_ids, pairs, eos):
        print("\nE0 FAILED. Stopping before E1; nothing below would be interpretable.")
        sys.exit(1)
    e1(model, tok, layer, pairs, pools, eos, gen)


if __name__ == "__main__":
    main()
