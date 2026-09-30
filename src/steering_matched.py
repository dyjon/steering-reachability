r"""At genuinely matched distance, does donor kind still change the graft?

Every attempt so far has let the donor construction decide the distance, and then tried to
compare across constructions after the fact. `steering_kind.py` fitted three of twelve ladders
on under 2x of range. `steering_graded.py` tried to fix that by interpolating in prompt space
and made it worse on the instruction arm, spanning 1.16x, because for an instruction base every
donor containing any passage is already far and varying how much barely moves the distance.

So stop constructing donors and start **selecting** them.

Build one labelled pool — passages, instructions, and graded blends of the two — embed all of
it, then for each base prompt, each kind, and each target distance, take the pool member of that
kind whose activation sits closest to the target. Distance is matched by construction. Kind is
the only thing that varies.

    own     a donor of the same kind as the base
    cross   a donor of the other kind
    mixed   a graded blend

A selection is only accepted when the achieved distance lands within 15% of the target.
Otherwise the cell is **uncovered** and reported as such, because "no donor of this kind exists
at this distance" is itself a finding and is exactly what the earlier runs were papering over
by using whatever donor was nearest.

TWO READINGS.

  Coverage tells you whether the kinds even occupy the same distance range. If cross-kind
  donors cannot be found near a base at small distances, no construction will fix that and the
  cross-kind curve genuinely does not exist there.

  Within covered cells, `graft` across kinds at the same achieved distance answers the question
  runs 10 through 12 circled: is kind a real second variable, or was it standing in for
  distance all along?

REFUSAL IS THE INTERNAL CHECK. It is a fixed direction fired at the achieved magnitude, so at
matched distance it must come back nearly identical across kinds. If it does not, the matching
failed and the graft column cannot be read.

No ablation and no jailbreak. Only AdvBench's `goal` column is read.

Run: python src/steering_matched.py
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
MIX_N = 256
N_VEC = 96
SEQ_LEN = 96
TARGETS = [0.2, 0.3, 0.4, 0.5, 0.7]
TOL = 0.15
BATCH = 16
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


def logprobs(model, layer, ids, mask, delta=None):
    out = []
    for i in range(0, ids.shape[0], BATCH):
        hd = layer.register_forward_hook(add_at_last(delta[i:i + BATCH])) if delta is not None else None
        try:
            lg = model(input_ids=ids[i:i + BATCH], attention_mask=mask[i:i + BATCH]).logits[:, -1, :]
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
            model(input_ids=ids[i:i + BATCH], attention_mask=mask[i:i + BATCH])
    finally:
        h.remove()
    return torch.cat(got, 0)


def kl(p, q):
    return (p.exp() * (p - q)).sum(-1)


def chat(tok, prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True) for p in prompts]
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    return e["input_ids"], e["attention_mask"]


def load_harmful():
    """AdvBench goals only; the `target` column of affirmative prefixes is never read."""
    try:
        return [r.strip() for r in load_dataset("walledai/AdvBench", split="train")["prompt"]]
    except Exception:
        raw = urllib.request.urlopen(ADVBENCH_CSV, timeout=60).read().decode("utf-8")
        return [r["goal"].strip() for r in csv.DictReader(io.StringIO(raw)) if r.get("goal")]


def wiki_texts(tok, n):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    out, ids = [], []
    for t in ds["text"]:
        t = t.strip()
        if len(t) <= 600:
            continue
        e = tok(t, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(e) == SEQ_LEN:
            ids.append(e)
            out.append(tok.decode(e))
            if len(out) == n:
                break
    return out, torch.tensor(ids, device=DEVICE)


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
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"].strip() for r in alp
                if not r["input"].strip() and 20 < len(r["instruction"]) < 160]
    wtexts, wids = wiki_texts(tok, BASE_N + POOL_N)
    print(f"AdvBench {len(harmful)}, Alpaca {len(harmless)}, wikitext {len(wtexts)}")

    p_base, p_pool = wtexts[:BASE_N], wtexts[BASE_N:]
    i_base = harmless[N_VEC:N_VEC + BASE_N]
    i_pool = harmless[N_VEC + BASE_N:N_VEC + BASE_N + POOL_N]

    # graded blends: a passage truncated to (1-a) of its tokens, then an instruction
    mixed = []
    for j in range(MIX_N):
        a = [0.25, 0.5, 0.75][j % 3]
        keep = int(round((1 - a) * SEQ_LEN))
        head = tok.decode(wids[BASE_N + j, :keep])
        mixed.append((head + " " + i_pool[j % len(i_pool)]).strip())

    pb, pbm = chat(tok, p_base)
    ib, ibm = chat(tok, i_base)
    pools = {"passage": chat(tok, p_pool), "instruction": chat(tok, i_pool), "mixed": chat(tok, mixed)}
    print(f"pools: passage {len(p_pool)}, instruction {len(i_pool)}, mixed {len(mixed)}\n")

    vh, vhm = chat(tok, harmful[:N_VEC])
    vs, vsm = chat(tok, harmless[:N_VEC])

    for L in LAYERS:
        lay = layers[L]
        ref = hidden(model, lay, vh, vhm).mean(0) - hidden(model, lay, vs, vsm).mean(0)
        ref = ref / ref.norm()
        pool_h = {k: hidden(model, lay, *v) for k, v in pools.items()}

        for arm, (ids, mask), own in (("passage-base", (pb, pbm), "passage"),
                                      ("instruction-base", (ib, ibm), "instruction")):
            base_h = hidden(model, lay, ids, mask)
            mn = base_h.norm(dim=1).median().item()
            base_lp = logprobs(model, lay, ids, mask)
            cross = "instruction" if own == "passage" else "passage"
            kinds = {"own": pool_h[own], "cross": pool_h[cross], "mixed": pool_h["mixed"]}

            print(f"\n{'=' * 90}\n{arm}, layer {L}   median norm {mn:.1f}\n{'=' * 90}")
            print(f"{'target':>7}{'kind':>8}{'covered':>9}{'achieved':>10}"
                  f"{'graft':>11}{'refusal':>10}")
            print("-" * 55)
            for t in TARGETS:
                want = t * mn
                for kname, ph in kinds.items():
                    d_all = torch.cdist(base_h, ph)
                    j = (d_all - want).abs().argmin(dim=1)
                    d = ph[j] - base_h
                    mag = d.norm(dim=1)
                    ok = ((mag - want).abs() / want) < TOL
                    frac = ok.float().mean().item()
                    if frac < 0.5:
                        print(f"{t:>7.1f}{kname:>8}{frac:>8.0%}{mag[ok].mean().item() / mn if ok.any() else float('nan'):>10.2f}"
                              f"{'--':>11}{'--':>10}   uncovered")
                        continue
                    g = kl(base_lp, logprobs(model, lay, ids, mask, d))[ok].mean().item()
                    r = kl(base_lp, logprobs(model, lay, ids, mask,
                                             ref[None] * mag[:, None]))[ok].mean().item()
                    print(f"{t:>7.1f}{kname:>8}{frac:>8.0%}{mag[ok].mean().item() / mn:>10.2f}"
                          f"{g:>11.4f}{r:>10.4f}", flush=True)

    print("\nCHECK: refusal must be near-identical across kinds within a target row.")
    print("       It is a fixed direction at the achieved magnitude, so a spread there")
    print("       means the distance matching failed and graft cannot be read.")
    print("READ:  coverage says whether the kinds occupy the same distance range at all.")
    print("       Within covered rows, graft across kinds at one achieved distance is the")
    print("       clean answer to whether kind is a real second variable.")


if __name__ == "__main__":
    main()
