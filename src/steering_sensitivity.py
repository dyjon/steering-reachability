"""How much does behaviour move per unit of activation-space distance?

Gate experiment for the reachability question, and a replacement for
steering_corpus_scale.py, which was aimed wrongly: no feasible corpus approaches Im(F),
so nearest-neighbour distance to one cannot bear on the theorem.

The question this gates. Mishra et al. prove steered activations are almost surely not
*exactly* reachable, and caution against reading steering success as evidence of
prompt-based vulnerability. But vulnerability is about behaviour. If a steered activation
is unreachable yet close to something reachable, and behaviour is continuous in the
activation, it might still be reproducible in behaviour. That step needs a magnitude, not
just continuity, and nobody has measured it.

Splitting it:

    (a) how close can a prompt get in activation space?   intractable, needs SipIt
    (b) at that distance, how different is behaviour?     cheap, measured here

Measure (b). Perturb the residual stream at one layer by a controlled magnitude and see how
far the next-token distribution moves.

THE CONTROL IS THE POINT. Every earlier version of this experiment lacked one. Here each
steering perturbation is compared against RANDOM directions of the SAME magnitude. Without
that, "steering changes behaviour a lot" says nothing, because any perturbation of that
size might.

Reading the result:

    steering >> random    the direction matters, not just the magnitude
    steering ~= random    the behavioural effect is about magnitude; steering is not
                          special, and near-misses in activation space would behave
                          similarly
    both small at the steering magnitude
                          behaviour is insensitive at this scale, so formal
                          unreachability need not imply behavioural unreachability, and
                          the expensive prompt search is worth running
    both large            behaviour is sensitive here, near-misses do not help, and the
                          paper's caution is well founded without further work

Model is Qwen2.5-0.5B-Instruct, one of the three they use, and the smallest.

Run: python src/steering_sensitivity.py
"""
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 200
N_RANDOM = 4                                    # random directions averaged per magnitude
DELTAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]  # multiples of the median activation norm
BATCH = 16
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
    """Decoder layers return either a bare hidden-states tensor or a tuple starting with
    one, depending on the transformers version. Qwen2 returns a bare tensor as of 4.5x."""
    return output[0] if isinstance(output, tuple) else output


def repack(output, h):
    return (h,) + output[1:] if isinstance(output, tuple) else h


def add_at_last_position(vec):
    """Forward hook adding `vec` to the residual stream at the final token position."""
    def hook(module, args, output):
        h = unpack(output).clone()
        h[:, -1, :] += vec
        return repack(output, h)
    return hook


def logprobs(model, layer, tok, texts, vec=None):
    """Next-token log probabilities, optionally perturbing the residual stream."""
    handle = layer.register_forward_hook(add_at_last_position(vec)) if vec is not None else None
    try:
        out = []
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(DEVICE)
            logits = model(**enc).logits[:, -1, :]
            out.append(F.log_softmax(logits.float(), dim=-1))
        return torch.cat(out, 0)
    finally:
        if handle is not None:
            handle.remove()


def hidden_at_last(model, layer, tok, texts):
    """Residual stream at `layer`, final token position."""
    grabbed = []
    handle = layer.register_forward_hook(
        lambda m, a, o: grabbed.append(unpack(o)[:, -1, :]))
    try:
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(DEVICE)
            model(**enc)
    finally:
        handle.remove()
    return torch.cat(grabbed, 0)


def kl(p_log, q_log):
    """KL(P || Q) per row, from log probabilities."""
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def main():
    print(f"device: {DEVICE}, model: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()

    layers = model.model.layers
    j = len(layers) // 2                        # middle layer, as they use for persona vectors
    layer = layers[j]
    print(f"{len(layers)} layers, hooking layer {j}")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 200][:N_PROMPTS]
    print(f"{len(texts)} prompts")

    steering = (hidden_at_last(model, layer, tok, POSITIVE).mean(0)
                - hidden_at_last(model, layer, tok, NEGATIVE).mean(0))
    steering = steering / steering.norm()

    h = hidden_at_last(model, layer, tok, texts)
    median_norm = h.norm(dim=1).median().item()
    print(f"median activation norm at layer {j}: {median_norm:.1f}\n")

    base = logprobs(model, layer, tok, texts)
    dim = h.shape[1]

    print(f"{'delta':>7}{'|v|':>9}{'steering KL':>14}{'random KL':>12}{'ratio':>8}")
    print("-" * 50)
    for d in DELTAS:
        mag = d * median_norm
        s_kl = kl(base, logprobs(model, layer, tok, texts, steering * mag)).mean().item()

        r_kls = []
        for k in range(N_RANDOM):
            g = torch.randn(dim, generator=torch.Generator(DEVICE).manual_seed(k),
                            device=DEVICE)
            r_kls.append(kl(base, logprobs(model, layer, tok, texts,
                                           g / g.norm() * mag)).mean().item())
        r_kl = sum(r_kls) / len(r_kls)

        ratio = s_kl / r_kl if r_kl > 1e-9 else float("nan")
        print(f"{d:>7.2f}{mag:>9.1f}{s_kl:>14.4f}{r_kl:>12.4f}{ratio:>8.2f}", flush=True)

    print("\ndelta = 0 is the sanity check: both columns must be ~0.")
    print("ratio near 1 means the direction does not matter, only the magnitude.")


if __name__ == "__main__":
    main()
