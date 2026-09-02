"""Is the steering direction behaviourally special, against a baseline that can compete?

steering_sensitivity.py found the steering direction produces only ~1.23x the KL of an
isotropic random direction of the same norm, flat across a 20x magnitude range, with
KL ~ 0.42 * delta^2.

That comparison is too easy to win, and the conclusion drawn from it does not hold.
Activations occupy a low-dimensional manifold inside a 896-dimensional space, so an
isotropic Gaussian direction is almost entirely orthogonal to anywhere the data goes.
"Steering beats random by 23%" may only mean "steering beats a direction pointing
nowhere."

So this replaces one baseline with four, all at matched magnitude:

    steering      difference in means, as before
    subspace      random inside the top-k principal subspace of natural activations
    complement    random inside the orthogonal complement, i.e. deliberately off-manifold
    natdiff       difference between two natural activations, a direction that actually
                  connects two reachable points

Reading it:

    steering ~= subspace ~= natdiff
        steering is not a special direction. Its behavioural effect is what any
        on-manifold perturbation of that size does, which is a real claim about steering
        rather than an artefact of a weak control.

    steering >> subspace
        steering is genuinely special, and the 1.23 against isotropic random was
        understating it because isotropic random is a straw baseline.

    subspace >> complement
        confirms the manifold story: direction matters, but what matters is being on the
        manifold, not being the steering vector specifically.

Also reports what fraction of each direction's energy falls inside the top-k subspace.
For an isotropic random vector the expectation is k/D, so that number says directly
whether steering is an on-manifold direction.

Run: python src/steering_subspace.py
"""
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 256
N_RANDOM = 5                    # draws per random baseline, gives a spread
DELTAS = [0.0, 0.25, 1.0]       # KL is quadratic, so three points suffice
TOP_K = 32                      # principal subspace dimension
BATCH = 32
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
    return output[0] if isinstance(output, tuple) else output


def repack(output, h):
    return (h,) + output[1:] if isinstance(output, tuple) else h


def add_at_last_position(vec):
    def hook(module, args, output):
        h = unpack(output).clone()
        h[:, -1, :] += vec
        return repack(output, h)
    return hook


def encode(tok, texts):
    return tok(texts, return_tensors="pt", padding=True, truncation=True,
               max_length=128).to(DEVICE)


def logprobs(model, layer, tok, texts, vec=None):
    handle = layer.register_forward_hook(add_at_last_position(vec)) if vec is not None else None
    try:
        out = []
        for i in range(0, len(texts), BATCH):
            logits = model(**encode(tok, texts[i:i + BATCH])).logits[:, -1, :]
            out.append(F.log_softmax(logits.float(), dim=-1))
        return torch.cat(out, 0)
    finally:
        if handle is not None:
            handle.remove()


def hidden_at_last(model, layer, tok, texts):
    grabbed = []
    handle = layer.register_forward_hook(
        lambda m, a, o: grabbed.append(unpack(o)[:, -1, :]))
    try:
        for i in range(0, len(texts), BATCH):
            model(**encode(tok, texts[i:i + BATCH]))
    finally:
        handle.remove()
    return torch.cat(grabbed, 0)


def kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def unit(v):
    return v / v.norm()


def energy_in(basis, v):
    """Fraction of v's squared norm lying in the span of `basis` (D x k, orthonormal)."""
    return ((basis.T @ unit(v)) ** 2).sum().item()


def main():
    print(f"device: {DEVICE}, model: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()

    layers = model.model.layers
    j = len(layers) // 2
    layer = layers[j]

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 200][:N_PROMPTS]
    print(f"{len(layers)} layers, hooking layer {j}, {len(texts)} prompts")

    steering = unit(hidden_at_last(model, layer, tok, POSITIVE).mean(0)
                    - hidden_at_last(model, layer, tok, NEGATIVE).mean(0))

    h = hidden_at_last(model, layer, tok, texts)
    median_norm = h.norm(dim=1).median().item()
    dim = h.shape[1]

    # principal subspace of the natural activations
    hc = h - h.mean(0)
    _, S, Vh = torch.linalg.svd(hc, full_matrices=False)
    basis = Vh[:TOP_K].T.contiguous()                       # D x k, orthonormal
    var = (S ** 2)
    explained = (var[:TOP_K].sum() / var.sum()).item()

    print(f"dim {dim}, median norm {median_norm:.1f}")
    print(f"top-{TOP_K} subspace holds {explained:.1%} of activation variance")
    print(f"steering energy in that subspace: {energy_in(basis, steering):.1%}"
          f"   (isotropic null is {TOP_K / dim:.1%})\n")

    gen = torch.Generator(device=DEVICE).manual_seed(0)

    def draw(kind):
        if kind == "subspace":
            c = torch.randn(TOP_K, generator=gen, device=DEVICE)
            return unit(basis @ c)
        if kind == "complement":
            g = torch.randn(dim, generator=gen, device=DEVICE)
            return unit(g - basis @ (basis.T @ g))
        if kind == "natdiff":
            i, k = torch.randint(0, h.shape[0], (2,), generator=gen, device=DEVICE)
            d = h[i] - h[k]
            return unit(d) if d.norm() > 1e-6 else draw("natdiff")
        raise ValueError(kind)

    base = logprobs(model, layer, tok, texts)
    kinds = ["subspace", "complement", "natdiff"]

    print(f"{'delta':>7}{'steering':>11}" +
          "".join(f"{k:>20}" for k in kinds))
    print("-" * (18 + 20 * len(kinds)))

    for d in DELTAS:
        mag = d * median_norm
        row = f"{d:>7.2f}"
        row += f"{kl(base, logprobs(model, layer, tok, texts, steering * mag)).mean().item():>11.4f}"
        for kind in kinds:
            vals = [kl(base, logprobs(model, layer, tok, texts, draw(kind) * mag)).mean().item()
                    for _ in range(N_RANDOM)]
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            row += f"{m:>13.4f}±{sd:.4f}"
        print(row, flush=True)

    print("\ndelta = 0 must be 0.0000 across the row.")
    print("steering vs subspace is the comparison that matters.")
    print("subspace vs complement says whether being on-manifold is what counts.")


if __name__ == "__main__":
    main()
