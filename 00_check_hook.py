import argparse
import torch

from common import load_model, decoder_layers

PROMPTS = [
    "Explain why the sky looks blue during the day.",
    "A train travels 120 km in 1.5 hours. What is its average speed? Think step by step.",
    "Write two sentences about a quiet afternoon in a small town.",
]

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--directions", default="out/directions.pt")
p.add_argument("--layer", type=int, default=10)
p.add_argument("--alpha", type=float, default=1.5)
p.add_argument("--tol", type=float, default=0.10)
args = p.parse_args()

L = args.layer
model, processor = load_model(args.model)
layers = decoder_layers(model)
dtype = next(model.parameters()).dtype
vec = torch.load(args.directions, weights_only=False)["directions"][L].float()
vec = (vec / vec.norm()).to(model.device, dtype)
v32 = vec.float()

def inputs_for(prompt):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt").to(model.device)

def out_tensor(output):
    return output[0] if isinstance(output, tuple) else output

def run(inp, push_layer=None, want_hidden=False):
    raw, handles = {}, []
    if push_layer is not None:
        def push(module, inputs, output):
            h = out_tensor(output) + args.alpha * vec.view(1, 1, -1)
            return (h,) + output[1:] if isinstance(output, tuple) else h
        handles.append(layers[push_layer].register_forward_hook(push))
    for j, layer in enumerate(layers):

        handles.append(layer.register_forward_hook(
            lambda m, i, o, j=j: raw.__setitem__(j, out_tensor(o).detach().float().clone())))
    try:
        with torch.inference_mode():
            out = model(**inp, output_hidden_states=want_hidden, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return (out.hidden_states if want_hidden else None), raw

inp0 = inputs_for(PROMPTS[0])
hs, raw0 = run(inp0, want_hidden=True)
print("1. index map: max |hidden_states[k] - output of layers[k-1]| (should all be 0)")
index_ok = True
checked_layers = 0
for k in range(1, min(len(hs) - 1, len(layers))):
    difference = (hs[k].float() - raw0[k - 1]).abs().max().item()
    print(f"   k={k:2d}: {difference:.3g}")
    index_ok = index_ok and torch.allclose(hs[k].float(), raw0[k - 1], atol=1e-5, rtol=1e-4)
    checked_layers += 1
index_ok = bool(index_ok and checked_layers > 0)

shift = {"fixed": [], "legacy": []}
for prompt in PROMPTS:
    inp = inputs_for(prompt)
    _, base = run(inp)
    for name, push_layer in (("fixed", L - 1), ("legacy", L)):
        _, steered = run(inp, push_layer)
        shift[name].append(((steered[L - 1] - base[L - 1]) @ v32).mean().item())

fixed = sum(shift["fixed"]) / len(PROMPTS)
legacy = sum(shift["legacy"]) / len(PROMPTS)
print(f"\n2. push alpha={args.alpha:+.4f} along directions[{L}]; shift of the output of layers[{L - 1}]")
print(f"   fixed  (hook layers[{L - 1}]): {fixed:+.4f}")
print(f"   legacy (hook layers[{L}]):   {legacy:+.4f}")
ok = (index_ok and abs(fixed - args.alpha) <= args.tol * abs(args.alpha)
      and abs(legacy) <= args.tol * abs(args.alpha))
print("\nVERDICT:", "PASS (the fixed hook lands on the measured layer; the old hook does not)" if ok
      else "CHECK (send this output to the team before running any steering)")
