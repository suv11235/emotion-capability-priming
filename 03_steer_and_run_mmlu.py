"""Step 3: push the model along the direction and ask it MMLU-Pro questions.

The push is added at the layer the direction was measured on. A direction for
layer L comes from hidden_states[L], which is the output of decoder layer L - 1,
so the hook goes on layers[L - 1]. (An earlier version of this script hooked
layers[L], one layer too late; results from it are not comparable.)

Reads : out/directions.pt, out/frozen.json, MMLU-Pro from Hugging Face
Writes: out/<run_name>/run_meta.json   (settings, written first)
        out/<run_name>/records.jsonl   (raw answers, written BEFORE any scoring)

How a push works: at one layer we add  alpha * direction  to the model's
internal numbers, where  alpha = multiplier * delta.  multiplier 0 is the
normal model, +4 is four natural steps toward sad, -4 is four steps away.

Same questions for every multiplier, greedy decoding (no randomness), batch
size 1 (batching changes the answers a bit, see the results doc).

Re-running into an existing run folder is refused, so records are never
duplicated. Use --resume to continue an interrupted run with the same settings;
it skips (multiplier, question) pairs that are already in records.jsonl.

Equal-norm random control: --random_index i replaces the direction with a random
unit vector. It is the same vector as random_i in the main pipeline
(numpy default_rng(42), row i of a standard-normal draw, normalised), up to
float rounding.
"""
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import torch
import transformers
from datasets import load_dataset

from common import SEED, load_model, decoder_layers, mmlu_prompt

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", default="out")
p.add_argument("--directions", default=None, help="default: out/directions.pt")
p.add_argument("--frozen", default=None, help="default: out/frozen.json")
p.add_argument("--name", default=None, help="label stored in each record (default: sad, or random_<i>)")
p.add_argument("--multipliers", default="-4,-2,-1,-0.5,0,0.5,1,2,4")
p.add_argument("--n_questions", type=int, default=1000)
p.add_argument("--max_new_tokens", type=int, default=2048)
p.add_argument("--run_name", default="mmlu")
p.add_argument("--random_index", type=int, default=None,
               help="use random direction i (equal norm) instead of the loaded direction")
p.add_argument("--resume", action="store_true",
               help="continue an interrupted run in the same folder; skips answers already saved")
args = p.parse_args()

frozen = json.load(open(args.frozen or f"{args.out}/frozen.json"))
layer, delta = frozen["layer"], frozen["delta"]
vec = torch.load(args.directions or f"{args.out}/directions.pt", weights_only=False)["directions"][layer].float()
vec = vec / vec.norm()
if args.random_index is not None:
    if args.random_index < 0:
        sys.exit("--random_index must be >= 0")
    raw = np.random.default_rng(SEED).standard_normal((args.random_index + 1, vec.numel()))[args.random_index]
    raw = raw.astype(np.float32)
    vec = torch.from_numpy(raw / (np.linalg.norm(raw) + 1e-8))
name = args.name or (f"random_{args.random_index}" if args.random_index is not None else "sad")
multipliers = [float(x) for x in args.multipliers.split(",")]

# ---- the same questions every time: shuffle with seed 42, take the first N --
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=SEED).select(range(args.n_questions))
rows = [dict(r) for r in ds]

out_dir = f"{args.out}/{args.run_name}"
records_path = f"{out_dir}/records.jsonl"
meta_path = f"{out_dir}/run_meta.json"
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
meta = dict(model=args.model, layer=layer, delta=delta, multipliers=multipliers,
            n_questions=args.n_questions, seed=SEED, do_sample=False, batch_size=1,
            max_new_tokens=args.max_new_tokens, git_commit=commit, hook_layer=layer - 1,
            direction_name=name, random_index=args.random_index,
            torch=torch.__version__, transformers=transformers.__version__)

done = set()
if os.path.exists(records_path):
    if not args.resume:
        sys.exit(f"{records_path} already exists. Use a new --run_name, or --resume to continue it.")
    old = json.load(open(meta_path))
    for k in ("model", "layer", "delta", "n_questions", "max_new_tokens", "direction_name", "random_index", "multipliers"):
        if old.get(k) != meta[k]:
            sys.exit(f"--resume: setting {k!r} differs from the saved run ({old.get(k)!r} vs {meta[k]!r})")
    for line in open(records_path):
        if line.strip():
            r = json.loads(line)
            done.add((float(r["alpha_multiplier"]), r["question_id"]))
    print(f"resuming: {len(done)} answers already saved", file=sys.stderr)

model, processor = load_model(args.model)
vec = vec.to(model.device, next(model.parameters()).dtype)

# ---- save the settings first (so the run can be checked later) -------------
os.makedirs(out_dir, exist_ok=True)
if not done:
    json.dump(meta, open(meta_path, "w"), indent=2)


def make_hook(alpha):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * vec.view(1, 1, -1)            # the push
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def generate(prompt, alpha):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt").to(model.device)
    handle = None
    if alpha != 0:
        # directions[layer] came from hidden_states[layer], which is the output of layers[layer - 1]
        handle = decoder_layers(model)[layer - 1].register_forward_hook(make_hook(alpha))
    try:
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
    finally:
        if handle:
            handle.remove()                           # always remove the push
    ids = out[0, inp["input_ids"].shape[1]:]
    eos = model.config.get_text_config().eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    truncated = len(ids) >= args.max_new_tokens and not (set(ids.tolist()) & eos)
    return processor.decode(ids, skip_special_tokens=True), len(ids), truncated


with open(records_path, "a") as f:
    for m in multipliers:
        alpha = m * delta
        for row in rows:
            qid = str(row["question_id"])
            if (m, qid) in done:
                continue
            text, n_tok, trunc = generate(mmlu_prompt(row), alpha)
            f.write(json.dumps(dict(
                question_id=qid, category=row["category"], direction=name,
                alpha_multiplier=m, alpha=alpha, gold=row["answer"], n_options=len(row["options"]),
                response=text, n_new_tokens=n_tok, truncated=trunc)) + "\n")
            f.flush()
        print(f"finished multiplier {m}", file=sys.stderr)
