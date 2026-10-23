import argparse
import json
import os
import subprocess
import sys
import zlib
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
p.add_argument("--random_subspace", default=None, help="e.g. 0.5:0.95 (needs --random_index and --acts)")
p.add_argument("--random_scale", choices=["equal", "std"], default="equal")
p.add_argument("--acts", default=None, help="activations_train_test.npz from step 1 (neutral train stories)")
p.add_argument("--question_offset", type=int, default=0)
p.add_argument("--sample_seed", type=int, default=None, help="sample instead of greedy decoding")
p.add_argument("--temperature", type=float, default=0.7)
args = p.parse_args()

frozen = json.load(open(args.frozen or f"{args.out}/frozen.json"))
layer, delta = frozen["layer"], frozen["delta"]
vec = torch.load(args.directions or f"{args.out}/directions.pt", weights_only=False)["directions"][layer].float()
vec = vec / vec.norm()
sad_vec = vec.numpy().copy()
random_info = None
if (args.random_subspace or args.random_scale == "std") and args.random_index is None:
    sys.exit("--random_subspace / --random_scale std apply to a random direction; add --random_index")
if args.random_index is not None:
    if args.random_index < 0:
        sys.exit("--random_index must be >= 0")
    raw = np.random.default_rng(SEED).standard_normal((args.random_index + 1, vec.numel()))[args.random_index]
    raw = raw.astype(np.float32)
    r = raw / (np.linalg.norm(raw) + 1e-8)
    tag, random_info = "", {}
    if args.random_subspace or args.random_scale == "std":
        if not args.acts:
            sys.exit("--random_subspace / --random_scale std need --acts")
        neutral = np.load(args.acts)["neu_tr"][:, layer].astype(np.float32)
    if args.random_subspace:
        from sklearn.decomposition import PCA
        lo, hi = (float(x) for x in args.random_subspace.split(":"))
        pca = PCA().fit(neutral)
        cum = np.cumsum(pca.explained_variance_ratio_)
        start = 0 if lo <= 0 else int(np.searchsorted(cum, lo) + 1)
        end = int(np.searchsorted(cum, hi) + 1)
        band = pca.components_[start:end]
        inside = band.T @ (band @ r)
        print(f"subspace {lo}:{hi} = components {start} to {end}; "
              f"{np.linalg.norm(inside):.2f} of the random vector's length is inside", file=sys.stderr)
        r = inside / np.linalg.norm(inside)
        tag += "sub"
        random_info.update(subspace=[lo, hi], components=[start, end])
    if args.random_scale == "std":
        scale = float((neutral @ r).std() / (neutral @ sad_vec).std())
        print(f"variance-matched scale {scale:.3f}", file=sys.stderr)
        r = r * scale
        tag += "_std" if tag else "std"
        random_info.update(scale=scale)
    vec = torch.from_numpy(r.astype(np.float32))
name = args.name or (f"random_{tag + '_' if tag else ''}{args.random_index}"
                     if args.random_index is not None else "sad")
multipliers = [float(x) for x in args.multipliers.split(",")]
do_sample = args.sample_seed is not None

if 0 < args.question_offset < 1000:
    sys.exit("--question_offset 1 to 999 overlaps the main 1,000 questions; use 0 or >= 1000")
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=SEED)
ds = ds.select(range(args.question_offset, args.question_offset + args.n_questions))
rows = [dict(r) for r in ds]

out_dir = f"{args.out}/{args.run_name}"
records_path = f"{out_dir}/records.jsonl"
meta_path = f"{out_dir}/run_meta.json"
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
meta = dict(model=args.model, layer=layer, delta=delta, multipliers=multipliers,
            n_questions=args.n_questions, question_offset=args.question_offset, seed=SEED,
            do_sample=do_sample, temperature=args.temperature if do_sample else None,
            sample_seed=args.sample_seed, batch_size=1,
            max_new_tokens=args.max_new_tokens, git_commit=commit, hook_layer=layer - 1,
            direction_name=name, random_index=args.random_index, random_control=random_info,
            torch=torch.__version__, transformers=transformers.__version__)

done = set()
if os.path.exists(records_path):
    if not args.resume:
        sys.exit(f"{records_path} already exists. Use a new --run_name, or --resume to continue it.")
    old = json.load(open(meta_path))
    for k in ("model", "layer", "delta", "n_questions", "question_offset", "max_new_tokens",
              "direction_name", "random_index", "random_control", "do_sample", "temperature", "sample_seed", "multipliers"):
        if old.get(k) != meta[k]:
            sys.exit(f"--resume: setting {k!r} differs from the saved run ({old.get(k)!r} vs {meta[k]!r})")
    for line in open(records_path):
        if line.strip():
            r = json.loads(line)
            done.add((float(r["alpha_multiplier"]), r["question_id"]))
    print(f"resuming: {len(done)} answers already saved", file=sys.stderr)

model, processor = load_model(args.model)
vec = vec.to(model.device, next(model.parameters()).dtype)

os.makedirs(out_dir, exist_ok=True)
if not done:
    json.dump(meta, open(meta_path, "w"), indent=2)

def make_hook(alpha):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * vec.view(1, 1, -1)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook

def generate(prompt, alpha, item_seed=None):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt").to(model.device)
    handle = None
    if alpha != 0:

        handle = decoder_layers(model)[layer - 1].register_forward_hook(make_hook(alpha))
    sampling = {}
    if do_sample:
        torch.manual_seed(item_seed)
        torch.cuda.manual_seed_all(item_seed)
        sampling = dict(do_sample=True, temperature=args.temperature, top_p=1.0)
    try:
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 **(sampling or dict(do_sample=False)))
    finally:
        if handle:
            handle.remove()
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
            seed = zlib.crc32(f"{qid}|{name}|{m:g}|{args.sample_seed}".encode()) if do_sample else None
            text, n_tok, trunc = generate(mmlu_prompt(row), alpha, seed)
            f.write(json.dumps(dict(
                question_id=qid, category=row["category"], direction=name,
                alpha_multiplier=m, alpha=alpha, gold=row["answer"], n_options=len(row["options"]),
                response=text, n_new_tokens=n_tok, truncated=trunc, sample_seed=args.sample_seed)) + "\n")
            f.flush()
        print(f"finished multiplier {m}", file=sys.stderr)
