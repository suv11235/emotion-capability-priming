import re

LETTERS = "ABCDEFGHIJ"
SEED = 42

def load_model(model_path):

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, processor

def decoder_layers(model):

    return model.model.language_model.layers

def story_activations(model, processor, texts, start_token=50):

    import numpy as np
    import torch
    out = []
    for t in texts:
        inp = processor(text=t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.inference_mode():
            hs = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
        n = inp["input_ids"].shape[1]
        s = min(start_token, n // 2)
        out.append(torch.stack([h[0, s:n].float().mean(0).cpu() for h in hs]).numpy())
    return np.stack(out)

def build_direction(pos, neg, neutral, pca=0.50):

    import numpy as np
    from sklearn.decomposition import PCA
    raw = pos.mean(0) - neg.mean(0)
    directions = np.zeros_like(raw)
    for L in range(raw.shape[0]):
        v = raw[L].copy()
        if pca > 0:

            p = PCA().fit(neutral[:, L])
            k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), pca) + 1)
            pcs = p.components_[:k]
            v = v - pcs.T @ (pcs @ v)
        directions[L] = v / (np.linalg.norm(v) + 1e-8)
    return directions

def auc(pos, neg, direction):

    import numpy as np
    from sklearn.metrics import roc_auc_score
    a, b = pos @ direction, neg @ direction
    return roc_auc_score([1] * len(a) + [0] * len(b), np.r_[a, b])

def mcnemar_p(b, c):

    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def mmlu_prompt(row):

    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return (
        "Answer the following multiple-choice question.\n\n"
        f"{row['question']}\n\n{options}\n\n"
        "Use no more than 150 words for your reasoning. "
        "Then finish with exactly:\nAnswer: X\n\n"
        "where X is the letter of the correct answer."
    )

ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-J])\b", re.IGNORECASE)
REFUSAL_RE = re.compile(
    r"\bi (can'?t|cannot|won'?t|will not) (help|answer|provide|assist)\b"
    r"|\bi don'?t know\b|\bi refuse\b|\bunable to determine\b", re.IGNORECASE)

def parse_answer(text):
    found = ANSWER_RE.findall(text or "")
    return found[-1].upper() if found else None

def outcome(text, gold, truncated):

    pred = parse_answer(text)
    refusal = pred is None and bool(REFUSAL_RE.search(text or ""))
    return {
        "correct": pred == gold,
        "refusal": refusal,
        "parse_failure": pred is None and not refusal,
        "truncated": bool(truncated),
    }
