"""Small helpers shared by the scripts: loading the model, the MMLU prompt,
and the answer parser. Nothing here is specific to one step."""
import re

LETTERS = "ABCDEFGHIJ"
SEED = 42


def load_model(model_path):
    # Gemma 4 E2B is a chat model that also takes images, so it loads this way.
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, processor


def decoder_layers(model):
    # Where the list of transformer layers lives in this model.
    return model.model.language_model.layers


def mmlu_prompt(row):
    # Same wording for every question and every steering strength.
    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return (
        "Answer the following multiple-choice question.\n\n"
        f"{row['question']}\n\n{options}\n\n"
        "Use no more than 150 words for your reasoning. "
        "Then finish with exactly:\nAnswer: X\n\n"
        "where X is the letter of the correct answer."
    )


# Only an explicit "Answer: X" line counts. Anything else is a parse failure,
# and a parse failure is NOT the same thing as a refusal.
ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-J])\b", re.IGNORECASE)
REFUSAL_RE = re.compile(
    r"\bi (can'?t|cannot|won'?t|will not) (help|answer|provide|assist)\b"
    r"|\bi don'?t know\b|\bi refuse\b|\bunable to determine\b", re.IGNORECASE)


def parse_answer(text):
    found = ANSWER_RE.findall(text or "")
    return found[-1].upper() if found else None


def outcome(text, gold, truncated):
    """One answer -> separate flags (so we never mix them up)."""
    pred = parse_answer(text)
    refusal = pred is None and bool(REFUSAL_RE.search(text or ""))
    return {
        "correct": pred == gold,
        "refusal": refusal,
        "parse_failure": pred is None and not refusal,
        "truncated": bool(truncated),
    }
