"""The round-3 student's question format, shared by training (train3.py) and use (vllm_classify.py).

The paper comes first, the question after: many questions about one paper (each level of a hierarchy, several
schemes) reuse the paper's cached computation. The answer is ONE token, read without generating:
  - a list of options: each option has a two-letter code that is a single token (codes.json, 562 of them); the
    distribution is the softmax of the codes' logits, restricted to the offered codes
  - one topic: "yes" or "no" (a two-option list with fixed codes)
Standard library only (vLLM's Python imports it too).
"""
import json
import random
import re
from pathlib import Path

CODES = json.loads(Path(__file__).with_name("codes.json").read_text())
SYSTEM = ("You classify scientific papers. You get a paper (its title, and its abstract when available), then a "
          "question. For a list of options, answer with the code of the option that best describes the research "
          "content of the paper, following the classification's own meaning of each option. For a single topic, answer "
          "yes if the topic is a central subject of the paper, else no. On topic: the paper studies the topic itself as "
          "a main subject: the phenomenon, its causes, consequences, mechanisms, measurement, methods, or how it is "
          "managed, in any system or organism. Not on topic: the topic is absent or only mentioned in passing; or the "
          "words mean something else in this paper; or the paper studies a neighbouring subject without studying the "
          "topic itself.")
NONE = "None of these (outside the options listed)"


def option_line(code, o, describe=True):
    s = f"{code}. {o['name']}"
    extra = [x for x in ((o.get("about") if describe else None), (f"in {o['parent']}" if o.get("parent") else None)) if x]
    return s + (" — " + "; ".join(extra) if extra else "")


def pick_prompt(paper, scheme, question, options, codes, describe=True):
    lines = "\n".join(option_line(c, o, describe) for c, o in zip(codes, options))
    return (f"{paper}\n\n## Classification: {scheme}\n{question}\nOptions:\n{lines}\n"
            "Answer with the option code only.")


def yesno_prompt(paper, topic, definition=None):
    d = f" (what it covers: {definition})" if definition else ""
    return f"{paper}\n\n## Topic: {topic}{d}\nIs this topic a central subject of the paper? Answer yes or no."


def messages(user):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def assign_codes(k, rng: random.Random | None = None):
    """Inference: the first k codes (AA, AB, ...). Training: half the time a random sorted subset, so the model
    reads codes instead of counting positions."""
    if k > len(CODES):
        raise ValueError(f"at most {len(CODES)} options in one question; split by level")
    if rng is None or rng.random() < 0.5:
        return CODES[:k]
    return sorted(rng.sample(CODES, k))


def clean_paper(text, max_chars=2200):
    return re.sub(r"[ \t]+", " ", text).strip()[:max_chars]
