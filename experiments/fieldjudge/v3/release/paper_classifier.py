"""paper-classifier: give a scientific paper and ANY list of options (a classification at any level, your own labels,
or a single topic), get a probability for every option. One forward pass, no generated text.

    from paper_classifier import PaperClassifier, paper_text
    clf = PaperClassifier("maximerivest/paper-classifier-2B")          # or a local folder
    paper = paper_text("Habitat fragmentation reduces bird diversity", "We surveyed birds in 40 forest fragments...")
    clf.classify(paper, ["Ecology", "Ornithology", "Particle physics"], scheme="My fields")   # {option: probability}
    clf.is_on_topic(paper, "habitat fragmentation")                                         # P(yes)

The prompt below is exactly the one the model was trained on. Requires transformers >= 5.3 and torch.
Apache-2.0.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

SYSTEM = ("You classify scientific papers. You get a paper (its title, and its abstract when available), then a "
          "question. For a list of options, answer with the code of the option that best describes the research "
          "content of the paper, following the classification's own meaning of each option. For a single topic, answer "
          "yes if the topic is a central subject of the paper, else no. On topic: the paper studies the topic itself as "
          "a main subject: the phenomenon, its causes, consequences, mechanisms, measurement, methods, or how it is "
          "managed, in any system or organism. Not on topic: the topic is absent or only mentioned in passing; or the "
          "words mean something else in this paper; or the paper studies a neighbouring subject without studying the "
          "topic itself.")
NONE = "None of these (outside the options listed)"
PAPER_TOKENS = 700            # the paper part was cut to at most 700 tokens in training


def paper_text(title: str, abstract: str | None = None, venue: str | None = None, year: int | str | None = None) -> str:
    """The paper as the model saw it in training: Title / Venue / Year / Abstract, or "(no abstract)"."""
    s = f"Title: {title.strip()}"
    if venue:
        s += f"\nVenue: {venue.strip()}"
    if year:
        s += f"\nYear: {year}"
    abstract = " ".join((abstract or "").split())
    return s + (f"\nAbstract: {abstract[:1500]}" if abstract else "\n(no abstract)")


def _option(o) -> dict:
    return {"name": o} if isinstance(o, str) else dict(o)


def option_line(code: str, o: dict, describe: bool = True) -> str:
    s = f"{code}. {o['name']}"
    extra = [x for x in ((o.get("about") if describe else None), (f"in {o['parent']}" if o.get("parent") else None)) if x]
    return s + (" — " + "; ".join(extra) if extra else "")


def pick_prompt(paper: str, scheme: str, question: str, options: list[dict], codes: list[str], describe: bool = True) -> str:
    lines = "\n".join(option_line(c, o, describe) for c, o in zip(codes, options))
    return (f"{paper}\n\n## Classification: {scheme}\n{question}\nOptions:\n{lines}\n"
            "Answer with the option code only.")


def yesno_prompt(paper: str, topic: str, definition: str | None = None) -> str:
    d = f" (what it covers: {definition})" if definition else ""
    return f"{paper}\n\n## Topic: {topic}{d}\nIs this topic a central subject of the paper? Answer yes or no."


class PaperClassifier:
    def __init__(self, model: str = "maximerivest/paper-classifier-2B", device: str | None = None,
                 dtype=torch.bfloat16):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        local = Path(model) / "classifier.json"
        cfg = json.loads((local if local.exists() else Path(hf_hub_download(model, "classifier.json"))).read_text())
        self.codes, self.code_ids = cfg["codes"], cfg["code_ids"]
        self.yes, self.no = cfg["yes"], cfg["no"]

    # ------------------------------------------------------------------ low level
    def _ids(self, user: str) -> list[int]:
        paper, sep, rest = user.partition("\n\n## ")
        ids = self.tok(paper, add_special_tokens=False)["input_ids"]
        if len(ids) > PAPER_TOKENS:                          # cut the paper, never the question
            user = self.tok.decode(ids[:PAPER_TOKENS]) + sep + rest
        text = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def scores(self, users: list[str], answer_ids: list[list[int]], batch_size: int = 8) -> list[list[float]]:
        """For each prompt, the softmax over ITS answer tokens of the next-token logits (no generation)."""
        out = []
        for i in range(0, len(users), batch_size):
            enc = self.tok.pad({"input_ids": [self._ids(u) for u in users[i:i + batch_size]]}, return_tensors="pt")
            logits = self.model(**enc.to(self.device), logits_to_keep=1).logits[:, -1, :].float()
            for row, ans in zip(logits, answer_ids[i:i + batch_size]):
                out.append(torch.softmax(row[ans], -1).tolist())
        return out

    # ------------------------------------------------------------------ tasks
    def classify(self, paper: str, options: list, scheme: str = "Classification of research",
                 question: str = "Which option best describes the research content of this paper?",
                 describe: bool = True, orders: int = 1, none_option: bool = False, seed: int = 0) -> dict:
        """{option name: probability}. options: names, or dicts {"name", "about" (optional), "parent" (optional)}.
        orders > 1: also ask with the options shuffled and average (steadier picks, `orders` x the cost)."""
        opts = [_option(o) for o in options] + ([{"name": NONE}] if none_option else [])
        if len(opts) > len(self.codes):
            raise ValueError(f"at most {len(self.codes)} options in one question: ask level by level (classify_tree)")
        rng = random.Random(seed)
        perms = [list(range(len(opts)))] + [rng.sample(range(len(opts)), len(opts)) for _ in range(orders - 1)]
        codes = self.codes[:len(opts)]
        users = [pick_prompt(paper, scheme, question, [opts[j] for j in p], codes, describe) for p in perms]
        res = self.scores(users, [self.code_ids[:len(opts)]] * len(users))
        total = [0.0] * len(opts)
        for p, r in zip(perms, res):
            for pos, j in enumerate(p):
                total[j] += r[pos] / len(perms)
        return dict(sorted(((o["name"], q) for o, q in zip(opts, total)), key=lambda x: -x[1]))

    def is_on_topic(self, paper: str, topic: str, definition: str | None = None) -> float:
        """P(the topic is a central subject of the paper)."""
        return self.scores([yesno_prompt(paper, topic, definition)], [[self.yes, self.no]])[0][0]

    def classify_tree(self, paper: str, tree: dict[str, list], scheme: str = "Classification of research",
                      levels: tuple[str, str] = ("field", "subfield")) -> dict:
        """Two levels: P(child) = P(parent) x P(child | parent). tree = {parent: [children]}.
        For lists that fit in one question (<= 562 options), one flat classify() is usually better."""
        top = self.classify(paper, list(tree), scheme, f"Which {levels[0]} best describes the research content of this paper?")
        out = {}
        for parent, children in tree.items():
            sub = self.classify(paper, children, f"{scheme} — within: {parent}",
                                f"Which {levels[1]} best describes the research content of this paper?")
            for c, q in sub.items():
                out[c] = top[parent] * q
        return dict(sorted(out.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    import sys
    clf = PaperClassifier(sys.argv[1] if len(sys.argv) > 1 else "maximerivest/paper-classifier-2B")
    p = paper_text("Habitat fragmentation reduces bird diversity in tropical forests",
                   "We surveyed bird communities in 40 forest fragments of different sizes and isolation in the Atlantic "
                   "Forest of Brazil. Species richness declined with fragment area and increased with connectivity.")
    print(clf.classify(p, ["Ecology", "Ornithology", "Particle physics", "Economics"], scheme="Fields of science"))
    print(clf.is_on_topic(p, "habitat fragmentation"), clf.is_on_topic(p, "dark matter"))
