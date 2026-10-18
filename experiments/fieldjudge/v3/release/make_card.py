"""Writes README.md (the model card) into both release folders."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]
NUM = {
    "0.8B": dict(other="2B", disc="64%", multi="53%", top3="87%", field="80% / 70%", jel="80.5%", inv="78.4%", pap="86.4%",
                 ap="0.73", ap_before="0.68", ece="0.076", shuffle="74%", per_hour="60,000", one="0.065 s", yes_one="0.028 s",
                 yes_hour="330,000", usd="$4", params="0.8B", vram="about 3 GB", steps="2,694", example="{'Ecology': 0.58, 'Ornithology': 0.41, 'Particle physics': 0.0006, 'Economics': 0.0006}"),
    "2B": dict(other="0.8B", disc="62%", multi="61%", top3="90%", field="79% / 77%", jel="82.2%", inv="79.9%", pap="87.2%",
               ap="0.80", ap_before="0.75", ece="0.053", shuffle="76%", per_hour="28,000", one="0.14 s", yes_one="0.035 s",
               yes_hour="155,000", usd="$9", params="2B", vram="about 6 GB", steps="2,694", example="{'Ornithology': 0.65, 'Ecology': 0.35, 'Economics': 0.0004, 'Particle physics': 0.0001}"),
}

CARD = '''---
license: apache-2.0
base_model: Qwen/Qwen3.5-{size}
language:
- en
library_name: transformers
pipeline_tag: zero-shot-classification
tags:
- scientific-papers
- classification
- zero-shot-classification
- field-of-science
- taxonomy
- research-classification
- science-of-science
---

# paper-classifier-{size}

**Give it a scientific paper and any list of options: it returns a probability for every option.** The options can be
any classification of research, at any level (OpenAlex fields, arXiv categories, MeSH, your own labels, the groups of
a map you made), or a single topic: "is this topic a central subject of the paper?" (yes / no).

- **No text is generated.** One forward pass; the answer is read from the next-token logits of the option codes.
- **Fast and cheap:** about {per_hour} papers an hour on one RTX 3090 for a 176-option question ({usd} per million papers at
  $0.25 an hour), {one} for one paper on its own; yes/no questions {yes_one} ({yes_hour} an hour).
- **Works on classifications it never saw:** tested on a held-out expert benchmark and on held-out schemes (below).
- {params} parameters, fine-tuned from [Qwen/Qwen3.5-{size}](https://huggingface.co/Qwen/Qwen3.5-{size}).
  Sibling: [paper-classifier-{other}](https://huggingface.co/maximerivest/paper-classifier-{other})
  ({sibling_note}).
- Made for [Science Made Readable](https://sciencemadereadable.com). Apache 2.0.

## Quick start

```bash
pip install "transformers>=5.3" torch huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import importlib.util, sys
spec = importlib.util.spec_from_file_location("paper_classifier",
          hf_hub_download("maximerivest/paper-classifier-{size}", "paper_classifier.py"))
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)

clf = pc.PaperClassifier("maximerivest/paper-classifier-{size}")
paper = pc.paper_text(
    title="Habitat fragmentation reduces bird diversity in tropical forests",
    abstract="We surveyed bird communities in 40 forest fragments ...",
    venue="Biological Conservation", year=2019)          # venue and year are optional

clf.classify(paper, ["Ecology", "Ornithology", "Particle physics", "Economics"], scheme="Fields of science")
# {example}

clf.is_on_topic(paper, "habitat fragmentation")   # 1.0
clf.is_on_topic(paper, "dark matter")             # 0.0
```

`paper_classifier.py` (in this repo, standard transformers + torch) builds exactly the prompt the model was trained
on. Use it, or copy its ~40 lines of prompt code, rather than writing your own prompt.

## The two tasks, as functions

```
classify(paper, options, scheme, question) -> {{option: probability}}     # probabilities sum to 1
    paper     str   "Title: ...\\nVenue: ...\\nYear: ...\\nAbstract: ..."   (paper_text(); "(no abstract)" if none)
    options   list  names, or {{"name", "about" (optional), "parent" (optional)}}; 2 to 562 of them
    scheme    str   name of the classification, e.g. "OpenAlex topic classification" or "My lab's reading groups"
    question  str   e.g. "Which subfield best describes the research content of this paper?"

is_on_topic(paper, topic, definition=None) -> P(yes)
    topic       str  e.g. "habitat fragmentation"
    definition  str  optional: what the topic covers, to settle ambiguous words
```

### Exactly what the model sees

Every question uses the same system message and the Qwen3.5 chat template **with thinking off**
(`enable_thinking=False`, so the assistant turn starts with an empty `<think>` block). The paper comes first, the
question after.

System message (verbatim, also `SYSTEM` in `paper_classifier.py`):

```
{system}
```

User message, list of options (here 3 of the 176 Science-Metrix subfields; each option gets a two-letter code):

```
Title: Combined androgen excess and Western-style diet accelerates adipose tissue dysfunction ...
Venue: Human Reproduction
Year: 2017
Abstract: STUDY QUESTION: What are the separate and combined effects of ...

## Classification: Science-Metrix classification of science (176 subfields within 22 fields and 6 domains)
Which subfield best describes the research content of this paper?
Options:
AA. Agronomy & Agriculture — in Agriculture, Fisheries & Forestry
AB. Dairy & Animal Science — in Agriculture, Fisheries & Forestry
AC. Fisheries — in Agriculture, Fisheries & Forestry
Answer with the option code only.
```

An option line is `CODE. name`, then optionally ` — ` followed by its description and/or `in <parent>`, joined by
`; `. Codes are `AA, AB, AC, ...` in the order listed: the 562 two-letter codes that are single tokens
(`classifier.json` → `codes`, with their token ids in `code_ids`).

User message, one topic:

```
Title: ...
Abstract: ...

## Topic: habitat fragmentation
Is this topic a central subject of the paper? Answer yes or no.
```

(with a definition: `## Topic: habitat fragmentation (what it covers: ...)`)

### Reading the answer (no generation)

Run the prompt once and take the logits at the next position. Keep only the logits of the codes you offered
(`code_ids[:len(options)]`; for yes/no the `yes` and `no` ids in `classifier.json`) and apply a softmax over those.
That is the distribution the model was trained to match. Do not sample or parse generated text.

```python
logits = model(**inputs, logits_to_keep=1).logits[0, -1]        # transformers
probs = logits[code_ids[:len(options)]].float().softmax(-1)
```

## Tips for the best results (closest to training)

- **Use `paper_text()`**: title, venue and year when known, then the abstract (the first 1,500 characters; the
  paper part was cut to 700 tokens in training). Title-only works too (`(no abstract)`), a little less well.
- **Name the classification** in `scheme`, and ask about the level by its name: "Which *field* ...", "Which
  *topic* ...". The model follows the scheme's own meaning of each option.
- **Short descriptions help** when option names are ambiguous (`about`). For a step down a hierarchy, put the
  parent in the scheme: `scheme="OpenAlex topic classification — within: Ecology"` (what `classify_tree` does), or
  give each option its parent (`parent`).
- **Up to 562 options in one question.** Bigger schemes: ask level by level (`classify_tree`). For lists that fit,
  one flat question was better than going level by level on the expert benchmark.
- **The order of the options matters somewhat** (shuffling them keeps the same top pick {shuffle} of the time).
  For important decisions use `classify(..., orders=3)`: it averages 3 orders, at 3x the cost.
- **"None of these"**: `none_option=True` adds the option the model was trained with for papers outside the list.
- Probabilities are fairly well calibrated (expected calibration error {ece} on the expert benchmark): use them,
  not just the top pick.

## Fast bulk scoring with vLLM

vLLM's text generation path reads at most 128 token probabilities per request and re-reads the prompt for more.
`vllm_fast.py` in this repo uses vLLM's pooling runner instead: it returns the hidden state at the answer position and
computes the option scores with the model's output layer, one request per question, any number of options. Same
results as transformers (same top pick, probabilities within about 0.02). Tested with vLLM 0.29 on an RTX 3090:
throughput was the same from 16 to 256 questions at a time (the GPU is busy computing the prompt), CUDA graphs on
gave +13%.

## Results

### Expert benchmark: 176 subfields of science, a classification it never trained on

200 articles, each classified by 5 or 6 experts into the 176 Science-Metrix subfields (Rivest, Vignola-Gagné &
Archambault 2021, PLOS ONE, [doi:10.1371/journal.pone.0251493](https://doi.org/10.1371/journal.pone.0251493),
"Benchmark 2"). Score: agreement with the experts' most common first choice. The Science-Metrix scheme and these
articles were excluded from training (checked: no training question used the scheme; none of the 200 titles is in
any training data). One question per article, with title, venue, year and abstract.

| | specialist journals | general journals | $ per 1,000 papers |
|---|---:|---:|---:|
| paper-classifier-0.8B | 64% | 53% | $0.004 |
| paper-classifier-2B | 62% | 61% | $0.009 |
| paper-classifier-2B, 3 option orders | 68% | 61% | $0.027 |
| Qwen3.8-27B (direct) | 57% | 64% | $0.022 |
| Jev (TypeSafe), flat | 64% | 61% | $0.19 |
| Jev (TypeSafe), field then subfield | 68% | 65% | $0.21 |
| GPT-6.1 Sol, 20 papers a request | 66% | 69% | $1.40 |
| one human expert vs the others | 63% | 57% | ~$550 (reading the abstract at $50/h) |

![Best-value chart](benchmark.png)

Each number has about ±10 points of uncertainty (100 articles per half). This model ({size}): right subfield in its
top 3 for {top3} of articles; right broad field (of 22) from its subfield pick {field} (specialist / general).
Costs of local models: one RTX 3090 at $0.25 an hour (Vast.ai and RunPod, 2026-10-18).

### Classifications it never saw, against the teachers

Held-out schemes and papers, labelled by GPT-6.1 Sol (the two teachers agree with each other 90-92% of the time,
the practical ceiling):

| | same top pick as Sol |
|---|---:|
| JEL economics codes (held-out scheme) | {jel} |
| 18 invented classifications (held out) | {inv} |
| known classifications, held-out papers | {pap} |

### Topic yes / no

8 topics held out of all training (4,800 papers, a Kimi K3 reference): macro average precision {ap} (the round-2
judge it started from: {ap_before}). Training on lists improved the yes/no skill rather than eroding it.

## How it was trained

1. **Round 2 (start):** Qwen3.5-{size} fine-tuned (LoRA) as a yes/no judge of "is this topic a central subject of
   this paper", on about 85,000 topic-paper pairs labelled by GPT-6.1 Sol and Claude Opus 5.5.
2. **Round 3 (this model):** from the round-2 judge, a new LoRA (rank 32, all linear layers, merged) and the output
   rows of the answer tokens, {steps} steps of 32, on
   - 46,179 classification questions: 2,500 lists of options from 321 schemes (OpenAlex at four levels, arXiv, ACM
     CCS, MSC 2020, MeSH, ANZSRC FoR, OECD FORD, 28 maps of research fields with named groups, and 284 schemes invented
     for the purpose: by method, organism, application, study design, ...), as whole levels, subsets or the children
     of one parent, some with a "none of these" option, 20 papers each. Labels: the share of experts who would pick
     each option (up to 3), from GPT-6.1 Sol and Claude Opus 5.5, each on different schemes and papers.
   - 40,000 yes/no examples replayed from round 2.
   Loss: cross-entropy between the teacher's shares and the softmax over the offered codes. Every time a question was
   shown, its options were shuffled (80%), given a random set of codes (50%) and stripped of descriptions (30%).
3. Papers come from OpenAlex (titles, abstracts, venues, years); 15% were shown title-only.

## Limitations

- **English** titles and abstracts; other languages were not tested.
- **Order sensitivity** (see tips): average a few orders when it matters.
- **Very broad, oddly named levels** are its weak spot: asked directly for one of 22 broad fields, it was right less
  often than when the field was read off its subfield pick. Prefer the finest level you have and add up.
- Labels came from language models, not people, except the expert benchmark; the model inherits their views of where
  one field ends and another begins.
- It judges what a paper is *about* from its title and abstract, not its quality or correctness.

## License

Apache 2.0, like the base model (Qwen3.5, Apache 2.0, see `LICENSE`).
'''

SIB = {"0.8B": "slightly better on general journals and yes/no, half the speed",
       "2B": "50% faster, a little less accurate"}
import sys
sys.path.insert(0, str(R / "common"))
from paper_classifier import SYSTEM  # noqa: E402

for size, n in NUM.items():
    sib = SIB["0.8B" if size == "0.8B" else "2B"]
    sib = "slightly more accurate, half the speed" if size == "0.8B" else "about twice as fast, a little less accurate"
    text = CARD.format(size=size, system=SYSTEM, sibling_note=sib, **n)
    (R / f"paper-classifier-{size}" / "README.md").write_text(text)
    import shutil
    shutil.copy(R / "common" / "benchmark.png", R / f"paper-classifier-{size}" / "benchmark.png")
    print(size, len(text), "characters")
