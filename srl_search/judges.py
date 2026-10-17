"""Judges of "is this paper about the topic?" for whole-field maps. Nothing is trained on Jev's answers (TypeSafe's
terms): Jev only checks papers at the end (`Jev`). Measured on habitat fragmentation (experiments/topicmap/README.md, "Judging papers without Jev"):

  teacher   Kimi K3 (open weights; its providers' terms allow training on its answers), ~375 papers for <= $0.50
  decider   Qwen3.5-0.8B with a yes/no head (Perplexity's pplx-decider recipe), LoRA, 75 steps on the teacher's
            labels (~2.5 min); then it labels ~10,000 papers in vLLM (~2 min). AUC 0.961 against the teacher.
  students  logistic regression on the papers' meaning vectors (instant, any number of papers; AUC 0.950) and
            Ettin-17M on their text (3,600 papers/s; AUC 0.951; the only one that can judge papers with a title
            only), both trained on the teacher's and the decider's labels.

GPU work runs in subprocesses (`python -m srl_search.judges ...`; vLLM from SRL_VLLM_PYTHON), one at a time, on the
card SRL_FIELDS_GPU (PCI order, default 1), only when enough of its memory is free: the household services on that
card (dictation, speech, search, OCR) are never stopped, and a job waits while /run/gpu1-reserved exists (Figurer
training). Data goes through JSONL files in the job's folder.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

GPU = os.environ.get("SRL_FIELDS_GPU", "1")
RESERVED = Path("/run/gpu1-reserved")
DECIDER_BASE = "Qwen/Qwen3.5-0.8B"
STUDENT_BASE = "jhu-clsp/ettin-encoder-17m"
LABELS = ("on topic", "not on topic")
# name: url, model, extras that turn thinking off, USD per 1M tokens (input, output) (tiny-classifiers' label.py)
CAPS = {"fireworks": 512, "together": 320, "deepinfra": 32, "parasail": 256, "openrouter": 256}   # in flight
KIMI = {
    "fireworks": ("https://api.fireworks.ai/inference/v1/chat/completions", "accounts/fireworks/models/kimi-k3",
                  {"reasoning_effort": "none"}, (3.0, 15.0), "FIREWORKS_API_KEY"),
    "together": ("https://api.together.ai/v1/chat/completions", "moonshotai/Kimi-K3",
                 {"reasoning": {"enabled": False}}, (3.0, 15.0), "TOGETHER_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai/chat/completions", "moonshotai/Kimi-K3",
                  {"reasoning": {"enabled": False}}, (2.85, 14.25), "DEEPINFRA_API_KEY"),
    "parasail": ("https://api.parasail.io/v1/chat/completions", "parasail-kimi-k3",
                 {"reasoning_effort": "none"}, (3.0, 15.0), "PARASAIL_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "moonshotai/kimi-k3",
                   {"reasoning": {"enabled": False}}, (3.0, 15.0), "OPENROUTER_API_KEY"),
}


def instruction(topic: str, definition: str | None) -> str:
    d = f" (what it covers: {definition})" if definition else ""
    return (f"You judge scientific papers for a map of one research topic: {topic}{d}. Is {topic} a central subject of "
            "the paper below (its title and abstract, or its title alone)? On topic: the paper studies the topic itself "
            "as a main subject: the phenomenon, its causes, consequences, mechanisms, measurement, methods, or how it "
            "is managed, in any system or organism. Not on topic: the topic is absent or only mentioned in passing; or "
            "the words mean something else in this paper; or the paper studies a neighbouring subject without "
            "studying the topic itself.")


def paper_text(title: str | None, abstract: str | None) -> str:
    t = re.sub(r"<[^>]+>", "", title or "").strip()
    a = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", abstract or "")).strip()
    return f"Title: {t}\nAbstract: {a[:1500]}" if a else f"Title: {t}\n(no abstract)"


# ------------------------------------------------------------------------------------------------ the teacher
class _Retry(Exception):
    pass


class Teacher:
    """Kimi K3 through whichever providers have a key; stops before `budget_usd`."""

    def __init__(self, budget_usd: float):
        self.providers = [(n, *v) for n, v in KIMI.items() if os.environ.get(v[4])]
        if not self.providers:
            raise RuntimeError("no Kimi K3 provider key in the environment")
        self.budget = budget_usd
        self.usd = 0.0
        self.lock = threading.Lock()
        self.failed = 0
        self.dead: set[str] = set()
        self.limited: dict[str, int] = {}
        self.slots = {n: threading.BoundedSemaphore(CAPS.get(n, 32)) for n, *_ in self.providers}
        self.slowest: dict[str, float] = {}           # seconds, per provider: one stuck call holds up a whole label()

    def _one(self, i: int, system: str, text: str) -> float:
        """One answer: providers in turn from this paper's own offset; a full provider is skipped, a failed call is
        retried elsewhere. Gives up after 6 failed calls, or 2 minutes without a free slot."""
        calls, t0, k = 0, time.time(), i
        while calls < 6 and time.time() - t0 < 120:
            live = [p for p in self.providers if p[0] not in self.dead]
            if not live:
                break
            name, url, model, extra, price, key = live[k % len(live)]
            k += 1
            if not self.slots[name].acquire(timeout=0.2):
                continue
            body = json.dumps({"model": model, "temperature": 0, "max_tokens": 8, **extra, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": text}]}).encode()
            try:
                calls += 1
                return self._call(name, url, body, price, key)
            except _Retry:
                continue
            finally:
                self.slots[name].release()
        with self.lock:
            self.failed += 1
        return float("nan")

    def _call(self, name, url, body, price, key) -> float:
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {os.environ[key]}", "Content-Type": "application/json",
                "User-Agent": "scholarsreadinglist-fields/1"})              # Together's Cloudflare refuses urllib's
            # answers take 1-6 s (8 output tokens): a call still waiting at 20 s is stuck; another provider is asked
            with urllib.request.urlopen(req, timeout=20) as r:
                out = json.load(r)
            u = out.get("usage") or {}
            with self.lock:
                self.slowest[name] = round(max(self.slowest.get(name, 0.0), time.time() - t0), 1)
                self.usd += (u.get("prompt_tokens", 0) * price[0] + u.get("completion_tokens", 0) * price[1]) / 1e6
            ans = (out["choices"][0]["message"].get("content") or "").strip().lower().strip(".\"'")
            if ans.startswith("not on topic"):
                return 0.0
            if ans.startswith("on topic"):
                return 1.0
            raise _Retry()                                # not one of the labels: ask again elsewhere
        except urllib.error.HTTPError as e:
            with self.lock:
                if e.code == 429:
                    self.limited[name] = self.limited.get(name, 0) + 1
                if e.code in (401, 402, 403) or self.limited.get(name, 0) >= 10:
                    self.dead.add(name)                   # refused, no credit, or always rate-limited: not this run
            if e.code not in (401, 402, 403, 429):
                time.sleep(1)
            raise _Retry()
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
            time.sleep(1)
            raise _Retry()

    def label(self, texts: list[str], topic: str, definition: str | None) -> np.ndarray:
        system = (instruction(topic, definition) + " Answer with exactly one label from this list and nothing else:\n"
                  + "\n".join(f"- {l}" for l in LABELS))
        out = np.full(len(texts), np.nan)

        def go(i):
            if self.usd >= self.budget:
                return
            out[i] = self._one(i, system, texts[i])
        with ThreadPoolExecutor(256) as ex:              # answers take 1-4 s: many in flight (I/O only)
            list(ex.map(go, range(len(texts))))
        return out


# ------------------------------------------------------------------------------------------------ the checker
JEV_MODEL = "jev-1.13.0"                 # pinned: the answers' meaning must not move under a running job
JEV_USD_PER_TOKEN = 0.042 / 1e6          # input tokens; output is free (docs.typesafe.ai/models, 2026-10)


class Jev:
    """TypeSafe's Jev as a checker: it gives final answers for papers readers will see (the field's most cited near
    the cut-off). Its answers decide those papers and are kept in field.json; no model is ever trained on them."""

    def __init__(self, budget_usd: float, threads: int = 192):
        self.key = os.environ.get("TYPESAFE_API_KEY")
        if not self.key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.budget, self.threads = budget_usd, threads
        self.tokens = self.calls = self.failed = 0
        self.lock = threading.Lock()

    @staticmethod
    def question(topic: str, definition: str | None) -> dict:
        ins = {"topic": topic}
        if definition:
            ins["definition"] = definition
        ins["question"] = ("Is `topic` a central subject of the paper described in `paper`?"
                           + (" `definition` says what the topic covers." if definition else ""))
        return {"type": "noul", "instructions": ins, "criteria": {
            "true": "The paper studies the topic itself as a main subject: the phenomenon, its causes, consequences, "
                    "mechanisms, measurement, methods, or how it is managed, in any system or organism.",
            "false": "The topic is absent or only mentioned in passing; or the words mean something else in this "
                     "paper; or the paper studies a neighbouring subject without studying the topic itself."}}

    def _one(self, paper: dict, question: dict) -> float:
        if self.usd >= self.budget:
            return float("nan")
        body = json.dumps({"state": {"paper": paper}, "model": JEV_MODEL, "questions": {"q": question}}).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=body, headers={
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    out = json.load(r)
                with self.lock:
                    self.tokens += int(out.get("usage", {}).get("input_tokens", 0))
                    self.calls += 1
                return float(out["answers"]["q"]["noul"])
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(min(float(e.headers.get("retry-after") or 0) or 2 ** attempt, 30))
            except (OSError, ValueError, KeyError):
                time.sleep(2 ** attempt)
        with self.lock:
            self.failed += 1
        return float("nan")

    def label(self, papers: list[tuple[str, str]], topic: str, definition: str | None) -> np.ndarray:
        """P(on topic) for each (title, abstract); NaN when the service did not answer or the budget is spent."""
        q = self.question(topic, definition)
        clean = [{"title": re.sub(r"<[^>]+>", "", t or ""), "abstract": re.sub(r"<[^>]+>", " ", a or "")[:3000]}
                 for t, a in papers]
        with ThreadPoolExecutor(self.threads) as ex:
            return np.array(list(ex.map(lambda p: self._one(p, q), clean)), dtype=np.float64)

    @property
    def usd(self) -> float:
        return self.tokens * JEV_USD_PER_TOKEN


# ------------------------------------------------------------------------------------------------ the GPU
def gpu_free_mb() -> int:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30).stdout
    for line in q.splitlines():
        i, free = [x.strip() for x in line.split(",")]
        if i == GPU:
            return int(free)
    return 0


def wait_for_gpu(need_mb: int, say=lambda s: None, timeout_s: int = 7200) -> None:
    """Wait (saying why) until the card has `need_mb` free and is not reserved; never stops anything on it."""
    t0 = time.time()
    while True:
        reserved = RESERVED.exists() and GPU == "1"
        free = gpu_free_mb()
        if not reserved and free >= need_mb:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"GPU {GPU} busy for {timeout_s // 60} min")
        say("waiting for the GPU (reserved for training)" if reserved else
            f"waiting for {need_mb // 1024} GB free on the GPU ({free // 1024} GB now)")
        time.sleep(30)


def gpu_env() -> dict:
    return {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU,
            "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TRITON_LIBCUDA_PATH": "/run/opengl-driver/lib"}


def run(args: list[str], log: Path, env: dict | None = None) -> None:
    with open(log, "a") as f:
        r = subprocess.run(args, env={**gpu_env(), **(env or {})}, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = log.read_text()[-1500:]
        raise RuntimeError(f"{args[1:3]} failed ({r.returncode}): {tail}")


def train_decider(rows: list[tuple[str, float]], topic: str, definition: str | None, out: Path, say) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps({"text": t, "y": y}) + "\n" for t, y in rows))
    (out / "task.json").write_text(json.dumps({"system": instruction(topic, definition) +
                                               ' Answer with one word: "yes" (on topic) or "no" (not on topic).'}))
    wait_for_gpu(8000, say)                     # it saves memory by itself when less than 17 GB is free
    run([sys.executable, "-m", "srl_search.judges", "decider-train", str(out)], out / "log.txt")


def decider_scores(model_dir: Path, texts: list[str], say) -> np.ndarray:
    f = model_dir / "score.jsonl"
    f.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    vpy = os.environ.get("SRL_VLLM_PYTHON") or "python"
    args = [vpy, str(Path(__file__).with_name("vllm_decider.py")), str(model_dir / "merged"), str(f),
            str(model_dir / "scores.json"), GPU]
    wait_for_gpu(10000, say)
    try:
        run(args, model_dir / "log.txt")
    except RuntimeError:                        # a household service grew under it: once more, smaller
        say("the GPU filled up; trying again with less memory")
        time.sleep(20)
        wait_for_gpu(8000, say)
        run(args, model_dir / "log.txt", {"SRL_VLLM_UTIL": "0.3"})
    return np.array(json.loads((model_dir / "scores.json").read_text()))


# ------------------------------------------------------------------------------------------------ the general judge
# One model for every topic (experiments/fieldjudge/v2): Qwen3.5-0.8B trained on 85,303 topic-paper answers of
# GPT-6.1 Sol and Claude Opus 5.5 over ~660 topic families; the topic's question goes in each prompt. It replaces
# Kimi K3 (teacher), the per-topic decider and Jev (check): no paid AI per map. Replay on the two test topics
# (2026-10-16): fragmentation reading lists 80% on topic (Kimi+Jev: 80%), LLM post-training 52% (49%); 4-6 points
# fewer of the field's classics. SRL_FIELD_JUDGE=paid brings back Kimi + per-topic decider + Jev.
JUDGE_MODEL = Path(os.environ.get("SRL_FIELD_JUDGE_MODEL", "/mnt/fast/scholarsreadinglist/models/fieldjudge-0.8b-v2"))
JUDGE_ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'
LOCAL = os.environ.get("SRL_FIELD_JUDGE", "local") != "paid" and (JUDGE_MODEL / "model.safetensors").exists()


def _vllm(model: Path, rows_file: Path, out: Path, log: Path, say) -> np.ndarray:
    vpy = os.environ.get("SRL_VLLM_PYTHON") or "python"
    args = [vpy, str(Path(__file__).with_name("vllm_decider.py")), str(model), str(rows_file), str(out), GPU]
    # 9 GB: the household services on GPU 1 hold ~14.4 of 24 GB since 2026-10-16, leaving 9.5-10.3; vLLM then takes
    # ~20% of the card (measured: 3,000 papers in 74 s with start-up). 10 GB made maps wait for hours
    wait_for_gpu(9000, say)
    try:
        run(args, log)
    except RuntimeError:                        # a household service grew under it: once more, smaller
        say("the GPU filled up; trying again with less memory")
        time.sleep(20)
        wait_for_gpu(8000, say)
        run(args, log, {"SRL_VLLM_UTIL": "0.3"})
    return np.array(json.loads(out.read_text()))


class LocalJudge:
    """The general judge with the Teacher/Jev interface: P(on topic) for texts, $0, on GPU 1 (vLLM)."""

    def __init__(self, work: Path, say=lambda s: None):
        self.work, self.say = Path(work), say
        self.usd, self.failed, self.slowest, self.calls = 0.0, 0, {}, 0

    def label(self, texts, topic: str, definition: str | None) -> np.ndarray:
        texts = [paper_text(*t) if isinstance(t, tuple) else t for t in texts]   # Jev's (title, abstract) pairs too
        if not texts:
            return np.zeros(0)
        self.calls += 1
        d = self.work / f"call{self.calls}"
        d.mkdir(parents=True, exist_ok=True)
        system = instruction(topic, definition) + JUDGE_ANSWER
        (d / "score.jsonl").write_text("".join(json.dumps({"text": t, "system": system}) + "\n" for t in texts))
        p = _vllm(JUDGE_MODEL, d / "score.jsonl", d / "scores.json", d / "log.txt", self.say)
        if len(p) != len(texts) or not np.isfinite(p).all():
            raise RuntimeError("the general judge returned an unexpected number of scores")
        return p


def student_scores(train: list[tuple[str, float]], texts: list[str], work: Path, say) -> np.ndarray:
    work.mkdir(parents=True, exist_ok=True)
    (work / "train.jsonl").write_text("".join(json.dumps({"text": t, "y": y}) + "\n" for t, y in train))
    (work / "score.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    wait_for_gpu(4000, say)
    run([sys.executable, "-m", "srl_search.judges", "student", str(work)], work / "log.txt")
    return np.array(json.loads((work / "scores.json").read_text()))


# ------------------------------------------------------------------------------------------------ subprocesses
def _decider_train(d: Path, steps: int = 75) -> None:
    """LoRA on Qwen3.5-0.8B with the model's own yes/no output rows as the head; saves a merged model whose yes/no
    rows (in an untied output layer) are the trained head, for vLLM."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    rng = random.Random(0)
    rows = [json.loads(l) for l in open(d / "train.jsonl")]
    system = json.loads((d / "task.json").read_text())["system"]
    tok = AutoTokenizer.from_pretrained(DECIDER_BASE)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(DECIDER_BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    yes, no = tok.encode("yes", add_special_tokens=False)[0], tok.encode("no", add_special_tokens=False)[0]
    head = torch.nn.Parameter(model.get_output_embeddings().weight[[yes, no]].detach().clone().float())
    free = torch.cuda.mem_get_info()[0]
    # 16 papers a step in any case. With flash-linear-attention's kernels (5x faster than transformers' torch fallback
    # for Qwen3.5's linear-attention layers) 4 papers of 600 tokens need ~7 GB; below 6 GB, recompute activations
    bs = 8 if free >= 14 * 2**30 else 4 if free >= 9 * 2**30 else 2
    if free < 6 * 2**30:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))

    def prompt(t):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def margin(texts):
        # lengths rounded up to 128: the kernels are tuned once per shape (~1-2 s each), not once per batch
        enc = tok([prompt(t) for t in texts], return_tensors="pt", padding=True, truncation=True, max_length=640,
                  pad_to_multiple_of=128, add_special_tokens=False).to("cuda")
        h = model.get_base_model().model(**enc).last_hidden_state[:, -1, :].float()
        lg = h @ head.T
        return lg[:, 0] - lg[:, 1]
    params = [p for p in model.parameters() if p.requires_grad] + [head]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    order: list[int] = []
    for step in range(steps):
        for _ in range(16 // bs):
            if len(order) < bs:
                order += rng.sample(range(len(rows)), len(rows))
            b, order = order[:bs], order[bs:]
            y = torch.tensor([float(rows[i]["y"] >= 0.5) for i in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m = margin([rows[i]["text"] for i in b])
            (F.binary_cross_entropy_with_logits(m.float(), y) / (16 // bs)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    merged = model.merge_and_unload()
    W = merged.get_input_embeddings().weight.detach().clone()
    W[[yes, no]] = head.detach().to(W.dtype)
    merged.lm_head = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype, device=W.device)
    merged.lm_head.weight = torch.nn.Parameter(W)
    merged.config.tie_word_embeddings = False
    if getattr(merged.config, "text_config", None) is not None:
        merged.config.text_config.tie_word_embeddings = False
    merged.save_pretrained(d / "merged", safe_serialization=True)
    tok.save_pretrained(d / "merged")
    json.dump({"system": system, "yes": yes, "no": no}, open(d / "merged" / "decider.json", "w"))


def _student(d: Path, epochs: int = 2) -> None:
    """Ettin-17M, all weights (tiny-classifiers' recipe), trained on the rows (y: P(on topic)), then scores score.jsonl."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(0)
    rng = random.Random(0)
    train = [json.loads(l) for l in open(d / "train.jsonl")]
    score = [json.loads(l)["text"] for l in open(d / "score.jsonl")]
    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    lens = sorted(len(tok(r["text"])["input_ids"]) for r in train[:2000])
    maxlen = min(512, 64 * math.ceil(lens[int(.95 * len(lens))] / 64))
    enc = lambda ts: [tok(t, truncation=True, max_length=maxlen)["input_ids"] for t in ts]
    X = enc([r["text"] for r in train])
    # soft targets: the decider's probabilities as they are (distillation), the teacher's answers as 0/1. No class
    # weights: they shift every probability towards "yes", and most candidates are not on topic
    y = torch.tensor([[float(r["y"]), 1.0 - float(r["y"])] for r in train])
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_BASE, num_labels=2, attn_implementation="sdpa",
                                                               dtype=torch.float32).to("cuda")

    def batches(xs, shuffle, bs):
        order = list(range(len(xs)))
        rng.shuffle(order) if shuffle else order.sort(key=lambda i: len(xs[i]))
        for i in range(0, len(xs), bs):
            b = order[i:i + bs]
            yield b, tok.pad({"input_ids": [xs[j] for j in b]}, return_tensors="pt").to("cuda")
    steps = epochs * math.ceil(len(X) / 32)
    warm = max(1, int(0.05 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    model.train()
    for _ in range(epochs):
        for b, e in batches(X, True, 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(**e).logits.float(), y[b].to("cuda"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    xs = enc(score)
    p = np.zeros(len(xs))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, e in batches(xs, False, 256):
            p[b] = torch.softmax(model(**e).logits.float(), 1)[:, 0].cpu().numpy()
    (d / "scores.json").write_text(json.dumps([round(float(x), 4) for x in p]))


def _knn(d: Path, k: int) -> None:
    """Exact k nearest neighbours by cosine of the unit rows in d/vectors.npy (float16), on the GPU: d/knn.npz."""
    import torch
    V = torch.from_numpy(np.load(d / "vectors.npy")).cuda()
    m = V.shape[0]
    nn = torch.empty((m, k), dtype=torch.int64)
    sim = torch.empty((m, k), dtype=torch.float32)
    block = max(256, min(8192, int(2 * 2**30 // (m * 2))))      # ~2 GB of scores at a time
    with torch.no_grad():
        for s0 in range(0, m, block):
            S = V[s0:s0 + block] @ V.T                             # fp16 inputs, fp32 accumulation
            S[torch.arange(S.shape[0]), torch.arange(s0, s0 + S.shape[0])] = -float("inf")
            v, i = torch.topk(S.float(), k, dim=1)
            nn[s0:s0 + len(i)], sim[s0:s0 + len(i)] = i.cpu(), v.cpu()
    np.savez(d / "knn.npz", nn=nn.numpy(), d=(1 - sim.numpy()).astype(np.float64))


def knn_on_gpu(V: np.ndarray, k: int, work: Path, wait_s: int = 1800, say=lambda s: None) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact neighbours on the field GPU (a 2M-paper field: ~1-2 min), or None when it stays busy for wait_s."""
    work.mkdir(parents=True, exist_ok=True)
    need = int(V.shape[0] * V.shape[1] * 2 / 2**20) + 4000
    t0 = time.time()
    np.save(work / "vectors.npy", V.astype(np.float16))
    try:
        while True:                             # a household service can take the memory between check and start
            try:
                wait_for_gpu(need, say, timeout_s=max(0, int(wait_s - (time.time() - t0))))
            except RuntimeError:
                return None
            try:
                run([sys.executable, "-m", "srl_search.judges", "knn", str(work), str(k)], work / "log.txt")
                z = np.load(work / "knn.npz")
                return z["nn"], z["d"]
            except RuntimeError:
                if time.time() - t0 > wait_s:
                    return None
                say("the GPU filled up; trying the neighbours again in a minute")
                time.sleep(60)
    finally:
        (work / "vectors.npy").unlink(missing_ok=True)
        (work / "knn.npz").unlink(missing_ok=True)


if __name__ == "__main__":
    if sys.argv[1] == "knn":
        _knn(Path(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "decider-train":
        _decider_train(Path(sys.argv[2]))
    elif sys.argv[1] == "student":
        _student(Path(sys.argv[2]))
