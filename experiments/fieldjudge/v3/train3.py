"""Round 3: one model that gives a distribution over ANY option list (any scheme, any level), and yes/no for a topic.

Starts from the round-2 judge (its LoRA merged). Trains a new LoRA (rank 32, all linear layers) and the output rows
of the answer tokens (the 562 option codes, "yes", "no"; round 2 untied the output layer, so this does not touch the
input embeddings). One token is read at the answer position:
  pick rows   cross-entropy between the teacher's shares and the softmax over the OFFERED codes only
  yes/no rows cross-entropy over (yes, no): round 2's examples, replayed so the judge keeps its skill
Every time a pick row is seen it is shown differently: options shuffled (80%), codes a random subset (50%),
descriptions dropped (30%), and the target follows the options, so position and code carry no answer.
Exit 3 yields the GPU to a live map (run3.sh restarts). The export is the final state.

    ../../../.venv/bin/python train3.py --base <round-2 merged> --out <dir>
"""
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2])); sys.path.insert(0, str(HERE.parent / "v2"))
import fmt                                                  # noqa: E402
from train_incremental import wanted as _wanted             # noqa: E402  (live maps first)
wanted = (lambda: False) if os.environ.get("FJ3_GPU") == "0" else _wanted   # live maps use GPU 1 only

DATA = Path(os.environ.get("FJ3_DATA", "/mnt/fast/scholarsreadinglist/fieldjudge/v3/training"))
MAXLEN, BATCH, LR, RANK = 3584, 32, 1e-4, 32


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return [json.loads(l) for l in open(p)]


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    def say(**v):
        v["time"] = time.time(); s = json.dumps(v); print(s, flush=True); log.write(s + "\n"); log.flush()
    train_p, val_p = DATA / "train.jsonl", DATA / "validation.jsonl"
    contract = {"base": a.base, "rank": RANK, "lr": LR, "batch": BATCH, "maxlen": MAXLEN,
                "train_sha256": sha(train_p), "validation_sha256": sha(val_p)}
    cp = out / "checkpoint.pt"
    state = {"cursor": 0, "step": 0, "order": [], "best": None}
    saved = torch.load(cp, map_location="cpu", weights_only=False) if cp.exists() else None
    if saved:
        if saved["contract"] != contract:
            raise ValueError("checkpoint contract differs")
        state = saved["state"]
    if (out / "DONE.json").exists():
        return
    if wanted():
        raise SystemExit(3)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    tid = lambda s: tok.encode(s, add_special_tokens=False)
    answer_ids = [tid(c)[0] for c in fmt.CODES] + [tid("yes")[0], tid("no")[0]]
    assert all(len(tid(c)) == 1 for c in fmt.CODES)
    slot = {c: i for i, c in enumerate(fmt.CODES)}; YES, NO = len(fmt.CODES), len(fmt.CODES) + 1
    head = torch.nn.Parameter(model.get_output_embeddings().weight[answer_ids].detach().clone().float())
    model.config.use_cache = False
    if getattr(model.config, "text_config", None):
        model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=.05, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad] + [head]
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad]},
                             {"params": [head], "lr_scale": 0.5}], lr=LR, weight_decay=0.)
    if saved:
        set_peft_model_state_dict(model, saved["lora"]); head.data.copy_(saved["head"].to("cuda"))
        opt.load_state_dict(saved["optimizer"]); torch.set_rng_state(saved["torch_rng"])
        del saved
    free = torch.cuda.mem_get_info()[0] / 2 ** 30
    budget = int((9000 if "0.8B" in a.base else 5000) * max(0.5, min(float(os.environ.get("FJ3_SCALE_MAX", "1.5")), (free - 2) / 8)))   # tokens per micro-batch

    def checkpoint():
        tmp = cp.with_suffix(".tmp")
        torch.save({"contract": contract, "state": state, "lora": get_peft_model_state_dict(model), "head": head.detach(),
                    "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, cp)

    def render(user):
        return tok(tok.apply_chat_template(fmt.messages(user), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False), add_special_tokens=False)["input_ids"]

    def paper_budget(text, other_tokens):
        ids = tid(text)
        room = max(48, min(700, MAXLEN - other_tokens - 64))
        return tok.decode(ids[:room])

    def example(r, rng):
        """-> (token ids, head slots offered, target over them)"""
        if r["kind"] == "yesno":
            user = fmt.yesno_prompt(paper_budget(r["text"], 200), r["topic"], r.get("definition"))
            return render(user), [YES, NO], [float(r["y"]), 1 - float(r["y"])]
        opts = list(range(len(r["options"])))
        if rng is not None and rng.random() < 0.8:
            rng.shuffle(opts)
        codes = fmt.assign_codes(len(opts), rng)
        options = [r["options"][i] for i in opts]
        describe = r["describe"] and (rng is None or rng.random() >= 0.3)
        user = fmt.pick_prompt("{PAPER}", r["scheme_text"], r["question"], options, codes, describe)
        if len(tid(user)) > MAXLEN - 200 and describe:
            describe = False; user = fmt.pick_prompt("{PAPER}", r["scheme_text"], r["question"], options, codes, False)
        user = user.replace("{PAPER}", paper_budget(r["paper"], len(tid(user)) + 120))
        ids = render(user)[-MAXLEN:]
        target = [r["target"].get(o["node"], 0.0) for o in options]
        return ids, [slot[c] for c in codes], target

    def forward(batch):
        enc = tok.pad({"input_ids": [b[0] for b in batch]}, return_tensors="pt", pad_to_multiple_of=64).to("cuda")
        h = model.get_base_model().model(**enc, use_cache=False).last_hidden_state[:, -1, :].float()
        losses, logps = [], []
        for i, (_, slots, target) in enumerate(batch):
            lp = F.log_softmax(h[i] @ head[slots].T, -1)
            t = torch.tensor(target, device="cuda"); t = t / t.sum()
            losses.append(-(t * lp).sum()); logps.append(lp.detach())
        return torch.stack(losses), logps

    def micro_batches(rows):
        rows = sorted(rows, key=lambda x: len(x[0]))
        out, cur = [], []
        for x in rows:
            if cur and len(x[0]) * (len(cur) + 1) > budget:
                out.append(cur); cur = []
            cur.append(x)
        return out + ([cur] if cur else [])

    val = read(val_p)
    vrng = random.Random(5)
    Xv = [example(r, vrng) for r in val]                     # one fixed presentation, shuffled like training

    def validate():
        model.eval(); res = {"pick": [], "yesno": []}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            idx = sorted(range(len(val)), key=lambda i: len(Xv[i][0]))
            for mb in micro_batches([(Xv[i][0], Xv[i][1], Xv[i][2], i) for i in idx]):
                loss, lps = forward([m[:3] for m in mb])
                for m, l, lp in zip(mb, loss.tolist(), lps):
                    r = val[m[3]]; t = np.array(m[2]); p = lp.exp().cpu().numpy()
                    res["yesno" if r["kind"] == "yesno" else "pick"].append((l, int(p.argmax() == t.argmax()), float(p[0]), t[0]))
        v = {k: {"loss": float(np.mean([x[0] for x in xs])), "agree": float(np.mean([x[1] for x in xs])), "n": len(xs)}
             for k, xs in res.items() if xs}
        score = v["pick"]["loss"] + v["yesno"]["loss"]
        say(event="validation", step=state["step"], **v)
        if state["best"] is None or score < state["best"]:
            state["best"] = score
            tmp = out / "best.tmp"
            torch.save({"lora": get_peft_model_state_dict(model), "head": head.detach(), "step": state["step"], "v": v}, tmp)
            os.replace(tmp, out / "best.pt")
        model.train()

    train = read(train_p)
    if not state["order"]:
        state["order"] = list(range(len(train))); random.Random(31).shuffle(state["order"])
    total_steps = math.ceil(len(train) / BATCH)
    say(event="start_or_resume", base=a.base, step=state["step"], rows=len(train), validation=len(val),
        token_budget=budget, free_gib=round(free, 1))
    if state["step"] == 0:
        validate()
    model.train(); tick = time.time(); seen = 0
    while state["cursor"] < len(train):
        if state["step"] % 10 == 0 and wanted():
            checkpoint(); say(event="yielding_to_live_map"); raise SystemExit(3)
        b = state["order"][state["cursor"]:state["cursor"] + BATCH]
        rng = random.Random(1000003 * state["step"] + 17)
        lr = LR * min(1, (state["step"] + 1) / 50) * .5 * (1 + math.cos(math.pi * state["step"] / total_steps))
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        loss_sum = 0.0
        for mb in micro_batches([example(train[i], rng) for i in b]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = forward(mb)
            (loss.sum() / len(b)).backward(); loss_sum += float(loss.sum()) / len(b)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        state["step"] += 1; state["cursor"] += len(b); seen += len(b)
        if state["step"] % 20 == 0:
            say(event="train", step=state["step"], of=total_steps, loss=round(loss_sum, 4),
                rows_s=round(seen / (time.time() - tick), 1)); seen = 0; tick = time.time()
        if state["step"] % 100 == 0:
            checkpoint()
        if state["step"] % 400 == 0:
            validate(); checkpoint(); tick = time.time(); seen = 0
    validate(); checkpoint()
    opt.state.clear(); torch.cuda.empty_cache()
    merged = model.merge_and_unload()
    W = merged.get_output_embeddings().weight.detach().clone()
    W[answer_ids] = head.detach().to(W.dtype)
    merged.lm_head = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype, device=W.device)
    merged.lm_head.weight = torch.nn.Parameter(W); merged.config.tie_word_embeddings = False
    if getattr(merged.config, "text_config", None):
        merged.config.text_config.tie_word_embeddings = False
    merged.save_pretrained(out / "merged", safe_serialization=True); tok.save_pretrained(out / "merged")
    (out / "merged/classifier.json").write_text(json.dumps({"format": "fieldjudge-v3", "codes": fmt.CODES,
        "code_ids": answer_ids[:len(fmt.CODES)], "yes": answer_ids[YES], "no": answer_ids[NO], "base": a.base}))
    (out / "DONE.json").write_text(json.dumps({"contract": contract, "steps": state["step"], "best_validation": state["best"]}))
    say(event="done", steps=state["step"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    run(ap.parse_args())
