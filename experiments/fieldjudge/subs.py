"""Extra judges, asked exactly what the reference model is asked (srl_search/judges.py). Which models, through which
accounts, with which reasoning effort and output cap: the private settings (teachers.py, JUDGES)."""
import sys, time
from pathlib import Path
import lm15
sys.path.insert(0, "/home/maxime/Projects/scholarsreadinglist/search-dev")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # teachers.py
from srl_search import judges as J
import teachers

MODELS = teachers.judges()  # public name: {"account", "model", "effort", "max_tokens", ...}
_CLIENTS = {}               # one client per account, shared by its judges


def client(account):
    if account not in _CLIENTS:
        _CLIENTS[account] = teachers.client(account)
    return _CLIENTS[account]


def system(topic, definition):
    return (J.instruction(topic, definition) + " Answer with exactly one label from this list and nothing else:\n"
            + "\n".join(f"- {l}" for l in J.LABELS))


def ask(name, topic, definition, text):
    """1.0 on topic, 0.0 not, None unreadable. Raises lm15.RateLimitError."""
    j = MODELS[name]
    req = lm15.Request(model=j["model"], messages=(lm15.Message.user(text),), system=system(topic, definition),
                       config=lm15.Config(max_tokens=j.get("max_tokens"),      # None: some backends refuse a cap
                                          reasoning=lm15.Reasoning(effort=j["effort"])))
    ans = (client(j["account"]).complete(req).text or "").strip().lower().strip(".\"'*` -")
    return 0.0 if ans.startswith("not on topic") else 1.0 if ans.startswith("on topic") else None


if __name__ == "__main__":
    paper = "Title: Woodlots as biogeographic islands in southeastern Wisconsin\n(no abstract)"
    for n in MODELS:
        t = time.time()
        try:
            print(n, ask(n, "habitat fragmentation", None, paper), f"{time.time() - t:.1f}s", flush=True)
        except Exception as e:
            print(n, "FAILED", type(e).__name__, str(e)[:250], flush=True)
