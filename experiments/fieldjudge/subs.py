"""Subscription judges (Claude login, ChatGPT/Codex login) asked exactly what Kimi K3 is asked (srl_search/judges.py)."""
import sys, time
import lm15
sys.path.insert(0, "/home/maxime/Projects/scholarsreadinglist/search-dev")
from srl_search import judges as J

CLAUDE = lm15.ClaudeCodeLM(claude_code_version="2.1.284")
CODEX = lm15.OpenAICodexLM()
MODELS = {  # name: (client, model id, reasoning)
    "opus-5.5": (CLAUDE, "claude-opus-5-5", "off"), "sonnet-5.5": (CLAUDE, "claude-sonnet-5-5", "off"),
    "haiku-5.5": (CLAUDE, "claude-haiku-5-5", "off"),
    "sol-6.1": (CODEX, "gpt-6.1-sol", "low"), "astra-6": (CODEX, "gpt-6-astra", "low"), "luna-6": (CODEX, "gpt-6-luna", "low"),
}


def system(topic, definition):
    return (J.instruction(topic, definition) + " Answer with exactly one label from this list and nothing else:\n"
            + "\n".join(f"- {l}" for l in J.LABELS))


def ask(name, topic, definition, text):
    """1.0 on topic, 0.0 not, None unreadable. Raises lm15.RateLimitError."""
    client, model, eff = MODELS[name]
    req = lm15.Request(model=model, messages=(lm15.Message.user(text),), system=system(topic, definition),
                       config=lm15.Config(max_tokens=None if client is CODEX else 400,      # Codex refuses a cap
                                          reasoning=lm15.Reasoning(effort=eff)))
    ans = (client.complete(req).text or "").strip().lower().strip(".\"'*` -")
    return 0.0 if ans.startswith("not on topic") else 1.0 if ans.startswith("on topic") else None


if __name__ == "__main__":
    paper = "Title: Woodlots as biogeographic islands in southeastern Wisconsin\n(no abstract)"
    for n in MODELS:
        t = time.time()
        try:
            print(n, ask(n, "habitat fragmentation", None, paper), f"{time.time() - t:.1f}s", flush=True)
        except Exception as e:
            print(n, "FAILED", type(e).__name__, str(e)[:250], flush=True)
