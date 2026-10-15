"""Fail-closed allowance checks before every bounded wave. Never log credentials.

Checks reduce risk of consuming extra usage; they cannot guarantee zero spend if a
provider reports usage late or other processes consume the shared allowance. Disable
subscription extra-usage billing in the account settings for a hard billing limit.
"""
from __future__ import annotations
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, write_json

class Pause(RuntimeError):
    pass

def get_usage(model):
    if model == "opus":
        tokens = json.loads((Path.home() / ".claude/.credentials.json").read_text())["claudeAiOauth"]
        req = urllib.request.Request("https://api.anthropic.com/api/oauth/usage", headers={
            "Authorization": f"Bearer {tokens['accessToken']}", "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "claude-cli/2.1.284"})
    else:
        tokens = json.loads((Path.home() / ".codex/auth.json").read_text())["tokens"]
        req = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers={
            "Authorization": f"Bearer {tokens['access_token']}", "chatgpt-account-id": tokens.get("account_id") or "",
            "User-Agent": "codex_cli_rs", "originator": "codex_cli_rs"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def check(model, force=False):
    """Share one usage observation per minute; querying the usage endpoint itself is rate-limited."""
    try:
        status_path = ROOT / f"allowance-{model}.json"
        if not force and status_path.exists():
            previous = json.loads(status_path.read_text())
            age = time.time() - datetime.fromisoformat(previous['checked']).timestamp()
            if 0 <= age < 60:
                if any(w['used'] >= 85 for w in previous['windows']):
                    raise Pause(f"allowance guard: {previous['windows']}")
                initial_path = ROOT / f"credits-start-{model}.json"
                initial = json.loads(initial_path.read_text())['credits'] if initial_path.exists() else None
                if initial is not None and (previous['credits'] is None or previous['credits'] < initial):
                    raise Pause('credit balance fell or became unreadable')
                if previous.get('blocked'):
                    raise Pause('provider reports allowance exhausted')
                return previous
        u = get_usage(model)
        windows = []
        if model == "opus":
            for name in ("five_hour", "seven_day"):
                w = u[name]
                windows.append({"window": name, "used": float(w["utilization"]), "resets": w.get("resets_at")})
            credits = None
            blocked = False
        else:
            rate = u["rate_limit"]
            blocked = bool(rate["limit_reached"])
            for name in ("primary_window", "secondary_window"):
                if rate.get(name):
                    w = rate[name]
                    windows.append({"window": name, "used": float(w["used_percent"]), "resets": w.get("reset_at")})
            if not windows:
                raise ValueError("no Codex allowance windows")
            credits = (u.get("credits") or {}).get("balance")
            credits = float(credits) if credits is not None else None
        state = {"checked": datetime.now(timezone.utc).isoformat(), "windows": windows, "credits": credits, "blocked": blocked}
        write_json(ROOT / f"allowance-{model}.json", state)
        initial_path = ROOT / f"credits-start-{model}.json"
        if initial_path.exists():
            initial = json.loads(initial_path.read_text())["credits"]
        else:
            write_json(initial_path, {"credits": credits})
            initial = credits
        if initial is not None and (credits is None or credits < initial):
            raise Pause("credit balance fell or became unreadable; inspect billing before resuming")
        if blocked or any(w["used"] >= 85 for w in windows):
            raise Pause(f"allowance guard: {windows}")
        return state
    except Pause:
        raise
    except Exception as error:
        # Do not include an HTTP request or response that could carry credentials.
        raise Pause(f"usage unavailable ({type(error).__name__}, code={getattr(error, 'code', None)}); no new calls") from None
