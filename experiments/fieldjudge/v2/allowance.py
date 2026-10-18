"""Fail-closed allowance checks before every bounded wave. Never log credentials.

Checks reduce risk of consuming extra usage; they cannot guarantee zero spend if a
provider reports usage late or other processes consume the shared allowance. Disable
extra-usage billing in the account settings for a hard billing limit.
"""
from __future__ import annotations
import json
import time
import urllib.error
from datetime import datetime, timezone
from common import ROOT, write_json, teachers

class Pause(RuntimeError):
    pass

def get_usage(model):
    """The teacher account's usage report (endpoint and credentials: private settings, ../teachers.py)."""
    return teachers.usage(model)

def normalise(model, usage):
    """{"windows": [{"window", "used", "resets"}], "credits", "blocked"}; raises if no window is reported."""
    return teachers.windows(model, usage)

def check(model, force=False):
    """Share one usage observation per minute; querying the usage endpoint itself is rate-limited."""
    try:
        status_path = ROOT / f"allowance-{model}.json"
        cooldown = ROOT / f"usage-cooldown-{model}.json"
        if cooldown.exists():
            remaining = json.loads(cooldown.read_text()).get('until', 0) - time.time()
            if remaining > 0:
                print(f'{model}: usage endpoint cooling down; no calls for {remaining:.0f}s', flush=True)
                time.sleep(remaining)
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
        while True:
            try:
                u = get_usage(model)
                break
            except urllib.error.HTTPError as error:
                if error.code != 429:
                    raise
                try:
                    delay = max(300.0, float(error.headers.get('Retry-After') or 0))
                except (TypeError, ValueError, AttributeError):
                    delay = 300.0
                write_json(cooldown, {'until': time.time() + delay})
                print(f'{model}: usage endpoint rate-limited; no model calls, waiting {delay:.0f}s', flush=True)
                time.sleep(delay)
        observed = normalise(model, u)
        windows = [dict(w, used=float(w["used"])) for w in observed["windows"]]
        if not windows:
            raise ValueError("no allowance windows")
        credits = observed.get("credits")
        credits = float(credits) if credits is not None else None
        blocked = bool(observed.get("blocked"))
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
