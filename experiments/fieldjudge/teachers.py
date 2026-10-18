"""The teacher models of the field-judge experiments: a private setting, outside this repository.

The two models that label training data ("teacher A" and "teacher B"), how to reach them, and how to read their
usage allowance are set in a private Python file: $FIELDJUDGE_TEACHERS, or by default
~/.config/scholarsreadinglist/fieldjudge_teachers.py. teachers.example.py, next to this file, documents what it
defines; copy it there and fill it in. Choose models whose terms allow training on their outputs.

Standard library only: the private file imports its own client libraries inside its functions, so this module (and
ROLES) can be imported anywhere.

    ROLES                     the two teacher roles, as named on disk (answer folders, JSON keys); ("a", "b") by default
    settings(role, **extra)   FunctAI .using(...) keyword arguments (lm, client, reasoning, max_tokens ...) + extra
    overrides(role, purpose)  extra settings a script applies for one purpose (e.g. "fast")
    client(role, transport)   the client object for that role (optionally with a given transport)
    usage(role)               the provider's usage report, as returned by its usage endpoint
    windows(role, report)     that report normalised: {"windows": [{"window", "used" (%), "resets"}], "credits", "blocked"}
    judges()                  the extra judges of compare.py / subs.py: {public name: {...}}
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

DEFAULT_PATH = Path.home() / ".config" / "scholarsreadinglist" / "fieldjudge_teachers.py"
PATH = Path(os.environ.get("FIELDJUDGE_TEACHERS") or DEFAULT_PATH).expanduser()
EXAMPLE = Path(__file__).resolve().with_name("teachers.example.py")


class MissingSettings(RuntimeError):
    """The private teacher settings file is absent."""


def _load():
    if not PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("fieldjudge_teachers_private", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PRIVATE = _load()
ROLES: tuple[str, str] = tuple(getattr(_PRIVATE, "ROLES", ("a", "b")))
if len(ROLES) != 2 or len(set(ROLES)) != 2 or not all(isinstance(r, str) and r for r in ROLES):
    raise ValueError(f"{PATH}: ROLES must name two different teacher roles, got {ROLES!r}")


def _private():
    if _PRIVATE is None:
        where = "$FIELDJUDGE_TEACHERS" if os.environ.get("FIELDJUDGE_TEACHERS") else "the default location"
        raise MissingSettings(
            f"No private teacher settings at {PATH} ({where}). The teacher models are not part of this repository: "
            f"copy {EXAMPLE} to {DEFAULT_PATH} (or set FIELDJUDGE_TEACHERS to another file) and fill in the models, "
            "choosing models whose terms allow training on their outputs.")
    return _PRIVATE


def _check(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown teacher role {role!r}; the roles are {ROLES}")
    return role


def _need(name: str):
    p = _private()
    if not hasattr(p, name):
        raise MissingSettings(f"{PATH} does not define {name} (see {EXAMPLE})")
    return getattr(p, name)


def settings(role: str, **extra) -> dict:
    """FunctAI `.using(...)` keyword arguments for this teacher, with `extra` (cache and log paths ...) added."""
    return {**_need("settings")(_check(role)), **extra}


def overrides(role: str, purpose: str) -> dict:
    """Settings a script changes for one purpose (OVERRIDES[purpose][role] in the private file; none by default)."""
    _check(role)
    return dict(((getattr(_private(), "OVERRIDES", None) or {}).get(purpose) or {}).get(role) or {})


def client(role: str, transport=None):
    """The client for this teacher; `transport` (e.g. a transport with longer timeouts) if given."""
    return _need("client")(_check(role), transport=transport)


def usage(role: str) -> dict:
    """The provider's own usage report for this teacher's account. Never log it: it can be personal."""
    return _need("usage")(_check(role))


def windows(role: str, report: dict) -> dict:
    """`report` (from usage) as {"windows": [{"window": name, "used": percent, "resets": ...}], "credits": float or
    None, "blocked": bool}. Raises if the report has no window."""
    return _need("windows")(_check(role), report)


def judges() -> dict:
    """{public name: {"account": a role, "model": id, "effort": ..., "max_tokens": ..., "stored": name in
    compare_answers.jsonl, "stop_windows": window names checked before each block, or None for all}}."""
    return dict(_need("JUDGES"))


def judge_run_order() -> dict:
    """{account role: [judge names in the order they are asked]}; by default the order of JUDGES."""
    order = getattr(_private(), "JUDGE_RUN_ORDER", None)
    if order:
        return {k: list(v) for k, v in order.items()}
    out: dict = {}
    for name, j in judges().items():
        out.setdefault(j["account"], []).append(name)
    return out
