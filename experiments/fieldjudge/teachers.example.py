"""Template of the PRIVATE teacher settings of the field-judge experiments (read by teachers.py).

Copy this file to ~/.config/scholarsreadinglist/fieldjudge_teachers.py (or point $FIELDJUDGE_TEACHERS at your copy),
keep it out of the repository, and fill in your own models. Choose models whose terms allow training on their
outputs. Import client libraries inside the functions: teachers.py must stay importable without them.
"""

# Teacher A, teacher B: names used on disk (answer folders, allowance files, JSON keys of on-disk records).
# Changing them later orphans existing answers.
ROLES = ("a", "b")

MODELS = {"a": "provider:model-a", "b": "provider:model-b"}       # FunctAI model ids: placeholders


def client(role, transport=None):
    """The client object FunctAI calls for this role, or None to let FunctAI pick one from the model id.
    `transport`, when given, is an lm15 transport the caller wants (v2/fast.py passes longer timeouts)."""
    return None


def settings(role):
    """FunctAI .using(...) keyword arguments for this role. Callers add their own cache and log paths."""
    import lm15
    if role not in MODELS:
        raise ValueError(role)
    out = dict(lm=MODELS[role], reasoning=lm15.Reasoning(effort="low"), max_tokens=8192)
    c = client(role)
    if c is not None:
        out["client"] = c
    return out


# Changes a script applies for one purpose: {purpose: {role: {setting: value}}}. "fast" is v2/fast.py.
OVERRIDES = {"fast": {"b": {"max_tokens": 2048}}}


def usage(role):
    """The provider's usage report for this role's account, as a dict (any shape: windows() reads it).
    Called before every wave of requests (v2/allowance.py); raise if it cannot be read: the run then pauses."""
    raise NotImplementedError("read your provider's usage report here")


def windows(role, report):
    """`report` normalised. used = percent of the window used (the runs pause at 85%); credits = a balance that must
    not fall (None if there is none); blocked = the provider says the allowance is exhausted."""
    return {"windows": [{"window": "daily", "used": float(report["used_percent"]), "resets": report.get("resets")}],
            "credits": None, "blocked": False}


# compare.py / subs.py: extra judges asked the reference model's question. account: a role above (its client and
# usage report). max_tokens: output cap, or None. stored: the judge's name in compare_answers.jsonl. stop_windows:
# window names checked against 90% before each block of questions (None: all windows). label: console only.
JUDGES = {
    "judge-1": dict(account="a", model="model-a", effort="low", max_tokens=400, stored="judge-1",
                    stop_windows=None, label="judge 1"),
}
# Optional: {account: [judge names, in the order they are asked]}; default: the order of JUDGES.
JUDGE_RUN_ORDER = {"a": ["judge-1"]}
