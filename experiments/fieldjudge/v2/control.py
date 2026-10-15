"""Exit codes for checkpointed jobs: retry temporary outages, pause on unsafe changes.

75: systemd retries after a delay. 78: manual inspection needed; no automatic retry.
Provider refusals, invalid data, depleted allowances and credit changes never
qualify as temporary transport failures.
"""
import re
import urllib.error

RETRY = 75
BLOCKED = 78

class TemporaryFailure(RuntimeError):
    pass

class NeedsAttention(RuntimeError):
    pass


def transient(error) -> bool:
    text = str(error)
    # A nested transport message must not override a safety/account/data decision.
    if any(word in text.casefold() for word in (
        'refusal', 'content_filter', 'credit balance', 'allowance guard',
        'allowance exhausted', 'authentication', 'autherror', 'billingerror',
        'invalidrequest', 'permission', 'invalid ids', 'ids mismatch',
    )):
        return False
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429, 500, 502, 503, 504)
    if isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError)):
        return True
    # FunctAI maps preserve error types in their exported error strings.
    if any(word in text.casefold() for word in (
        'transporterror', 'timeouterror', 'ratelimiterror', 'servererror',
        'connection closed mid-chunk', 'connection reset', 'connection refused',
        'remote disconnected', 'temporarily unavailable',
    )):
        return True
    return bool(re.search(r'code=(408|429|500|502|503|504)\b', text))


def provider_failure(error, where):
    """Classify a saved FunctAI row error without putting raw provider data in logs."""
    cls = TemporaryFailure if transient(error) else NeedsAttention
    raise cls(f'{where}: {"temporary provider failure" if cls is TemporaryFailure else "provider/parser failure needs inspection"}; raw result saved')
