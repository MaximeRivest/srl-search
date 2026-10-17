"""Round-3 live dashboard: progress, learning curves, test results. Counts and curves only, never papers or credentials.

    python3 dashboard_server.py --port 8944      (tailnet: https://lambda.tail69222b.ts.net:18944)
"""
import argparse, json, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = Path("/mnt/fast/scholarsreadinglist/fieldjudge/v3")
BATCH = 32


def read(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def logs(p):
    out = []
    try:
        for l in Path(p).read_text().splitlines():
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def active(u):
    r = subprocess.run(["systemctl", "--user", "show", u, "-p", "ActiveState", "--value"], capture_output=True, text=True, timeout=5)
    return r.stdout.strip()


def gpu():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout
        return [dict(zip(("index", "free_mb", "total_mb", "util"), map(int, l.split(", ")))) for l in q.strip().splitlines()]
    except Exception:                                       # noqa: BLE001
        return []


def model(size):
    d = V3 / "models" / size
    ev = logs(d / "log.jsonl")
    train = [e for e in ev if e.get("event") == "train"]
    val = [e for e in ev if e.get("event") == "validation"]
    yields = sum(e.get("event") == "yielding_to_live_map" for e in ev)
    done = (d / "DONE.json").exists()
    last = train[-1] if train else {}
    total = last.get("of")
    step = last.get("step", 0)
    recent = [e["rows_s"] for e in train[-10:] if e.get("rows_s")]
    rate = sum(recent) / len(recent) if recent else None
    eta = (total - step) * BATCH / rate if (rate and total and not done) else None
    stale = (time.time() - ev[-1]["time"]) if ev else None
    state = "done" if done else "not started" if not ev else "paused for a live map" if ev[-1].get("event") == "yielding_to_live_map" \
        else "training" if stale is not None and stale < 600 else "waiting"
    return {"size": size, "state": state, "step": step, "total": total, "rows_s": rate, "eta_s": eta, "yields": yields,
            "train": [[e["step"], e["loss"]] for e in train],
            "validation": [[e["step"], e["pick"]["agree"], e["yesno"]["agree"], e["pick"]["loss"], e["yesno"]["loss"]] for e in val],
            "updated": ev[-1]["time"] if ev else None}


def snapshot():
    teachers = {}
    for m in ("sol", "opus"):
        f = list((V3 / "answers" / m).glob("*.json"))
        teachers[m] = {"units": len(f), "errors": len(list((V3 / "answers" / m / "errors").glob("*.json")))}
    evals = {n: read(V3 / "evaluation" / f"{n}.json") for n in ("round2-0.8B", "round2-2B", "v3-0.8B", "v3-2B")}
    return {"now": time.time(), "manifest": read(V3 / "training" / "manifest.json", {}),
            "teachers": teachers, "queue": {s: len(list((V3 / "queue" / s).glob("*.json"))) for s in ("train", "eval")},
            "schemes": len(list((V3 / "schemes").glob("*.json"))),
            "models": [model("0.8B"), model("2B")],
            "services": {u: active(u) for u in ("fj3-train", "fj3-chain")},
            "gpu": gpu(), "evaluation": {k: v for k, v in evals.items() if v}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/status.json", "/status"):
            try:
                body, code = json.dumps(snapshot()).encode(), 200
            except Exception as e:                          # noqa: BLE001
                body, code = json.dumps({"error": type(e).__name__}).encode(), 503
            kind = "application/json"
        elif p in ("/", "/index.html"):
            body, kind, code = (HERE / "dashboard" / "index.html").read_bytes(), "text/html; charset=utf-8", 200
        else:
            body, kind, code = b"Not found", "text/plain", 404
        self.send_response(code); self.send_header("Content-Type", kind); self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8944)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), Handler).serve_forever()
