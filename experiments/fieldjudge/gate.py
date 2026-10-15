"""Exit 0 once GPU 1 has NEED_MB free and no whole-field map needs it soon (train.py's own test); checks every 30 s."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import judges as J
from srl_search.field import FieldStore, ROOT
need = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
while True:
    pl = FieldStore(ROOT).plan()
    busy = J.RESERVED.exists() or (pl["running"] and pl["running"]["st"].get("kind") == "full") or \
        any(w["st"].get("kind") == "full" and w["start"] - pl["now"] < 300 for w in pl["waiting"][:1])
    free = J.gpu_free_mb()
    if not busy and free >= need:
        sys.exit(0)
    print(f"waiting: {'a map needs the GPU' if busy else f'{free} MB free of {need}'}", flush=True)
    time.sleep(30)
