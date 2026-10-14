importlib.reload(J); importlib.reload(Fm)
class KwTeacher:
    def __init__(self, b): self.usd, self.failed, self.slowest = 0.0, 0, {}
    def label(self, texts, q, d): self.usd += 0.0012 * len(texts); return np.array([1.0 if "fragment" in t.lower() else 0.0 for t in texts])
class FakeJev:
    def __init__(self, b): self.usd, self.failed, self.budget = 0.0, 0, b
    def label(self, pairs, q, d): self.usd += 0.000027 * len(pairs); return np.array([0.9 if "fragment" in (t or "").lower() else 0.1 for t, a in pairs])
fs = lambda texts: np.array([0.9 if "fragment" in t.lower() else 0.05 for t in texts])
sv = (J.Teacher, J.train_decider, J.decider_scores, J.student_scores, J.Jev)
J.Teacher, J.Jev = KwTeacher, FakeJev
J.train_decider = lambda *a, **k: None; J.decider_scores = lambda md, t, say: fs(t); J.student_scores = lambda tr, t, w, say: fs(t)
jd = Fm.Job(R, "dry2"); jd.dir.mkdir(exist_ok=True)
try: sel = Fm.select_field(ctx, jd, "habitat fragmentation", None, None)
finally: J.Teacher, J.train_decider, J.decider_scores, J.student_scores, J.Jev = sv
r = sel["report"]; print(r["check"], r["usd"], r["papers"], len(sel["labels"]["jev"]))
