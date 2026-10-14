import importlib
importlib.reload(Fm)
class KwTeacher:
    def __init__(self, b): self.usd, self.failed, self.slowest = 0.0, 0, {}
    def label(self, texts, q, d): self.usd += 0.0012 * len(texts); return np.array([1.0 if "fragment" in t.lower() else 0.0 for t in texts])
fake_scores = lambda texts: np.array([0.9 if "fragment" in t.lower() else 0.05 for t in texts])
saved = (J.Teacher, J.train_decider, J.decider_scores, J.student_scores)
J.Teacher = KwTeacher
J.train_decider = lambda rows, q, d, out, say: None
J.decider_scores = lambda md, texts, say: fake_scores(texts)
J.student_scores = lambda train, texts, work, say: fake_scores(texts)
jd = Fm.Job(R2, "dry"); jd.dir.mkdir(exist_ok=True)
try:
    t = time.time(); sel = Fm.select_field(ctx, jd, "habitat fragmentation", None, None); print("dry run", round(time.time() - t), "s")
finally:
    J.Teacher, J.train_decider, J.decider_scores, J.student_scores = saved
r = sel["report"]
for k in ("teacher_labels", "candidates", "decider_labels", "students", "judging", "papers", "by_route"): print(k, r.get(k))
print("stage seconds", jd.st.get("stage_seconds"))
