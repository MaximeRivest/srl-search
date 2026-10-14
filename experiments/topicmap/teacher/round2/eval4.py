D4 = "/mnt/fast/scholarsreadinglist/search/fields-test4/a699318c9a209d289e4c3702"
f4 = json.load(open(D4 + "/field.json"))
r4 = f4["report"]
for k in ("teacher_labels", "candidates", "decider_labels", "students", "judging", "papers", "by_route", "teacher_usd"): print(k, r4.get(k))
print()
evaluate(set(int(w) for w in f3["papers"]), "v3 ($0.41)")
evaluate(set(int(w) for w in f4["papers"]), "v4 ($%.2f)" % r4["teacher_usd"])
evaluate(set(int(w) for w in v1), "v1 Jev ($1.56)")
