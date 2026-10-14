D5 = "/mnt/fast/scholarsreadinglist/search/fields-test5/a699318c9a209d289e4c3702"
f5 = json.load(open(D5 + "/field.json")); r5 = f5["report"]
print("check", r5["check"]); print("usd", r5["usd"], "teacher", r5["teacher_usd"], "papers", r5["papers"], r5["by_route"])
v1 = json.load(open("/mnt/fast/scholarsreadinglist/search/fields/a699318c9a209d289e4c3702/field.json"))["papers"]
def measure(name, chosen):
    def wp(ids, y, w, ok):
        s = np.array([int(x) in chosen for x in ids], float); tp = (w * s * y)[ok].sum()
        return tp / max(1e-9, (w * s)[ok].sum()), tp / max(1e-9, (w * y)[ok].sum())
    p, r = wp(ev_ids, np.nan_to_num(ev_y), w_e, ok_e); pc, rc = wp(cl_ids_a, np.nan_to_num(cy), w_c, ok_c)
    mr = np.array([w in chosen for w in ref_pos])
    print(f"{name:16s} papers {len(chosen):6,d}  population P {p:.2f} R {r:.2f}  cited>=50 P {pc:.2f} R {rc:.2f}  meta {mr.mean():.2f} (title only {mr[~ref_ab].mean():.2f})")
measure("v1 Jev $1.56", set(int(w) for w in v1))
measure("v4 run $0.48", set(int(w) for w in f4["papers"]))
measure("v5 run $%.2f" % r5["usd"], set(int(w) for w in f5["papers"]))
