P_ids_a = np.array(P_ids); P_st_a = np.array(P_st); w6 = np.array([PN[s] / Pn[s] for s in P_st]); ok6 = ~np.isnan(P_y); y6 = np.nan_to_num(P_y)
def pr6(chosen):
    s = np.array([int(w) in chosen for w in P_ids_a], float); tp = (w6 * s * y6)[ok6].sum()
    return tp / max(1e-9, (w6 * s)[ok6].sum()), tp / max(1e-9, (w6 * y6)[ok6].sum())
for name, ch in (("before Jev", pre), ("v5 (after Jev)", v5)):
    p, r = pr6(ch); print(f"{name:16s} papers {len(ch):6,d}  population P {p:.2f} R {r:.2f}  landmarks {sum(w in ch for w in lm)}/44")
print("estimated on-topic papers among candidates:", round(sum(PN[s] * np.nanmean(P_y[P_st_a == s]) for s in PN)))
fp = [i for i in range(len(P_ids)) if P_ids[i] in v5 and P_y[i] < .5]
print("wrong papers in the map (sample):", len(fp))
for i in rng6.choice(fp, 18, replace=False): print("  ", P_st[i], "|", (P_pairs[i][0] or "")[:110])
n_dpo = [w for w in lm if "Direct Preference" in (Fm._details(ctx, [w]).get(w) or {}).get("title", "")]
print("Jev on DPO:", [f6["labels"]["jev"].get(str(w)) for w in n_dpo])
