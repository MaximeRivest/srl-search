jy6 = np.nan_to_num(P_j) >= .5; ky6 = y6 >= .5
est = lambda yes: sum(PN[s] * yes[P_st_a == s].mean() for s in PN)
print(f"on-topic papers in the pool, estimated from the yardstick: Kimi {est(ky6):,.0f}   Jev {est(jy6):,.0f}")
dis = np.flatnonzero(jy6 & ~ky6 & ok6); dis2 = np.flatnonzero(~jy6 & ky6 & ok6)
print(f"Jev yes / Kimi no: {len(dis)}   Kimi yes / Jev no: {len(dis2)}  (of {ok6.sum()})")
for i in rng6.choice(dis, 16, replace=False): print("  J+K-", P_st[i], "|", (P_pairs[i][0] or "")[:105])
for i in dis2[:5]: print("  K+J-", P_st[i], "|", (P_pairs[i][0] or "")[:105])
