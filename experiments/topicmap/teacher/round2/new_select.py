def _plugin_cut(p: np.ndarray) -> tuple[float, float]:
    """The cut-off that maximises the F1 expected from probabilities `p` of every candidate (no labels needed; Lipton et
    al. 2014: with calibrated probabilities the best cut is about half the best F1): (cut, expected F1)."""
    o = np.sort(p)[::-1]
    f1 = 2 * np.cumsum(o) / (np.arange(1, len(o) + 1) + o.sum())
    k = int(np.argmax(f1))
    return round(float(o[k]), 3), round(float(f1[k]), 3)


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def select_field(ctx: Ctx, job: Job, q: str, definition: str | None, keywords: str | None) -> dict:
    import faiss
    from scipy.optimize import brentq
    from sklearn.linear_model import LogisticRegression
    from srl_search import judges as J
    from srl_search.engine import passes
    from srl_search.scope import NON_PUBLICATION_TYPES
    eng, cit = ctx.engine, ctx.cit
    faiss.omp_set_num_threads(16)
    report: dict = {"topic": q, "definition": definition, "teacher": "Kimi K3", "decider": J.DECIDER_BASE,
                    "student": J.STUDENT_BASE}
    work = job.dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    say = lambda detail: job.stage(job.st.get("stage_index", 0), detail)
    rng = np.random.default_rng(0)
    bad_types = [i for i, tn in enumerate(cit.types) if tn in NON_PUBLICATION_TYPES]

    def texts_of(ws) -> list[str]:
        det = _details(ctx, ws)
        out = []
        for w in ws:
            d = det.get(int(w))
            if d:
                out.append(J.paper_text(d.get("title"), d.get("abstract")))
            else:
                nd = int(cit.nodes([int(w)])[0])
                out.append(J.paper_text(cit.title(nd) if nd >= 0 else "", None))
        return out

    def publications(ws) -> np.ndarray:
        det = _details(ctx, ws)
        ok = []
        for w in ws:
            d = det.get(int(w))
            if d is not None:
                ok.append(passes(d, {}))
            else:
                nd = int(cit.nodes([int(w)])[0])
                ok.append(nd >= 0 and int(cit.kind[nd]) not in bad_types and len(cit.title(nd)) >= 10)
        return np.array(ok, bool)

    def cited_by(ws, k=CITED_K):
        nodes = cit.nodes(np.unique(np.asarray(ws, np.int64)))
        nodes = nodes[nodes >= 0]
        allrefs = np.concatenate([cit.refs(int(x)) for x in nodes]) if len(nodes) else np.zeros(0, np.uint32)
        u, cnt = np.unique(allrefs, return_counts=True)
        while (cnt >= k).sum() > EXPANSION_MAX:
            k += 1
        cn = u[cnt >= k]
        cn = cn[~np.isin(np.asarray(cit.kind[cn]), bad_types)]
        return np.array([cit.work(int(x)) for x in cn], np.int64), k

    def lr_hard(ws_y: dict[int, float]):
        """First classifier (teacher's answers only): a search direction and strata, not a judge."""
        ws = np.array(list(ws_y), np.int64)
        X, h = _vectors(ctx, ws)
        y = np.array([ws_y[int(w)] >= 0.5 for w in ws])
        return LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(X[h], y[h])

    def lr_soft(ws_p: dict[int, float]):
        """Vector student: learns probabilities (each paper counts as yes with weight p and no with weight 1 - p)."""
        ws = np.array(list(ws_p), np.int64)
        X, h = _vectors(ctx, ws)
        X, p = X[h], np.array([ws_p[int(w)] for w in ws])[h]
        return LogisticRegression(C=2.0, max_iter=4000).fit(np.concatenate([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))],
                                                             sample_weight=np.r_[p, 1 - p])

    def proba(model, ws) -> tuple[np.ndarray, np.ndarray]:
        out = np.full(len(ws), np.nan)
        X, has = _vectors(ctx, ws)
        if has.any():
            out[has] = model.predict_proba(X[has])[:, 1]
        return out, has

    # 1 the teacher, first round: papers spread from the nearest to the farthest of the 300k closest to the topic, and a
    #   quarter from the works the nearest papers cite (many have only a title: the decider must learn those too)
    t = time.time()
    job.stage(0, f"gathering the {POOL:,} papers closest to the topic")
    qv = eng.encoder.encode(q)[0]
    _, ca = eng.index.search(qv.reshape(1, -1), POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_a = ca[0][ca[0] >= 0]
    VA = unit(eng.vectors.get(rows_a))
    wa = np.asarray(eng.work_ids[rows_a], dtype=np.int64)
    o = np.argsort(-(VA @ qv), kind="stable")
    rows_a, VA, wa = rows_a[o], VA[o], wa[o]
    okA = ctx.quality.junk(wa, VA) < 0.5
    idxA = np.flatnonzero(okA)
    n = len(idxA)
    n_cited = TEACHER_R1 // 4
    edges = sorted({0, *[int(n * f) for f in (.002, .004, .007, .01, .017, .025, .035, .05, .07, .1, .14, .2, .27, .4,
                                                .55, .75)], n})
    per = math.ceil((TEACHER_R1 - n_cited) * 1.25 / (len(edges) - 1))
    sample, bucket = [], []
    for bi, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if hi > lo:
            pick = rng.choice(np.arange(lo, hi), size=min(per, hi - lo), replace=False)
            sample += idxA[pick].tolist()
            bucket += [bi] * len(pick)
    sw = wa[sample]
    ok = publications(sw)
    keep = rng.choice(np.flatnonzero(ok), size=min(TEACHER_R1 - n_cited, int(ok.sum())), replace=False)
    sw_pool, bucket = sw[np.sort(keep)], np.array(bucket)[np.sort(keep)]
    pre, _ = cited_by(wa[idxA[:5000]])
    pre = np.setdiff1d(pre, wa)
    _, hv = _vectors(ctx, pre)
    title_only, with_vec = pre[~hv], pre[hv]
    pick = list(rng.choice(title_only, size=min(len(title_only), n_cited * 2), replace=False)) + \
        list(rng.choice(with_vec, size=min(len(with_vec), n_cited), replace=False))
    pick = np.array(pick, np.int64)
    pick = pick[publications(pick)]
    n_t = min(len(pick), n_cited)
    title_set = set(title_only.tolist())
    t_only = [w for w in pick if w in title_set][:round(n_t * 0.7)]
    t_only_set = set(t_only)
    sw_cit = np.array(t_only + [w for w in pick if w not in t_only_set][:n_t - len(t_only)], np.int64)
    sw = np.concatenate([sw_pool, sw_cit])
    job.stage(0, f"the teacher (Kimi K3) judges {len(sw)} papers")
    teacher = J.Teacher(TEACHER_USD)
    texts_sw = texts_of(sw)
    t_label = time.time()
    lab = teacher.label(texts_sw, q, definition)
    secs = [round(time.time() - t_label, 1)]
    _spend(job.dir.parent, teacher.usd)
    labels = {int(w): float(x) for w, x in zip(sw, lab) if not math.isnan(x)}
    on_pool = np.nan_to_num(lab[:len(sw_pool)]) >= 0.5
    last = bucket == bucket.max()
    tail = float(on_pool[last].mean()) if last.any() else 0.0
    if int(on_pool.sum()) < MIN_ON_TOPIC:
        raise FinalError(f"Only {int(on_pool.sum())} of {len(sw_pool)} papers judged near “{q}” are about it: too few to "
                         "find the field. Phrase the topic as papers about it would, or give a definition.")
    if tail > TOO_BROAD_TAIL:
        raise FinalError(f"“{q}” is broader than a whole-field map can hold today: {round(100 * tail)}% of the papers at "
                         f"the edge of the {POOL:,} closest are still about it. Map a narrower topic, or use the "
                         "1,000-paper map.")
    job.done_stage(0, t)

    # 2 candidates: a classifier on meaning searches a second pool along its direction; the exact phrase; what the
    #   likely field cites. Then the teacher's second round (active learning): papers this classifier is unsure about,
    #   where the field's neighbours are (measured: +0.07 F1 for the decider over spreading the same answers)
    t = time.time()
    job.stage(1, "searching along a first classifier; following citations")
    lr0 = lr_hard(labels)
    wdir = unit(lr0.coef_.astype(np.float32))[0]
    _, cb = eng.index.search(wdir.reshape(1, -1), POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_b = np.setdiff1d(cb[0][cb[0] >= 0], rows_a)
    VB = unit(eng.vectors.get(rows_b)) if len(rows_b) else np.zeros((0, 768), np.float32)
    wb = np.asarray(eng.work_ids[rows_b], dtype=np.int64)
    okB = ctx.quality.junk(wb, VB) < 0.5 if len(wb) else np.zeros(0, bool)
    U_w = np.concatenate([wa[okA], wb[okB]])
    U_V = np.concatenate([VA[okA], VB[okB]])
    del VA, VB
    p0 = lr0.predict_proba(U_V)[:, 1]
    phrase = keywords or (f'"{q}"' if len(q.split()) <= 6 and not re.search(r'["()*:]|\b(AND|OR|NOT)\b', q) else None)
    H = np.zeros(0, np.int64)
    if phrase:
        hits, matches, _ = ctx.keywords.search(phrase, 200_000, {}, "all")
        H = np.setdiff1d(np.array([h[0] for h in hits], np.int64), U_w)
        report["phrases"] = {"query": phrase, "matches": int(matches), "outside_pool": int(len(H))}
    C, k = cited_by(np.concatenate([U_w[p0 >= 0.5], H]))
    C = np.setdiff1d(C, np.concatenate([U_w, H]))
    pH0, hH = proba(lr0, H)
    pC0, hC = proba(lr0, C)
    report["candidates"] = {"pool": int(len(U_w)), "likely_by_first_classifier": int((p0 >= 0.5).sum()),
                            "phrase_outside_pool": int(len(H)), "cited_at_least": k, "cited_candidates": int(len(C)),
                            "cited_title_only": int((~hC).sum())}
    vec_w = np.concatenate([U_w, H[hH], C[hC]])
    vec_p = np.concatenate([p0, pH0[hH], pC0[hC]])
    band = vec_w[(vec_p > ACTIVE_BAND[0]) & (vec_p < ACTIVE_BAND[1])]
    band = band[~np.isin(band, sw)]
    cit_t = C[~hC]
    cit_t = cit_t[~np.isin(cit_t, sw)]
    n_band = round(TEACHER_R2 * 0.75)
    r2 = np.concatenate([rng.choice(band, size=min(len(band), n_band * 2), replace=False),
                         rng.choice(cit_t, size=min(len(cit_t), (TEACHER_R2 - n_band) * 2), replace=False)])
    okr = publications(r2)
    nb = min(len(band), n_band * 2)
    r2 = np.concatenate([r2[:nb][okr[:nb]][:n_band], r2[nb:][okr[nb:]][:TEACHER_R2 - n_band]])
    job.stage(1, f"the teacher judges {len(r2)} papers the first classifier is unsure about")
    texts_r2 = texts_of(r2)
    t_label = time.time()
    lab2 = teacher.label(texts_r2, q, definition)
    secs.append(round(time.time() - t_label, 1))
    _spend(job.dir.parent, teacher.usd - sum(0 for _ in ()))  # placeholder replaced below
    labels.update({int(w): float(x) for w, x in zip(r2, lab2) if not math.isnan(x)})
    report["teacher_labels"] = {"papers": len(labels), "on_topic": int(sum(v >= 0.5 for v in labels.values())),
                                "first_round": int(len(sw)), "cited": int(len(sw_cit)),
                                "title_only": int(sum(1 for w in sw_cit if w in title_set)) + int(len(r2) - min(len(r2), n_band)),
                                "active": int(len(r2)), "active_on_topic": int(np.nansum(lab2 >= 0.5)),
                                "failed": teacher.failed, "usd": round(teacher.usd, 3),
                                "farthest_on_topic": round(tail, 3), "seconds": secs,
                                "slowest_answer_s": teacher.slowest}
    job.done_stage(1, t)

    # 3 the decider learns from all the teacher's answers
    t = time.time()
    job.stage(2, "teaching the 0.8B decider")
    tw = list(labels)
    J.train_decider(list(zip(texts_of(tw), [labels[w] for w in tw])), q, definition, work / "decider", say)
    job.done_stage(2, t)

    # 4 the decider judges a balanced set for the students: likely papers, unsure ones, a spread, and cited works
    t = time.time()
    job.stage(3, f"the decider judges {DECIDER_N:,} papers")
    taken: set[int] = set(labels)

    def take(cands, m):
        cands = [int(w) for w in cands if int(w) not in taken]
        out = [cands[i] for i in rng.choice(len(cands), size=min(m, len(cands)), replace=False)] if cands else []
        taken.update(out)
        return out
    dws = take(U_w[p0 >= 0.5], int(DECIDER_N * .30)) + take(U_w[(p0 > 0.1) & (p0 < 0.9)], int(DECIDER_N * .25)) \
        + take(U_w, int(DECIDER_N * .15)) + take(C[~hC], int(DECIDER_N * .20)) + take(C[hC], int(DECIDER_N * .10))
    dp = J.decider_scores(work / "decider", texts_of(dws), say)
    decided = dict(zip(dws, dp.tolist()))
    report["decider_labels"] = {"papers": len(decided), "expected_on_topic": round(float(dp.sum()))}
    job.done_stage(3, t)

    # 5 students learn the decider's probabilities (not the teacher's answers: those stay unseen, to fit the
    #   combiner). The vector student judges every paper with a vector; Ettin judges papers with a title only, those
    #   the vector student is unsure about, and the teacher's papers
    t = time.time()
    job.stage(4, "training the students")
    univ = np.concatenate([U_w, H, C])
    lr1 = lr_soft(decided)
    pv = np.full(len(univ), np.nan)
    pv[:len(U_w)] = lr1.predict_proba(U_V)[:, 1]
    pv[len(U_w):len(U_w) + len(H)] = proba(lr1, H)[0]
    pv[len(U_w) + len(H):] = proba(lr1, C)[0]
    del U_V
    is_teacher = np.isin(univ, np.array(list(labels), np.int64))
    e_mask = np.isnan(pv) | ((pv > BAND[0]) & (pv < BAND[1])) | is_teacher
    E = univ[e_mask]
    say(f"the text student judges {len(E):,} papers (title only, unsure, or judged by the teacher)")
    dw = list(decided)
    pt = np.full(len(univ), np.nan)
    pt[e_mask] = J.student_scores(list(zip(texts_of(dw), [decided[w] for w in dw])), texts_of(E), work / "student", say)
    both = ~np.isnan(pv) & ~np.isnan(pt)
    stud = np.where(both, _sigmoid((_logit(np.nan_to_num(pv, nan=.5)) + _logit(np.nan_to_num(pt, nan=.5))) / 2),
                    np.where(np.isnan(pv), pt, pv))
    stud = np.nan_to_num(stud, nan=0.0)
    report["students"] = {"judged_by_text_student": int(len(E)), "expected_on_topic": round(float(stud.sum()))}
    job.done_stage(4, t)

    # 6 judging. Citations: works the likely field cites a lot, and papers whose references are mostly the field's,
    #   are on topic more often than their text alone says (the field's classics in older words: "woodlots as
    #   islands", "forest remnants"). A small combiner (student score + citation signals) is fitted on the teacher's
    #   answers, its baseline shifted so it expects as many papers on topic as the students; the teacher's and decider's
    #   own answers come first; the cut-off maximises the expected F1 over all candidates
    t = time.time()
    job.stage(5, "judging every candidate")
    pos = {w: i for i, w in enumerate(univ.tolist())}
    fixed = np.zeros(len(univ), bool)
    s1 = stud.copy()
    for w, v in decided.items():
        s1[pos[w]] = v
        fixed[pos[w]] = True
    feats = _citation_features(cit, univ, s1, _logit(stud))
    tix = np.array([pos[w] for w in labels if w in pos])
    ty = np.array([labels[int(univ[i])] >= 0.5 for i in tix])
    comb = LogisticRegression(C=1.0, max_iter=2000).fit(feats[tix], ty)
    z = feats @ comb.coef_[0] + comb.intercept_[0]
    free = ~fixed & ~is_teacher
    target = float(stud[free].sum())
    shift = brentq(lambda s: _sigmoid(z[free] + s).sum() - target, -20, 20)
    final = np.where(fixed, s1, _sigmoid(z + shift))
    for w, v in labels.items():
        if w in pos:
            final[pos[w]] = v
    cut, ef1 = _plugin_cut(final)
    names = ("student", "cited by the field", "share of its citations", "share of its references", "no references")
    report["judging"] = {"cut": cut, "expected_f1": ef1, "expected_on_topic": round(float(final.sum())),
                         "combiner": {n_: round(float(c_), 3) for n_, c_ in zip(names, comb.coef_[0])},
                         "baseline_shift": round(float(shift), 3)}
    srcs = (["meaning"] * len(U_w)) + (["phrase"] * len(H)) + (["cited"] * len(C))
    yes = np.flatnonzero(final >= cut)
    ok = publications(univ[yes])
    field = {int(univ[i]): {"via": srcs[i], "p": round(float(final[i]), 3)} for i, o_ in zip(yes, ok) if o_}
    if len(field) > TOO_BROAD_PAPERS:
        raise FinalError(f"“{q}” has about {len(field):,} papers: too broad for a whole-field map today. Map a narrower topic.")
    report["papers"] = len(field)
    report["by_route"] = {v: sum(1 for f in field.values() if f["via"] == v) for v in ("meaning", "phrase", "cited")}
    report["teacher_usd"] = round(teacher.usd, 3)
    job.done_stage(5, t)
    return {"version": SELECTION_VERSION, "made": _now(), "release": ctx.release["id"], "topic": q,
            "definition": definition, "keywords": keywords, "report": report,
            "papers": {str(w): f for w, f in field.items()},
            "labels": {"teacher": {str(w): v for w, v in labels.items()},
                       "decider": {str(w): round(v, 3) for w, v in decided.items()}}}


def _citation_features(cit, univ: np.ndarray, score: np.ndarray, student_logit: np.ndarray) -> np.ndarray:
    """Per candidate: its student logit; log(1 + candidates citing it, each counted by its score); the share of all its
    citations that come from them; the score-weighted share of its references that are candidates; no references."""
    nodes = cit.nodes(univ)
    order = np.argsort(nodes)
    sn = nodes[order]

    def at(ns):
        i = np.minimum(np.searchsorted(sn, ns), len(sn) - 1)
        return np.where(sn[i] == ns, order[i], -1)
    offs = np.asarray(cit.fwd_off)
    has = np.flatnonzero(nodes >= 0)
    lens = (offs[nodes[has] + 1] - offs[nodes[has]]).astype(np.int64)
    refs = np.concatenate([cit.refs(int(x)) for x in nodes[has]]).astype(np.int64) if len(has) else np.zeros(0, np.int64)
    owner = np.repeat(has, lens)
    tgt = at(refs)
    hit = tgt >= 0
    fc = np.bincount(tgt[hit], weights=score[owner[hit]], minlength=len(univ))       # in: likely field citing it
    tot = np.zeros(len(univ))
    tot[has] = np.asarray(cit.cites)[nodes[has]]
    out_sum = np.bincount(owner[hit], weights=score[tgt[hit]], minlength=len(univ))  # out: its references in the field
    nref = np.zeros(len(univ))
    nref[has] = lens
    return np.column_stack([student_logit, np.log1p(fc), fc / np.maximum(tot, 1), out_sum / np.maximum(nref, 1),
                            (nref == 0).astype(float)])
