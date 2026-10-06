"""Does a human-defined science tree route searches better than automatic clusters?

Same papers, same queries, same exact scoring inside the searched groups: only the way the
collection is split into groups differs. Measures recall@10 against exact search over the
whole sample, as a function of how many papers each search has to scan.

  A  k-means:       4,096 automatic groups (FAISS k-means on 400k papers)
  B  topic means:   one group per OpenAlex topic (centre = mean of its real papers);
                    each paper goes to the nearest centre
  C  topic labels:  same centres, but each paper stays in its labelled topic's group
"""
import glob, random, time, json
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc, faiss

faiss.omp_set_num_threads(16)
SH = '/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards'
R = '/mnt/fast/scholarsreadinglist/retrieval-bench/emb/denseon/'
t0 = time.time()
parts = sorted(glob.glob(f'{SH}/year=202[45]/rg-*/part-*/'))
random.seed(1); parts = random.sample(parts, 500)
X = np.concatenate([np.load(p + 'vectors.npy') for p in parts]).astype(np.float32)
ids = np.concatenate([np.load(p + 'work_ids.npy') for p in parts])
print('papers', len(X), f'{time.time()-t0:.0f}s', flush=True)

# topic label per paper (years 2024, 2025 of the content table)
want = pa.array(ids.astype(np.int64))
lab = {}
for y in (2024, 2025):
    t = pq.read_table(f'/mnt/bulk/openalex/embedding-inputs/denseon-complete/year={y}/data.parquet', columns=['id', 'topic'])
    num = pc.cast(pc.replace_substring(t['id'], 'https://openalex.org/W', ''), pa.int64())
    m = pc.is_in(num, value_set=want)
    for k, v in zip(num.filter(m).to_pylist(), t['topic'].filter(m).to_pylist()):
        lab[k] = v
topics = np.array([lab.get(int(i)) or '' for i in ids], dtype=object)
ok = topics != ''
X, ids, topics = X[ok], ids[ok], topics[ok]
names, tidx = np.unique(topics, return_inverse=True)
print('labelled', len(X), 'topics', len(names), f'{time.time()-t0:.0f}s', flush=True)

# queries: 600 real questions (BEIR biomedical/science) + 600 held-out papers spread over all fields
Q1 = np.concatenate([np.load(R + d + '/queries.npy')[:150] for d in ['scifact', 'nfcorpus', 'scidocs', 'trec-covid']]).astype(np.float32)
Q1 /= np.linalg.norm(Q1, axis=1, keepdims=True)
rng = np.random.default_rng(0)
hold = rng.choice(len(X), 600, replace=False)
Q2 = X[hold].copy()
keep = np.ones(len(X), bool); keep[hold] = False
X, tidx = X[keep], tidx[keep]
fl = faiss.IndexFlatIP(768); fl.add(X)
GT1 = fl.search(Q1, 10)[1]; GT2 = fl.search(Q2, 10)[1]; del fl

def evaluate(name, C, assign):
    sizes = np.bincount(assign, minlength=len(C))
    order = np.argsort(assign, kind='stable'); starts = np.concatenate([[0], np.cumsum(sizes)])
    cq = faiss.IndexFlatIP(C.shape[1]); cq.add(C.astype(np.float32))
    nz = sizes[sizes > 0]
    print(f'\n{name}: {len(C)} groups, size mean {nz.mean():.0f}, max {sizes.max()}, '
          f'largest 1% of groups hold {np.sort(sizes)[::-1][:max(1,len(C)//100)].sum()/len(X):.0%} of papers', flush=True)
    out = {}
    for npb in (4, 8, 16, 32, 64):
        row = []
        for Q, GT in ((Q1, GT1), (Q2, GT2)):
            _, L = cq.search(Q, npb)
            rec = []; scanned = []
            for i in range(len(Q)):
                cand = np.concatenate([order[starts[l]:starts[l + 1]] for l in L[i]])
                scanned.append(len(cand))
                top = cand[np.argsort(-(X[cand] @ Q[i]))[:10]] if len(cand) else []
                rec.append(len(set(top) & set(GT[i])) / 10)
            row.append((np.mean(rec), np.mean(scanned)))
        out[npb] = row
        print(f'  {npb:3d} groups searched: questions recall {row[0][0]:.3f} (scanned {row[0][1]:,.0f}) | '
              f'papers-as-queries recall {row[1][0]:.3f} (scanned {row[1][1]:,.0f})', flush=True)
    return out

# A: k-means
km = faiss.Kmeans(768, 4096, niter=20, spherical=True, seed=1)
km.train(X[rng.choice(len(X), 400_000, replace=False)])
_, a = km.index.search(X, 1)
evaluate('A  k-means', km.centroids, a[:, 0])

# B/C: OpenAlex topics
C = np.zeros((len(names), 768), np.float32); np.add.at(C, tidx, X)
C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
cq = faiss.IndexFlatIP(768); cq.add(C); _, b = cq.search(X, 1)
evaluate('B  topic centres, nearest-centre assignment', C, b[:, 0])
evaluate('C  topic centres, labelled-topic assignment', C, tidx)
print(f'\ntotal {time.time()-t0:.0f}s')
