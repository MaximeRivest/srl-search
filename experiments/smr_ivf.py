"""SMR collection: exact flat (360 ms/query) vs IVF over the same fp16 vectors. Recall@15 vs exact."""
import time, numpy as np, faiss, json
from pathlib import Path
from srl_search.engine import Encoder
from eval_index import QUESTIONS
D = Path('/mnt/fast/scholarsreadinglist/search/smr-2026-10-08')
flat = faiss.read_index(str(D / 'index.faiss'))
ids = faiss.vector_to_array(flat.id_map).astype(np.int64)
X = flat.index.reconstruct_n(0, flat.ntotal)
print('vectors', X.shape, flush=True)
enc = Encoder(4)
extra = ["why do we need sleep", "can exercise help depression", "how do bees find flowers", "microplastics in drinking water",
         "does coffee protect the liver", "gut bacteria and autism", "how do vaccines train the immune system",
         "CRISPR gene therapy for blindness", "why do cats purr", "is screen time bad for toddlers"]
Q = np.stack([enc.encode(t)[0] for t in QUESTIONS + extra])
B = np.load('/mnt/fast/scholarsreadinglist/retrieval-bench/emb/denseon/nfcorpus/queries.npy')[:100].astype(np.float32)
B /= np.linalg.norm(B, axis=1, keepdims=True)
Q = np.concatenate([Q, B])
faiss.omp_set_num_threads(16)
_, GT = flat.search(Q, 15)
for nlist in (4096, 8192):
    t = time.time()
    q = faiss.IndexFlatIP(768)
    ivf = faiss.IndexIVFScalarQuantizer(q, 768, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    ivf.cp.niter = 15
    ivf.train(X[np.random.default_rng(0).choice(len(X), min(len(X), 80 * nlist), replace=False)])
    ivf.add_with_ids(X, ids)
    print(f'nlist {nlist}: built in {time.time()-t:.0f}s', flush=True)
    faiss.omp_set_num_threads(4)
    for npb in (32, 64, 128, 256):
        ivf.nprobe = npb
        lat = []
        res = []
        for i in range(len(Q)):
            t = time.perf_counter(); _, I = ivf.search(Q[i:i+1], 15); lat.append(time.perf_counter() - t); res.append(I[0])
        rec = np.mean([len(set(res[i]) & set(GT[i])) / 15 for i in range(len(Q))])
        lat = np.array(lat) * 1000
        print(f'  nprobe {npb:3d}: recall@15 {rec:.3f}  p50 {np.median(lat):.1f} ms  p95 {np.percentile(lat,95):.1f} ms', flush=True)
    faiss.omp_set_num_threads(16)
    if nlist == 4096:
        faiss.write_index(ivf, str(D / 'ivf4096.faiss'))
