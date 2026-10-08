"""SMR collection: graph index (HNSW over fp16 vectors) vs exact; plus IVF4096 at wider settings. Recall@15, split by query set."""
import time, numpy as np, faiss
from pathlib import Path
from srl_search.engine import Encoder
from eval_index import QUESTIONS
D = Path('/mnt/fast/scholarsreadinglist/search/smr-2026-10-08')
flat = faiss.read_index(str(D / 'index.faiss'))
ids = faiss.vector_to_array(flat.id_map).astype(np.int64)
X = flat.index.reconstruct_n(0, flat.ntotal)
enc = Encoder(4)
extra = ["why do we need sleep", "can exercise help depression", "how do bees find flowers", "microplastics in drinking water",
         "does coffee protect the liver", "gut bacteria and autism", "how do vaccines train the immune system",
         "CRISPR gene therapy for blindness", "why do cats purr", "is screen time bad for toddlers"]
P = np.stack([enc.encode(t)[0] for t in QUESTIONS + extra])
B = np.load('/mnt/fast/scholarsreadinglist/retrieval-bench/emb/denseon/nfcorpus/queries.npy')[:100].astype(np.float32)
B /= np.linalg.norm(B, axis=1, keepdims=True)
sets = {'plain questions': P, 'NFCorpus': B}
faiss.omp_set_num_threads(16)
GT = {k: flat.search(v, 15)[1] for k, v in sets.items()}
del flat

def run(name, index, setter, values):
    faiss.omp_set_num_threads(4)
    for v in values:
        setter(v)
        out = []
        for k, Q in sets.items():
            lat, rec = [], []
            for i in range(len(Q)):
                t = time.perf_counter(); _, I = index.search(Q[i:i+1], 15); lat.append(time.perf_counter() - t)
                rec.append(len(set(I[0]) & set(GT[k][i])) / 15)
            out.append(f'{k} {np.mean(rec):.3f} ({np.median(np.array(lat))*1000:.1f} ms)')
        print(f'  {name} {v}: ' + ' | '.join(out), flush=True)
    faiss.omp_set_num_threads(16)

ivf = faiss.read_index(str(D / 'ivf4096.faiss'))
run('IVF4096 nprobe', ivf, lambda v: setattr(ivf, 'nprobe', v), (256, 512, 1024))
del ivf

t = time.time()
h = faiss.IndexHNSWSQ(768, faiss.ScalarQuantizer.QT_fp16, 32, faiss.METRIC_INNER_PRODUCT)
h.hnsw.efConstruction = 80
h.train(X[:100000])
h.add(X)
print(f'HNSW M=32 efConstruction=80 built in {time.time()-t:.0f}s', flush=True)
hm = faiss.IndexIDMap2(h)
hm.id_map.resize(0)
faiss.copy_array_to_vector(ids, hm.id_map)
hm.construct_rev_map()
run('HNSW efSearch', hm, lambda v: setattr(h.hnsw, 'efSearch', v), (64, 128, 256, 512))
faiss.write_index(hm, str(D / 'hnsw.faiss'))
print('saved')
