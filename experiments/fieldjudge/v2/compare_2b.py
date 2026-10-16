"""The round-2 2B on the same matched cases as compare_08.py (old 0.8B = v1, round-2 0.8B = v2).
Reuses v1/v2 scores; scores only the 2B. Kimi labels are a silver reference; fixed 0.5 threshold, nothing tuned here.
Writes evaluation-08/comparison-2b.json.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from compare_08 import data, metrics, OUT, ANSWER, ROOT, J, VPY, gate, digest, write_json

rows, removed = data()
fingerprint = digest(rows)
assert json.loads((OUT / 'case-manifest.json').read_text())['case_hash'] == fingerprint, 'cases changed since compare_08'
model = ROOT / 'models/2B/merged'
file, meta = OUT / 'v2-2B-scores.json', OUT / 'v2-2B-manifest.json'
contract = {'cases': fingerprint, 'model': str(model), 'model_mtime': (model / 'model.safetensors').stat().st_mtime}
if not (file.exists() and meta.exists() and json.loads(meta.read_text()) == contract):
    gate()
    src = OUT / 'v2-2B-input.jsonl'
    src.write_text(''.join(json.dumps({'text': r['text'], 'system': J.instruction(r['topic'], r['definition']) + ANSWER}) + '\n'
                           for r in rows))
    J.run([VPY, str(Path(J.__file__).with_name('vllm_decider.py')), str(model), str(src), str(file), J.GPU], OUT / 'v2-2B-log.txt')
    write_json(meta, contract)
scores = {n: np.array(json.loads((OUT / f'{n}-scores.json').read_text())) for n in ('v1', 'v2', 'v2-2B')}
for n, p in scores.items():
    if len(p) != len(rows) or not np.isfinite(p).all(): raise ValueError(f'invalid scores for {n}')

results = {}
for name, p in scores.items():
    results[name] = {}
    for key in sorted({(r['suite'], r['topic'], r['kind']) for r in rows}):
        idx = [i for i, r in enumerate(rows) if (r['suite'], r['topic'], r['kind']) == key]
        results[name][' / '.join(key)] = metrics([rows[i] for i in idx], p[idx])
topics = sorted({r['topic'] for r in rows if r['suite'] == 'new_topics' and r['kind'] == 'benchmark'})
macro = {n: {m: float(np.mean([results[n][f'new_topics / {t} / benchmark'][m] for t in topics]))
             for m in ('precision', 'recall', 'f1', 'auc', 'average_precision')} for n in scores}
strata = defaultdict(list)
for i, r in enumerate(rows):
    if r['suite'] == 'new_topics' and r['kind'] == 'benchmark': strata[(r['topic'], r['stratum'])].append(i)
rng = np.random.default_rng(22026); deltas = {'f1': [], 'average_precision': []}
for _ in range(400):
    by = defaultdict(list)
    for (topic, s), idx in strata.items(): by[topic].extend(rng.choice(idx, len(idx), replace=True).tolist())
    for m in deltas:
        v = {n: np.mean([metrics([rows[i] for i in idx], scores[n][idx])[m] for idx in by.values()]) for n in ('v2', 'v2-2B')}
        deltas[m].append(v['v2-2B'] - v['v2'])
report = {'reference': 'Kimi silver labels, not human gold. Fixed 0.5 threshold.', 'excluded': removed, 'macro': macro,
          '2B_minus_08_ci95': {m: np.percentile(d, [2.5, 97.5]).tolist() for m, d in deltas.items()},
          'per_topic': results}
write_json(OUT / 'comparison-2b.json', report)
dev = ['development / habitat fragmentation / cited50', 'development / habitat fragmentation / meta_refs',
       'development / LLM post-training / landmarks', 'development / habitat fragmentation / population',
       'development / LLM post-training / population']
print(json.dumps({'macro': macro, 'ci95': report['2B_minus_08_ci95'],
                  'dev': {k: {n: {m: round(results[n][k][m], 3) for m in ('precision', 'recall')} for n in scores} for k in dev}},
                 indent=1))
