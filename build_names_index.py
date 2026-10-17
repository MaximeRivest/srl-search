"""Author names into a tantivy index for /authors?q= (srl_search/people.py); with --institutions, the
institution index's names for /institutions?q= (into <out>/<generation>-institutions).

    search/.venv/bin/python search/build_names_index.py [--citations DIR] [--out DIR] [--min-works 1] [--institutions]

One document per author with at least --min-works publications: the display name (lower-cased, accents folded by
the "names" analyzer), the publication count (fast field: results are ordered by it) and the author's row in
the authorship index. Documents are added most productive first, so equal counts keep that order.
Output: <out>/<generation>/{index/, meta.json}; default /mnt/fast/scholarsreadinglist/search/names-index.
~110M names take about 15 minutes and 4 GB.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srl_search.people import analyzer, schema  # noqa: E402


def log(msg):
    print(time.strftime('%H:%M:%S ') + msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--citations', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=Path('/mnt/fast/scholarsreadinglist/search/names-index'))
    ap.add_argument('--min-works', type=int, default=1)
    ap.add_argument('--limit', type=int, help='smoke test: only the first N authors')
    ap.add_argument('--institutions', action='store_true', help='index the institution names instead')
    args = ap.parse_args()
    import tantivy
    cit = args.citations
    if cit is None:                                       # the citation index of the current data release
        rel = Path('/mnt/fast/scholarsreadinglist/release/current/manifest.json')
        cit = Path(json.loads(rel.read_text())['paths']['citations'])
    A = cit / ('institutions' if args.institutions else 'authors')
    gen = json.loads((cit / 'meta.json').read_text())['generation']
    out = args.out / (gen + ('-institutions' if args.institutions else '') + (f'-first{args.limit}' if args.limit else ''))
    if (out / 'meta.json').exists():
        log(f'already built at {out}')
        return
    shutil.rmtree(out, ignore_errors=True)
    (out / 'index').mkdir(parents=True)
    t0 = time.time()
    works = np.diff(np.load(A / 'i_off.npy')).astype(np.int64) if args.institutions else np.load(A / 'a_works.npy')
    # authors are found best known first (citations); institutions by size
    cites = works if args.institutions else np.load(A / 'a_cites.npy')
    name_off = np.load(A / 'name_off.npy', mmap_mode='r')
    names = np.memmap(A / 'names.bin', np.uint8, 'r')
    rows = np.flatnonzero(works >= args.min_works)
    rows = rows[np.argsort(-cites[rows].astype(np.int64), kind='stable')]
    if args.limit:
        rows = rows[: args.limit]
    log(f'{len(rows):,} authors with >= {args.min_works} publications')
    S = schema(tantivy)
    index = tantivy.Index(S, path=str(out / 'index'))
    index.register_tokenizer('names', analyzer(tantivy))
    writer = index.writer(heap_size=1_500_000_000, num_threads=8)
    Doc = tantivy.Document
    for k, r in enumerate(rows.tolist()):
        a, b = int(name_off[r]), int(name_off[r + 1])
        if b <= a:
            continue
        writer.add_document(Doc(name=bytes(names[a:b]).decode('utf-8', 'replace'), works=int(works[r]), cites=int(cites[r]), row=r))
        if (k + 1) % 5_000_000 == 0:
            log(f'  {k + 1:,} added')
    writer.commit()
    writer.wait_merging_threads()
    (out / 'meta.json').write_text(json.dumps(dict(generation=gen, authors=int(len(rows)), min_works=args.min_works, order='works' if args.institutions else 'cites',
                                                   built_at=time.time()), indent=1))
    log(f'done: {len(rows):,} names in {(time.time() - t0) / 60:.1f} min, '
        f'{sum(f.stat().st_size for f in (out / "index").iterdir()) / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
