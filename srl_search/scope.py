"""Publication types the site's search and map cover (same set as citewalk/science/scope.py; a test
checks they agree). Datasets, figures, instrument logs and other records stay in the corpus and
can still be asked for with an explicit `type=`."""
SCHOLARLY_TYPES = frozenset({
    'article', 'review', 'conference-paper', 'conference-abstract', 'preprint', 'book', 'book-chapter',
    'dissertation', 'report', 'letter', 'editorial', 'data-paper', 'software-paper'})

# Every other OpenAlex work type (the citation index's type list minus the above).
NON_PUBLICATION_TYPES = ('dataset', 'other', 'paratext', 'supplementary-materials', 'software', 'book-review',
                         'reference-entry', 'peer-review', 'libguides', 'erratum', 'standard', 'retraction')
