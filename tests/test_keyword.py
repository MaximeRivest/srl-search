import pytest

from srl_search.keyword import looks_boolean, normalize


@pytest.mark.parametrize("q,want", [
    ('"gut microbiome" autism', '"gut microbiome" AND autism'),
    ('sleep OR insomnia', 'sleep OR insomnia'),
    ('(sleep OR insomnia) adolescents NOT caffeine', '( sleep OR insomnia ) AND adolescents NOT caffeine'),
    ('teens AND NOT caffeine', 'teens NOT caffeine'),
    ('title:(sleep AND teens) "drinking water"', 'title:( sleep AND teens ) AND "drinking water"'),
    ('title:"sleep deprivation" mice', 'title:"sleep deprivation" AND mice'),
    ('microplastic* -review', 'microplastic* AND -review'),
    ('"BRCA1 ovarian"~5', '"BRCA1 ovarian"~5'),
    ('AND sleep OR', 'sleep'),
    ('(sleep OR insomnia', '( sleep OR insomnia )'),
    ('sleep) teens', 'sleep AND teens'),
    ('"unbalanced quote sleep', 'unbalanced AND quote AND sleep'),
])
def test_normalize(q, want):
    assert normalize(q) == want


def test_looks_boolean():
    for q in ['"gut microbiome"', "sleep AND teens", "a OR b", "x NOT y", "(a b)", "microplastic*",
              "sleep -review", "title:sleep"]:
        assert looks_boolean(q), q
    for q in ["why do we need sleep", "can exercise help depression", "and or not lowercase",
              "covid-19 vaccines", "e-cigarettes in teens"]:
        assert not looks_boolean(q), q


def test_keyword_index(tmp_path):
    import tantivy
    from srl_search.keyword import KeywordIndex, schema
    ix = tantivy.Index(schema(), path=str(tmp_path))
    w = ix.writer(heap_size=50_000_000, num_threads=1)
    docs = [
        (1, "Gut microbiome in autism", "children with autism spectrum disorder", 2021, "article", 1),
        (2, "Sleep in adolescents", "insomnia and caffeine use in teens", 2020, "article", 0),
        (3, "Sleep in adolescents without caffeine", "adolescents sleeping", 2019, "review", 1),
        (4, "Journal cover", "gut microbiome autism", 2022, "paratext", 1),
        (5, "Microplastics in drinking water", "microplastic particles in tap water", 2023, "article", 1),
    ]
    for i, title, ab, year, typ, smr in docs:
        w.add_document(tantivy.Document(id=i, title=title, abstract=ab, keywords="", year=year, type=typ,
                                        license="cc-by", oa=1, retracted=0, pmc_oa=smr, smr=smr))
    w.commit(); w.wait_merging_threads()
    k = KeywordIndex(tmp_path)
    ids = lambda q, f={}, c="all": [h for h, _ in k.search(q, 10, f, c)[0]]
    assert ids('"gut microbiome" autism') == [1]                 # paratext left out by default
    assert ids('"gut microbiome" autism', {"type": "paratext"}) == [4]
    assert set(ids("sleep adolescents")) == {2, 3}
    assert ids("sleep adolescents NOT caffeine") == []           # 3 has caffeine in its title
    assert ids("sleep AND adolescents", c="smr") == [3]
    assert ids("sleep", {"year_from": 2020}) == [2]
    assert ids("microplastic*") == [5]
    assert ids("title:(drinking water)") == [5]
    hits, matches, understood = k.search("sleep OR autism", 1, {})
    assert matches == 3 and len(hits) == 1 and understood == "sleep OR autism"
