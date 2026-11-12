"""Parser and chunker tests for hfzakon (synthetic HTML, no network)."""
from __future__ import annotations

from hfzakon import nn, zakon_hr
from hfzakon import config as zconfig

LAW_HTML = """
<html><body>
<div class="main-zakon-header">
  <h1>Zakon o testiranju</h1>
  <p>NN 10/24, 5/25</p>
  <p>na snazi od 01.01.2025.</p>
</div>
<span class="status-zakona" id="na-snazi">Na snazi</span>
<div id="pretrazitekst">
  <div class="cms-block"><div><h3 class="cms-zakon-h3">I. OPĆE ODREDBE</h3></div></div>
  <div class="cms-block">
    <div><p class="cms-zakon-clanak">Članak 1.</p></div>
    <div><p class="cms-zakon-clanak">Predmet Zakona</p></div>
    <div><p>(1) Ovim se Zakonom uređuje testiranje parsera i ništa drugo posebno.</p>
         <p>(2) Testiranje se provodi redovito i temeljito, bez iznimke i odgode.</p></div>
  </div>
  <div class="cms-block">
    <div><p class="cms-zakon-clanak">Članak 2.a</p></div>
    <div><p>Ovaj članak nema naslov, ali ima dovoljno dugačak sadržaj da prođe minimum.</p></div>
  </div>
</div>
</body></html>
"""


def test_parse_law_articles_and_meta():
    law = zakon_hr.parse_law(LAW_HTML.encode(), "https://www.zakon.hr/z/42/zakon-o-testiranju")
    assert law["title"] == "Zakon o testiranju"
    assert law["nn_refs"] == "NN 10/24, 5/25"
    assert law["status"] == "Na snazi"
    assert [a["number"] for a in law["articles"]] == ["1", "2a"]
    a1 = law["articles"][0]
    assert a1["title"] == "Predmet Zakona"
    assert a1["section"] == "I. OPĆE ODREDBE"
    assert len(a1["paragraphs"]) == 2


def test_chunk_law_header_and_articles():
    law = zakon_hr.parse_law(LAW_HTML.encode(), "https://www.zakon.hr/z/42/zakon-o-testiranju")
    chunks = list(zakon_hr.chunk_law(law))
    ids = [c[0] for c in chunks]
    assert ids == ["zakon:42:head", "zakon:42:cl1", "zakon:42:cl2a"]
    head_text = chunks[0][1]
    assert "Zakon o testiranju" in head_text and "NN 10/24" in head_text
    cl1_id, cl1_text, cl1_meta = chunks[1]
    assert cl1_text.startswith("Zakon o testiranju — Članak 1. (Predmet Zakona)")
    assert "(2) Testiranje" in cl1_text
    assert cl1_meta["clanak"] == "1" and cl1_meta["law_id"] == "42"


def test_long_article_splits_by_stavak():
    paras = [f"({i}) " + "x" * 400 for i in range(1, 11)]
    law = {"id": "7", "url": "u", "title": "Zakon o dugim člancima", "nn_refs": "NN 1/25",
           "effective": "", "status": "", "preamble": [],
           "articles": [{"number": "5", "title": "Dugi", "section": "", "paragraphs": paras}]}
    chunks = list(zakon_hr.chunk_law(law))
    parts = [c for c in chunks if c[0].startswith("zakon:7:cl5:p")]
    assert len(parts) >= 3
    assert all(len(text) <= zconfig.MAX_CHUNK_CHARS for _, text, _ in parts)
    # each part repeats the law+article header and labels its stavak range
    assert all(text.startswith("Zakon o dugim člancima — Članak 5. (Dugi) st.") for _, text, _ in parts)
    joined = "".join(text for _, text, _ in parts)
    assert all(f"({i})" in joined for i in range(1, 11))


def test_old_law_inline_clan_fallback():
    html = """
    <html><body><div class="main-zakon-header"><h1>Zakon o starom pravu</h1></div>
    <div id="pretrazitekst"><div class="cms-block"><div>
    <p>Član 1. Prvi član uređuje nešto staro i važno za sve građane podjednako.
    Član 2. Drugi član uređuje nešto drugo, jednako staro i jednako važno.</p>
    </div></div></div></body></html>"""
    law = zakon_hr.parse_law(html.encode(), "https://www.zakon.hr/z/9/zakon-o-starom-pravu")
    assert [a["number"] for a in law["articles"]] == ["1", "2"]
    assert "Prvi član uređuje" in law["articles"][0]["paragraphs"][0]


def test_c_zakon_layout_and_expired_suffix():
    html = """
    <html><body><h1>Zakon o isteklom</h1>
    <p><a>10/20</a> , <a>5/24</a> na snazi od 01.01.2024. do 01.09.2025.</p>
    <div class="cms-text-content">
    <h3>I. OPĆE ODREDBE</h3>
    <p>Članak 1.</p>
    <p>Ovaj zakon je istekao, ali njegov tekst ostaje dostupan radi povijesti primjene.</p>
    <p>Članak 2.</p>
    <p>Drugi članak isteklog zakona, s dovoljno teksta da prijeđe minimalnu duljinu.</p>
    </div></body></html>"""
    law = zakon_hr.parse_law(html.encode(), "https://www.zakon.hr/z/8/zakon-o-isteklom")
    assert law["title"] == "Zakon o isteklom"
    assert law["nn_refs"].startswith("NN 10/20")
    assert law["effective"].startswith("na snazi od 01.01.2024.")
    assert [a["number"] for a in law["articles"]] == ["1", "2"]
    chunks = list(zakon_hr.chunk_law(law))
    cl1 = next(text for cid, text, _ in chunks if cid == "zakon:8:cl1")
    assert cl1.startswith("Zakon o isteklom (na snazi do 01.09.2025.) — Članak 1.")


NN_HTML = """
<html><head><title>t</title></head><body>
<div id="sticky" class="title"><h2>Odluka o testnoj naknadi</h2></div>
<div id="html-content-frame">
  <span class="title-for-print"><h3>NN 7/2025 (15.1.2025.), Odluka o testnoj naknadi</h3></span>
  <style>.x{}</style>
  <p>Na temelju članka 1. Zakona o testiranju donosi se sljedeća odluka o naknadi.</p>
  <p>Članak 1.</p>
  <p>Naknada iznosi 10 eura mjesečno za svakoga obveznika u Republici Hrvatskoj.</p>
  <p>Članak 2.</p>
  <p>Ova odluka stupa na snagu osmoga dana od dana objave u Narodnim novinama.</p>
</div>
<div class="contentRight metadata">
  <div>Vrsta dokumenta: Odluka</div>
  <div>Donositelj: Testno tijelo</div>
  <a href="https://narodne-novine.nn.hr/eli/sluzbeni/2025/7/123">ELI</a>
</div>
</body></html>
"""


def test_parse_and_chunk_nn_doc_by_article():
    doc = nn.parse_doc(NN_HTML.encode(), "https://narodne-novine.nn.hr/clanci/sluzbeni/2025_01_7_123.html")
    assert doc["title"] == "Odluka o testnoj naknadi"
    assert doc["issue"] == "7/2025" and doc["date"] == "15.1.2025."
    assert doc["vrsta"] == "Odluka" and doc["donositelj"] == "Testno tijelo"
    assert doc["eli"].endswith("/eli/sluzbeni/2025/7/123")
    chunks = list(nn.chunk_doc(doc, "2025_01_7_123"))
    ids = [c[0] for c in chunks]
    assert ids == ["nn:2025_01_7_123:uvod", "nn:2025_01_7_123:cl1", "nn:2025_01_7_123:cl2"]
    assert all(text.startswith("Odluka o testnoj naknadi\nNN 7/2025, 15.1.2025., Odluka, Testno tijelo")
               for _, text, _ in chunks)
    assert "Naknada iznosi 10 eura" in chunks[1][1]


def test_nn_prose_doc_single_chunk():
    html = NN_HTML.replace("Članak 1.", "Prvo poglavlje.").replace("Članak 2.", "Drugo poglavlje.")
    doc = nn.parse_doc(html.encode(), "u")
    chunks = list(nn.chunk_doc(doc, "k"))
    assert [c[0] for c in chunks] == ["nn:k:c1"]


def test_refresh_state_and_plan(tmp_path, monkeypatch):
    from hfzakon import refresh

    monkeypatch.setattr(refresh, "STATE_FILE", tmp_path / "last_refresh.json")
    monkeypatch.setattr(zconfig, "RAW_NN_SEARCH_DIR", tmp_path / "search")
    monkeypatch.setattr(zconfig, "ensure_dirs", lambda: None)
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "2025_158.html").write_text("x")
    monkeypatch.setattr(refresh.nn, "editions", lambda year: {2025: 158, 2026: 120}.get(year, 0))
    monkeypatch.setattr(refresh, "nn_years", lambda: [2025, 2026])

    levels = refresh.plan()
    assert levels[2025] == (158, 158)      # recorded from cached search pages, nothing new
    assert levels[2026] == (0, 120)        # nothing cached in tmp -> everything is new

    refresh.write_state({"nn": {"2025": 158}})
    st = refresh.read_state()
    assert st["nn"]["2025"] == 158 and st["ts"]
    assert refresh.plan()[2025] == (158, 158)


def test_maybe_run_skips_fresh_state(tmp_path, monkeypatch):
    from hfzakon import refresh

    monkeypatch.setattr(refresh, "STATE_FILE", tmp_path / "last_refresh.json")
    monkeypatch.setattr(zconfig, "ensure_dirs", lambda: None)
    called = []
    monkeypatch.setattr(refresh, "run", lambda progress=None: called.append(1) or {})
    refresh.write_state({"nn": {}})        # just refreshed -> the gate must skip
    assert refresh.maybe_run() is None
    assert not called
    refresh.STATE_FILE.unlink()            # no state at all -> the gate must run
    assert refresh.maybe_run() == {}
    assert called
