"""hffiles for Word, PowerPoint and PDF: the mechanics, the plan contract, PDF export.

Nothing here touches the real files/ workspace (the ``files_dir`` fixture redirects it) and
nothing talks to a model. The PDF-export tests skip themselves when the machine has neither
Microsoft Office nor LibreOffice, because that export is a rendering we cannot fake.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document
from pptx import Presentation

from hffiles import config, convert, files, ops, pdf, plan, slides, word
from hffiles.errors import FileError

RENDERER = any(convert.backends().values())
needs_renderer = pytest.mark.skipif(not RENDERER, reason="no Office or LibreOffice installed")

LETTER = {"save_as": "dopis.docx", "ops": [
    {"op": "add_heading", "text": "Ponuda", "level": 1},
    {"op": "add_paragraph", "text": "Postovani, u nastavku su cijene."},
    {"op": "add_bullets", "items": ["Kruh - 1,50", "Mlijeko - 1,20"]},
    {"op": "add_table", "rows": [["Stavka", "Cijena"], ["Kruh", "1,50"]]},
]}
DECK = {"save_as": "deck.pptx", "ops": [
    {"op": "add_slide", "kind": "title", "title": "Naslov", "subtitle": "Podnaslov"},
    {"op": "add_slide", "kind": "bullets", "title": "Tocke", "bullets": ["prva", "druga"],
     "notes": "govoriti kratko"},
]}
REPORT = {"save_as": "izvjestaj.pdf", "ops": [
    {"op": "write_blocks", "blocks": [
        {"block": "heading", "text": "Izvjestaj"},
        {"block": "para", "text": "Tekst bez dijakritike."},
        {"block": "bullets", "items": ["prva", "druga"]},
        {"block": "table", "rows": [["Stavka", "Iznos"], ["Kruh", "1,50"]]}]},
]}


def make_docx(path: Path, paragraphs=("Prvi odlomak", "Drugi odlomak")) -> Path:
    doc = Document()
    doc.add_heading("Naslov", level=1)
    for text in paragraphs:
        doc.add_paragraph(text)
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Stavka"
    table.cell(0, 1).text = "Kruh"
    doc.save(str(path))
    return path


def make_pptx(path: Path, titles=("Prvi", "Drugi")) -> Path:
    prs = Presentation()
    for title in titles:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
    prs.save(str(path))
    return path


# ------------------------------------------------------------------- routing
@pytest.mark.parametrize("name, kind", [
    ("a.xlsx", "excel"), ("a.xlsm", "excel"), ("a.docx", "word"),
    ("a.pptx", "slides"), ("a.pdf", "pdf"),
])
def test_kind_follows_the_extension(name, kind):
    assert files.kind_of(name) == kind


def test_an_unsupported_extension_is_refused():
    with pytest.raises(FileError, match="unsupported file type"):
        files.kind_of("slike.png")


def test_describe_and_digest_route_by_kind(files_dir, tmp_path, make_xlsx):
    for path in (make_xlsx(), make_docx(tmp_path / "a.docx"), make_pptx(tmp_path / "a.pptx")):
        desc = files.describe(path)
        assert desc["kind"] == files.kind_of(path)
        assert files.digest_text(desc).startswith("file: ")
        assert config.KIND_LABELS[desc["kind"]] in files.summary(desc)


def test_upload_works_for_every_format(files_dir, tmp_path):
    copy = files.upload(make_docx(tmp_path / "a.docx"))
    assert copy.parent == config.IN_DIR and copy.suffix == ".docx"


def test_a_plan_cannot_mix_formats():
    with pytest.raises(FileError, match="cannot mix"):
        ops.parse_plan({"ops": [{"op": "autofit"}, {"op": "add_heading", "text": "x"}]})


def test_operations_must_match_the_file(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "add_heading", "text": "x"}]},
                            source=files.upload(make_xlsx()))
    assert not result.ok
    assert "those are Word operations" in result.error


def test_every_operation_name_is_unique_across_formats():
    assert len(ops.REGISTRY) == sum(len(ops.ops_for(kind)) for kind in config.KINDS)


# ---------------------------------------------------------------------- Word
def test_word_document_is_created_from_a_plan(files_dir):
    result = ops.apply_plan(LETTER)
    assert result.ok and result.saved and result.target.suffix == ".docx"
    desc = files.describe(result.target)
    assert desc["outline"] == [{"level": 1, "text": "Ponuda"}]
    assert desc["tables"][0]["rows"] == 2
    assert desc["words"] > 5


def test_word_bullets_use_list_styles(files_dir):
    result = ops.apply_plan(LETTER)
    doc = Document(str(result.target))
    styles = [p.style.name for p in doc.paragraphs if p.text.startswith("Kruh")]
    assert "List Bullet" in styles


def test_word_replace_covers_body_and_tables(files_dir, tmp_path):
    source = files.upload(make_docx(tmp_path / "a.docx", paragraphs=("Kruh je skup",)))
    result = ops.apply_plan({"ops": [{"op": "replace_text", "find": "Kruh",
                                      "replace": "Pecivo"}]}, source=source)
    assert result.ok
    assert result.results[0].value == 2  # the paragraph and the table cell
    text = " ".join(p.text for p in Document(str(result.target)).paragraphs)
    assert "Pecivo" in text and "Kruh" not in text


def test_word_delete_paragraphs_where(files_dir, tmp_path):
    source = files.upload(make_docx(tmp_path / "a.docx"))
    result = ops.apply_plan({"ops": [{"op": "delete_paragraphs_where",
                                      "contains": "Drugi"}]}, source=source)
    assert result.results[0].value == 1
    assert all("Drugi" not in p.text for p in Document(str(result.target)).paragraphs)


def test_word_read_only_plan_saves_nothing(files_dir, tmp_path):
    source = files.upload(make_docx(tmp_path / "a.docx"))
    result = ops.apply_plan({"ops": [{"op": "count_words"}, {"op": "read_outline"}]},
                            source=source)
    assert result.ok and not result.saved
    assert result.results[0].value > 0
    assert result.results[1].value[0]["text"] == "Naslov"
    assert list(config.OUT_DIR.iterdir()) == []


def test_unknown_paragraph_style_is_a_message(files_dir):
    result = ops.apply_plan({"ops": [{"op": "add_paragraph", "text": "x",
                                      "style": "Nepostojeci"}]})
    assert not result.ok and "no paragraph style" in result.results[0].message


@pytest.mark.parametrize("name, hint", [("staro.doc", "save it as .docx"),
                                        ("makro.docm", "macro-enabled")])
def test_old_and_macro_word_formats_are_refused(tmp_path, name, hint):
    (tmp_path / name).write_bytes(b"x")
    with pytest.raises(FileError, match=hint):
        word.check_path(tmp_path / name)


# ---------------------------------------------------------------- PowerPoint
def test_deck_is_created_from_a_plan(files_dir):
    result = ops.apply_plan(DECK)
    assert result.ok and result.saved and result.target.suffix == ".pptx"
    desc = files.describe(result.target)
    assert desc["slide_count"] == 2
    assert desc["slides"][0]["title"] == "Naslov"
    assert desc["slides"][1]["bullets"] == ["prva", "druga"]
    assert desc["slides"][1]["notes"] == "govoriti kratko"


def test_a_new_deck_is_widescreen(files_dir):
    result = ops.apply_plan(DECK)
    assert files.describe(result.target)["size"].startswith("13.33")


def test_layouts_are_chosen_by_placeholders_not_by_index():
    prs = slides.new_presentation()
    assert "Title Slide" in slides.pick_layout(prs, "title").name
    assert slides.pick_layout(prs, "bullets").name == "Title and Content"
    assert "Section" in slides.pick_layout(prs, "section").name
    assert slides.pick_layout(prs, "blank").name == "Blank"


def test_too_many_bullets_spill_onto_a_continuation_slide(files_dir):
    many = [f"tocka {n}" for n in range(slides.MAX_BULLETS + 3)]
    result = ops.apply_plan({"ops": [{"op": "add_slide", "title": "Puno", "bullets": many}]})
    desc = files.describe(result.target)
    assert desc["slide_count"] == 2
    assert desc["slides"][1]["title"].endswith(slides.CONTINUED)


def test_chart_and_table_slides(files_dir):
    result = ops.apply_plan({"ops": [
        {"op": "add_chart_slide", "kind": "column", "title": "Graf",
         "categories": ["Q1", "Q2"], "values": [10, 20]},
        {"op": "add_table_slide", "title": "Tablica", "rows": [["A", "B"], [1, 2]]},
    ]})
    assert result.ok
    prs = Presentation(str(result.target))
    assert any(shape.has_chart for shape in prs.slides[0].shapes)
    assert any(shape.has_table for shape in prs.slides[1].shapes)


def test_chart_values_must_match_the_categories(files_dir):
    result = ops.apply_plan({"ops": [{"op": "add_chart_slide", "kind": "bar",
                                      "categories": ["Q1", "Q2"], "values": [1]}]})
    assert not result.ok and "they must match" in result.results[0].message


def test_delete_slide_and_notes(files_dir, tmp_path):
    source = files.upload(make_pptx(tmp_path / "a.pptx"))
    result = ops.apply_plan({"ops": [{"op": "set_notes", "index": 1, "text": "biljeska"},
                                     {"op": "delete_slide", "index": 2}]}, source=source)
    assert result.ok
    desc = files.describe(result.target)
    assert desc["slide_count"] == 1
    assert desc["slides"][0]["notes"] == "biljeska"


def test_deleting_a_slide_that_is_not_there_is_a_message(files_dir, tmp_path):
    source = files.upload(make_pptx(tmp_path / "a.pptx"))
    result = ops.apply_plan({"ops": [{"op": "delete_slide", "index": 9}]}, source=source)
    assert not result.ok and "the deck has 2" in result.results[0].message


def test_an_unknown_slide_kind_is_a_message(files_dir):
    result = ops.apply_plan({"ops": [{"op": "add_slide", "kind": "fancy", "title": "x"}]})
    assert not result.ok and "unknown slide kind" in result.results[0].message


# ----------------------------------------------------------------------- PDF
def test_pdf_is_rendered_from_blocks(files_dir):
    result = ops.apply_plan(REPORT)
    assert result.ok and result.saved and result.target.suffix == ".pdf"
    desc = files.describe(result.target)
    assert desc["pages"] == 1 and desc["has_text"]


def test_pdf_keeps_croatian_diacritics(files_dir):
    """reportlab's built-in font cannot write c-caron, so a system TTF must be registered."""
    if pdf.fonts()[0] == "Helvetica":
        pytest.skip("no TrueType font with Latin Extended-A on this machine")
    result = ops.apply_plan({"ops": [{"op": "write_blocks", "blocks": [
        {"block": "para", "text": "čćžšđ"}]}]})
    text = ops.apply_plan({"ops": [{"op": "read_pdf_text"}]}, source=result.target)
    assert "čćžšđ" in text.results[0].value


def test_pdf_pages_can_be_merged_kept_deleted_and_rotated(files_dir):
    first = ops.apply_plan(REPORT).target
    built = ops.apply_plan({"save_as": "spojeno.pdf", "ops": [
        {"op": "merge_pdf", "path": first.name},
        {"op": "merge_pdf", "path": first.name},
        {"op": "rotate_pages", "pages": "1", "degrees": 90},
    ]})
    assert built.ok and files.describe(built.target)["pages"] == 2
    kept = ops.apply_plan({"save_as": "jedna.pdf", "ops": [
        {"op": "merge_pdf", "path": built.target.name},
        {"op": "keep_pages", "pages": "2"}]})
    assert files.describe(kept.target)["pages"] == 1
    deleted = ops.apply_plan({"ops": [{"op": "delete_pages", "pages": "1"}]},
                             source=built.target)
    assert files.describe(deleted.target)["pages"] == 1


def test_deleting_every_page_is_refused(files_dir):
    source = ops.apply_plan(REPORT).target
    result = ops.apply_plan({"ops": [{"op": "delete_pages", "pages": "1"}]}, source=source)
    assert not result.ok and "every page" in result.results[0].message


def test_merge_only_looks_inside_the_workspace(files_dir, tmp_path):
    outside = tmp_path / "tajna.pdf"
    outside.write_bytes(b"%PDF-1.4")
    result = ops.apply_plan({"ops": [{"op": "merge_pdf", "path": str(outside)}]})
    assert not result.ok and "no PDF named" in result.results[0].message


@pytest.mark.parametrize("spec, expected", [
    ("1-3,7", [1, 2, 3, 7]), ("2", [2]), ([1, 3], [1, 3]), ("3-3", [3]),
])
def test_page_specs(spec, expected):
    assert pdf.page_numbers(spec, 10) == expected


@pytest.mark.parametrize("spec, hint", [
    ("12", "does not exist"), ("5-2", "makes no sense"), ("prva", "not a page or range"),
])
def test_bad_page_specs(spec, hint):
    with pytest.raises(FileError, match=hint):
        pdf.page_numbers(spec, 10)


def test_pdf_read_only_plan_saves_nothing(files_dir):
    source = ops.apply_plan(REPORT).target
    before = sorted(p.name for p in config.OUT_DIR.iterdir())
    result = ops.apply_plan({"ops": [{"op": "count_pages"}, {"op": "read_pdf_text"}]},
                            source=source)
    assert result.ok and not result.saved
    assert result.results[0].value == 1
    assert sorted(p.name for p in config.OUT_DIR.iterdir()) == before


def test_an_unknown_block_is_a_message(files_dir):
    result = ops.apply_plan({"ops": [{"op": "write_blocks",
                                      "blocks": [{"block": "video", "text": "x"}]}]})
    assert not result.ok and "unknown block" in result.results[0].message


# ---------------------------------------------------------------- PDF export
def test_backends_are_reported():
    have = convert.backends()
    assert set(have) == {"office", "libreoffice"}
    assert isinstance(have["office"], bool)


def test_a_pdf_source_is_just_copied(files_dir):
    source = ops.apply_plan(REPORT).target
    out = files.to_pdf(source, config.OUT_DIR / "kopija.pdf")
    assert out.read_bytes() == source.read_bytes()


def test_export_to_pdf_refuses_a_non_pdf_target(files_dir):
    source = ops.apply_plan(REPORT).target
    with pytest.raises(FileError, match="must end in .pdf"):
        files.to_pdf(source, config.OUT_DIR / "x.docx")


def test_an_unknown_renderer_is_refused(files_dir):
    source = ops.apply_plan(LETTER).target
    with pytest.raises(FileError, match="office"):
        files.to_pdf(source, prefer="inkscape")


@needs_renderer
def test_a_word_document_exports_to_pdf(files_dir):
    source = ops.apply_plan(LETTER).target
    out = files.to_pdf(source)
    assert out.suffix == ".pdf" and out.stat().st_size > 500
    assert files.describe(out)["pages"] >= 1


@needs_renderer
def test_a_plan_can_ask_for_the_pdf_as_well(files_dir):
    result = ops.apply_plan(dict(DECK, export_pdf=True))
    assert result.ok and result.saved
    assert result.pdf is not None and result.pdf.suffix == ".pdf"
    assert f"PDF: {result.pdf}" in result.lines()


def test_a_failed_pdf_export_does_not_fail_the_plan(files_dir, monkeypatch):
    monkeypatch.setattr(convert, "backends", lambda: {"office": False, "libreoffice": None})
    result = ops.apply_plan(dict(LETTER, export_pdf=True))
    assert result.ok and result.saved  # the .docx is still there
    assert result.pdf is None
    assert any("PDF export failed" in note for note in result.notes)


# ------------------------------------------------------------------- prompts
@pytest.mark.parametrize("kind", ["excel", "word", "slides", "pdf"])
def test_each_format_has_its_own_prompt_and_examples(kind):
    messages = plan.build_messages("Napravi nesto", kind=kind)
    system = messages[0]["content"]
    assert config.KIND_LABELS[kind] in system
    assert plan.EMPTY[kind] in messages[-1]["content"]
    for name in plan.CORE_OPS[kind][:3]:
        assert f"{name}(" in system
    for other in set(config.KINDS) - {kind}:  # no other format's operations leak in
        for name in plan.CORE_OPS[other]:
            if name not in plan.CORE_OPS[kind]:
                assert f"- {name}(" not in system


def test_the_prompt_follows_the_described_file(files_dir, tmp_path):
    desc = files.describe(make_pptx(tmp_path / "a.pptx"))
    system = plan.build_messages("Dodaj slajd", desc)[0]["content"]
    assert "add_slide(" in system and "sort_rows(" not in system


# ----------------------------------------------------------------------- CLI
def test_cli_describe_works_for_every_format(files_dir, tmp_path, make_xlsx, capsys):
    from hffiles import cli

    paths = [make_xlsx(), make_docx(tmp_path / "a.docx"), make_pptx(tmp_path / "a.pptx"),
             ops.apply_plan(REPORT).target]
    for path in paths:
        assert cli.main(["describe", str(path)]) == 0
        out = capsys.readouterr().out
        assert out.startswith("file: ") and path.name in out


def test_cli_describe_json_and_ops_and_where(files_dir, capsys):
    import json as _json

    from hffiles import cli

    target = ops.apply_plan(LETTER).target
    assert cli.main(["describe", str(target), "--json"]) == 0
    assert _json.loads(capsys.readouterr().out)["kind"] == "word"
    for kind in config.KINDS:
        assert cli.main(["ops", "--kind", kind]) == 0
        printed = capsys.readouterr().out
        assert all(f"{name}(" in printed for name in plan.CORE_OPS[kind])
    assert cli.main(["where"]) == 0
    assert "PDF export:" in capsys.readouterr().out


def test_cli_reports_a_bad_file_without_a_traceback(files_dir, tmp_path, capsys):
    from hffiles import cli

    assert cli.main(["describe", str(tmp_path / "nema.xlsx")]) == 1
    assert "error: no such file" in capsys.readouterr().err


@needs_renderer
def test_cli_exports_to_pdf(files_dir, capsys):
    from hffiles import cli

    source = ops.apply_plan(LETTER).target
    assert cli.main(["pdf", str(source)]) == 0
    produced = Path(capsys.readouterr().out.strip())
    assert produced.is_file() and produced.suffix == ".pdf"
