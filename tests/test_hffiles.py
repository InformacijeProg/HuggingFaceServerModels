"""hffiles: path hygiene, workbook mechanics, plan validation and plan execution.

Nothing here touches the real files/ workspace (the ``files_dir`` fixture redirects it)
and nothing talks to a model: the plans are the JSON a model would have produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from hffiles import config, excel, ops, plan
from hffiles.errors import FileError


# ------------------------------------------------------------- path hygiene
@pytest.mark.parametrize("given, expected", [
    ("../../Windows/evil.xlsx", "evil.xlsx"),
    ("C:/secret.xlsx", "secret.xlsx"),
    ("sub/dir/report.xlsm", "report.xlsm"),
    ("budget", "budget.xlsx"),
    ("", "rezultat.xlsx"),
    ("payload.exe", "payload.exe.xlsx"),
])
def test_safe_name_keeps_the_name_and_drops_the_path(given, expected):
    assert config.safe_name(given) == expected


def test_out_path_stays_inside_the_workspace(files_dir):
    target = config.out_path("../../Windows/evil.xlsx")
    assert target.parent == config.OUT_DIR
    assert config.is_within(target, files_dir)


def test_unique_path_never_overwrites(files_dir):
    first = config.OUT_DIR / "a.xlsx"
    first.write_bytes(b"x")
    assert config.unique_path(first).name == "a-1.xlsx"


def test_xls_is_refused_with_a_hint(tmp_path):
    old = tmp_path / "staro.xls"
    old.write_bytes(b"x")
    with pytest.raises(FileError, match="save it as .xlsx"):
        excel.check_path(old)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(FileError, match="no such file"):
        excel.check_path(tmp_path / "nema.xlsx")


# ------------------------------------------------------- upload and copying
def test_upload_copies_into_files_in_and_leaves_the_original(files_dir, make_xlsx):
    src = make_xlsx()
    before = src.read_bytes()
    copy = excel.upload(src)
    assert copy.parent == config.IN_DIR
    assert copy.read_bytes() == before
    assert src.read_bytes() == before


def test_upload_twice_keeps_both(files_dir, make_xlsx):
    src = make_xlsx()
    first, second = excel.upload(src), excel.upload(src)
    assert first != second
    assert first.exists() and second.exists()


def test_copy_workbook_is_byte_for_byte(files_dir, make_xlsx):
    src = make_xlsx()
    copy = excel.copy_workbook(src)
    assert copy.read_bytes() == src.read_bytes()
    assert copy.parent == config.OUT_DIR


def test_copy_onto_itself_is_refused(make_xlsx):
    src = make_xlsx()
    with pytest.raises(FileError, match="same file"):
        excel.copy_workbook(src, src)


# --------------------------------------------------------------- describing
def test_describe_finds_sheets_headers_and_kinds(make_xlsx):
    path = make_xlsx(extra_sheets={"Biljeske": [["tekst"]]})
    desc = excel.describe(path)
    assert desc["file"] == path.name
    assert desc["sheet_names"] == ["Prihodi", "Biljeske"]
    first = desc["sheets"][0]
    assert first["headers"] == ["Datum", "Opis", "Iznos", "Kategorija"]
    assert first["column_kinds"][2] == "number"
    assert first["rows"] == 4


def test_digest_is_short_and_mentions_the_columns(make_xlsx):
    digest = excel.digest_text(excel.describe(make_xlsx()))
    assert "C=Iznos (number)" in digest
    assert len(digest) < 600


def test_describe_falls_back_to_formula_text_without_cached_values(files_dir):
    result = ops.create_workbook({"List": [["a"], [None]]})
    ops.apply_plan({"ops": [{"op": "set_cell", "cell": "A2", "value": "=1+1"}]},
                   source=result.target, target=result.target)
    digest = excel.digest_text(excel.describe(result.target))
    assert "=1+1" in digest


# ----------------------------------------------------------- plan validation
def test_unknown_operation_is_refused_with_a_suggestion():
    with pytest.raises(FileError, match="did you mean"):
        ops.parse_plan({"ops": [{"op": "sort_row"}]})


def test_missing_parameter_is_refused_before_any_work():
    with pytest.raises(FileError, match="missing value"):
        ops.parse_plan({"ops": [{"op": "set_cell", "cell": "A1"}]})


def test_unknown_parameter_is_dropped_and_reported(files_dir):
    """An invented parameter is noise, not an instruction - but the user is told."""
    parsed = ops.parse_plan({"ops": [{"op": "autofit", "colour_scheme": "dark"}]})
    assert parsed.steps[0][1] == {}
    assert "ignored unknown parameter(s) colour_scheme" in parsed.warnings[0]
    result = ops.create_workbook({"L": [["a"]]})
    result = ops.apply_plan({"ops": [{"op": "autofit", "colour_scheme": "dark"}]},
                            source=result.target)
    assert result.ok and result.saved
    assert any(line.startswith("note: op 1 (autofit): ignored unknown parameter(s) "
                               "colour_scheme") for line in result.lines())


def test_too_many_operations_is_refused():
    with pytest.raises(FileError, match="over the"):
        ops.parse_plan({"ops": [{"op": "autofit"}] * (config.MAX_PLAN_OPS + 1)})


def test_a_bare_list_and_a_single_op_are_both_plans():
    assert len(ops.parse_plan([{"op": "autofit"}]).steps) == 1
    assert len(ops.parse_plan({"op": "autofit"}).steps) == 1


def test_aliases_and_nested_params_are_accepted():
    parsed = ops.parse_plan({"ops": [{"action": "sort_rows",
                                      "args": {"col": "Iznos", "desc": True}}]})
    spec, params = parsed.steps[0]
    assert spec.name == "sort_rows"
    assert params == {"column": "Iznos", "descending": True}


def test_a_sheet_may_be_named_like_an_operation():
    parsed = ops.parse_plan({"ops": [{"op": "create_sheet", "name": "autofit"}]})
    assert parsed.steps[0][1]["name"] == "autofit"


def test_read_only_plans_do_not_count_as_mutating():
    assert not ops.parse_plan({"ops": [{"op": "aggregate", "func": "sum",
                                        "column": "A"}]}).mutates
    assert ops.parse_plan({"ops": [{"op": "autofit"}]}).mutates


# ------------------------------------------------------------ plan execution
def test_create_workbook_writes_rows_and_bold_header(files_dir):
    result = ops.create_workbook({"Roba": [["Proizvod", "Cijena"], ["Kruh", 1.5]]})
    assert result.ok and result.saved
    ws = load_workbook(result.target)["Roba"]
    assert ws["A1"].value == "Proizvod"
    assert ws["B2"].value == 1.5
    assert ws["A1"].font.bold


def test_a_plan_writes_a_copy_and_leaves_the_source_alone(files_dir, make_xlsx):
    src = excel.upload(make_xlsx())
    before = src.read_bytes()
    result = ops.apply_plan({"ops": [{"op": "set_cell", "cell": "B2", "value": "Promjena"}]},
                            source=src)
    assert result.saved
    assert result.target.parent == config.OUT_DIR
    assert src.read_bytes() == before
    assert load_workbook(result.target)["Prihodi"]["B2"].value == "Promjena"


def test_editing_a_file_already_in_out_updates_it_in_place(files_dir, make_xlsx):
    first = ops.apply_plan({"ops": [{"op": "set_cell", "cell": "A1", "value": "x"}]},
                           source=excel.upload(make_xlsx()))
    second = ops.apply_plan({"ops": [{"op": "set_cell", "cell": "A1", "value": "y"}]},
                            source=first.target)
    assert second.target == first.target
    assert load_workbook(first.target)["Prihodi"]["A1"].value == "y"


def test_save_as_from_a_plan_cannot_escape_the_workspace(files_dir, make_xlsx):
    result = ops.apply_plan({"save_as": "../../Windows/evil.xlsx",
                             "ops": [{"op": "autofit"}]}, source=excel.upload(make_xlsx()))
    assert result.saved
    assert result.target.parent == config.OUT_DIR
    assert result.target.name == "evil.xlsx"


def test_read_only_plan_saves_nothing(files_dir, make_xlsx):
    src = excel.upload(make_xlsx())
    result = ops.apply_plan({"ops": [{"op": "aggregate", "func": "sum", "column": "Iznos"},
                                     {"op": "count_rows"}]}, source=src)
    assert result.ok and not result.saved and result.target is None
    assert list(config.OUT_DIR.iterdir()) == []
    assert result.results[0].value == pytest.approx(1942.5)
    assert result.results[1].value == 3


def test_strict_plan_writes_nothing_when_one_operation_fails(files_dir, make_xlsx):
    src = excel.upload(make_xlsx())
    result = ops.apply_plan({"ops": [{"op": "autofit"},
                                     {"op": "sort_rows", "column": "Nepostojeci"}]}, source=src)
    assert not result.ok and not result.saved
    assert "no column" in result.results[-1].message
    assert list(config.OUT_DIR.iterdir()) == []


def test_loose_plan_saves_what_worked(files_dir, make_xlsx):
    src = excel.upload(make_xlsx())
    result = ops.apply_plan({"ops": [{"op": "set_cell", "cell": "A1", "value": "Dan"},
                                     {"op": "sort_rows", "column": "Nepostojeci"}]},
                            source=src, strict=False)
    assert not result.ok and result.saved
    assert load_workbook(result.target)["Prihodi"]["A1"].value == "Dan"


def test_editing_keeps_existing_formulas(files_dir, make_xlsx):
    src = excel.upload(make_xlsx(rows=[["Iznos"], [10], [20]]))
    first = ops.apply_plan({"ops": [{"op": "set_formula", "cell": "B1",
                                     "formula": "=SUM(A2:A3)"}]}, source=src)
    second = ops.apply_plan({"ops": [{"op": "autofit"}]}, source=first.target)
    assert load_workbook(second.target)["Prihodi"]["B1"].value == "=SUM(A2:A3)"


# ------------------------------------------------------------- the operations
def test_sort_and_total_row_and_formula_column(files_dir, make_xlsx):
    src = excel.upload(make_xlsx())
    result = ops.apply_plan({"ops": [
        {"op": "sort_rows", "column": "Iznos", "descending": True},
        {"op": "add_column", "header": "PDV", "formula": "=C{row}*0.25"},
        {"op": "total_row", "columns": ["Iznos"], "label": "Ukupno"},
    ]}, source=src)
    assert result.ok
    ws = load_workbook(result.target)["Prihodi"]
    assert [ws.cell(row=r, column=3).value for r in (2, 3, 4)] == [1500, 430, 12.5]
    assert ws["E2"].value == "=C2*0.25"
    assert ws["A5"].value == "Ukupno"
    assert ws["C5"].value == "=SUM(C2:C4)"


def test_total_row_as_value_is_computed_by_python(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "total_row", "columns": ["Iznos"], "as": "value"}]},
                            source=excel.upload(make_xlsx()))
    assert load_workbook(result.target)["Prihodi"]["C5"].value == pytest.approx(1942.5)


def test_delete_rows_where_removes_only_matching_rows(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "delete_rows_where", "column": "Kategorija",
                                      "equals": "trosak"}]}, source=excel.upload(make_xlsx()))
    ws = load_workbook(result.target)["Prihodi"]
    assert ws.max_row == 2
    assert ws["D2"].value == "prihod"


def test_find_replace_leaves_formulas_alone(files_dir, make_xlsx):
    src = excel.upload(make_xlsx(rows=[["Opis"], ["Kava"], ['=UPPER("Kava")']]))
    result = ops.apply_plan({"ops": [{"op": "find_replace", "find": "Kava",
                                      "replace": "Caj"}]}, source=src)
    ws = load_workbook(result.target)["Prihodi"]
    assert ws["A2"].value == "Caj"
    assert ws["A3"].value == '=UPPER("Kava")'
    assert "1 formula cells left alone" in result.results[0].message


def test_aggregate_says_what_it_left_out(files_dir, make_xlsx):
    """A read-only plan sees cached values, and openpyxl never computed that formula."""
    src = excel.upload(make_xlsx(rows=[["Iznos"], [10], ["=5+5"]]))
    result = ops.apply_plan({"ops": [{"op": "aggregate", "func": "sum", "column": "Iznos"}]},
                            source=src)
    assert result.results[0].value == 10
    assert "1 empty or uncomputed cells ignored" in result.results[0].message


def test_aggregate_in_a_writing_plan_reports_skipped_formulas(files_dir, make_xlsx):
    """A writing plan keeps formulas, so the same cell is counted as a formula."""
    src = excel.upload(make_xlsx(rows=[["Iznos"], [10], ["=5+5"]]))
    result = ops.apply_plan({"ops": [{"op": "aggregate", "func": "sum", "column": "Iznos"},
                                     {"op": "autofit"}]}, source=src)
    assert "1 formula cells skipped" in result.results[0].message


def test_column_by_header_or_letter_and_an_unknown_one(make_xlsx):
    ws = excel.open_workbook(make_xlsx())["Prihodi"]
    assert excel.column_index(ws, "Iznos") == 3
    assert excel.column_index(ws, "C") == 3
    assert excel.column_index(ws, 3) == 3
    with pytest.raises(FileError, match="headers: Datum"):
        excel.column_index(ws, "Nepostojeci")


def test_a_header_name_still_works_when_has_header_was_set_wrong(files_dir, make_xlsx):
    """Models pass has_header: false by mistake; the name must not then be rejected."""
    result = ops.apply_plan({"ops": [{"op": "sort_rows", "column": "Iznos",
                                      "has_header": False}]},
                            source=excel.upload(make_xlsx()))
    assert result.ok


def test_a_column_with_no_values_says_so(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "add_column", "header": "PDV"}]},
                            source=excel.upload(make_xlsx()))
    assert result.ok
    assert "header only" in result.results[0].message


def test_sheet_lookup_tolerates_case_and_rejects_unknown(make_xlsx):
    wb = excel.open_workbook(make_xlsx())
    assert excel.sheet(wb, "prihodi ").title == "Prihodi"
    assert excel.sheet(wb, None).title == "Prihodi"
    with pytest.raises(FileError, match="no sheet named"):
        excel.sheet(wb, "Nema")


def test_numeric_text_becomes_a_number_but_codes_stay_text(files_dir):
    result = ops.create_workbook({"L": [["a", "b", "c"], ["1500", "0031", "3.5"]]})
    ws = load_workbook(result.target)["L"]
    assert ws["A2"].value == 1500
    assert ws["B2"].value == "0031"
    assert ws["C2"].value == pytest.approx(3.5)


def test_explicit_types_are_honoured(files_dir):
    result = ops.apply_plan({"ops": [
        {"op": "set_cell", "cell": "A1", "value": "2026-01-31", "type": "date"},
        {"op": "set_cell", "cell": "A2", "value": "31.01.2026", "type": "date"},
        {"op": "set_cell", "cell": "A3", "value": "1500", "type": "text"},
        {"op": "set_cell", "cell": "A4", "value": "SUM(A1:A2)", "type": "formula"},
    ]})
    ws = load_workbook(result.target).active
    assert ws["A1"].value.strftime("%Y-%m-%d") == "2026-01-31"
    assert ws["A2"].value.strftime("%Y-%m-%d") == "2026-01-31"
    assert ws["A3"].value == "1500"
    assert ws["A4"].value == "=SUM(A1:A2)"


def test_a_bad_date_is_a_message_not_a_crash(files_dir):
    result = ops.apply_plan({"ops": [{"op": "set_cell", "cell": "A1", "value": "jucer",
                                      "type": "date"}]})
    assert not result.ok and not result.saved
    assert "not a date" in result.results[0].message


def test_format_range_sets_bold_fill_and_number_format(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "format_range", "range": "1", "bold": True,
                                      "fill": "lightblue"},
                                     {"op": "format_range", "column": "Iznos",
                                      "number_format": "#,##0.00"}]},
                            source=excel.upload(make_xlsx()))
    ws = load_workbook(result.target)["Prihodi"]
    assert ws["A1"].font.bold
    assert str(ws["A1"].fill.start_color.rgb).endswith("DDEBF7")
    assert ws["C2"].number_format == "#,##0.00"


def test_an_unknown_colour_is_a_message(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "format_range", "range": "1", "fill": "nebesko"}]},
                            source=excel.upload(make_xlsx()))
    assert not result.ok and "not a colour" in result.results[0].message


def test_sheet_operations(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "create_sheet", "name": "Sazetak"},
                                     {"op": "rename_sheet", "sheet": "Prihodi", "to": "Promet"},
                                     {"op": "copy_sheet", "sheet": "Promet", "to": "Promet 2"}]},
                            source=excel.upload(make_xlsx()))
    assert load_workbook(result.target).sheetnames == ["Promet", "Sazetak", "Promet 2"]


def test_the_last_sheet_cannot_be_deleted(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "delete_sheet", "sheet": "Prihodi"}]},
                            source=excel.upload(make_xlsx()))
    assert not result.ok and "at least one sheet" in result.results[0].message


def test_chart_is_added(files_dir, make_xlsx):
    result = ops.apply_plan({"ops": [{"op": "add_chart", "kind": "bar", "data": "C1:C4",
                                      "categories": "B2:B4", "title": "Iznosi"}]},
                            source=excel.upload(make_xlsx()))
    assert result.ok
    assert len(load_workbook(result.target)["Prihodi"]._charts) == 1


# --------------------------------------------------------------------- csv
def test_csv_roundtrip_with_croatian_decimal_commas(files_dir):
    (config.IN_DIR / "roba.csv").write_text("Proizvod;Cijena\nKruh;1,50\nMlijeko;1,20\n",
                                            encoding="utf-8")
    created = ops.apply_plan({"save_as": "roba.xlsx",
                              "ops": [{"op": "import_csv", "path": "roba.csv"}]})
    ws = load_workbook(created.target).active
    assert ws["B2"].value == pytest.approx(1.5)
    exported = ops.apply_plan({"ops": [{"op": "export_csv", "name": "izvoz.csv"}]},
                              source=created.target)
    text = Path(exported.results[0].value).read_text(encoding="utf-8-sig")
    assert text.splitlines()[1] == "Kruh;1.5"


def test_import_csv_only_looks_inside_the_workspace(files_dir, tmp_path):
    outside = tmp_path / "tajna.csv"
    outside.write_text("a;b\n", encoding="utf-8")
    result = ops.apply_plan({"ops": [{"op": "import_csv", "path": str(outside)}]})
    assert not result.ok and "no csv named" in result.results[0].message


# -------------------------------------------------------- extracting the plan
@pytest.mark.parametrize("answer", [
    '{"ops": [{"op": "autofit"}]}',
    'Sure!\n```json\n{"ops": [{"op": "autofit"}]}\n```\nHope that helps.',
    'Here is the plan: {"ops": [{"op": "autofit"},]}',
    '{"ops": [{"op": "autofit", "sheet": None}]}',
    '{"ops": [{"op": "autofit"}]} and then save it',
])
def test_extract_plan_survives_the_usual_wrapping(answer):
    raw = plan.extract_plan(answer)
    assert ops.parse_plan(raw).steps[0][0].name == "autofit"


def test_extract_plan_returns_none_without_json():
    assert plan.extract_plan("Naravno, sortirat cu tablicu!") is None
    assert plan.extract_plan("") is None


def test_prose_only_answer_is_reported_as_a_bad_plan():
    result = ops.apply_plan("Naravno!")
    assert not result.ok and result.error == "the plan is text, not JSON"


def test_the_examples_name_nothing_that_could_be_mistaken_for_real_data():
    """The 1.7B copies what it is shown, so the examples must be unmistakably fake."""
    for kind, shots in plan.FEW_SHOT.items():
        shown = " ".join(m["content"] for m in shots)
        for real_looking in ("Iznos", "Prihodi", "Kruh", "Proizvod"):
            assert real_looking not in shown, f"{kind} examples mention {real_looking}"


def test_build_messages_shows_the_rules_the_ops_and_the_file(make_xlsx):
    messages = plan.build_messages("Sortiraj po iznosu", excel.describe(make_xlsx()))
    assert messages[0]["role"] == "system"
    assert "JSON only" in messages[0]["content"]
    assert "sort_rows(" in messages[0]["content"]
    assert messages[-1]["content"].endswith("Request: Sortiraj po iznosu")
    assert "C=Iznos (number)" in messages[-1]["content"]


def test_build_messages_without_a_file_says_so():
    assert plan.NEW_WORKBOOK in plan.build_messages("Napravi tablicu")[-1]["content"]


def test_every_registered_op_has_help_and_the_core_set_exists():
    assert len(ops.REGISTRY) > 20
    for name, spec in ops.REGISTRY.items():
        assert spec.help, f"{name} has no help text"
        assert name == name.lower()
    assert set(plan.CORE_OPS) == set(config.KINDS)
    for kind, names in plan.CORE_OPS.items():
        assert set(names) <= set(ops.ops_for(kind)), f"{kind} core ops are not all registered"
    json.dumps(ops.reference())
