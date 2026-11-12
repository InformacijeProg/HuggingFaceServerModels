# Working with files (hffiles)

`hffiles` lets a model work on real files without ever writing one. The model turns a
request into a **plan** — a small JSON list of operations — and Python validates that plan
and executes it with openpyxl, python-docx, python-pptx or pypdf/reportlab.

```
request ──▶ prompt (rules + operations for this format + what is in the file)
                        │
                     model
                        │
                  plan (JSON)  ──▶ validate ──▶ the format's library ──▶ files/out/<name>
                                      │
                                      └── unknown op / missing argument / bad path ──▶ error, nothing written
```

Why this way round: a 1.7B model cannot be trusted to produce a correct spreadsheet or a
readable slide, but it is good enough at picking `sort_rows` or `add_slide` and naming the
content. Totals, dates, sorting, layout and page breaks are then exactly as correct as the
Python that does them, and the same request twice gives the same file.

## Formats

| Format | Library | Create | Modify | Not supported |
|---|---|---|---|---|
| Excel `.xlsx` `.xlsm` | openpyxl | yes | yes | `.xls` (save as .xlsx first) |
| Word `.docx` | python-docx | yes | yes | `.doc`, `.docm` (macros would be lost) |
| PowerPoint `.pptx` | python-pptx | yes | yes | `.ppt`, `.pptm` |
| PDF `.pdf` | pypdf + reportlab | yes (rendered pages) | yes (pages, metadata) | editing existing text in place |

All four are MIT/BSD licensed pure-Python libraries — no Office installation is needed to
read or write any of these files. Office or LibreOffice is needed only to **export to PDF**
(see below), because that is a rendering, not a file conversion.

The format follows from the extension, and operation names are unique across formats, so a
plan never has to say which format it means — and a plan that mixes two is refused.

## The workspace

| Path | Content |
|---|---|
| `files/in/` | uploads, copied in untouched — the user's own file is never opened for writing |
| `files/out/` | everything hffiles produces; the only place a model-supplied name can reach |

`HFFILES_HOME` moves the workspace (default `HFSERVE_HOME/files`).

Rules that hold for every format:

* A plan that only reads (`aggregate`, `count_words`, `list_slides`, `read_pdf_text`, …)
  **saves nothing**.
* A plan that writes never touches its source. The result goes to `files/out/`; a source
  that already lives there is the working copy of an ongoing edit and is updated in place,
  so a conversation of small changes does not leave twenty near-identical files behind.
* `strict` (the default) is all-or-nothing: the first failing operation stops the plan and
  nothing is saved.
* `save_as` from a plan is reduced to a bare file name with the right extension.
  `"../../Windows/evil.xlsx"` becomes `files/out/evil.xlsx`, and existing files are never
  silently overwritten (`evil-1.xlsx`).
* An unknown *operation* is an error; an unknown *parameter* is dropped and reported as a
  `note:` line, because small models decorate operations with arguments that do not exist.

## In the GUI

`python -m hfserve gui` (or `HFServe-Chat.cmd`) has a **Datoteka** row:

1. **Priloži …** — pick an `.xlsx`, `.docx`, `.pptx` or `.pdf`. It is copied into
   `files/in/`, read, and its digest (sheets and columns, or headings, or slide titles, or
   pages) is printed in the transcript: exactly what the model will see.
2. Ask for what you want in plain language — *"sortiraj po iznosu i dodaj zbroj"*, *"dodaj
   slajd s tri točke"*, *"zadrži prve tri stranice"*. While a file is attached every message
   becomes a plan, and the RAG toggles are ignored.
3. The plan and the result of each operation are printed, ending in `saved: …`. That result
   becomes the new working file, so the next request builds on it.
4. **Spremi kao …** is the download button: it copies the file being worked on wherever you
   want it. **U PDF** renders it to PDF. **Otvori mapu** opens `files/out/`, and **Odvoji**
   goes back to ordinary chat.

Nothing is written if the model's answer is not a plan, or if any operation fails.

## From the command line

```powershell
python -m hffiles describe <file>                # what the model gets to see (any format)
python -m hffiles ops --kind word [--all]        # the operations a plan may use
python -m hffiles copy <file>                    # byte-for-byte copy into files/out
python -m hffiles upload <file>                  # copy into files/in
python -m hffiles new roba.xlsx --csv roba.csv --sheet Roba
python -m hffiles pdf dopis.docx [izlaz.pdf] [--prefer libreoffice]
python -m hffiles apply plan.json --source files/in/prihodi.xlsx [--pdf]
python -m hffiles ask "dodaj stupac PDV, 25% iznosa" --file files/in/prihodi.xlsx
python -m hffiles ask "napravi prezentaciju o prihodima" --kind slides --dry-run
python -m hffiles where                          # folders, formats and PDF renderers
```

`ask` needs a running server with a model loaded (`hfserve serve`, `hfserve load`). It is the
whole loop without the GUI; `--dry-run` is the honest way to see what the model actually
produced, `--show-prompt` prints what was sent, and `--pdf` exports the result as well.

## In code

```python
from hffiles import files, ops, plan

desc = files.describe("files/in/prihodi.xlsx")               # works for any format
messages = plan.build_messages("sortiraj po iznosu", desc)   # ask any model with these
raw = plan.extract_plan(answer)                              # JSON out of fences and prose
result = ops.apply_plan(raw, source="files/in/prihodi.xlsx")
print(result.ok, result.target, result.summary())

files.upload("C:/Users/me/ponuda.docx")   # into files/in, original untouched
files.copy_file("files/in/ponuda.docx")   # byte-for-byte copy into files/out
files.to_pdf("files/out/ponuda.docx")     # rendered PDF next to it

# no model needed when the data is already in hand
ops.create_workbook({"Roba": [["Proizvod", "Cijena"], ["Kruh", 1.5]]}, target="roba.xlsx")
```

## The plan format

```json
{
  "save_as": "ponuda.docx",
  "export_pdf": true,
  "ops": [
    {"op": "add_heading", "text": "Ponuda za 2026.", "level": 1},
    {"op": "add_paragraph", "text": "Poštovani, u nastavku su naše cijene."},
    {"op": "add_table", "rows": [["Stavka", "Cijena"], ["Kruh", "1,50"]]}
  ]
}
```

`sheet` (Excel) is optional everywhere and means the first sheet. `"export_pdf": true`
renders the result to PDF as well. `python -m hffiles ops --kind <format> --all` prints
every operation with its parameters. The vocabularies:

**Excel** (33 operations) — `set_cell`, `set_formula`, `write_rows`, `append_row`,
`add_column`, `total_row`, `insert_rows`, `delete_rows`, `insert_columns`,
`delete_columns`, `delete_rows_where`, `sort_rows`, `find_replace`, `create_sheet`,
`delete_sheet`, `rename_sheet`, `copy_sheet`, `list_sheets`, `format_range`,
`column_width`, `autofit`, `freeze_panes`, `merge_cells`, `unmerge_cells`, `auto_filter`,
`add_chart`, `read_cell`, `read_range`, `aggregate`, `count_rows`, `find_rows`,
`import_csv`, `export_csv`.

* Columns may be named by header text (`"Iznos"`), letter (`"C"`) or number (`3`).
* `range` takes `"B2:D10"`, a cell, columns (`"C"`, `"C:E"`) or rows (`"1"`, `"2:10"`).
* `add_column` expands `{row}` in a formula for every data row, so the model never counts rows.
* `set_cell` takes a `type`: `auto` (default), `text`, `number`, `date`, `bool`, `formula`.
  Under `auto`, `"1500"` becomes the number 1500 while `"0031"` stays text, because a
  leading zero means a code. Dates are ISO (`2026-01-31`) or Croatian (`31.01.2026`) and are
  written with the `dd.mm.yyyy` format; a date-looking string under `auto` stays text, so
  ask for `"type": "date"`.

**Word** (11) — `add_heading`, `add_paragraph`, `add_bullets`, `add_table`,
`add_page_break`, `replace_text`, `delete_paragraphs_where`, `set_properties`,
`read_outline`, `read_document`, `count_words`.

A document is a stream of blocks, so every writing operation appends to the end and there is
deliberately no "edit paragraph 7" — a positional edit breaks as soon as anything above it
shifts. Reshaping an existing document is `replace_text` and `delete_paragraphs_where`;
`add_bullets` items may be `"text"` or `{"text", "level"}`.

**PowerPoint** (7) — `add_slide`, `add_table_slide`, `add_chart_slide`, `delete_slide`,
`set_notes`, `replace_slide_text`, `list_slides`.

The model never gets a coordinate. It names a slide *kind* — `title`, `bullets` (default),
`section`, `two`, `blank` — and Python picks the matching layout from the template by
inspecting its placeholders, fills them, sizes the font to the number of bullets, and spills
more than seven bullets onto a continuation slide. A new deck is 16:9.

**PDF** (8) — `write_blocks`, `merge_pdf`, `keep_pages`, `delete_pages`, `rotate_pages`,
`set_pdf_metadata`, `read_pdf_text`, `count_pages`.

`write_blocks` renders new pages with reportlab from the same vocabulary Word uses
(`heading`, `para`, `bullets`, `numbers`, `table`, `page_break`); the others rearrange the
pages that are already there, losslessly. Pages are given as `"1-3,7"`.

A model's wording is tolerated where it is unambiguous: `action`/`name` for `op`,
`params`/`args` for nested arguments, `col`→`column`, `desc`→`descending`, `ref`→`range`.

## Export to PDF

`files.to_pdf(path)`, `python -m hffiles pdf <file>`, the GUI's **U PDF** button and
`"export_pdf": true` in a plan all go through one place, which uses whichever renderer the
machine has:

1. **Microsoft Office** through COM (pywin32): highest fidelity, Windows only, and it also
   recalculates a workbook's formulas on the way out. It runs in a **child process** with
   `DispatchEx`, so it starts its own Office instance instead of driving — and then
   quitting — the one the user has open, and Office's slow asynchronous shutdown cannot
   disturb the GUI.
2. **LibreOffice** `soffice --headless --convert-to pdf`: cross-platform, good fidelity.

With neither installed the export fails with a message saying so. It does not fall back to
re-typesetting the content with reportlab, because that would silently produce a document
that differs from the file the user has. A PDF source is simply copied.

## Limits worth knowing

* **openpyxl does not calculate.** A formula is stored, not evaluated, so reading a workbook
  Excel has never saved shows no value for it, and `aggregate` reports how many cells it had
  to ignore. `total_row` writes `=SUM(...)` by default, or `"as": "value"` for a number
  Python computed. (A PDF export through Office *does* recalculate.)
* **Editing rewrites the file.** None of these libraries round-trips everything: openpyxl
  drops pivot tables and pre-existing charts, python-pptx drops some animations, python-docx
  drops comments and tracked changes. For files like those `hffiles copy` (or
  `files.copy_file`) copies bytes; the plan path is for content and layout.
* **Sorting moves values, not formatting** (cell styles belong to cells in the file format).
* **`replace_text` in Word** keeps run formatting when the match sits inside one run; a match
  spanning differently formatted runs collapses to the first run's formatting.
* **PDF text is a rendering**, so there is no editing of existing text in place — add pages,
  or rebuild the content and render it again.
* **Croatian in PDFs** needs a TrueType font: reportlab's built-in Helvetica is
  WinAnsi-encoded and has no č, ć or đ, so `pdf.fonts()` registers Calibri, Arial or
  DejaVuSans from the system. With none of them present the text would lose those letters,
  and the test for it skips rather than pretending.
* Guard rails: 64 MB per file, 80 operations per plan, 100 000 rows / 500 000 cells per Excel
  write, 12×8 cells for a table on a slide, 100 slides, 5000 PDF pages.

## How well does the 1.7B plan?

Measured with `utter-project/EuroLLM-1.7B-Instruct` on 2026-10-16, four Croatian requests
through `hffiles ask` on a spreadsheet:

| Request | Plan | Result |
|---|---|---|
| "Sortiraj po iznosu od najvećeg" | `sort_rows` | correct |
| "Dodaj stupac PDV koji je 25% iznosa" | `add_column` without `formula` | header only, the percentage was lost |
| "Napravi tablicu s tri grada i brojem stanovnika" | `write_rows` | valid file, nonsense content ("Grad1", "Grad2") |
| "Koliki je ukupan iznos?" | `total_row` on the wrong column | wrong operation picked |

What this says: the *plumbing* is sound — every bad plan was rejected or reported, and no
file was ever corrupted — but the planner is the weak link. Two things improved it
noticeably: examples with obviously fake names (the model copies whatever it is shown; with
`"Proizvodi"` in the examples it kept addressing a sheet that did not exist) and a rule that
one request is usually one operation (without it, plans were padded with eight extra
operations copied from the examples).

So: structural errors are cheap (they fail loudly), semantic errors are not (a wrong column
is a plausible-looking file). If plans matter, use a larger model for this prompt, or
fine-tune on request/plan pairs — plans are ideal training data, since each one is short,
verifiable by `ops.parse_plan`, and gradable by running it.

## Adding another format

`excel.py`, `word.py`, `slides.py` and `pdf.py` are the mechanics; `ops.py` is the contract
(registry, validation, execution); `plan.py` is the prompt and the JSON extraction;
`files.py` is the switchboard that routes by extension. A new format adds its own mechanics
module, registers operations with `@op(..., kind="<format>")`, adds its suffixes and labels
to `config.py`, its prompt pieces to `plan.py`, and itself to `files.MODULES`. Keep the two
rules that make this safe: every parameter is validated before anything is touched, and the
model never supplies a path.
