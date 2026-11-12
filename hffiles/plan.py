"""The model-facing half: ask for a plan, get a plan out of the answer.

``build_messages`` writes the prompt (rules, the operation list straight from
``ops.REGISTRY``, the file's digest, worked examples) for whichever format the file is,
and ``extract_plan`` digs the JSON back out of whatever the model wrapped it in. Both are
deliberately boring: the model's only job is to name operations and their arguments, and a
plan that comes out wrong is caught by ``ops.parse_plan`` instead of corrupting a file.

Small models follow demonstrations far better than instructions (the same lesson as
hfrag's few-shot prompts), so the examples below do most of the work. They use obviously
fake names - "List1", "Stupac2", "Naslov1" - because a 1.7B copies whatever it is shown,
and with real-looking names it addresses sheets and slides that do not exist.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import config, ops

# The subset shown by default, per format: the full registry is far longer, and a small
# model picks the wrong operation more often the longer the list gets.
CORE_OPS = {
    "excel": (
        "set_cell", "set_formula", "write_rows", "append_row", "add_column", "total_row",
        "create_sheet", "rename_sheet", "sort_rows", "delete_rows_where", "find_replace",
        "format_range", "autofit", "freeze_panes", "add_chart",
        "aggregate", "count_rows", "find_rows", "read_range",
    ),
    "word": (
        "add_heading", "add_paragraph", "add_bullets", "add_table", "add_page_break",
        "replace_text", "delete_paragraphs_where", "set_properties",
        "read_outline", "read_document", "count_words",
    ),
    "slides": (
        "add_slide", "add_table_slide", "add_chart_slide", "delete_slide", "set_notes",
        "replace_slide_text", "list_slides",
    ),
    "pdf": (
        "write_blocks", "merge_pdf", "keep_pages", "delete_pages", "rotate_pages",
        "set_pdf_metadata", "read_pdf_text", "count_pages",
    ),
}

SHARED_RULES = (
    "Rules:\n"
    "- only the operations the request actually needs: one request is usually one "
    "operation, and anything extra is wrong\n"
    "- use only the operations listed, with exactly those parameter names; invent no "
    "parameters\n"
    "- use only the names and contents from the file shown below, never from the examples\n"
)
KIND_RULES = {
    "excel": ('- "sheet" is optional and defaults to the first sheet\n'
              "- name columns by their header text or by letter\n"
              "- for a question about the data, use a reading operation (aggregate, "
              "count_rows, find_rows, read_range); Python computes the answer\n"),
    "word": ("- a document is written in order: each operation appends to the end\n"
             "- for a question about the document, use read_outline, read_document or "
             "count_words\n"),
    "slides": ("- one add_slide per slide; Python picks the layout, so never give positions "
               "or sizes\n"
               "- bullets are short lines of text; Python splits them over slides when "
               "there are too many\n"),
    "pdf": ('- pages are given as "1-3,7"\n'
            "- write_blocks renders new pages; merge_pdf, keep_pages, delete_pages and "
            "rotate_pages rearrange the pages that are already there\n"),
}
SAVE_RULES = {
    "excel": '- add "save_as": "name.xlsx" only when the user asks for a particular file name',
    "word": '- add "save_as": "name.docx" only when the user asks for a particular file name',
    "slides": '- add "save_as": "name.pptx" only when the user asks for a particular file name',
    "pdf": '- add "save_as": "name.pdf" only when the user asks for a particular file name',
}
PDF_RULE = '- add "export_pdf": true when the user also wants the result as a PDF'

_EXCEL_FILE = ('file: primjer.xlsx\nsheet "List1": 20 rows x 2 cols\n'
               "  columns: A=Stupac1 (text), B=Stupac2 (number)")
_WORD_FILE = ("file: primjer.docx (Word)\n12 paragraphs, 300 words, 0 tables\n"
              "outline:\n  H1 Naslov1\n  H2 Naslov2")
_SLIDES_FILE = ("file: primjer.pptx (PowerPoint)\n2 slides, 13.33x7.50 in\n"
                "  slide 1 [Title Slide]: Naslov1\n  slide 2 [Title and Content]: Naslov2")
_PDF_FILE = "file: primjer.pdf (PDF)\n8 pages, 210x297 mm"

FEW_SHOT = {
    "excel": [
        {"role": "user", "content": f"{_EXCEL_FILE}\n\nRequest: Sortiraj po Stupac2 od najveceg."},
        {"role": "assistant", "content": '{"ops": [{"op": "sort_rows", "column": "Stupac2", '
                                         '"descending": true}]}'},
        {"role": "user", "content": f"{_EXCEL_FILE}\n\nRequest: Koliki je zbroj Stupac2?"},
        {"role": "assistant", "content": '{"ops": [{"op": "aggregate", "func": "sum", '
                                         '"column": "Stupac2"}]}'},
        {"role": "user", "content": f"{_EXCEL_FILE}\n\nRequest: Podebljaj naslove i prosiri "
                                    "stupce."},
        {"role": "assistant", "content": '{"ops": [{"op": "format_range", "range": "1", '
                                         '"bold": true}, {"op": "autofit"}]}'},
        {"role": "user", "content": "(new empty workbook)\n\nRequest: Napravi tablicu s dva "
                                    "unosa i njihovim brojevima."},
        {"role": "assistant", "content": '{"ops": [{"op": "write_rows", "rows": '
                                         '[["Naslov1", "Naslov2"], ["prvi", 1], '
                                         '["drugi", 2]]}]}'},
    ],
    "word": [
        {"role": "user", "content": f"{_WORD_FILE}\n\nRequest: Zamijeni Naslov2 s Naslov3."},
        {"role": "assistant", "content": '{"ops": [{"op": "replace_text", "find": "Naslov2", '
                                         '"replace": "Naslov3"}]}'},
        {"role": "user", "content": f"{_WORD_FILE}\n\nRequest: Koliko rijeci ima dokument?"},
        {"role": "assistant", "content": '{"ops": [{"op": "count_words"}]}'},
        {"role": "user", "content": "(new empty document)\n\nRequest: Napravi dopis s naslovom, "
                                    "jednim odlomkom i dvije tocke."},
        {"role": "assistant", "content": '{"ops": [{"op": "add_heading", "text": "Naslov1", '
                                         '"level": 1}, {"op": "add_paragraph", "text": '
                                         '"Prvi odlomak."}, {"op": "add_bullets", "items": '
                                         '["prva tocka", "druga tocka"]}]}'},
    ],
    "slides": [
        {"role": "user", "content": f"{_SLIDES_FILE}\n\nRequest: Dodaj slajd s tri tocke."},
        {"role": "assistant", "content": '{"ops": [{"op": "add_slide", "kind": "bullets", '
                                         '"title": "Naslov3", "bullets": ["prva", "druga", '
                                         '"treca"]}]}'},
        {"role": "user", "content": f"{_SLIDES_FILE}\n\nRequest: Obrisi drugi slajd."},
        {"role": "assistant", "content": '{"ops": [{"op": "delete_slide", "index": 2}]}'},
        {"role": "user", "content": "(new empty presentation)\n\nRequest: Napravi prezentaciju "
                                    "s naslovnom stranicom i jednim slajdom s dvije tocke."},
        {"role": "assistant", "content": '{"ops": [{"op": "add_slide", "kind": "title", '
                                         '"title": "Naslov1", "subtitle": "Podnaslov"}, '
                                         '{"op": "add_slide", "kind": "bullets", "title": '
                                         '"Naslov2", "bullets": ["prva", "druga"]}]}'},
    ],
    "pdf": [
        {"role": "user", "content": f"{_PDF_FILE}\n\nRequest: Zadrzi prve tri stranice."},
        {"role": "assistant", "content": '{"ops": [{"op": "keep_pages", "pages": "1-3"}]}'},
        {"role": "user", "content": f"{_PDF_FILE}\n\nRequest: Okreni stranicu 4."},
        {"role": "assistant", "content": '{"ops": [{"op": "rotate_pages", "pages": "4", '
                                         '"degrees": 90}]}'},
        {"role": "user", "content": "(new empty PDF)\n\nRequest: Napravi PDF s naslovom i "
                                    "tablicom od dva reda."},
        {"role": "assistant", "content": '{"ops": [{"op": "write_blocks", "blocks": '
                                         '[{"block": "heading", "text": "Naslov1"}, '
                                         '{"block": "table", "rows": [["Naslov2", "Naslov3"], '
                                         '["prvi", 1]]}]}]}'},
    ],
}

EMPTY = {"excel": "(new empty workbook)", "word": "(new empty document)",
         "slides": "(new empty presentation)", "pdf": "(new empty PDF)"}
NEW_WORKBOOK = EMPTY["excel"]  # kept for callers that only deal with spreadsheets


def system_prompt(kind: str, *, all_ops: bool = False) -> str:
    label = config.KIND_LABELS[kind]
    listed = None if all_ops else CORE_OPS[kind]
    rules = SHARED_RULES + KIND_RULES[kind] + SAVE_RULES[kind]
    if kind != "pdf":
        rules += "\n" + PDF_RULE
    return (f"You turn a request about a {label} file into a JSON plan. Python executes the "
            f"plan and writes the file; you never write file contents yourself.\n"
            "Answer with JSON only - no explanation, no code fences:\n"
            '{"ops": [{"op": "...", ...}]}\n'
            f"{rules}\n\nOperations:\n{ops.reference(listed, kind=kind)}")


def build_messages(request: str, describe: dict | str | None = None, *,
                   kind: str | None = None, all_ops: bool = False) -> list[dict]:
    """Chat messages that ask for a plan: rules, operations, the file, the request."""
    from . import files  # here, so plan.py keeps no import cycle with the dispatcher

    if isinstance(describe, dict):
        kind = kind or str(describe.get("kind") or "excel")
        digest = files.digest_text(describe)
    else:
        kind = kind or "excel"
        digest = describe or EMPTY[kind]
    if kind not in config.KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    return ([{"role": "system", "content": system_prompt(kind, all_ops=all_ops)}]
            + list(FEW_SHOT[kind])
            + [{"role": "user", "content": f"{digest}\n\nRequest: {request.strip()}"}])


# ------------------------------------------------------------------- parsing
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_PY_LITERALS = ((re.compile(r"\bTrue\b"), "true"), (re.compile(r"\bFalse\b"), "false"),
                (re.compile(r"\bNone\b"), "null"))


def repair(text: str) -> str:
    """Fix what small models get wrong *around* valid JSON: Python literals, trailing commas."""
    for pattern, replacement in _PY_LITERALS:
        text = pattern.sub(replacement, text)
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _blocks(text: str) -> list[str]:
    """Every balanced {...} or [...] in the text, outermost only, string contents respected."""
    found: list[str] = []
    depth = 0
    start = -1
    in_string = escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    found.append(text[start:i + 1])
                    start = -1
    return found


def extract_plan(text: str) -> Any | None:
    """The plan JSON from a model answer, or None when the answer holds no JSON at all."""
    if not text or not text.strip():
        return None
    for raw in _FENCE_RE.findall(text) + [text]:
        for attempt in (raw, repair(raw)):
            try:
                return json.loads(attempt)
            except ValueError:
                pass
        for block in _blocks(raw):
            for attempt in (block, repair(block)):
                try:
                    return json.loads(attempt)
                except ValueError:
                    continue
    return None
