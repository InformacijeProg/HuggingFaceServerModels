"""hffiles - deterministic file work for hfserve models, starting with Excel.

The model never writes a file. It only turns a request into a *plan*: a small
JSON list of operations (``set_cell``, ``write_rows``, ``sort_rows``, ...).
``hffiles.ops`` validates that plan and executes it with openpyxl, so the bytes
on disk are always produced by Python, never by generated text.

    from hffiles import excel, ops, plan

    desc = excel.describe("budget.xlsx")                 # what is in the file
    messages = plan.build_messages(question, desc)       # ask the model for a plan
    result = ops.apply_plan(plan.extract_plan(answer), source="budget.xlsx")

See docs/FILES.md and ``python -m hffiles --help``.
"""

__version__ = "0.1.0"
