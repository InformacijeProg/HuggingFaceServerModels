"""hftrain - fine-tune the installed EuroLLM model on Croatian question-answer data.

Sibling of the ``hfserve`` package: it reuses hfserve's model registry and paths, but
nothing in hfserve imports it, so the server keeps working without the training extras
(``pip install -r requirements-train.txt``).

Workflow (see docs/TRAINING.md)::

    python -m hftrain prepare --name qa-v1           # from training/QA (the default source)
    python -m hftrain train --data qa-v1 --run qa-lora-1 --lr 1e-4
    python -m hftrain eval --run qa-lora-1 --compare-base
    python -m hftrain export --run qa-lora-1 --name local/EuroLLM-1.7B-hr
    python -m hfserve run local/EuroLLM-1.7B-hr
"""

__version__ = "0.1.0"
