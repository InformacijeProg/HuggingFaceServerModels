"""The one exception type every hffiles entry point is allowed to show the user."""
from __future__ import annotations


class FileError(Exception):
    """A bad request, bad plan or bad file: shown as a message, never a traceback."""
