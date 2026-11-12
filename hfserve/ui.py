"""Tiny console helpers shared by cli.py and chat.py.

Colours are plain ANSI SGR codes. They are disabled when the ``NO_COLOR``
environment variable is set (https://no-color.org) or when stdout is not a
terminal, so piping output into a file never produces escape codes.
"""
from __future__ import annotations

import os
import sys

# SGR 90 (bright black / grey) is used for "dim": classic conhost ignores SGR 2.
_CODES = {
    "dim": "\x1b[90m",
    "grey": "\x1b[90m",
    "bold": "\x1b[1m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}
_RESET = "\x1b[0m"


def setup_console() -> None:
    """UTF-8 output and ANSI colour support on Windows. Safe to call repeatedly."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:  # piped/redirected input (hfserve run 1 < prompts.txt) is read as UTF-8 too
        if not sys.stdin.isatty():
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    try:
        import colorama

        colorama.just_fix_windows_console()
    except (ImportError, AttributeError):
        pass


def colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def c(text: object, color: str) -> str:
    """Wrap *text* in the ANSI code for *color* (a key of ``_CODES``) when colours are on."""
    text = str(text)
    code = _CODES.get(color)
    if not code or not text or not colors_enabled():
        return text
    return f"{code}{text}{_RESET}"


def dim(text: object) -> str:
    return c(text, "dim")


def grey(text: object) -> str:
    return c(text, "grey")


def bold(text: object) -> str:
    return c(text, "bold")


def red(text: object) -> str:
    return c(text, "red")


def green(text: object) -> str:
    return c(text, "green")


def yellow(text: object) -> str:
    return c(text, "yellow")


def cyan(text: object) -> str:
    return c(text, "cyan")
