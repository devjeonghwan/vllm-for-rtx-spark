"""UTF-8 defaults at Windows startup/configuration boundaries; CPU only."""

import io
import os
import sys


def configure_standard_streams():
    """Use UTF-8 for redirected Python stdio unless the caller chose a policy.

    Console handles and application-owned stream wrappers are left alone.
    PYTHONIOENCODING, PYTHONUTF8 (including 0), and -X utf8 are authoritative.
    This runs in every fresh API/worker process, without changing environment
    variables or replacing streams and logging handlers globally.
    """
    if sys.platform != "win32":
        return
    if ("PYTHONIOENCODING" in os.environ or "PYTHONUTF8" in os.environ
            or "utf8" in sys._xoptions):
        return
    for stream in (sys.stdout, sys.stderr):
        if not isinstance(stream, io.TextIOWrapper) or stream.closed or stream.isatty():
            continue
        stream.reconfigure(encoding="utf-8", errors=stream.errors)


_FILE_HANDLERS = frozenset({
    "logging.FileHandler",
    "logging.handlers.RotatingFileHandler",
    "logging.handlers.TimedRotatingFileHandler",
    "logging.handlers.WatchedFileHandler",
})


def with_file_handler_encoding(config):
    """Default known stdlib file handlers to UTF-8 in this dictConfig only.

    Explicit encoding (including None), custom subclasses/factories and handlers
    created outside vLLM's logging configuration boundary remain authoritative.
    The caller-owned configuration is not mutated.
    """
    if sys.platform != "win32" or not isinstance(config, dict):
        return config
    handlers = config.get("handlers")
    if not isinstance(handlers, dict):
        return config  # Let dictConfig validate malformed configurations.
    changed = {}
    for name, handler in handlers.items():
        if (isinstance(handler, dict) and "()" not in handler
                and handler.get("class") in _FILE_HANDLERS and "encoding" not in handler):
            changed[name] = dict(handler, encoding="utf-8")
    return dict(config, handlers=dict(handlers, **changed)) if changed else config
