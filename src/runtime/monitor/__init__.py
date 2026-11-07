"""Opt-in local monitoring. Importing CLI helpers does not import Torch or a UI."""
import argparse
import os


def add_cli_args(parser):
    group = parser.add_argument_group("Local monitor")
    group.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=False,
                       help="Enable the local monitor at /monitor/ (recent 100 generations).")
    group.add_argument("--tray", action=argparse.BooleanOptionalAction, default=False,
                       help="Show a Windows tray icon for --monitor.")
    return parser


def validate_args(args):
    if getattr(args, "tray", False) and not getattr(args, "monitor", False):
        raise ValueError("--tray requires --monitor")
    if not getattr(args, "monitor", False):
        return
    if getattr(args, "disable_log_stats", False):
        raise ValueError("--monitor requires engine statistics; remove --disable-log-stats")
    count = getattr(args, "api_server_count", None)
    if count is None:
        count = getattr(args, "data_parallel_size", 1)
    if count != 1:
        raise ValueError("--monitor requires --api-server-count 1")
    if (getattr(args, "headless", False) or getattr(args, "grpc", False)
            or getattr(args, "uds", None) or os.environ.get("VLLM_USE_RUST_FRONTEND") == "1"
            or getattr(args, "subparser", "serve") not in ("serve", None)):
        raise ValueError("--monitor requires the Python HTTP vllm serve frontend with a TCP port")
    if getattr(args, "tray", False) and os.name != "nt":
        raise ValueError("--tray requires Windows")
    if getattr(args, "tray", False) and getattr(args, "ssl_certfile", None):
        raise ValueError("--tray currently requires local HTTP; open HTTPS /monitor/ manually")


def install(app, args, engine):
    if not getattr(args, "monitor", False):
        return None
    validate_args(args)
    from .web import install_monitor

    return install_monitor(app, args, engine)
