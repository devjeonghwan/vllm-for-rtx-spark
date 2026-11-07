"""Routes and application-owned startup/cleanup. HTTP access stays on loopback."""
import asyncio
from contextlib import asynccontextmanager
import ipaddress
import logging
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .state import Monitor
from ...version import PROJECT_VERSION

ASSETS = Path(__file__).with_name("static")
LOCAL_NAMES = {"localhost", "127.0.0.1", "::1"}
logger = logging.getLogger("vllm.rtx_spark.monitor")


async def local_only(request: Request):
    try:
        local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        local = False
    # A remote Origin or arbitrary DNS name must not read local prompts via CORS/rebinding.
    origin = request.headers.get("origin")
    if (not local or request.url.hostname not in LOCAL_NAMES
            or request.headers.get("sec-fetch-site") == "cross-site"
            or (origin and origin != f"{request.url.scheme}://{request.url.netloc}")):
        raise HTTPException(403, "Monitor is local-only")


def install_monitor(app, args, engine):
    if getattr(app.state, "rtx_spark_monitor", None) is not None:
        return app.state.rtx_spark_monitor
    monitor = Monitor(engine, args, version=PROJECT_VERSION)
    app.state.rtx_spark_monitor = monitor
    router = APIRouter(prefix="/monitor", dependencies=[Depends(local_only)], include_in_schema=False)
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
               "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"}

    @router.get("")
    async def redirect(request: Request):
        return RedirectResponse(str(request.url.replace(path=request.url.path + "/")), headers=headers)

    @router.get("/")
    async def index():
        return FileResponse(ASSETS / "index.html", headers=headers)

    @router.get("/{asset}")
    async def asset(asset: str):
        if asset not in ("app.js", "styles.css", "vllm.png", "favicon.ico"):
            raise HTTPException(404, "Not found")
        return FileResponse(ASSETS / asset, headers=headers)

    @router.get("/api/state")
    async def state():
        return JSONResponse(monitor.snapshot(), headers=headers)

    @router.get("/api/requests/{key}")
    async def detail(key: int):
        row = monitor.rows.get(key)
        if row is None:
            raise HTTPException(404, "Request no longer retained")
        return JSONResponse(monitor.public_row(row, detail=True), headers=headers)

    @router.post("/api/stop")
    async def stop(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise HTTPException(415, "JSON required")
        if await request.json() != {"confirm": True}:
            raise HTTPException(400, "Confirmation required")
        if monitor.shutdown is None:
            raise HTTPException(503, "Shutdown is not available yet")
        asyncio.get_running_loop().call_later(0.1, monitor.shutdown)
        return JSONResponse({"stopping": True}, headers=headers)

    app.include_router(router)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        tray = None
        try:
            async with original_lifespan(application) as lifetime:
                try:
                    if args.tray:
                        from .tray import Tray

                        scheme = "http"
                        root = (getattr(args, "root_path", "") or "").rstrip("/")
                        url = f"{scheme}://127.0.0.1:{args.port}{root}/monitor/"
                        tray = Tray(url)
                        await asyncio.to_thread(tray.start)
                    logger.info("Local monitor: /monitor/")
                    yield lifetime
                finally:
                    if tray is not None:
                        await asyncio.to_thread(tray.close)
                    monitor.close()
        finally:
            # Also restore hooks when the original lifespan's startup fails.
            monitor.close()

    app.router.lifespan_context = lifespan
    return monitor
