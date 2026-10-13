"""Native macOS Cocoa window for the existing local NixScribe Flask UI."""
from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path
from typing import Any, Callable


def _instance_lock():
    path = Path.home() / "Library" / "Application Support" / "NixScribe"
    path.mkdir(parents=True, exist_ok=True)
    handle = (path / "instance.lock").open("a+", encoding="ascii")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            handle.seek(0)
            pid = int(handle.read().strip())
            from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
            running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if running is not None:
                running.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        except (ValueError, ImportError, OSError):
            pass
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def run_native(app: Any, host: str, port: int, is_running: Callable[[], bool],
               register_quit: Callable[[Callable[[], None] | None], None]) -> None:
    """Keep the HTTP backend alive only while the native app window is open."""
    from werkzeug.serving import make_server
    import webview
    from webview.menu import Menu, MenuAction

    handle = _instance_lock()
    if handle is None:
        return

    server = None
    worker = None
    quit_event = threading.Event()
    closed_event = threading.Event()
    try:
        server = make_server(host, port, app, threaded=True)
        worker = threading.Thread(target=server.serve_forever,
                                  name="NixScribe-local-web", daemon=True)
        worker.start()
        register_quit(quit_event.set)
        window = webview.create_window(
            "NixScribe", f"http://{host}:{port}/",
            width=1100, height=900, min_size=(760, 620),
            resizable=True, text_select=True, background_color="#F3F4F7",
        )
        window.events.closing += lambda: not is_running()
        window.events.closed += closed_event.set

        def watch_quit() -> None:
            while not closed_event.is_set():
                if quit_event.wait(timeout=0.25):
                    if not closed_event.is_set() and not is_running():
                        window.destroy()
                    return

        def request_quit() -> None:
            if not is_running():
                quit_event.set()

        menu = [Menu("__app__", [MenuAction("Quit NixScribe", request_quit)])]
        # Cocoa/WebKit requires its UI loop on the main thread.
        webview.start(watch_quit, menu=menu, private_mode=True)
    finally:
        register_quit(None)
        if server is not None:
            server.shutdown()
            server.server_close()
        if worker is not None:
            worker.join(timeout=4)
        handle.close()
