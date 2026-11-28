"""Native startup port regression without models, media or a GUI."""
import importlib.util
import os
import socket
import sys
import types
import unittest
from urllib.request import urlopen
from unittest.mock import MagicMock, patch

from native_app import run_native


@unittest.skipUnless(importlib.util.find_spec("werkzeug"), "Werkzeug required; run in build venv")
class NativeStartupTests(unittest.TestCase):
    def test_native_launch_uses_available_port_while_old_port_is_occupied(self):
        webview = types.ModuleType("webview")
        webview.create_window = MagicMock()
        menu = types.ModuleType("webview.menu")
        menu.Menu = MagicMock()
        menu.MenuAction = MagicMock()
        lock = MagicMock()

        def application(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"NixScribe test"]

        def start(*args, **kwargs):
            url = webview.create_window.call_args.args[1]
            self.assertNotEqual(url, f"http://127.0.0.1:{occupied_port}/")
            with urlopen(url, timeout=5) as response:
                self.assertEqual(response.read(), b"NixScribe test")
            self.assertEqual(occupied.getsockname()[1], occupied_port)

        webview.start = start
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            occupied_port = occupied.getsockname()[1]
            with patch.dict(os.environ), patch.dict(sys.modules, {"webview": webview, "webview.menu": menu}), \
                 patch("native_app._instance_lock", return_value=lock):
                os.environ.pop("WHISPERX_UI_PORT", None)
                run_native(application, "127.0.0.1", occupied_port, lambda: False, lambda _: None)
        lock.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
