"""Desktop-compatible launcher for the local web interface."""

import webbrowser


def open_interface(url="http://127.0.0.1:8000/"):
    if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("Only loopback interface URLs are permitted.")
    return bool(webbrowser.open(url))
