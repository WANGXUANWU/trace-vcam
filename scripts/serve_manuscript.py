"""Serve the manuscript tree over localhost with permissive CORS headers.

The Overleaf project is updated from the browser: the page fetches each file
from this server and posts it to the project's own upload endpoint.  Chrome
treats http://localhost as a trustworthy origin, so the fetch is allowed from
the https Overleaf page.  The server is read-only and binds to the loopback
interface only.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"

TRACKED = (
    ["main.tex", "supplement.tex", "references.bib"]
    + [f"sections/{path.name}" for path in sorted((MANUSCRIPT / "sections").glob("*.tex"))]
    + [f"supplement/{path.name}" for path in sorted((MANUSCRIPT / "supplement").glob("*.tex"))]
    + [f"tables/{path.name}" for path in sorted((MANUSCRIPT / "tables").glob("*.tex"))]
    + [f"figures/{path.name}" for path in sorted((MANUSCRIPT / "figures").glob("*.pdf"))]
)


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/manifest.json":
            payload = json.dumps({"files": TRACKED}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = partial(CORSHandler, directory=str(MANUSCRIPT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving {MANUSCRIPT} on http://127.0.0.1:{args.port} ({len(TRACKED)} files)")
    server.serve_forever()


if __name__ == "__main__":
    main()
