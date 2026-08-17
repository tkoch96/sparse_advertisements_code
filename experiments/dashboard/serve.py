"""Dash server with no-store headers (browser-cache-proof)."""
import http.server
import os
import sys


class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()


if __name__ == '__main__':
    os.chdir(sys.argv[1] if len(sys.argv) > 1 else 'dashboard_site')
    http.server.ThreadingHTTPServer(('127.0.0.1', 8643), H).serve_forever()
