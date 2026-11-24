"""
FolioWise Local Development Server
==================================
Serves the FolioWise web application cleanly from any working directory
without 404 or path resolution issues.
"""

import sys
import os
import socket
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Determine base directory containing frontend
CWD = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(CWD, "frontend", "dashboard.html")):
    SERVE_DIR = CWD
elif os.path.exists(os.path.join(CWD, "foliowise", "frontend", "dashboard.html")):
    SERVE_DIR = os.path.join(CWD, "foliowise")
else:
    # Look one level up
    parent = os.path.abspath(os.path.join(CWD, ".."))
    if os.path.exists(os.path.join(parent, "frontend", "dashboard.html")):
        SERVE_DIR = parent
    elif os.path.exists(os.path.join(parent, "foliowise", "frontend", "dashboard.html")):
        SERVE_DIR = os.path.join(parent, "foliowise")
    else:
        SERVE_DIR = CWD


class FolioWiseHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def do_GET(self):
        # Route root '/' directly to user website (frontend/index.html)
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/frontend/index.html")
            self.end_headers()
            return
        # Route '/admin' directly to admin panel (frontend/dashboard.html)
        if self.path in ("/admin", "/admin/"):
            self.send_response(302)
            self.send_header("Location", "/frontend/dashboard.html")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format, *args):
        # Clean request logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def find_available_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    port = find_available_port(8000)
    server_address = ("", port)
    
    # Change to serve directory for reliable relative resolution
    os.chdir(SERVE_DIR)
    
    httpd = ThreadingHTTPServer(server_address, FolioWiseHandler)
    user_url = f"http://localhost:{port}/frontend/index.html"
    admin_url = f"http://localhost:{port}/frontend/dashboard.html"
    root_url = f"http://localhost:{port}/"

    print("=" * 64)
    print("   FolioWise Portfolio Management & Quantitative Engine")
    print("=" * 64)
    print(f" Serving directory : {SERVE_DIR}")
    print(f" Status            : Active on port {port}")
    print("-" * 64)
    print(" Clickable URLs:")
    print(f"  • Public Investor Website : {user_url}")
    print(f"  • Admin Panel (Protected) : {admin_url} (or /admin)")
    print(f"  • Default Home            : {root_url}")
    print("-" * 64)
    print(" Admin Access Gate: Client-Side Demo Lock (Passcode: admin123)")
    print(" Architecture Note: Academic/local presentation mode; see docs for production JWT auth.")
    print(" Press Ctrl+C in this terminal to stop the server.")
    print("=" * 64)

    # Automatically open user website in browser
    try:
        webbrowser.open(user_url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    main()
