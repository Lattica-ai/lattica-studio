#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys

class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        # Allow Authorization header in preflight (OPTIONS) request
        self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Origin', '*')  # Allow all origins for now - need to change to more restrictive

        # For OPTIONS requests (preflight), we return early with the necessary headers
        if self.command == 'OPTIONS':
            self.send_response(200)
            self.end_headers()
            return

        # Additional headers for normal requests
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Authorization', 'Bearer <QUERY_TOKEN>')

        SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    port = 3010
    print(f'Starting server at http://localhost:{port}/RunExample.html')
    test(CORSRequestHandler, HTTPServer, port=int(sys.argv[1]) if len(sys.argv) > 1 else port)