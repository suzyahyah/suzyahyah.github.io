from http.server import HTTPServer, BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        # the container-side address this request actually arrived on (container IP on the docker network)
        container_ip, container_port = self.connection.getsockname()[:2]
        self.wfile.write(f"hi from app1 at {container_ip}:{container_port}\n".encode())


# must bind 0.0.0.0, not 127.0.0.1, otherwise traffic from outside the container can't reach it
HTTPServer(("0.0.0.0", 5000), Handler).serve_forever()
