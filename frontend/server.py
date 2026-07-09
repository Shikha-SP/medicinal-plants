import errno
import json
import os
import re
import sys
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = ROOT / "index.html"

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}
ADDRESS_IN_USE_ERRORS = {errno.EADDRINUSE, errno.EACCES, getattr(errno, "WSAEADDRINUSE", 10048)}
SEARCH_PLANT = None


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = True


def get_search_plant():
    global SEARCH_PLANT

    if SEARCH_PLANT is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        from plant_pipeline.search import search_plant

        SEARCH_PLANT = search_plant

    return SEARCH_PLANT


def parse_image_upload(content_type, raw_body):
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data upload.")

    if "boundary=" not in content_type:
        raise ValueError("Upload is missing a multipart boundary.")

    headers = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("latin-1", errors="replace")
    message = BytesParser(policy=policy.default).parsebytes(headers + raw_body)

    if not message.is_multipart():
        raise ValueError("Upload body was not a valid multipart request.")

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue

        field_name = part.get_param("name", header="content-disposition")
        if field_name != "image":
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            raise ValueError("No image file received.")

        return part.get_filename() or "upload.jpg", payload

    raise ValueError("No image file received.")


def next_upload_path(filename):
    suffix = Path(filename).suffix.lower()
    if suffix not in VALID_IMAGE_EXTS:
        suffix = ".jpg"

    stem = Path(filename).stem or "upload"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-") or "upload"

    query_dir = DATA_DIR / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)

    destination = query_dir / f"{safe_stem}{suffix}"
    counter = 1
    while destination.exists():
        destination = query_dir / f"{safe_stem}_{counter}{suffix}"
        counter += 1

    return destination


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status_code, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path):
        content_type = STATIC_TYPES.get(path.suffix.lower(), "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_not_found(self):
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        parsed = urlparse(self.path)
        if parsed.path != "/analyze":
            self.send_not_found()
            return

        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_path = unquote(urlparse(self.path).path)
        if parsed_path in {"/", "/index.html"}:
            self.send_file(INDEX_PATH)
            return

        requested = (ROOT / parsed_path.lstrip("/")).resolve()
        try:
            requested.relative_to(ROOT)
        except ValueError:
            self.send_not_found()
            return

        if requested.is_file() and requested.suffix.lower() in STATIC_TYPES:
            self.send_file(requested)
            return

        self.send_not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/analyze":
            self.send_json(404, {"error": "Endpoint not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Invalid Content-Length header."})
            return

        if content_length <= 0:
            self.send_json(400, {"error": "Empty request body."})
            return

        content_type = self.headers.get("Content-Type", "")
        raw_body = self.rfile.read(content_length)

        try:
            image_name, image_bytes = parse_image_upload(content_type, raw_body)
            destination_path = next_upload_path(image_name)
            destination_path.write_bytes(image_bytes)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        except OSError as exc:
            self.send_json(500, {"error": f"Could not save uploaded image: {exc}"})
            return

        try:
            search_plant = get_search_plant()
            result = search_plant(str(destination_path), verbose=False)
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(500, {"error": f"Analysis failed: {exc}"})

    def log_message(self, format, *args):
        return


def create_server(preferred_port):
    for port in range(preferred_port, preferred_port + 20):
        try:
            return LocalServer(("127.0.0.1", port), Handler), port
        except OSError as exc:
            error_code = exc.errno or getattr(exc, "winerror", None)
            if error_code not in ADDRESS_IN_USE_ERRORS:
                raise

    raise OSError(f"No available port found from {preferred_port} to {preferred_port + 19}.")


if __name__ == "__main__":
    preferred_port = int(os.environ.get("PORT", "8000"))
    server, port = create_server(preferred_port)
    print(f"Frontend server running at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
