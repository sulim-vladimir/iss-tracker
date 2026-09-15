"""MJPEG preview server for headless use: open http://<pi>:8080/ in a browser."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = """<!doctype html><html><head><title>ISS tracker</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{background:#111;color:#ddd;font:14px sans-serif;margin:8px}}
img{{max-width:100%;display:block;margin-bottom:8px}}</style></head>
<body>{imgs}</body></html>"""


def render(cam, cal, max_width=800):
    frame, det, _ = cam.latest()
    if frame is None:
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR) if cam.bayer else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    lo, hi = np.percentile(frame[::4, ::4], (1, 99.9))
    img = np.clip((img.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255).astype(np.uint8)
    k = min(1.0, max_width / img.shape[1])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    if cal:
        bx, by = (int(cal["boresight"][0] * k), int(cal["boresight"][1] * k))
        cv2.drawMarker(img, (bx, by), (0, 200, 0), cv2.MARKER_CROSS, 30, 1)
    if cam.gate:
        gx, gy, gr = cam.gate
        cv2.circle(img, (int(gx * k), int(gy * k)), int(gr * k), (200, 120, 0), 1)
    if det:
        cv2.circle(img, (int(det.x * k), int(det.y * k)), 12, (0, 0, 255), 1)
    cv2.putText(img, f"{cam.name} {cam.fps:.0f} fps", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return img


class Preview:
    def __init__(self, cams, state, port=8080, fps=5.0):
        self.cams, self.state, self.port, self.period = cams, state, port, 1.0 / fps

    def start(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.strip("/").removesuffix(".mjpg")
                if self.path in ("/", "/index.html"):
                    body = PAGE.format(imgs="".join(f'<img src="/{n}.mjpg">' for n in preview.cams)).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(body)
                elif name in preview.cams:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    cam = preview.cams[name]
                    try:
                        while True:
                            img = render(cam, preview.state.get("cameras", {}).get(name))
                            if img is not None:
                                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                            time.sleep(preview.period)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

        server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="preview", daemon=True).start()
        return server
