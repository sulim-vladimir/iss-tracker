"""MJPEG preview and control panel for headless use: open http://<pi>:8080/ in a browser.

Shows both cameras side by side with overlays, and gives live control over exposure, gain and
SER recording without touching the terminal.
"""

import json
import re
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

WEB = Path(__file__).resolve().parent / "web"


def asset(name):
    """Read a file from issctl/web. Never cached: the point of these files is that you edit them
    and reload, and a stale template is a confusing way to spend ten minutes."""
    return (WEB / name).read_text()


def fragments():
    """panels.html holds the page's building blocks, each after a `<!-- @name -->` marker."""
    out, name, buf = {}, None, []
    for line in asset("panels.html").splitlines():
        m = re.match(r"<!-- @([a-z_]+) -->\s*$", line)
        if m:
            if name:
                out[name] = "\n".join(buf).strip()
            name, buf = m.group(1), []
        elif name:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf).strip()
    return out


def fill(template, **values):
    """Fill {{name}} placeholders. Deliberately not str.format: these files are full of CSS and
    JavaScript braces, and doubling every one of them is what made the old inline templates so
    easy to break."""
    def sub(m):
        key = m.group(1)
        if key not in values:
            raise KeyError(f"no value for {{{{{key}}}}} in template")
        return str(values[key])
    return re.sub(r"\{\{([a-z_]+)\}\}", sub, template)


def draw_axes(img, cal, length=46, margin=10):
    """Show which way the mount axes push the image: the camera's rotation is rarely obvious.

    The origin is placed so both arrows and their labels always fit, whatever their directions.
    """
    J = np.array(cal["J"], dtype=float)
    arrows = []
    for col, colour, label in ((0, (255, 190, 40), "RA+"), (1, (230, 90, 230), "Dec+")):
        v = J[:, col]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            arrows.append((v / n, colour, label))
    if not arrows:
        return
    reach = [(v * (length + 18) + np.array([0, 4])) for v, _, _ in arrows]
    lo = np.minimum(np.min(reach, axis=0), 0) - np.array([16, 0])   # room for a label to the left
    hi = np.maximum(np.max(reach, axis=0), 0) + np.array([16, 4])
    origin = np.array([margin, margin]) - lo
    if origin[1] + hi[1] > img.shape[0]:
        origin[1] = img.shape[0] - margin - hi[1]
    for v, colour, label in arrows:
        tip = origin + v * length
        cv2.arrowedLine(img, tuple(origin.astype(int)), tuple(tip.astype(int)), colour, 1,
                        tipLength=0.22)
        pos = origin + v * (length + 16)
        cv2.putText(img, label, (int(pos[0]) - 12, int(pos[1]) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)


def render(cam, cal, max_width=800, corners=False):
    """Image only - status text lives under the frame on the page, not burned into the picture."""
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
        cx, cy = int((cam.width - 1) / 2 * k), int((cam.height - 1) / 2 * k)
        if abs(bx - cx) > 4 or abs(by - cy) > 4:   # frame centre, when it differs from the boresight
            cv2.drawMarker(img, (cx, cy), (160, 160, 160), cv2.MARKER_CROSS, 26, 1)
        cv2.drawMarker(img, (bx, by), (0, 220, 0), cv2.MARKER_CROSS, 34, 2)
    if cal:
        draw_axes(img, cal)     # top-left, positioned so the arrows always fit
    if cam.gate:
        gx, gy, gr = cam.gate
        cv2.circle(img, (int(gx * k), int(gy * k)), int(gr * k), (220, 130, 0), 2)
    if det:
        cv2.circle(img, (int(det.x * k), int(det.y * k)), 14, (0, 0, 255), 2)
    if corners:
        # Exactly the corners scene calibration will follow - same function, same settings - so
        # you can see whether there is anything to track before pressing the button.
        from .calib import scene_corners

        for x, y in scene_corners(frame, cam.bayer):
            cx, cy, r = int(x * k), int(y * k), 6
            cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 255), 2)
    return img


class Preview:
    def __init__(self, cams, state, port=8080, fps=5.0, status=None, controls=None):
        self.cams, self.state, self.port, self.period = cams, state, port, 1.0 / fps
        self.status = status      # callable(camera_name) -> list of overlay lines
        self.controls = controls  # dict of callables: state/exposure/gain/record
        self.show_corners = set()  # cameras drawing the corners scene calibration would use

    def page(self):
        can_record = bool(self.controls and self.controls.get("record"))
        can_move = bool(self.controls and self.controls.get("mount_action"))
        f = fragments()
        cam_panels = "".join(
            fill(f["panel"], name=n, title=n.capitalize(),
                   extra=f["record"] if (n == "main" and can_record) else "",
                   centre=(f["centre"].replace("EXTRA", "" if n == "main" else f["in_frame"])
                           .replace("NAME", n)
                           .replace("LABEL", "centre it" if n == "main" else "send to main")
                           if can_move else ""))
            for n in self.cams)
        estop = f["estop"] if (self.controls and self.controls.get("estop")) else ""
        return fill(asset("index.html"),
                      cam_panels=cam_panels,
                      sky_panel=f["sky"] if (self.controls and self.controls.get("sky")) else "",
                      mount_panel=fill(f["mount"], estop=estop) if can_move else "",
                      status_panel=f["status"] if can_move else "",
                      warnings_panel=f["warnings"] if can_move else "",
                      log_panel=f["log"] if can_move else "",
                      messages_panel=f["messages"] if can_move else "",
                      cams=json.dumps(list(self.cams)))

    def api_state(self):
        if self.controls and self.controls.get("state"):
            out = self.controls["state"]()
        else:
            out = {"cams": {n: {"fps": c.fps, "exposure_ms": c.exposure_ms, "gain": c.gain, "det": None}
                            for n, c in self.cams.items()}, "record": None}
        if self.status:
            try:
                out["status"] = {n: list(self.status(n)) for n in self.cams}
            except Exception:
                pass
        if self.controls and self.controls.get("mount_state"):
            out["mount"] = self.controls["mount_state"]()
        out["corners"] = sorted(self.show_corners)
        return out

    def start(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, content_type="application/json"):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                path = url.path
                ctl = preview.controls or {}

                if path in ("/", "/index.html"):
                    return self._send(preview.page().encode(), "text/html")
                if url.path in ("/style.css", "/app.js"):
                    kind = "text/css" if url.path.endswith(".css") else "application/javascript"
                    return self._send(asset(url.path.lstrip("/")).encode(), kind)
                if path == "/api/state":
                    return self._send(json.dumps(preview.api_state()).encode())
                if path == "/api/exposure" and ctl.get("exposure"):
                    ctl["exposure"](q.get("cam"), ms=q.get("ms"), factor=q.get("factor"))
                    return self._send(json.dumps(preview.api_state()).encode())
                if path == "/api/gain" and ctl.get("gain"):
                    ctl["gain"](q.get("cam"), value=q.get("value"), delta=q.get("delta"))
                    return self._send(json.dumps(preview.api_state()).encode())
                if path == "/api/record" and ctl.get("record"):
                    ctl["record"](q.get("on") not in (None, "0", "false"))
                    return self._send(json.dumps(preview.api_state()).encode())
                if path == "/api/corners":
                    cam_name = q.get("cam")
                    if cam_name in preview.cams:
                        if q.get("on") not in (None, "0", "false"):
                            preview.show_corners.add(cam_name)
                        else:
                            preview.show_corners.discard(cam_name)
                    return self._send(json.dumps(preview.api_state()).encode())
                if path == "/api/sky" and ctl.get("sky"):
                    return self._send(json.dumps(ctl["sky"]()).encode())
                if path == "/api/select" and ctl.get("select"):
                    ctl["select"](q.get("cam"), q.get("fx"), q.get("fy"), q.get("clear"))
                    return self._send(json.dumps(preview.api_state()).encode())
                if path == "/api/estop" and ctl.get("estop"):
                    ctl["estop"]()
                    return self._send(json.dumps(preview.api_state()).encode())
                if path == "/api/mount" and ctl.get("mount_action"):
                    ctl["mount_action"](q.get("action"), q)
                    return self._send(json.dumps(preview.api_state()).encode())

                name = path.strip("/").removesuffix(".mjpg")
                if name in preview.cams:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    cam = preview.cams[name]
                    try:
                        while True:
                            img = render(cam, preview.state.get("cameras", {}).get(name),
                                         corners=name in preview.show_corners)
                            if img is not None:
                                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                                 + jpg.tobytes() + b"\r\n")
                            time.sleep(preview.period)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_error(404)

        server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="preview", daemon=True).start()
        return server
