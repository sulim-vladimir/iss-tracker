"""MJPEG preview and control panel for headless use: open http://<pi>:8080/ in a browser.

Shows both cameras side by side with overlays, and gives live control over exposure, gain and
SER recording without touching the terminal.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PAGE = """<!doctype html><html><head><title>ISS tracker</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{{background:#2f3439;color:#e8e8e8;font:14px/1.4 system-ui,sans-serif;margin:10px}}
 .row{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}}
 .panel{{background:#3b4147;border-radius:6px;padding:8px;flex:1 1 380px;min-width:320px}}
 .panel img{{width:100%;display:block;border-radius:4px;background:#000}}
 h2{{font-size:15px;margin:0 0 6px}}
 .ctl{{display:flex;align-items:center;gap:6px;margin-top:8px;flex-wrap:wrap}}
 .ctl label{{width:62px;color:#b9c1c8}}
 .info{{font:13px/1.5 ui-monospace,monospace;color:#cfd6dd;white-space:pre;margin-top:6px;
        min-height:4.5em}}
 button{{background:#4d555d;color:#e8e8e8;border:0;border-radius:4px;padding:5px 11px;
         font-size:14px;cursor:pointer}}
 button:hover{{background:#5c656e}}
 input{{background:#262a2e;color:#e8e8e8;border:1px solid #5c656e;border-radius:4px;
        padding:4px;width:74px;font-size:14px}}
 .rec{{margin-top:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
 #recbtn.on{{background:#a33;}}
 #recinfo{{color:#b9c1c8}}
</style></head><body>
<div class="row">{panels}</div>
<script>
const CAMS = {cams};
async function api(path, params) {{
  const r = await fetch(path + '?' + new URLSearchParams(params));
  return apply(await r.json());
}}
function apply(s) {{
  for (const n of CAMS) {{
    const c = s.cams[n]; if (!c) continue;
    const e = document.getElementById('exp-' + n), g = document.getElementById('gain-' + n);
    if (document.activeElement !== e) e.value = c.exposure_ms.toFixed(2);
    if (document.activeElement !== g) g.value = c.gain;
    document.getElementById('stat-' + n).textContent =
      c.fps.toFixed(0) + ' fps  ' + (c.det ? 'detected ' + c.det[0] + ',' + c.det[1] : 'no detection');
    const info = document.getElementById('info-' + n);
    if (info) info.textContent = ((s.status || {{}})[n] || []).join('\\n');
  }}
  const r = s.record, btn = document.getElementById('recbtn');
  if (btn && r) {{
    btn.textContent = r.recording ? 'Stop recording' : 'Start recording';
    btn.className = r.recording ? 'on' : '';
    document.getElementById('recinfo').textContent = r.recording
      ? (r.path || '') + '  ' + r.frames + ' frames, ' + r.dropped + ' dropped'
        + (r.paused ? '  (paused: ISS not visible)' : '')
      : 'not recording';
  }}
}}
setInterval(async () => apply(await (await fetch('/api/state')).json()), 1000);
</script></body></html>"""

PANEL = """<div class="panel"><h2>{name} <span id="stat-{name}"></span></h2>
<img src="/{name}.mjpg">
<div class="info" id="info-{name}"></div>
<div class="ctl"><label>exposure</label>
 <button onclick="api('/api/exposure',{{cam:'{name}',factor:0.667}})">-</button>
 <input id="exp-{name}" onchange="api('/api/exposure',{{cam:'{name}',ms:this.value}})">ms
 <button onclick="api('/api/exposure',{{cam:'{name}',factor:1.5}})">+</button></div>
<div class="ctl"><label>gain</label>
 <button onclick="api('/api/gain',{{cam:'{name}',delta:-25}})">-</button>
 <input id="gain-{name}" onchange="api('/api/gain',{{cam:'{name}',value:this.value}})">
 <button onclick="api('/api/gain',{{cam:'{name}',delta:25}})">+</button></div>
{extra}</div>"""

# recording captures the main camera, so its button belongs in that panel
RECORD = """<div class="rec"><button id="recbtn"
 onclick="api('/api/record',{on: this.className!=='on' ? 1 : 0})">Start recording</button>
 <span id="recinfo">not recording</span></div>"""


def render(cam, cal, max_width=800):
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
        cv2.drawMarker(img, (bx, by), (0, 200, 0), cv2.MARKER_CROSS, 30, 1)
    if cam.gate:
        gx, gy, gr = cam.gate
        cv2.circle(img, (int(gx * k), int(gy * k)), int(gr * k), (200, 120, 0), 1)
    if det:
        cv2.circle(img, (int(det.x * k), int(det.y * k)), 12, (0, 0, 255), 1)
    return img


class Preview:
    def __init__(self, cams, state, port=8080, fps=5.0, status=None, controls=None):
        self.cams, self.state, self.port, self.period = cams, state, port, 1.0 / fps
        self.status = status      # callable(camera_name) -> list of overlay lines
        self.controls = controls  # dict of callables: state/exposure/gain/record

    def page(self):
        can_record = bool(self.controls and self.controls.get("record"))
        panels = "".join(PANEL.format(name=n, extra=RECORD if (n == "main" and can_record) else "")
                         for n in self.cams)
        return PAGE.format(panels=panels, cams=json.dumps(list(self.cams)))

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

                name = path.strip("/").removesuffix(".mjpg")
                if name in preview.cams:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    cam = preview.cams[name]
                    try:
                        while True:
                            img = render(cam, preview.state.get("cameras", {}).get(name))
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
