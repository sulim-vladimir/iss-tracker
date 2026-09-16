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
<meta charset="utf-8">
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
 .pad{{display:grid;grid-template-areas:". u ." "l c r" ". d .";gap:5px;width:170px;margin:8px 0}}
 .pad button{{padding:8px 0}}
 button{{background:#4d555d;color:#e8e8e8;border:0;border-radius:4px;padding:5px 11px;
         font-size:14px;cursor:pointer}}
 button:hover{{background:#5c656e}}
 input{{background:#262a2e;color:#e8e8e8;border:1px solid #5c656e;border-radius:4px;
        padding:4px;width:74px;font-size:14px}}
 .rec{{margin-top:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
 #recbtn.on,#trackbtn.on{{background:#a33;}}
 #estop{{background:#b32222;color:#fff;font-weight:600;font-size:15px;padding:9px 18px;
         margin-bottom:10px}}
 #estop:hover{{background:#c93030}}
 #recinfo{{color:#b9c1c8}}
</style></head><body>
{estop}<div class="row">{panels}</div>
<script>
const CAMS = {cams};
async function api(path, params) {{
  const r = await fetch(path + '?' + new URLSearchParams(params));
  return apply(await r.json());
}}
function tgt() {{ return document.getElementById('target').value; }}
function mnt(action, params) {{ return api('/api/mount', Object.assign({{action}}, params)); }}
let MODE = 'console';
function applyMount(m) {{
  MODE = m.mode || 'console';
  const tb = document.getElementById('trackbtn');
  if (tb) {{
    tb.textContent = MODE === 'track' ? 'Stop tracking' : 'Track next pass';
    tb.className = MODE === 'track' ? 'on' : '';
  }}
  for (const id of ['speedsel', 'framesel', 'target', 'passidx'])
    {{ const e = document.getElementById(id); if (e) e.disabled = (MODE === 'track' && id !== 'passidx'); }}
  const sel = document.getElementById('speedsel');
  if (sel && !sel.options.length)
    m.speeds.forEach((v, i) => sel.add(new Option(v, i)));
  if (sel) sel.value = m.speed_index;
  const fs = document.getElementById('framesel');
  if (fs && !fs.options.length) m.frames.forEach(f => fs.add(new Option(f, f)));
  if (fs) fs.value = m.frame;
  document.getElementById('frame-hint').textContent =
    m.frame === 'axes' ? 'raw mount axes' : 'move target in image';
  document.getElementById('mount-busy').textContent = m.busy ? 'working...' : '';
  document.getElementById('mount-info').textContent =
    `mode ${{MODE}}\\n`
    + `axis1 ${{m.axis1.toFixed(3)}}°  axis2 ${{m.axis2.toFixed(3)}}°\\n`
    + `alt ${{m.alt.toFixed(2)}}°  az ${{m.az.toFixed(2)}}° ${{m.compass}}\\n`
    + `jog ${{m.jog}}  sidereal ${{m.tracking ? 'on' : 'off'}}\\n${{m.msg || ''}}`;
  const cal = Object.entries(m.cal).map(([n, c]) =>
    `${{n}}: ${{c.scale}} px/°, rotation ${{c.rotation}}°`);
  let head = 'not calibrated yet';
  if (cal.length) {{
    head = 'calibrated';
    if (m.calibrated_at) {{
      const mins = (Date.now() / 1000 - m.calibrated_at) / 60;
      head += mins < 1 ? ' just now'
            : mins < 90 ? ` ${{Math.round(mins)}} min ago`
            : ` ${{(mins / 60).toFixed(1)}} h ago`;
    }}
  }}
  const warn = (m.cal_warnings || []).map(w => '! ' + w);
  document.getElementById('cal-info').textContent = [head].concat(cal, warn).join('\\n');
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
    const sel = document.getElementById('sel-' + n);
    if (sel) sel.textContent = c.manual
      ? (c.det ? 'locked on your pick' : 'your pick - nothing there, click again or go auto')
      : 'brightest in frame';
  }}
  if (s.mount) applyMount(s.mount);
  drawSky(s);
  const em = document.getElementById('estop-msg');
  if (em) em.textContent = (s.stopped || (s.mount && s.mount.aborted)) ? 'motors halted' : '';
  const r = s.record, btn = document.getElementById('recbtn');
  if (btn && r) {{
    btn.textContent = r.recording ? 'Stop recording' : 'Start recording';
    btn.className = r.recording ? 'on' : '';
    document.getElementById('recinfo').textContent = r.recording
      ? (r.waiting
          ? 'armed - waiting until the target is trackable'
          : (r.path || '') + '  ' + r.frames + ' frames, ' + r.dropped + ' dropped')
      : 'not recording';
  }}
}}
// ---- sky chart: zenith at the centre, horizon at the rim, north up, east right ----
const CX = 165, CY = 165, R = 140;
let SKY = null;
const SVGNS = 'http://www.w3.org/2000/svg';
function pos(az, alt) {{
  const r = (90 - Math.max(alt, 0)) / 90 * R, a = az * Math.PI / 180;
  return [CX + r * Math.sin(a), CY - r * Math.cos(a)];
}}
function el(tag, attrs) {{
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}}
function sector(az0, az1, alt0, alt1) {{
  const [r0, r1] = [(90 - Math.min(alt1, 90)) / 90 * R, (90 - Math.max(alt0, 0)) / 90 * R];
  let span = (az1 - az0 + 360) % 360; if (span === 0) span = 360;
  const big = span > 180 ? 1 : 0;
  const [ax, ay] = pos(az0, alt1), [bx, by] = pos(az1, alt1);
  const [cx2, cy2] = pos(az1, alt0), [dx, dy] = pos(az0, alt0);
  return `M ${{ax}} ${{ay}} A ${{r0}} ${{r0}} 0 ${{big}} 1 ${{bx}} ${{by}}`
       + ` L ${{cx2}} ${{cy2}} A ${{r1}} ${{r1}} 0 ${{big}} 0 ${{dx}} ${{dy}} Z`;
}}
function drawSky(s) {{
  const svg = document.getElementById('sky');
  if (!svg || !SKY) return;
  svg.textContent = '';
  for (const alt of [0, 30, 60]) {{
    svg.appendChild(el('circle', {{cx: CX, cy: CY, r: (90 - alt) / 90 * R,
      fill: 'none', stroke: '#4a525b'}}));
  }}
  if (SKY.min_alt > 0)
    svg.appendChild(el('circle', {{cx: CX, cy: CY, r: (90 - SKY.min_alt) / 90 * R,
      fill: 'none', stroke: '#7a4a2a', 'stroke-dasharray': '3 3'}}));
  for (const r of (SKY.mask.openings || []))
    svg.appendChild(el('path', {{d: sector(r[0], r[1], r[2], r[3]), fill: '#2e7d4b', opacity: 0.22}}));
  for (const r of (SKY.mask.blockers || []))
    svg.appendChild(el('path', {{d: sector(r[0], r[1], r[2], r[3]), fill: '#a33', opacity: 0.3}}));
  for (const [lbl, az] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {{
    const [x, y] = pos(az, -6);
    svg.appendChild(el('text', {{x: x, y: y + 4, fill: '#9aa4ae', 'font-size': 12,
      'text-anchor': 'middle'}})).textContent = lbl;
  }}
  // the pass, segment by segment: cyan while sunlit, grey in shadow, red behind an obstruction
  const trk = SKY.track || [];
  for (let i = 1; i < trk.length; i++) {{
    const [az0, alt0] = trk[i - 1], [az1, alt1, lit, open] = trk[i];
    const [x0, y0] = pos(az0, alt0), [x1, y1] = pos(az1, alt1);
    svg.appendChild(el('line', {{x1: x0, y1: y0, x2: x1, y2: y1, 'stroke-width': 2,
      stroke: !open ? '#c0504d' : (lit > 0.5 ? '#3fb9d6' : '#6b7580')}}));
  }}
  if (s.pointing) {{  // where the mount looks: open yellow circle
    const [x, y] = pos(s.pointing[1], s.pointing[0]);
    svg.appendChild(el('circle', {{cx: x, cy: y, r: 6, fill: 'none', stroke: '#ffd24a',
      'stroke-width': 2}}));
  }}
  if (s.target) {{  // the ISS itself: solid red dot, drawn on top
    const [x, y] = pos(s.target[1], s.target[0]);
    svg.appendChild(el('circle', {{cx: x, cy: y, r: 3.5, fill: '#e2483c'}}));
  }}
  const fmt = (p, name) => p ? `${{name}} alt ${{p[0].toFixed(1)}}°  az ${{p[1].toFixed(1)}}°` : '';
  document.getElementById('sky-info').textContent =
    [passLine(s.pass), fmt(s.pointing, 'mount'), fmt(s.target, 'ISS  ')].filter(Boolean).join('\\n');
}}
function clock(seconds) {{
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? h + 'h ' : '') + (h || m ? m + 'm ' : '') + sec + 's';
}}
function passLine(p) {{
  if (!p) return '';
  const rise = p.rise || p.start;           // horizon crossing, not the trackable segment
  if (p.now < rise)
    return `next pass in ${{clock(rise - p.now)}}\\n`
         + `rises ${{p.rise_at || p.starts_at}}, max alt ${{p.max_alt.toFixed(0)}}°`;
  if (p.now < p.start)
    return `ISS up, trackable in ${{clock(p.start - p.now)}} (at ${{p.starts_at}})`;
  if (p.now <= p.end)
    return `tracking  t+${{(p.now - p.start).toFixed(0)}}s  ${{clock(p.end - p.now)}} left`;
  return 'pass over';
}}
(async () => {{ try {{ SKY = await (await fetch('/api/sky')).json(); }} catch (e) {{}} }})();
setInterval(async () => apply(await (await fetch('/api/state')).json()), 1000);
</script></body></html>"""

PANEL = """<div class="panel"><h2>{name} <span id="stat-{name}"></span></h2>
<img src="/{name}.mjpg" title="click the object to track"
 onclick="api('/api/select',{{cam:'{name}',fx:event.offsetX/this.clientWidth,
                             fy:event.offsetY/this.clientHeight}})">
<div class="info" id="info-{name}"></div>
<div class="ctl"><label>target</label><span id="sel-{name}"></span>
 <button onclick="api('/api/select',{{cam:'{name}',clear:1}})">Auto</button></div>
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

ESTOP = """<button id="estop" onclick="api('/api/estop',{})">EMERGENCY STOP</button>
<span id="estop-msg"></span>"""

SKY = """<div class="panel"><h2>sky</h2>
<svg id="sky" viewBox="0 0 330 330" style="width:100%;max-width:340px;background:#20242a;
 border-radius:6px"></svg>
<div class="info" id="sky-info"></div></div>"""

MOUNT = """<div class="panel"><h2>mount <span id="mount-busy"></span></h2>
<div class="info" id="mount-info"></div>
<div class="pad">
 <button style="grid-area:u" onclick="mnt('jog',{axis:2,dir:1})">&#9650;</button>
 <button style="grid-area:l" onclick="mnt('jog',{axis:1,dir:-1})">&#9664;</button>
 <button style="grid-area:c" onclick="mnt('stop',{})">stop</button>
 <button style="grid-area:r" onclick="mnt('jog',{axis:1,dir:1})">&#9654;</button>
 <button style="grid-area:d" onclick="mnt('jog',{axis:2,dir:-1})">&#9660;</button></div>
<div class="ctl"><label>arrows</label>
 <select id="framesel" onchange="mnt('frame',{frame:this.value})"></select>
 <span id="frame-hint"></span></div>
<div class="ctl"><label>speed</label>
 <select id="speedsel" onchange="mnt('speed',{index:this.value})"></select> °/s
 <button onclick="mnt('track',{on:1})">sidereal on</button>
 <button onclick="mnt('track',{on:0})">off</button></div>
<div class="ctl"><label>target</label>
 <input id="target" placeholder="vega / jupiter / 18.6 38.8" style="width:150px">
 <button onclick="mnt('goto',{target:tgt()})">goto</button>
 <button onclick="mnt('sync',{target:tgt()})">sync</button></div>
<div class="ctl"><label>pass</label>
 <button id="trackbtn" onclick="mnt(MODE==='track' ? 'untrack' : 'track',
   {pass: document.getElementById('passidx').value})">Track next pass</button>
 <input id="passidx" placeholder="next" style="width:56px" title="pass index from 'passes', or blank for the next usable one"></div>
<div class="ctl"><label></label>
 <button onclick="if(confirm('Set current position as home?')) mnt('home',{})">set home</button>
 <button onclick="mnt('calibrate',{})">calibrate cameras</button>
 <button onclick="mnt('mask',{})">mask point</button></div>
<div class="info" id="cal-info"></div></div>"""


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
        cv2.drawMarker(img, (bx, by), (0, 220, 0), cv2.MARKER_CROSS, 34, 2)
    if cam.gate:
        gx, gy, gr = cam.gate
        cv2.circle(img, (int(gx * k), int(gy * k)), int(gr * k), (220, 130, 0), 2)
    if det:
        cv2.circle(img, (int(det.x * k), int(det.y * k)), 14, (0, 0, 255), 2)
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
        if self.controls and self.controls.get("mount_action"):
            panels += MOUNT
        if self.controls and self.controls.get("sky"):
            panels += SKY
        estop = ESTOP if (self.controls and self.controls.get("estop")) else ""
        return PAGE.format(panels=panels, cams=json.dumps(list(self.cams)), estop=estop)

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
