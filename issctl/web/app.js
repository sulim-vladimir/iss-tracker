async function api(path, params) {
  const r = await fetch(path + '?' + new URLSearchParams(params));
  return apply(await r.json());
}
function tgt() { return document.getElementById('target').value; }
function mnt(action, params) { return api('/api/mount', Object.assign({action}, params)); }
let MODE = 'console', MOTORS = true;
function applyMount(m) {
  MODE = m.mode || 'console';
  MOTORS = m.motors !== false;
  const mb = document.getElementById('motorbtn');
  if (mb) { mb.textContent = MOTORS ? 'motors off' : 'motors ON'; mb.className = MOTORS ? '' : 'on'; }
  const tb = document.getElementById('trackbtn');
  if (tb) {
    tb.textContent = MODE === 'track' ? 'Stop tracking' : 'Track next pass';
    tb.className = MODE === 'track' ? 'on' : '';
  }
  for (const id of ['speedsel', 'framesel', 'target', 'passidx'])
    { const e = document.getElementById(id); if (e) e.disabled = (MODE === 'track' && id !== 'passidx'); }
  const sel = document.getElementById('speedsel');
  if (sel && !sel.options.length)
    m.speeds.forEach((v, i) => sel.add(new Option(v, i)));
  if (sel) sel.value = m.speed_index;
  const fs = document.getElementById('framesel');
  if (fs && !fs.options.length) m.frames.forEach(f => fs.add(new Option(f, f)));
  if (fs) fs.value = m.frame;
  document.getElementById('frame-hint').textContent =
    m.frame === 'axes' ? 'raw axes'
    : m.jog_raw ? 'near pole: raw axes'
    : 'target in image';
  document.getElementById('mount-busy').textContent = m.busy ? 'working...' : '';
  // Alt/az sits under the sky chart and the axis angles are not something you read while
  // working, so this panel carries only what has nowhere else to go: what just happened.
  document.getElementById('mount-msg').textContent = m.msg || '\u2014';
  const sb = document.getElementById('siderealbtn');
  if (sb) {
    sb.textContent = m.tracking ? 'sidereal ON' : 'sidereal off';
    sb.className = m.tracking ? 'on' : '';
  }
  const cal = Object.entries(m.cal).map(([n, c]) =>
    `${n}: ${c.arcsec_px}"/px (${c.scale} px/°), rotation ${c.rotation}°`);
  let head = 'not calibrated yet';
  if (cal.length) {
    head = 'calibrated';
    if (m.calibrated_at) {
      const mins = (Date.now() / 1000 - m.calibrated_at) / 60;
      head += mins < 1 ? ' just now'
            : mins < 90 ? ` ${Math.round(mins)} min ago`
            : ` ${(mins / 60).toFixed(1)} h ago`;
    }
  }
  if (m.backlash_deg)
    cal.push('backlash: axis1 ' + (m.backlash_deg[0] * 60).toFixed(1) + "' axis2 "
             + (m.backlash_deg[1] * 60).toFixed(1) + "'");
  if (m.position_at) {
    const t = new Date(m.position_at * 1000);
    cal.push('position saved ' + t.toTimeString().slice(0, 8));
  }
  document.getElementById('cal-info').textContent = [head].concat(cal).join('\n');
  const wbox = document.getElementById('cal-warn');
  wbox.textContent = '';
  (m.cal_warnings || []).forEach(w => {
    const d = document.createElement('div');
    d.className = 'w';
    d.textContent = w;
    wbox.appendChild(d);
  });
}
function apply(s) {
  for (const n of CAMS) {
    const c = s.cams[n]; if (!c) continue;
    const e = document.getElementById('exp-' + n), g = document.getElementById('gain-' + n);
    if (document.activeElement !== e) e.value = c.exposure_ms.toFixed(2);
    if (document.activeElement !== g) g.value = c.gain;
    const eu = document.getElementById('expunit-' + n);
    if (eu) eu.textContent = c.exposure_unit || 'ms';
    document.getElementById('stat-' + n).textContent =
      c.fps.toFixed(0) + ' fps  ' + (c.det ? 'detected ' + c.det[0] + ',' + c.det[1] : 'no detection');
    const info = document.getElementById('info-' + n);
    if (info) info.textContent = ((s.status || {})[n] || []).join('\n');
    const sel = document.getElementById('sel-' + n);
    if (sel) sel.textContent = c.manual
      ? (c.det ? 'locked on your pick' : 'your pick - nothing there, click again or go auto')
      : 'brightest in frame';
  }
  if (s.mount) applyMount(s.mount);
  drawSky(s);
  const em = document.getElementById('estop-msg');
  if (em) em.textContent = (s.stopped || (s.mount && s.mount.aborted)) ? 'motors halted' : '';
  const r = s.record, btn = document.getElementById('recbtn');
  if (btn && r) {
    btn.textContent = r.recording ? 'Stop recording' : 'Start recording';
    btn.className = r.recording ? 'on' : '';
    document.getElementById('recinfo').textContent = r.recording
      ? (r.waiting
          ? 'armed - waiting until the target is trackable'
          : (r.path || '') + '  ' + r.frames + ' frames, ' + r.dropped + ' dropped')
      : 'not recording';
  }
}
// ---- sky chart: zenith at the centre, horizon at the rim, north up, east right ----
const CX = 165, CY = 165, R = 140;
let SKY = null;
const SVGNS = 'http://www.w3.org/2000/svg';
function pos(az, alt) {
  const r = (90 - Math.max(alt, 0)) / 90 * R, a = az * Math.PI / 180;
  return [CX + r * Math.sin(a), CY - r * Math.cos(a)];
}
function el(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}
function sector(az0, az1, alt0, alt1) {
  const [r0, r1] = [(90 - Math.min(alt1, 90)) / 90 * R, (90 - Math.max(alt0, 0)) / 90 * R];
  let span = (az1 - az0 + 360) % 360; if (span === 0) span = 360;
  const big = span > 180 ? 1 : 0;
  const [ax, ay] = pos(az0, alt1), [bx, by] = pos(az1, alt1);
  const [cx2, cy2] = pos(az1, alt0), [dx, dy] = pos(az0, alt0);
  return `M ${ax} ${ay} A ${r0} ${r0} 0 ${big} 1 ${bx} ${by}`
       + ` L ${cx2} ${cy2} A ${r1} ${r1} 0 ${big} 0 ${dx} ${dy} Z`;
}
function drawSky(s) {
  const svg = document.getElementById('sky');
  if (!svg || !SKY) return;
  svg.textContent = '';
  for (const alt of [0, 30, 60]) {
    svg.appendChild(el('circle', {cx: CX, cy: CY, r: (90 - alt) / 90 * R,
      fill: 'none', stroke: '#4a525b'}));
  }
  if (SKY.min_alt > 0)
    svg.appendChild(el('circle', {cx: CX, cy: CY, r: (90 - SKY.min_alt) / 90 * R,
      fill: 'none', stroke: '#7a4a2a', 'stroke-dasharray': '3 3'}));
  for (const r of (SKY.mask.openings || []))
    svg.appendChild(el('path', {d: sector(r[0], r[1], r[2], r[3]), fill: '#2e7d4b', opacity: 0.22}));
  for (const r of (SKY.mask.blockers || []))
    svg.appendChild(el('path', {d: sector(r[0], r[1], r[2], r[3]), fill: '#a33', opacity: 0.3}));
  for (const [lbl, az] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {
    const [x, y] = pos(az, -6);
    svg.appendChild(el('text', {x: x, y: y + 4, fill: '#9aa4ae', 'font-size': 12,
      'text-anchor': 'middle'})).textContent = lbl;
  }
  // the pass, segment by segment: cyan while sunlit, grey in shadow, red behind an obstruction
  const trk = SKY.track || [];
  for (let i = 1; i < trk.length; i++) {
    const [az0, alt0] = trk[i - 1], [az1, alt1, lit, open] = trk[i];
    const [x0, y0] = pos(az0, alt0), [x1, y1] = pos(az1, alt1);
    svg.appendChild(el('line', {x1: x0, y1: y0, x2: x1, y2: y1, 'stroke-width': 2,
      stroke: !open ? '#c0504d' : (lit > 0.5 ? '#3fb9d6' : '#6b7580')}));
  }
  if (s.pointing) {  // where the mount looks
    const [x, y] = pos(s.pointing[1], s.pointing[0]);
    svg.appendChild(el('circle', {cx: x, cy: y, r: 6, fill: 'none', stroke: '#ffd24a',
      'stroke-width': 2}));
    svg.appendChild(el('circle', {cx: x, cy: y, r: 1.5, fill: '#ffd24a'}));
  }
  if (s.target) {  // the ISS itself: solid red dot, drawn on top
    const [x, y] = pos(s.target[1], s.target[0]);
    svg.appendChild(el('circle', {cx: x, cy: y, r: 3.5, fill: '#e2483c'}));
  }
  const n5 = v => v.toFixed(1).padStart(5);
  const fmt = (p, name) => p ? `${name} alt ${n5(p[0])}°  az ${n5(p[1])}°` : '';
  document.getElementById('sky-info').textContent =
    [passLine(s.pass), fmt(s.pointing, 'mount'), fmt(s.target, 'ISS  ')].filter(Boolean).join('\n');
}
function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? h + 'h ' : '') + (h || m ? m + 'm ' : '') + sec + 's';
}
function passLine(p) {
  if (!p) return '';
  const rise = p.rise || p.start;           // horizon crossing, not the trackable segment
  if (p.now < rise)
    return `next pass in ${clock(rise - p.now)}\n`
         + `rises ${p.rise_at || p.starts_at}, max alt ${p.max_alt.toFixed(0)}°`;
  if (p.now < p.start)
    return `ISS up, trackable in ${clock(p.start - p.now)} (at ${p.starts_at})`;
  if (p.now <= p.end)
    return `tracking  t+${(p.now - p.start).toFixed(0)}s  ${clock(p.end - p.now)} left`;
  return 'pass over';
}
(async () => { try { SKY = await (await fetch('/api/sky')).json(); } catch (e) {} })();
setInterval(async () => apply(await (await fetch('/api/state')).json()), 1000);
