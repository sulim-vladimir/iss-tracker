async function api(path, params) {
  const r = await fetch(path + '?' + new URLSearchParams(params));
  return apply(await r.json());
}
function tgt() { return document.getElementById('target').value; }
function mnt(action, params) { return api('/api/mount', Object.assign({action}, params)); }
let MODE = 'console', MOTORS = true, PASS_MODE = false;
function applyMount(m) {
  MODE = m.mode || 'console';
  MOTORS = m.motors !== false;
  const mb = document.getElementById('motorbtn');
  if (mb) { mb.textContent = MOTORS ? 'motors off' : 'motors ON'; mb.className = MOTORS ? '' : 'on'; }
  const servo = MODE === 'track' && m.session === 'servo';
  PASS_MODE = MODE === 'track' && !servo;
  const vb = document.getElementById('servobtn');
  if (vb) {
    if (servo) FOLLOW_ARMED = false;
    vb.textContent = servo ? 'Stop following' : (FOLLOW_ARMED ? 'Click the object' : 'Follow');
    vb.className = (servo || FOLLOW_ARMED ? 'on ' : '') + 'right';
    vb.disabled = MODE === 'track' && !servo;
  }
  for (const id of ['speedsel', 'framesel', 'target'])
    { const e = document.getElementById(id); if (e) e.disabled = MODE === 'track'; }
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
  showComing(m.forecast, m.now);
  const spb = document.getElementById('spiralbtn');
  if (spb) {
    spb.dataset.running = m.spiral ? '1' : '0';
    spb.textContent = m.spiral ? 'Stop here' : 'Spiral search in main';
    spb.className = m.spiral ? 'on' : '';
  }
  const sb2 = document.getElementById('steerbtn');
  if (sb2) {
    sb2.dataset.on = m.main_steers ? '1' : '0';
    sb2.textContent = 'Main steers: ' + (m.main_steers ? 'ON' : 'off');
    sb2.className = (m.main_steers ? 'on ' : '') + 'right';
  }
  const ib = document.getElementById('identbtn');
  if (ib) { const on = m.identify_on !== false; ib.textContent = on ? 'naming ON' : 'naming off'; ib.className = on ? 'on' : ''; }
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
  if (m.alignment) cal.push('stars: ' + m.alignment);
  if (m.position_at) {
    const t = new Date(m.position_at * 1000);
    cal.push('position saved ' + t.toTimeString().slice(0, 8));
  }
  document.getElementById('cal-info').textContent = [head].concat(cal).join('\n');
  const lg = document.getElementById('log');
  if (lg) {
    const at_end = lg.scrollTop + lg.clientHeight >= lg.scrollHeight - 4;
    lg.textContent = (m.log || []).join('\n');
    if (at_end) lg.scrollTop = lg.scrollHeight;   // follow, unless you have scrolled back
  }
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
    const sat = n === 'guide' && s.mount && s.mount.sat_label ? '  \u00b7 ' + s.mount.sat_label : '';
    document.getElementById('stat-' + n).textContent =
      c.fps.toFixed(0) + ' fps  ' + (c.det ? 'detected' : 'no detection') + trackNote(n, s.mount && s.mount.track) + sat;
    const info = document.getElementById('info-' + n);
    if (info) info.textContent = ((s.status || {})[n] || []).join('\n');
    const sel = document.getElementById('sel-' + n);
    if (sel) sel.textContent = c.manual
      ? (c.det ? 'locked on your pick' : 'your pick - nothing there, click again or go auto')
      : 'brightest in frame';
  }
  if (s.mount) applyMount(s.mount);
  LAST_STATE = s;
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
let SKY = null, LAST_STATE = null, SKY_PICK = null;   // SKY_PICK: [az, alt] clicked on the chart
const SVGNS = 'http://www.w3.org/2000/svg';
function pickSky(ev) {     // a click on the chart: the alt/az under it, for "Go to point"
  const svg = document.getElementById('sky'), box = svg.getBoundingClientRect();
  const x = (ev.clientX - box.left) * 330 / box.width - CX, y = (ev.clientY - box.top) * 330 / box.height - CY;
  const alt = 90 - Math.hypot(x, y) / R * 90;
  if (alt < 0) return;
  SKY_PICK = [(Math.atan2(x, -y) * 180 / Math.PI + 360) % 360, alt];
  const b = document.getElementById('skygoto');
  if (b) b.disabled = MODE === 'track';
  if (LAST_STATE) drawSky(LAST_STATE);
}
function gotoSky() {
  if (SKY_PICK) mnt('goto_altaz', {az: SKY_PICK[0].toFixed(2), alt: SKY_PICK[1].toFixed(2)});
}
function skyCircle(alt, az, radius) {
  // A circle ON THE SKY of `radius` degrees round (alt, az), as chart points. The chart stretches
  // things sideways away from the zenith, so a plain SVG circle would be the wrong shape.
  const d = Math.PI / 180, a0 = alt * d, r = radius * d, pts = [];
  for (let b = 0; b < 360; b += 10) {
    const br = b * d;
    const a1 = Math.asin(Math.sin(a0) * Math.cos(r) + Math.cos(a0) * Math.sin(r) * Math.cos(br));
    const z1 = az * d + Math.atan2(Math.sin(br) * Math.sin(r) * Math.cos(a0),
                                   Math.cos(r) - Math.sin(a0) * Math.sin(a1));
    const [x, y] = pos(z1 / d, a1 / d);
    pts.push(x.toFixed(1) + ',' + y.toFixed(1));
  }
  return pts.join(' ');
}
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
  drawComing(svg, (s.mount && s.mount.now) || Date.now() / 1000);
  if (SKY_PICK) {            // the picked point: a yellow cross
    const [x, y] = pos(SKY_PICK[0], SKY_PICK[1]);
    for (const [dx, dy] of [[7, 0], [0, 7]])
      svg.appendChild(el('line', {x1: x - dx, y1: y - dy, x2: x + dx, y2: y + dy,
        stroke: '#ffd24a', 'stroke-width': 2}));
  }
  if (s.pointing) {  // where the mount looks, and what the guide sees round it
    const [x, y] = pos(s.pointing[1], s.pointing[0]);
    const ring = skyCircle(s.pointing[0], s.pointing[1], (SKY && SKY.guide_radius_deg) || 6.5);
    svg.appendChild(el('polygon', {points: ring, fill: '#ffd24a', 'fill-opacity': 0.08,
      stroke: '#ffd24a', 'stroke-width': 1.5}));
    svg.appendChild(el('circle', {cx: x, cy: y, r: 1.5, fill: '#ffd24a'}));
  }
  if (s.target) {  // the ISS itself: solid red dot, drawn on top
    const [x, y] = pos(s.target[1], s.target[0]);
    svg.appendChild(el('circle', {cx: x, cy: y, r: 3.5, fill: '#e2483c'}));
  }
  const n5 = v => v.toFixed(1).padStart(5);
  // az under alt, with the labels padded to the same width so the numbers line up
  const fmt = (p, name) => p ? `${name} alt ${n5(p[0])}°\n${' '.repeat(name.length)} az  ${n5(p[1])}°` : '';
  document.getElementById('sky-info').textContent =
    [passLine(s.pass), fmt(s.pointing, 'mount'), rateLine(s.mount && s.mount.rates),
     SKY_PICK ? fmt([SKY_PICK[1], SKY_PICK[0]], 'point') : '',
     fmt(s.target, ((s.pass && s.pass.name) || 'ISS').slice(0, 12))].filter(Boolean).join('\n');
}
function rateLine(r) {       // what the mount is being driven at right now, per axis
  if (!r) return '';
  const f = v => (v >= 0 ? '+' : '') + v.toFixed(4);
  return `rates axis1 ${f(r[0])}°/s\n      axis2 ${f(r[1])}°/s`;
}
function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? h + 'h ' : '') + (h || m ? m + 'm ' : '') + sec + 's';
}
function passLine(p) {
  if (!p) return '';
  const rise = p.rise || p.start;           // horizon crossing, not the trackable segment
  const who = p.name || 'ISS';
  if (p.now < rise)
    return `${who}: next pass in ${clock(rise - p.now)}\n`
         + `rises ${p.rise_at || p.starts_at}, max alt ${p.max_alt.toFixed(0)}°`;
  if (p.now < p.start)
    return `${who} up, trackable in ${clock(p.start - p.now)} (at ${p.starts_at})`;
  if (p.now <= p.end)
    return `tracking  t+${(p.now - p.start).toFixed(0)}s  ${clock(p.end - p.now)} left`;
  return 'pass over';
}
(async () => { try { SKY = await (await fetch('/api/sky')).json(); } catch (e) {} })();
setInterval(async () => apply(await (await fetch('/api/state')).json()), 1000);


// ---- clicking an image: normally picks a target, but "set boresight" claims the next click ----
let ARMED = null, FOLLOW_ARMED = false, BRIGHT_ARMED = false;
function armBright() {        // "Brightness" claims the next click in the guide image
  BRIGHT_ARMED = !BRIGHT_ARMED;
  const b = document.getElementById('brightbtn');
  if (b) { b.textContent = BRIGHT_ARMED ? 'Click a star' : 'Brightness'; b.className = BRIGHT_ARMED ? 'on' : ''; }
}
function armFollow() {        // "Follow" claims the next click in the guide image, like the boresight
  FOLLOW_ARMED = !FOLLOW_ARMED;
  if (FOLLOW_ARMED && ARMED) armBoresight(ARMED);
  const b = document.getElementById('servobtn');
  if (b) { b.textContent = FOLLOW_ARMED ? 'Click the object' : 'Follow'; b.className = FOLLOW_ARMED ? 'on right' : 'right'; }
}
function armBoresight(name) {
  ARMED = ARMED === name ? null : name;
  for (const n of CAMS) {
    const b = document.getElementById('bore-' + n);
    if (b) { b.className = ARMED === n ? 'on' : ''; b.textContent = ARMED === n ? 'click the spot' : 'set boresight'; }
  }
}
function imgClick(name, event, img) {
  const fx = event.offsetX / img.clientWidth, fy = event.offsetY / img.clientHeight;
  if (ARMED === name) { const n = name; armBoresight(name); return mnt('boresight', {cam: n, fx, fy}); }
  if (FOLLOW_ARMED && name === 'guide') { armFollow(); return mnt('servo', {fx, fy}); }
  if (BRIGHT_ARMED && name === 'guide') { armBright(); return mnt('brightness', {fx, fy}); }
  return api('/api/select', {cam: name, fx, fy});
}

function copyLog(btn) {
  const text = document.getElementById('log').textContent;
  const done = ok => { btn.textContent = ok ? 'copied' : 'failed';
                       setTimeout(() => { btn.textContent = 'copy'; }, 1200); };
  // navigator.clipboard needs a secure context, and this page is plain http on the LAN, so
  // fall back to the old selection trick rather than silently doing nothing.
  if (navigator.clipboard && window.isSecureContext)
    navigator.clipboard.writeText(text).then(() => done(true), () => done(false));
  else {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
    done(ok);
  }
}

// ---- coming up: bright satellites through the guide field, or anywhere visible ----
let COMING = [], COMING_SEL = null;   // the rows on show, and the key of the one picked
const comingKey = r => r.id + ':' + r.start;
function comingSelected() { return COMING.find(r => comingKey(r) === COMING_SEL) || null; }
function showComing(f, now) {
  const box = document.getElementById('coming');
  if (!box) return;
  if (!f) { box.textContent = ''; COMING = []; markComing(box); return; }
  if (f.busy && !f.items) { box.textContent = 'working it out...'; return; }
  const hm = t => new Date(t * 1000).toTimeString().slice(0, 8);
  const rows = (f.items || []).filter(r => r.end > now).slice(0, 12);
  COMING = rows;
  if (!comingSelected()) COMING_SEL = null;              // it has gone by
  const key = f.at + ':' + rows.map(comingKey).join(',');
  if (box.dataset.key !== key) {        // rebuild only when the list changes, not every poll
    box.dataset.key = key;
    box.replaceChildren();
    for (const r of rows) {
      const line = document.createElement('div');
      line.className = 'row';
      line.dataset.key = comingKey(r);
      line.title = `${r.name} (catalogue ${r.id}) - click to show it on the sky chart`;
      line.onclick = () => {
        COMING_SEL = COMING_SEL === line.dataset.key ? null : line.dataset.key;
        markComing(box);
        if (LAST_STATE) drawSky(LAST_STATE);
      };
      box.appendChild(line);
    }
  }
  const head = document.getElementById('coming-head');
  if (head) head.textContent = `${f.where} · next ${f.minutes || 60} min · updated ${hm(f.at).slice(0, 5)}`
    + (f.busy ? ' (updating...)' : '') + (rows.length ? '' : ' · nothing bright');
  box.querySelectorAll('.row').forEach((el, i) => {
    const r = rows[i];
    const when = r.start > now ? 'in ' + clock(r.start - now) : 'NOW, ' + clock(r.end - now) + ' left';
    const where = f.mode === 'field' ? `${r.sep.toFixed(1)}° from centre`
                                     : `alt ${r.alt.toFixed(0)}° az ${r.az.toFixed(0)}°`;
    el.textContent = `${hm(r.peak)}  ${when.padEnd(14)} mag ${r.mag.toFixed(1).padStart(4)}  ${r.name}\n`
                   + `          ${where}, ${r.range_km} km`;
  });
  markComing(box);
}
function trackComing() {           // "track selected", or "Stop tracking" while a pass runs
  if (PASS_MODE) return mnt('untrack', {});
  const r = comingSelected();
  if (r) mnt('track', {sat: r.id, at: r.peak});
}
function markComing(box) {
  box.querySelectorAll('.row').forEach(el => el.classList.toggle('sel', el.dataset.key === COMING_SEL));
  const b = document.getElementById('comingtrack');
  if (b) {
    b.textContent = PASS_MODE ? 'Stop tracking' : 'Track selected';
    b.className = PASS_MODE ? 'on' : '';
    b.disabled = PASS_MODE ? false : (!comingSelected() || MODE === 'track');
  }
}
// the picked pass on the sky chart: its path in violet, and where it is now (or where it comes in)
function drawComing(svg, now) {
  const r = comingSelected();
  if (!r || !r.path || r.path.length < 2) return;
  for (let i = 1; i < r.path.length; i++) {
    const [x0, y0] = pos(r.path[i - 1][0], r.path[i - 1][1]), [x1, y1] = pos(r.path[i][0], r.path[i][1]);
    svg.appendChild(el('line', {x1: x0, y1: y0, x2: x1, y2: y1, 'stroke-width': 2, stroke: '#b784f5'}));
  }
  // Where it is NOW, between the 10 s path points: jumping to the next point drew it up to
  // 10 s ahead, which made a mount sitting right on it look as if it were trailing.
  let k = r.path.findIndex(p => p[2] >= now);
  const inside = k > 0 && now >= r.path[0][2];
  let az, alt;
  if (inside) {
    const [a0, h0, t0] = r.path[k - 1], [a1, h1, t1] = r.path[k];
    const f = (now - t0) / Math.max(t1 - t0, 1e-6), da = ((a1 - a0 + 540) % 360) - 180;
    az = (a0 + f * da + 360) % 360;
    alt = h0 + f * (h1 - h0);
  } else {
    [az, alt] = r.path[0];
  }
  const [x, y] = pos(az, alt);
  svg.appendChild(el('circle', inside ? {cx: x, cy: y, r: 4, fill: '#b784f5'}
                                      : {cx: x, cy: y, r: 4, fill: 'none', stroke: '#b784f5', 'stroke-width': 2}));
  svg.appendChild(el('text', {x: x + 7, y: y - 6, fill: '#cdb0f7', 'font-size': 11})).textContent = r.name;
}

// ---- which camera the tracker is steering with, shown in each camera's caption ----
function trackNote(name, t) {
  if (!t) return '';
  if (t.source === name) return '  \u00b7 TRACKING uses this camera';
  if (t.source === 'guide' || t.source === 'main')
    return name === 'main' && t.handoff
      ? `  \u00b7 standby (handoff ${Math.min(t.main_streak, t.handoff)}/${t.handoff} frames)`
      : '  \u00b7 standby';
  const why = {predict: 'coasting on prediction', shadow: 'in Earth shadow - coasting',
               blocked: 'behind an obstruction - coasting'}[t.source] || t.source;
  return '  \u00b7 ' + why;
}
