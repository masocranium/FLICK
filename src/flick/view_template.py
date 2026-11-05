"""The page `flick view` writes. See `flick.view`. Everything is inline, so the file
works offline and can be opened directly. Data goes into the DOM with textContent."""

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Generated mouse paths</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --generated: #2a78d6; --real: #eb6834; --wash: rgba(11,11,11,0.05);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --generated: #3987e5; --real: #d95926; --wash: rgba(255,255,255,0.07);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --generated: #3987e5; --real: #d95926; --wash: rgba(255,255,255,0.07);
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body {
  margin: 0; padding: 24px 16px 48px; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 0; }
.sub { color: var(--ink-2); margin: 0; }
.controls {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 16px 0;
}
button, select {
  font: inherit; color: var(--ink); background: var(--surface); padding: 5px 12px;
  border: 1px solid var(--axis); border-radius: 6px;
}
button { cursor: pointer; font-weight: 600; }
button:hover, select:hover { background: var(--wash); }
label { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); }
#clock { color: var(--ink-2); font-variant-numeric: tabular-nums; min-width: 6ch; }
.card {
  position: relative; background: var(--surface); border-radius: 10px;
  box-shadow: 0 0 0 1px var(--ring); padding: 16px; margin-bottom: 16px;
}
.card .sub { margin-bottom: 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--ink-2); margin-bottom: 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { display: inline-block; width: 14px; height: 2px; border-radius: 1px; flex: none; }
.key.generated { background: var(--generated); }
.key.real { background: var(--real); }
svg { display: block; width: 100%; touch-action: pan-y; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .place {
  font-size: 12px; font-weight: 600; fill: var(--ink); paint-order: stroke;
  stroke: var(--surface); stroke-width: 3px; stroke-linejoin: round;
}
.tip {
  position: absolute; z-index: 2; pointer-events: none; display: none; min-width: 150px;
  background: var(--surface); box-shadow: 0 0 0 1px var(--ring), 0 4px 14px rgba(0,0,0,.14);
  border-radius: 8px; padding: 8px 10px; font-size: 12px;
}
.tip .head { color: var(--ink-2); margin-bottom: 4px; }
.tip .row { display: flex; align-items: center; gap: 8px; }
.tip b { font-weight: 600; font-variant-numeric: tabular-nums; }
.tip .name { color: var(--ink-2); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 5px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 600; vertical-align: bottom; white-space: normal; min-width: 72px; }
th:first-child, td:first-child { text-align: left; }
td { font-variant-numeric: tabular-nums; }
tbody tr { cursor: pointer; }
tbody tr:hover, tbody tr.active { background: var(--wash); }
tbody tr:focus-visible { outline: 2px solid var(--generated); outline-offset: -2px; }
td .key { margin-right: 8px; vertical-align: middle; }
</style>
</head>
<body>
<main>
  <h1>Generated mouse paths</h1>
  <p class="sub" id="summary"></p>

  <div class="controls">
    <button id="play" type="button">Replay</button>
    <label>Speed
      <select id="rate">
        <option value="1">Real time</option>
        <option value="0.5">1/2 speed</option>
        <option value="0.25">1/4 speed</option>
      </select>
    </label>
    <label id="real-toggle" hidden><input type="checkbox" id="show-real" checked> Show recorded movements</label>
    <span id="clock"></span>
  </div>

  <section class="card" id="path-card">
    <h2>Paths on screen</h2>
    <p class="sub">Pixels, with equal scales on both axes and y increasing downward, as on a screen.</p>
    <div class="legend" id="path-legend"></div>
    <svg id="path-chart" role="img"></svg>
    <div class="tip" id="path-tip"></div>
  </section>

  <section class="card" id="speed-card">
    <h2>Speed over time</h2>
    <p class="sub">Pixels per second, smoothed the same way the features are measured.</p>
    <div class="legend" id="speed-legend"></div>
    <svg id="speed-chart" role="img"></svg>
    <div class="tip" id="speed-tip"></div>
  </section>

  <section class="card">
    <h2>Drawn and measured</h2>
    <p class="sub">Drawn is what the model asked for. Measured is what the path has, from the same extractor as recordings. Select a row to highlight its path.</p>
    <div class="scroll"><table id="table"></table></div>
  </section>
</main>

<script id="flick-data" type="application/json">/*__FLICK_DATA__*/null</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("flick-data").textContent);
  var SVG_NS = "http://www.w3.org/2000/svg";
  var COLOR = { generated: "var(--generated)", real: "var(--real)" };
  var NAME = { generated: "Generated", real: "Recorded" };

  var all = DATA.movements;
  var hasReal = all.some(function (m) { return m.kind === "real"; });
  var state = { hover: null, pinned: null, showReal: true, playing: false, clock: null };
  var pathEls = {}, speedEls = {}, dotEls = {};

  function visible() {
    return all.filter(function (m) { return m.kind !== "real" || state.showReal; });
  }
  function active() { return state.hover || state.pinned; }
  function byId(id) { return all.filter(function (m) { return m.id === id; })[0]; }
  function fmt(v, digits) {
    if (v === null || v === undefined) return "—";
    return Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, tag, value, cls) {
    var e = document.createElement(tag);
    e.textContent = value;
    if (cls) e.className = cls;
    parent.appendChild(e);
    return e;
  }
  function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); }
  function median(values) {
    var s = values.slice().sort(function (a, b) { return a - b; });
    if (!s.length) return null;
    var i = s.length >> 1;
    return s.length % 2 ? s[i] : (s[i - 1] + s[i]) / 2;
  }
  function niceStep(span, count) {
    var raw = span / Math.max(count, 1), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var unit = raw / mag;
    return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * mag;
  }
  function tickLabel(v, step) {
    if (Math.abs(v) >= 10000) return (v / 1000).toLocaleString() + "K";
    return Number(v).toLocaleString(undefined, { maximumFractionDigits: step < 1 ? 2 : 0 });
  }
  function maxT() {
    return Math.max.apply(null, visible().map(function (m) { return m.t[m.t.length - 1]; }));
  }

  // Summary and legends ------------------------------------------------------
  var generatedCount = all.filter(function (m) { return m.kind === "generated"; }).length;
  var realCount = all.length - generatedCount;
  var parts = [generatedCount + " generated by the " + DATA.model + " model",
               "from (" + DATA.start.join(", ") + ") to (" + DATA.target.join(", ") + ")",
               fmt(DATA.distance_px, 0) + " px apart"];
  if (DATA.seed !== null) parts.push("seed " + DATA.seed);
  document.getElementById("summary").textContent = parts.join(", ") + "." +
    (realCount ? " " + realCount + " recorded clicks of a similar distance are rotated and scaled onto the same start and target." : "");
  if (hasReal) {
    document.getElementById("real-toggle").hidden = false;
    ["path-legend", "speed-legend"].forEach(function (id) {
      var box = document.getElementById(id);
      ["generated", "real"].forEach(function (kind) {
        var item = document.createElement("span");
        var key = text(item, "i", "", "key " + kind);
        key.setAttribute("aria-hidden", "true");
        item.appendChild(document.createTextNode(NAME[kind]));
        box.appendChild(item);
      });
    });
  }

  // Tooltips -----------------------------------------------------------------
  function placeTip(tip, card, x, y) {
    tip.style.display = "block";
    var box = card.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    var left = x + 14 + w > box.width - 8 ? x - w - 14 : x + 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, Math.min(y - h / 2, box.height - h - 8)) + "px";
  }
  function tipRow(tip, value, name, kind) {
    var row = text(tip, "div", "", "row");
    if (kind) text(row, "i", "", "key " + kind);
    text(row, "b", value);
    text(row, "span", name, "name");
  }

  // Path chart ---------------------------------------------------------------
  var pathChart = document.getElementById("path-chart");
  var pathScale = null, pathLayer = null, dotLayer = null;

  function drawPaths() {
    clear(pathChart);
    pathEls = {}; dotEls = {};
    var W = pathChart.clientWidth || 800;
    var H = Math.max(260, Math.min(520, Math.round(W * 0.5)));
    var m = { l: 46, r: 14, t: 14, b: 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    var xs = [], ys = [];
    all.forEach(function (mv) { xs = xs.concat(mv.x); ys = ys.concat(mv.y); });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var pad = Math.max(DATA.distance_px * 0.06, 6);
    x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
    var s = Math.min(iw / (x1 - x0), ih / (y1 - y0));
    var ox = m.l + (iw - s * (x1 - x0)) / 2 - s * x0, oy = m.t + (ih - s * (y1 - y0)) / 2 - s * y0;
    var sx = function (v) { return ox + s * v; }, sy = function (v) { return oy + s * v; };
    pathScale = { sx: sx, sy: sy, W: W, H: H };

    pathChart.setAttribute("viewBox", "0 0 " + W + " " + H);
    pathChart.setAttribute("height", H);
    pathChart.setAttribute("aria-label", "Generated paths from start to target, in screen pixels. The table below lists each path's values.");

    var step = niceStep(iw / s, W < 560 ? 4 : 8), grid = svg("g", {}, pathChart);
    var left = (m.l - ox) / s, right = (W - m.r - ox) / s, top = (m.t - oy) / s, bottom = (H - m.b - oy) / s;
    for (var gx = Math.ceil(left / step) * step; gx <= right; gx += step) {
      svg("line", { x1: sx(gx), x2: sx(gx), y1: m.t, y2: H - m.b, stroke: "var(--grid)" }, grid);
      svg("text", { x: sx(gx), y: H - 8, "text-anchor": "middle" }, grid).textContent = tickLabel(gx, step);
    }
    for (var gy = Math.ceil(top / step) * step; gy <= bottom; gy += step) {
      svg("line", { x1: m.l, x2: W - m.r, y1: sy(gy), y2: sy(gy), stroke: "var(--grid)" }, grid);
      svg("text", { x: m.l - 6, y: sy(gy) + 4, "text-anchor": "end" }, grid).textContent = tickLabel(gy, step);
    }

    pathLayer = svg("g", { fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, pathChart);
    all.forEach(function (mv) {
      var d = mv.x.map(function (v, i) { return (i ? "L" : "M") + sx(v).toFixed(1) + " " + sy(mv.y[i]).toFixed(1); }).join("");
      pathEls[mv.id] = svg("path", { d: d, stroke: COLOR[mv.kind] }, pathLayer);
    });

    dotLayer = svg("g", {}, pathChart);
    all.forEach(function (mv) {
      dotEls[mv.id] = svg("circle", { r: 4, fill: COLOR[mv.kind], stroke: "var(--surface)", "stroke-width": 2, display: "none" }, dotLayer);
    });

    // After the dots, so a replay can't cover the labels.
    [["Start", DATA.start], ["Target", DATA.target]].forEach(function (place) {
      var cx = sx(place[1][0]), cy = sy(place[1][1]);
      svg("circle", { cx: cx, cy: cy, r: 5, fill: "var(--ink)", stroke: "var(--surface)", "stroke-width": 2 }, pathChart);
      var label = svg("text", { x: cx, y: cy < m.t + 26 ? cy + 22 : cy - 12, "text-anchor": cx < m.l + 30 ? "start" : cx > W - m.r - 30 ? "end" : "middle", "class": "place" }, pathChart);
      label.textContent = place[0];
    });

    var hit = svg("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }, pathChart);
    hit.addEventListener("pointermove", onPathHover);
    hit.addEventListener("pointerleave", function () { setHover(null); document.getElementById("path-tip").style.display = "none"; });
    hit.addEventListener("click", function () { state.pinned = state.hover === state.pinned ? null : state.hover; emphasize(); });
  }

  // The nearest path within 24px, so nobody has to land on a 2px line.
  function onPathHover(ev) {
    var r = pathChart.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
    var best = null, bestD = 24 * 24;
    visible().forEach(function (mv) {
      for (var i = 0; i < mv.x.length; i++) {
        var dx = pathScale.sx(mv.x[i]) - px, dy = pathScale.sy(mv.y[i]) - py, d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = mv; }
      }
    });
    setHover(best ? best.id : null);
    var tip = document.getElementById("path-tip");
    if (!best) { tip.style.display = "none"; return; }
    clear(tip);
    text(tip, "div", best.id, "head");
    tipRow(tip, fmt(best.measured.duration_s, 2) + " s", "duration", best.kind);
    tipRow(tip, fmt(best.measured.v_peak, 0) + " px/s", "peak speed");
    tipRow(tip, fmt(best.measured.n_submovements, 0), "corrections");
    var card = document.getElementById("path-card").getBoundingClientRect();
    placeTip(tip, document.getElementById("path-card"), ev.clientX - card.left, ev.clientY - card.top);
  }

  // Speed chart --------------------------------------------------------------
  var speedChart = document.getElementById("speed-chart");
  var speedScale = null, crosshair = null;

  function drawSpeed() {
    clear(speedChart);
    speedEls = {};
    var W = speedChart.clientWidth || 800, H = 280;
    var m = { l: 46, r: 14, t: 10, b: 26 };
    var endT = Math.max.apply(null, all.map(function (mv) { return mv.t[mv.t.length - 1]; }));
    var top = 0;
    all.forEach(function (mv) { mv.speed.forEach(function (v) { if (v > top) top = v; }); });
    var yStep = niceStep(top, 4), yMax = Math.ceil(top / yStep) * yStep || 1;
    var sx = function (t) { return m.l + (W - m.l - m.r) * t / endT; };
    var sy = function (v) { return H - m.b - (H - m.t - m.b) * v / yMax; };
    speedScale = { sx: sx, sy: sy, W: W, H: H, m: m, endT: endT };

    speedChart.setAttribute("viewBox", "0 0 " + W + " " + H);
    speedChart.setAttribute("height", H);
    speedChart.setAttribute("aria-label", "Speed over time for each path. The table below lists each path's peak speed.");

    for (var v = 0; v <= yMax + 1e-9; v += yStep) {
      svg("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: v === 0 ? "var(--axis)" : "var(--grid)" }, speedChart);
      svg("text", { x: m.l - 6, y: sy(v) + 4, "text-anchor": "end" }, speedChart).textContent = tickLabel(v, yStep);
    }
    var tStep = niceStep(endT, W < 560 ? 4 : 8);
    for (var t = 0; t <= endT + 1e-9; t += tStep) {
      svg("text", { x: sx(t), y: H - 8, "text-anchor": "middle" }, speedChart).textContent = Number(t.toFixed(3)) + " s";
    }

    var layer = svg("g", { fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, speedChart);
    all.forEach(function (mv) {
      var d = mv.speed.map(function (v, i) { return (i ? "L" : "M") + sx(i / DATA.hz).toFixed(1) + " " + sy(v || 0).toFixed(1); }).join("");
      speedEls[mv.id] = svg("path", { d: d, stroke: COLOR[mv.kind] }, layer);
    });

    crosshair = svg("line", { y1: m.t, y2: H - m.b, stroke: "var(--muted)", display: "none" }, speedChart);
    var hit = svg("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }, speedChart);
    hit.addEventListener("pointermove", onSpeedHover);
    hit.addEventListener("pointerleave", function () {
      document.getElementById("speed-tip").style.display = "none";
      if (!state.playing) crosshair.setAttribute("display", "none");
    });
  }

  function speedAt(mv, t) {
    var i = Math.round(t * DATA.hz);
    return i < mv.speed.length ? mv.speed[i] : null;
  }

  function onSpeedHover(ev) {
    var r = speedChart.getBoundingClientRect(), s = speedScale;
    var raw = (ev.clientX - r.left - s.m.l) / (s.W - s.m.l - s.m.r) * s.endT;
    var t = Math.max(0, Math.min(s.endT, Math.round(raw * DATA.hz) / DATA.hz));
    moveCrosshair(t);
    var tip = document.getElementById("speed-tip");
    clear(tip);
    text(tip, "div", t.toFixed(2) + " s", "head");
    ["generated", "real"].forEach(function (kind) {
      if (kind === "real" && !(hasReal && state.showReal)) return;
      var values = all.filter(function (m) { return m.kind === kind; })
        .map(function (m) { return speedAt(m, t); }).filter(function (v) { return v !== null; });
      if (values.length) tipRow(tip, fmt(median(values), 0) + " px/s", NAME[kind] + " median, " + values.length + " moving", kind);
      else tipRow(tip, "—", NAME[kind] + ": all finished", kind);
    });
    var chosen = active() && byId(active());
    if (chosen && (chosen.kind !== "real" || state.showReal)) {
      var own = speedAt(chosen, t);
      tipRow(tip, own === null ? "—" : fmt(own, 0) + " px/s", chosen.id);
    }
    var card = document.getElementById("speed-card").getBoundingClientRect();
    placeTip(tip, document.getElementById("speed-card"), ev.clientX - card.left, ev.clientY - card.top);
  }

  function moveCrosshair(t) {
    crosshair.setAttribute("x1", speedScale.sx(t));
    crosshair.setAttribute("x2", speedScale.sx(t));
    crosshair.setAttribute("display", "inline");
  }

  // Table --------------------------------------------------------------------
  function drawTable() {
    var table = document.getElementById("table");
    clear(table);
    var head = table.appendChild(document.createElement("thead")).appendChild(document.createElement("tr"));
    ["Movement", "Duration (s)", "Peak speed (px/s)", "Straightness",
     "Corrections drawn", "Corrections measured", "Overshoot drawn (px)", "Overshoot measured (px)",
     "Peak timing drawn", "Peak timing measured"].forEach(function (name) {
      var th = text(head, "th", name);
      th.setAttribute("scope", "col");
    });
    var body = table.appendChild(document.createElement("tbody"));
    visible().forEach(function (mv) {
      var row = body.appendChild(document.createElement("tr"));
      row.tabIndex = 0;
      row.dataset.id = mv.id;
      var first = text(row, "td", "");
      text(first, "i", "", "key " + mv.kind);
      first.appendChild(document.createTextNode(mv.id));
      var d = mv.drawn, m = mv.measured;
      [fmt(m.duration_s, 2), fmt(m.v_peak, 0), fmt(m.straightness, 3),
       fmt(d.n_submovements, 0), fmt(m.n_submovements, 0),
       fmt(d.overshoot_px, 1), fmt(m.overshoot_px, 1),
       fmt(d.t_peak_frac, 2), fmt(m.t_peak_frac, 2)].forEach(function (v) { text(row, "td", v); });
      row.addEventListener("pointerenter", function () { setHover(mv.id); });
      row.addEventListener("pointerleave", function () { setHover(null); });
      row.addEventListener("focus", function () { setHover(mv.id); });
      row.addEventListener("blur", function () { setHover(null); });
      row.addEventListener("click", function () { togglePin(mv.id); });
      row.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); togglePin(mv.id); }
      });
    });
    emphasize();
  }
  function togglePin(id) { state.pinned = state.pinned === id ? null : id; emphasize(); }
  function setHover(id) {
    if (state.hover === id) return;
    state.hover = id;
    emphasize();
  }

  // Emphasis: one highlighted path, the rest recede, and identity stays in the hue.
  function emphasize() {
    var chosen = active();
    all.forEach(function (mv) {
      var shown = mv.kind !== "real" || state.showReal;
      var opacity = chosen ? (mv.id === chosen ? 1 : 0.14) : (state.playing ? 0.14 : 0.42);
      [pathEls[mv.id], speedEls[mv.id]].forEach(function (el) {
        if (!el) return;
        el.setAttribute("stroke-opacity", opacity);
        el.setAttribute("display", shown ? "inline" : "none");
        if (mv.id === chosen) el.parentNode.appendChild(el);
      });
    });
    Array.prototype.forEach.call(document.querySelectorAll("#table tbody tr"), function (row) {
      row.classList.toggle("active", row.dataset.id === chosen);
    });
  }

  // Replay -------------------------------------------------------------------
  var playButton = document.getElementById("play"), clock = document.getElementById("clock");
  var rafId = null;

  function positionAt(mv, t) {
    var n = mv.t.length, hi = 1;
    if (t >= mv.t[n - 1]) return [mv.x[n - 1], mv.y[n - 1]];
    while (hi < n && mv.t[hi] < t) hi++;
    var lo = hi - 1, span = mv.t[hi] - mv.t[lo], f = span > 0 ? (t - mv.t[lo]) / span : 0;
    return [mv.x[lo] + f * (mv.x[hi] - mv.x[lo]), mv.y[lo] + f * (mv.y[hi] - mv.y[lo])];
  }
  function showFrame(t) {
    visible().forEach(function (mv) {
      var p = positionAt(mv, t), dot = dotEls[mv.id];
      dot.setAttribute("cx", pathScale.sx(p[0]));
      dot.setAttribute("cy", pathScale.sy(p[1]));
      dot.setAttribute("display", "inline");
    });
    all.forEach(function (mv) {
      if (mv.kind === "real" && !state.showReal) dotEls[mv.id].setAttribute("display", "none");
    });
    moveCrosshair(Math.min(t, speedScale.endT));
    clock.textContent = t.toFixed(2) + " s";
  }
  function stop() {
    if (rafId !== null) cancelAnimationFrame(rafId);
    rafId = null;
    state.playing = false;
    playButton.textContent = "Replay";
    all.forEach(function (mv) { if (dotEls[mv.id]) dotEls[mv.id].setAttribute("display", "none"); });
    if (crosshair) crosshair.setAttribute("display", "none");
    emphasize();
  }
  function play() {
    state.playing = true;
    playButton.textContent = "Stop";
    emphasize();
    var rate = Number(document.getElementById("rate").value), began = performance.now(), end = maxT();
    function frame(now) {
      var t = (now - began) / 1000 * rate;
      showFrame(Math.min(t, end));
      if (t >= end) {
        rafId = null;
        state.playing = false;
        playButton.textContent = "Replay";
        emphasize();
        return;
      }
      rafId = requestAnimationFrame(frame);
    }
    rafId = requestAnimationFrame(frame);
  }
  playButton.addEventListener("click", function () { if (state.playing) stop(); else play(); });
  document.getElementById("show-real").addEventListener("change", function (ev) {
    state.showReal = ev.target.checked;
    stop();
    drawTable();
  });

  function drawAll() {
    stop();
    drawPaths();
    drawSpeed();
    drawTable();
  }
  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawAll, 120);
  });
  drawAll();
})();
</script>
</body>
</html>
"""
