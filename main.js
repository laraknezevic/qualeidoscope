'use strict';
// Qualeidoscope
// Each wedge is one qubit in a ring. Every dab of ink runs this circuit once:
//   RY(a) on every qubit -> RZZ on every ring bond -> RX(b) on every qubit -> measure.
// Each wedge's line is as thick and bright as that qubit's chance of reading 1, and
// its color shifts with how correlated the qubit is with its neighbors. Every wedge
// gets the same gates, so a perfect machine draws perfectly symmetric copies. A noisy
// machine gets each qubit slightly wrong, and each copy of your line drifts away. Exact statevector
// simulation, up to 12 qubits, same maths as Qiskit's Statevector (see main.py).

const TAU = Math.PI * 2;
const MAXQ = 12;

// ---------- quantum part ----------
function makeSim(m) {
  const dim = 1 << m;
  const re = new Float64Array(dim), im = new Float64Array(dim), p = new Float64Array(dim);
  const zz = [];
  for (let j = 0; j < m; j++) {
    const k = (j + 1) % m, t = new Int8Array(dim);
    for (let b = 0; b < dim; b++) t[b] = (((b >> j) ^ (b >> k)) & 1) ? -1 : 1;
    zz.push(t);
  }
  function run(a, phi, bx) {
    re[0] = 1; im.fill(0);
    let size = 1;
    for (let j = 0; j < m; j++) {
      const c = Math.cos(a[j] / 2), s = Math.sin(a[j] / 2);
      for (let b = 0; b < size; b++) { re[b + size] = re[b] * s; re[b] *= c; }
      size <<= 1;
    }
    for (let b = 0; b < dim; b++) {
      let th = 0;
      for (let j = 0; j < m; j++) th -= phi[j] * zz[j][b];
      const c = Math.cos(th), s = Math.sin(th), r0 = re[b], i0 = im[b];
      re[b] = r0 * c - i0 * s; im[b] = r0 * s + i0 * c;
    }
    for (let j = 0; j < m; j++) {
      const c = Math.cos(bx[j] / 2), s = Math.sin(bx[j] / 2), st = 1 << j;
      for (let b = 0; b < dim; b++) {
        if (b & st) continue;
        const b1 = b | st, r0 = re[b], i0 = im[b], r1 = re[b1], i1 = im[b1];
        re[b] = c * r0 + s * i1; im[b] = c * i0 - s * r1;
        re[b1] = c * r1 + s * i0; im[b1] = c * i1 - s * r0;
      }
    }
    for (let b = 0; b < dim; b++) p[b] = re[b] * re[b] + im[b] * im[b];
  }
  function sample() {
    let u = Math.random(), acc = 0;
    for (let b = 0; b < dim; b++) { acc += p[b]; if (u <= acc) return b; }
    return dim - 1;
  }
  // exact numbers from the state: chance each qubit reads 1, and how correlated each
  // neighboring pair is (the connected correlation <ZZ> - <Z><Z>, zero for an unentangled pair)
  function finish(p1, zzv) {
    const conn = new Float64Array(m);
    for (let j = 0; j < m; j++) conn[j] = zzv[j] - (1 - 2 * p1[j]) * (1 - 2 * p1[(j + 1) % m]);
    return { p1, zz: zzv, conn };
  }
  function stats() {
    const p1 = new Float64Array(m), zzv = new Float64Array(m);
    for (let b = 0; b < dim; b++) {
      const pb = p[b]; if (pb < 1e-12) continue;
      for (let j = 0; j < m; j++) { if ((b >> j) & 1) p1[j] += pb; zzv[j] += pb * zz[j][b]; }
    }
    return finish(p1, zzv);
  }
  // the same numbers estimated from N simulated measurements (shots). Each measured bit can
  // be flipped with probability flip, like a readout error.
  const cdf = new Float64Array(dim);
  function shots(N, flip) {
    let acc = 0;
    for (let b = 0; b < dim; b++) { acc += p[b]; cdf[b] = acc; }
    const ones = new Float64Array(m), zzs = new Float64Array(m);
    for (let s = 0; s < N; s++) {
      const u = Math.random() * acc;
      let lo = 0, hi = dim - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
      let b = lo;
      if (flip > 0) for (let j = 0; j < m; j++) if (Math.random() < flip) b ^= 1 << j;
      for (let j = 0; j < m; j++) { ones[j] += (b >> j) & 1; zzs[j] += zz[j][b]; }
    }
    for (let j = 0; j < m; j++) { ones[j] /= N; zzs[j] /= N; }
    return finish(ones, zzs);
  }
  return { m, dim, run, sample, stats, shots, p };
}

// fixed "manufacturing defects" so the imperfection slider is repeatable
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rng = mulberry32(20261022);
const EPS = { a: [], p: [], b: [] };
for (let i = 0; i < MAXQ; i++) { EPS.a.push(rng() * 2 - 1); EPS.p.push(rng() * 2 - 1); EPS.b.push(rng() * 2 - 1); }

if (typeof module !== 'undefined') module.exports = { makeSim, EPS };

// ---------- picture, controls ----------
if (typeof document !== 'undefined') {
  const $ = (id) => document.getElementById(id);
  const canvas = $('art'), ctx = canvas.getContext('2d');
  const ring = $('ring'), rctx = ring.getContext('2d');
  const FOLDS = { rot: [6, 8, 12], mir: [3, 4, 6] };
  const ERR = window.ERROR_PROFILES || {};
  const IONS = Object.keys(ERR).filter((k) => ERR[k].kind === 'ion');
  const SCS = Object.keys(ERR).filter((k) => ERR[k].kind === 'sc');
  const MAXREC = 1400;
  const ampOf = (v) => Math.pow(60, v);                 // slider 0..1 -> x1..x60
  const tauOf = (v) => Math.pow(10, -3 + 6 * v);        // slider 0..1 -> 1 ms .. 1000 s
  const fmtTime = (s) => (s < 1e-3 ? (s * 1e6).toFixed(0) + ' µs' : s < 1 ? (s * 1e3).toFixed(s < 1e-2 ? 1 : 0) + ' ms' : s < 100 ? s.toFixed(s < 10 ? 1 : 0) + ' s' : s.toFixed(0) + ' s');
  const fmtN = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e4 ? Math.round(n / 1e3) + 'k' : String(Math.round(n)));
  const COLORS = ['hsl(185,90%,66%)', 'hsl(320,90%,70%)'];

  // the seven error families, each with its own look
  const LOOKS = [
    { id: 'g1', title: 'Qubits are not identical', look: 'each wedge a different color', col: 'hsl(300,90%,72%)' },
    { id: 'g2', title: 'Two-qubit gates and connectivity', look: 'the line bends', col: 'hsl(38,95%,62%)' },
    { id: 'g4', title: 'Shots', look: 'closed shapes fill in on some wedges', col: 'hsl(140,80%,62%)' },
  ];

  const S = { view: 'single', machine: 'perfect', ion: IONS.includes('ionq_forte1') ? 'ionq_forte1' : IONS[0],
              sc: SCS.includes('ibm_fez') ? 'ibm_fez' : SCS[0], mirror: false, folds: 8, ent: 0.7, depth: 1, motion: 0.5, life: 9,
              ampV: 0, tauV: 0.5, overheadUs: 0, looks: LOOKS.map(() => true), auto: true, drawing: false, guides: false };
  let lanes = [], dpr = 1, W = 0, H = 0, cy = 0, unit = 1;
  let last = null, queue = [], speed = 0, t0 = performance.now(), frame = 0;
  if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) { S.auto = false; S.motion = 0; }

  function qubits() { return S.mirror ? 2 * S.folds : S.folds; }
  function nameOf(key) { return ERR[key] ? ERR[key].label : ''; }
  function sizeData(key) { return key && ERR[key] ? ERR[key].sizes[String(qubits())] : null; }
  function gauss() { let u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * Math.random()); }

  // how far each wedge is from the others that should behave identically, in units of a typical value
  function relDev(vals, ref, mirror) {
    const m = vals.length, out = new Float64Array(m), nc = mirror ? 2 : 1;
    if (!(ref > 0)) return out;
    for (let c = 0; c < nc; c++) {
      let sum = 0, cnt = 0;
      for (let j = c; j < m; j += nc) { sum += vals[j]; cnt++; }
      for (let j = c; j < m; j += nc) out[j] = 2.5 * Math.tanh(((vals[j] - sum / cnt) / ref) / 2.5);
    }
    return out;
  }

  // ----- lanes: one kaleidoscope each (one in single view, two side by side in compare view) -----
  // which way each wedge's copy bends. No machine publishes this, so it is made up (the same for every machine).
  function bendDir(m) {
    const d = new Float64Array(m);
    for (let j = 0; j < m; j++) d[j] = (EPS.p[j] < 0 ? -1 : 1) * (0.5 + 0.5 * Math.abs(EPS.p[j]));
    return d;
  }
  function chaosData(m) {
    const z = () => new Float64Array(m);
    const d = { d1: z(), d2: z(), d3: z(), d5: z(), e2: new Float64Array(m).fill(0.04), dir: bendDir(m), leak: new Float64Array(m).fill(0.003), loss: new Float64Array(m).fill(0.0003), g7: new Float64Array(m).fill(1) };
    for (let j = 0; j < m; j++) { d.d1[j] = 2 * EPS.a[j]; d.d2[j] = 2 * EPS.p[j]; d.d3[j] = 2 * EPS.b[j]; d.d5[j] = 2 * EPS.a[(j + 3) % 12]; }
    return d;
  }
  function laneData(kind, key, m) {
    if (kind === 'chaos') return chaosData(m);
    if (kind !== 'profile') return null;
    const P = sizeData(key); if (!P) return null;
    return { d1: relDev(P.g1, P.ref.g1, S.mirror), e2: Float64Array.from(P.g2), dir: bendDir(m), d3: relDev(P.g3, P.ref.g3, S.mirror), d5: relDev(P.g5, P.ref.g5, S.mirror),
             leak: Float64Array.from(P.leak), loss: Float64Array.from(P.loss), g7: Float64Array.from(P.g7) };
  }
  function newLane(kind, key) {
    const m = qubits();
    return { kind, key, sim: makeSim(m), data: laneData(kind, key, m), recs: [], lastDab: null, hue: 200, shots: 0, cx: 0, R: 0,
             dropUntil: new Float64Array(m), dropKind: new Uint8Array(m), rc: new Float64Array(m), seq: 0, sp: [], cut: 0, shapes: [], cur: null };
  }
  function buildLanes() {
    lanes = [];
    if (S.view === 'split') { lanes.push(newLane('profile', S.ion)); lanes.push(newLane('profile', S.sc)); }
    else if (S.machine === 'ionq') lanes.push(newLane('profile', S.ion));
    else if (S.machine === 'sc') lanes.push(newLane('profile', S.sc));
    else lanes.push(newLane(S.machine, null));
    layout();
    refreshUI();
  }
  function layout() {
    cy = H / 2;
    const n = lanes.length;
    lanes.forEach((L, i) => { L.cx = W * (i + 0.5) / n; L.R = Math.min(W / n, H) / 2 * 0.92; L.recs.length = 0; L.lastDab = null; L.sp.length = 0; L.cut = 0; L.shapes = []; });
    last = null; queue.length = 0;
  }
  function tshotOf(L) { const P = sizeData(L.key); return P && P.run_seconds ? P.run_seconds + S.overheadUs * 1e-6 : null; }
  function shotsOf(L) {
    if (L.kind === 'chaos') return 10;
    if (L.kind !== 'profile') return Infinity;
    const t = tshotOf(L); return t ? Math.max(1, Math.floor(tauOf(S.tauV) / t)) : 0;
  }

  function rebuild() {
    $('foldval').textContent = S.folds + ' folds, ' + qubits() + ' qubits';
    const list = S.mirror ? FOLDS.mir : FOLDS.rot;
    $('folds').max = list.length - 1;
    $('folds').value = list.indexOf(S.folds);
    document.querySelectorAll('.seg button[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String((b.dataset.mode === 'mir') === S.mirror)));
    buildLanes();
  }

  // ----- the panel: machines, the budget, and the seven looks with where each number comes from -----
  function buildLooks() {
    const box = $('looks'); box.textContent = '';
    LOOKS.forEach((lk, i) => {
      const row = document.createElement('div'); row.className = 'look';
      const lab = document.createElement('label'); lab.className = 'lk';
      const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = S.looks[i];
      cb.addEventListener('change', () => { S.looks[i] = cb.checked; });
      const sw = document.createElement('i'); sw.style.background = lk.col;
      const tx = document.createElement('span'); tx.innerHTML = '<b>' + (i + 1) + '. ' + lk.title + '</b> <em>' + lk.look + '</em>';
      lab.append(cb, sw, tx); row.appendChild(lab);
      lanes.forEach((L, li) => {
        const p = document.createElement('p'); p.className = 'src';
        let text, tag;
        if (L.kind === 'perfect') { text = 'none: a perfect machine'; tag = 'perfect'; }
        else if (L.kind === 'chaos') { text = 'made up and very large'; tag = 'not a real machine'; }
        else {
          const P = sizeData(L.key);
          text = P ? P.text[lk.id] : 'no data for this ring size'; tag = P ? P.tag[lk.id] : '';
          if (lk.id === 'g1' && P) { text = 'one-qubit gate error, idle error and readout error, compared across qubits. ' + ['g1', 'g3', 'g5'].map((g) => P.text[g]).join('; '); tag = [...new Set(['g1', 'g3', 'g5'].map((g) => P.tag[g]))].join(', '); }
          if (lk.id === 'g4' && P) text = fmtN(shotsOf(L)) + ' shots per dab (in 1 second)';
        }
        const who = lanes.length > 1 ? '<u style="color:' + COLORS[li] + '">' + (L.kind === 'profile' ? nameOf(L.key) : L.kind) + '</u>: ' : '';
        p.innerHTML = who + text + (tag ? ' <span class="tag t-' + tag.replace(/[^a-z]/gi, '').toLowerCase().slice(0, 9) + '">' + tag + '</span>' : '');
        row.appendChild(p);
      });
      box.appendChild(row);
    });
  }

  function refreshUI() {
    const split = S.view === 'split';
    document.querySelectorAll('[data-view]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === S.view)));
    document.querySelectorAll('.mach button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.m === S.machine)));
    $('machrow').hidden = split;
    const hasProfile = lanes.some((L) => L.kind === 'profile');
    $('ionrow').hidden = !(split || S.machine === 'ionq');
    $('screw').hidden = !(split || S.machine === 'sc');
    $('budget').hidden = !hasProfile;
    $('mdesc').textContent = S.machine === 'perfect' && !split ? 'No noise. Every qubit behaves identically, so every copy of your line is identical.'
      : S.machine === 'chaos' && !split ? 'Every error family is made up and turned up to the maximum, to show the extreme.'
      : 'Each of the three broken symmetries below is driven by that machine\'s own numbers. Tick one on or off to see what it does by itself.';
    buildLooks();
    updateBudget();
  }

  function updateBudget() {
    const parts = lanes.filter((L) => L.kind === 'profile').map((L) => nameOf(L.key) + ': ' + fmtN(shotsOf(L)));
    $('shotsline').textContent = 'Shots each machine fits in 1 second: ' + parts.join('  ·  ');
    buildLooks();
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.round(window.innerWidth * dpr); H = Math.round(window.innerHeight * dpr);
    canvas.width = W; canvas.height = H;
    unit = Math.max(1, Math.min(W, H) / 900);
    layout();
    placeLabels();
  }

  function wedgeAngles(th, n, m, mirror) {
    const sector = TAU / n, angs = new Array(m);
    let t = 0;
    if (!mirror) {
      for (let k = 0; k < n; k++) angs[k] = th + sector * k;
    } else {
      const u = ((th % sector) + sector) % sector;
      t = u > sector / 2 ? sector - u : u;
      for (let k = 0; k < n; k++) { angs[2 * k] = sector * k - t; angs[2 * k + 1] = sector * k + t; }
    }
    return { angs, t, sector };
  }

  const SPR = [];
  const SAT = [100, 100, 100, 55, 12], LIT = [30, 45, 60, 74, 90];   // deep, rich, normal, softer, pale
  for (let sb = 0; sb < 5; sb++) for (let i = 0; i < 36; i++) {
    const c = document.createElement('canvas'); c.width = c.height = 96;
    const g = c.getContext('2d'), gr = g.createRadialGradient(48, 48, 0, 48, 48, 48), h = i * 10, sa = SAT[sb];
    gr.addColorStop(0, 'hsla(' + h + ',' + sa + '%,' + Math.min(92, LIT[sb] - 4) + '%,0.65)');
    gr.addColorStop(0.45, 'hsla(' + h + ',' + sa + '%,' + Math.min(92, LIT[sb] - 12) + '%,0.24)');
    gr.addColorStop(1, 'hsla(' + h + ',' + sa + '%,45%,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 96, 96);
    SPR.push(c);
  }
  // ----- one dab on one lane: the perfect circuit sets the pattern; each error family adds its own look -----
  const A0 = [], B0 = [], P0 = [];
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  // a shape is closed when the drawn line crosses itself or comes back to a point it already passed
  const segHit = (a, b, c, d) => {
    const o = (p, q, r) => (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
    return o(a, b, c) * o(a, b, d) < 0 && o(c, d, a) * o(c, d, b) < 0;
  };
  function detectLoop(L, rec, ox, oy, tsec, broke) {
    if (broke) { L.sp.length = 0; L.cut = 0; }
    const sp = L.sp, p = { x: ox, y: oy, rec }, last = sp.length - 1, minSize = 0.12 * L.R;
    let close = -1;
    if (last >= 1) {
      for (let i = L.cut; i <= last - 2; i++) if (segHit(sp[i], sp[i + 1], sp[last], p)) { close = i + 1; break; }
      if (close < 0) for (let i = L.cut; i <= last - 4; i++) {                       // or it comes back onto an earlier part of the line
        const a = sp[i], b = sp[i + 1], dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy || 1;
        const u = Math.max(0, Math.min(1, ((ox - a.x) * dx + (oy - a.y) * dy) / l2));
        if (Math.hypot(a.x + u * dx - ox, a.y + u * dy - oy) < 0.07 * L.R) { close = i + 1; break; }
      }
    }
    sp.push(p);
    if (close >= 0) {
      const loop = sp.slice(close);
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      for (const q of loop) { x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y); }
      if (loop.length >= 6 && Math.min(x1 - x0, y1 - y0) > minSize * 0.5 && Math.max(x1 - x0, y1 - y0) > minSize) {
        L.shapes.push({ recs: loop.map((q) => q.rec), t: tsec, fill: Array.from(rec.pe, (v) => Math.random() < Math.min(0.95, 0.5 * v)) });   // each wedge flips its own coin for this shape; fewer shots, better odds of a fill
        L.cut = sp.length - 1;
      }
    }
    if (sp.length > 600) { sp.splice(0, 200); L.cut = Math.max(0, L.cut - 200); }
  }
  function dab(L, ox, oy, spd, tsec) {
    const sim = L.sim, m = sim.m, n = S.folds, x = L.cx + ox, y = cy + oy;
    const dx = ox, dy = oy, rg = Math.hypot(dx, dy) / L.R, r = Math.min(1, rg), th = Math.atan2(dy, dx);
    const cur = wedgeAngles(th, n, m, S.mirror), sector = cur.sector, t = cur.t;
    const Aang = Math.PI * (0.38 + 0.45 * r);
    const Bang = Math.PI * 0.5 * (0.35 + 0.65 * (0.5 + 0.5 * Math.sin(tsec * 0.7)));
    const base = S.ent * (0.4 + 2.2 * spd);
    for (let j = 0; j < m; j++) {
      const f = S.mirror ? (j % 2 === 0 ? 1 - t / sector : 0.5 + t / sector) : 1;
      A0[j] = Aang; B0[j] = Bang; P0[j] = base * f;
    }
    sim.run(A0, P0, B0);
    const ref = sim.stats();                         // the perfect, symmetric pattern
    L.hue = (tsec * 22 + r * 160) % 360;

    // the seven looks, one value per wedge
    const E = ampOf(S.ampV), on = S.looks, D = L.data, shots = shotsOf(L); L.shots = shots;
    const hs = new Float32Array(m), gl = new Float32Array(m).fill(1), tk = new Float32Array(m).fill(1), br = new Float32Array(m).fill(1), pl = new Float32Array(m), bend = new Float64Array(m);
    const sg = Number.isFinite(shots) ? Math.sqrt(0.25 / Math.max(1, shots)) : 0;
    const pe = new Float32Array(m), pp = new Uint8Array(m).fill(1);
    if (on[2] && sg > 0) {                                      // 3: shots -> every wedge keeps its own fixed amount of fill; each wedge randomly fills or skips each closed shape; fewer shots, better odds of a fill
      const nc = S.mirror ? 2 : 1, u = new Float64Array(m);
      for (let j = 0; j < m; j++) u[j] = ((7 * j + 3) % m) / (m - 1) * 2 - 1;                    // a fixed, made-up order of the wedges, evenly spread
      for (let c = 0; c < nc; c++) { let sum = 0, cnt = 0, mx = 0; for (let j = c; j < m; j += nc) { sum += u[j]; cnt++; } for (let j = c; j < m; j += nc) { u[j] -= sum / cnt; mx = Math.max(mx, Math.abs(u[j])); } for (let j = c; j < m; j += nc) u[j] /= mx || 1; }
      for (let j = 0; j < m; j++) { pe[j] = Math.max(0, sg * E * 25 * (1 + 0.5 * u[j])); pp[j] = 1 + Math.round(3 * Math.min(1, pe[j])); }
    }
    for (let j = 0; j < m; j++) {
      if (D) {
        if (on[0]) hs[j] = clamp(240 * E * (D.d1[j] + D.d3[j] + D.d5[j]) / 3, -180, 180);          // 1: qubits not identical (gate, idle and readout error compared across qubits) -> color
        if (on[1]) bend[j] = E * D.dir[j] * D.e2[j];                                              // 2: two-qubit gate error incl. swaps -> bend (size follows the error)
      }
    }
    L.cur = { hs, gl, tk, br, pl, bend, pe };

    const Fx = new Float64Array(m), Fy = new Float64Array(m);
    let jo = 0, best = 9;
    for (let j = 0; j < m; j++) {
      Fx[j] = rg * L.R * Math.cos(cur.angs[j]); Fy[j] = rg * L.R * Math.sin(cur.angs[j]);
      const d = Math.abs(Math.atan2(Math.sin(cur.angs[j] - th), Math.cos(cur.angs[j] - th)));
      if (d < best) { best = d; jo = j; }
    }
    const jump = L.lastDab && Math.hypot(x - L.lastDab.x, y - L.lastDab.y) > 36 * dpr;
    let T = L.lastDab, fresh = false;
    if (!T) {
      const z = () => new Float64Array(m);
      T = { Ax: Fx.slice(), Ay: Fy.slice(), Fx: Fx.slice(), Fy: Fy.slice(), ex: z(), ey: z(), H: z(), sd: z(), sp1: Float64Array.from(ref.p1), sco: Float64Array.from(ref.conn), x, y };
      fresh = true;
    }
    const noChord = fresh || jump;
    const pax = Float32Array.from(T.Ax), pay = Float32Array.from(T.Ay);
    for (let j = 0; j < m; j++) {
      const sx = noChord ? 0 : Fx[j] - T.Fx[j], sy = noChord ? 0 : Fy[j] - T.Fy[j];
      T.sp1[j] += 0.25 * (ref.p1[j] - T.sp1[j]);
      T.sco[j] += 0.25 * (ref.conn[j] - T.sco[j]);
      if (j === jo) { T.H[j] = 0; T.ex[j] = 0; T.ey[j] = 0; }
      else {
        T.H[j] = T.H[j] * 0.99 + bend[j];
        const c = Math.cos(T.H[j]), sn = Math.sin(T.H[j]);
        T.ex[j] = T.ex[j] * 0.97 + (c * sx - sn * sy) - sx;
        T.ey[j] = T.ey[j] * 0.97 + (sn * sx + c * sy) - sy;
      }
      T.Ax[j] = Fx[j] + T.ex[j]; T.Ay[j] = Fy[j] + T.ey[j];
      if (noChord) { pax[j] = T.Ax[j]; pay[j] = T.Ay[j]; }
      T.Fx[j] = Fx[j]; T.Fy[j] = Fy[j];
    }
    T.x = x; T.y = y; L.lastDab = T;
    const rec = { q: L.seq++, pe, pp, t0: tsec, hue: L.hue, m, n, sector, sp1: Float32Array.from(T.sp1), sco: Float32Array.from(T.sco), hs, gl, tk, br, pl, pax, pay, ax: Float32Array.from(T.Ax), ay: Float32Array.from(T.Ay) };
    L.recs.push(rec); detectLoop(L, rec, ox, oy, tsec, fresh || jump);
    if (L.recs.length > MAXREC) L.recs.splice(0, L.recs.length - MAXREC);
  }

  // ----- drawing: crisp solid lines that fade, melting into a glow that lingers and drifts -----
  function drawGuides(L) {
    const n = S.folds, count = S.mirror ? 2 * n : n;
    ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = dpr;
    ctx.beginPath();
    for (let k = 0; k < count; k++) {
      const a = (S.mirror ? Math.PI / n : TAU / n) * k;
      ctx.moveTo(L.cx, cy); ctx.lineTo(L.cx + L.R * Math.cos(a), cy + L.R * Math.sin(a));
    }
    ctx.stroke();
  }

  const hash = (x) => { const v = Math.sin(x * 12.9898) * 43758.5453; return v - Math.floor(v); };
  function renderLane(L, tsec) {
    if (S.guides) drawGuides(L);
    const life = S.life, gr = S.motion * 0.12 * tsec;
    const lines = new Map(), stamps = [], recs = L.recs, cx = L.cx;
    let w = 0;
    for (let i = 0; i < recs.length; i++) {
      const c = recs[i], age = tsec - c.t0;
      if (age > life * 1.8) continue;
      recs[w++] = c;
      const core = age < life ? Math.pow(1 - age / life, 1.3) : 0;
      const halo = Math.pow(Math.max(0, 1 - age / (life * 1.8)), 1.5);
      const drift = 1 + S.motion * 0.05 * age;
      for (let k = 0; k < S.depth; k++) {
        const sc = Math.pow(0.56, k) * drift, rot = gr + k * c.sector / 2, cr = Math.cos(rot) * sc, sr = Math.sin(rot) * sc;
        const fk = 1 / (1 + 0.3 * k), wk = Math.pow(0.78, k);
        for (let j = 0; j < c.m; j++) {
          if (c.tk[j] <= 0.01 || c.br[j] <= 0.01) continue;
          const pj = c.sp1[j], wd = unit * wk * (1.4 + 9 * pj * pj) * c.tk[j];
          const hue = (c.hue + 230 * c.sco[j] + k * 38 + c.hs[j] + 3600) % 360, hb = Math.floor(hue / 10) % 36;
          const x2 = cx + cr * c.ax[j] - sr * c.ay[j], y2 = cy + sr * c.ax[j] + cr * c.ay[j];
          const ai = Math.max(0, Math.min(5, Math.floor(Math.min(1, core * c.br[j]) * 6))), sb = c.pl[j] <= -0.6 ? 0 : c.pl[j] <= -0.2 ? 1 : c.pl[j] < 0.2 ? 2 : c.pl[j] < 0.6 ? 3 : 4;
          if (core * c.br[j] > 0.02) {
            const x1 = cx + cr * c.pax[j] - sr * c.pay[j], y1 = cy + sr * c.pax[j] + cr * c.pay[j];
            const wb = wd < 2.6 * unit ? 0 : wd < 5 * unit ? 1 : 2, key = ((hb * 6 + ai) * 3 + wb) * 5 + sb;
            let pth = lines.get(key);
            if (!pth) { pth = new Path2D(); lines.set(key, pth); }
            pth.moveTo(x1, y1); pth.lineTo(x2, y2);
          }
          stamps.push(x2, y2, hb + 36 * sb, wd * 9 * (1 + 0.9 * age / life) * c.gl[j], Math.min(1, 0.2 * fk * (0.4 + 1.2 * pj) * halo * Math.min(1.6, c.gl[j]) * c.br[j]));
        }
      }
    }
    recs.length = w;
    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < stamps.length; i += 5) {
      const d = stamps[i + 3];
      ctx.globalAlpha = stamps[i + 4];
      ctx.drawImage(SPR[stamps[i + 2]], stamps[i] - d / 2, stamps[i + 1] - d / 2, d, d);
    }
    ctx.globalCompositeOperation = 'source-over';
    L.shapes = L.shapes.filter((sh) => tsec - sh.recs[sh.recs.length - 1].t0 < life);
    for (const sh of L.shapes) {                                    // closed shapes: each wedge randomly fills its copy or leaves it empty; fewer shots, better odds of a fill
      const rs = sh.recs, last = rs[rs.length - 1];
      let mc = 0; for (const c of rs) { const ag = tsec - c.t0; mc += ag < life ? Math.pow(1 - ag / life, 1.3) : 0; } mc /= rs.length;
      const ramp = Math.min(1, (tsec - sh.t) / 0.35);
      for (let k = 0; k < S.depth; k++) for (let j = 0; j < last.m; j++) {
        const pe = last.pe[j]; if (!sh.fill[j] || last.br[j] <= 0.01) continue;
        ctx.beginPath();
        for (let i = 0; i < rs.length; i++) {
          const c = rs[i], ag = tsec - c.t0, sc = Math.pow(0.56, k) * (1 + S.motion * 0.05 * ag), rot = gr + k * c.sector / 2, cr = Math.cos(rot) * sc, sr = Math.sin(rot) * sc;
          const x = cx + cr * c.ax[j] - sr * c.ay[j], y = cy + sr * c.ax[j] + cr * c.ay[j];
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = 'hsl(' + Math.round((last.hue + 230 * last.sco[j] + k * 38 + last.hs[j] + 3600) % 360) + ',90%,60%)';
        ctx.globalAlpha = 0.65 * Math.min(1, pe) * mc * ramp / (1 + 0.5 * k);
        ctx.fill();
      }
    }
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const WIDTH = [1.7, 3.6, 6.2];
    for (const [key, pth] of lines) {
      const sb = key % 5, wb = Math.floor(key / 5) % 3, ai = Math.floor(key / 15) % 6, hb = Math.floor(key / 90);
      ctx.strokeStyle = 'hsl(' + (hb * 10) + ',' + SAT[sb] + '%,' + LIT[sb] + '%)';
      ctx.globalAlpha = 0.25 + 0.75 * (ai + 1) / 6;
      ctx.lineWidth = unit * WIDTH[wb];
      ctx.stroke(pth);
    }
    ctx.globalAlpha = 1;
  }

  function render(tsec) {
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    lanes.forEach((L, i) => {
      ctx.save();
      if (lanes.length > 1) { ctx.beginPath(); ctx.rect(i * W / lanes.length, 0, W / lanes.length, H); ctx.clip(); }
      renderLane(L, tsec);
      ctx.restore();
    });
    if (lanes.length > 1) { ctx.fillStyle = 'rgba(255,255,255,0.10)'; ctx.fillRect(W / 2 - dpr / 2, 0, dpr, H); }
  }

  // ----- labels over each kaleidoscope -----
  function placeLabels() {
    [0, 1].forEach((i) => {
      const el = $('lab' + i), L = lanes[i];
      el.hidden = !L;
      if (L) { el.style.left = (L.cx / dpr) + 'px'; }
    });
  }
  function labelText(L) {
    if (L.kind === 'perfect') return ['Perfect machine', 'no errors'];
    if (L.kind === 'chaos') return ['Max chaos (not a real machine)', 'every error family at the maximum'];
    return [ERR[L.key].kind === 'ion' ? nameOf(L.key) + ' (noise model)' : nameOf(L.key) + ' (simulated)', fmtN(L.shots || shotsOf(L)) + ' shots per dab'];
  }
  function updateLabels() {
    lanes.forEach((L, i) => {
      const [a, b] = labelText(L), el = $('lab' + i);
      el.children[0].textContent = a; el.children[1].textContent = b;
      el.style.setProperty('--c', lanes.length > 1 ? COLORS[i] : '#ece9ff');
    });
  }

  function pushPoint(ox, oy) {
    if (!last) { last = { ox, oy }; queue.push({ ox, oy, s: 0.2 }); return; }
    const dx = ox - last.ox, dy = oy - last.oy, d = Math.hypot(dx, dy), step = 7 * dpr;
    speed += (Math.min(1, d / (dpr * 30)) - speed) * 0.3;
    const k = Math.floor(d / step);
    for (let i = 1; i <= k; i++) queue.push({ ox: last.ox + dx * (i * step / d), oy: last.oy + dy * (i * step / d), s: speed });
    if (k > 0) last = { ox: last.ox + dx * (k * step / d), oy: last.oy + dy * (k * step / d) };
  }

  // the machine's qubits, one dot each, showing what every look is doing to that wedge right now
  function drawRing() {
    const nl = lanes.length, sz = 150 * dpr;
    ring.width = sz * nl; ring.height = sz; ring.style.width = (150 * nl) + 'px';
    rctx.clearRect(0, 0, sz * nl, sz);
    lanes.forEach((L, li) => {
      const C = L.cur; if (!C) return;
      const m = C.hs.length, c = sz * li + sz / 2, rr = sz * 0.34, pts = [];
      for (let j = 0; j < m; j++) { const a = -Math.PI / 2 + TAU * j / m; pts.push([c + rr * Math.cos(a), sz / 2 + rr * Math.sin(a)]); }
      for (let j = 0; j < m; j++) {
        const k = (j + 1) % m, bw = Math.min(1, Math.abs(C.bend[j]) / 0.03);
        rctx.strokeStyle = 'rgba(255,190,90,' + (0.2 + 0.7 * bw) + ')'; rctx.lineWidth = dpr * (0.8 + 3 * bw);
        rctx.beginPath(); rctx.moveTo(pts[j][0], pts[j][1]); rctx.lineTo(pts[k][0], pts[k][1]); rctx.stroke();
      }
      for (let j = 0; j < m; j++) {
        const hue = (L.hue + C.hs[j] + 3600) % 360, off = C.br[j] < 0.05;
        const rad = dpr * 5.5 * (0.5 + 0.5 * Math.min(2, 1 + C.pe[j])), gl = dpr * (9 + 8 * (C.gl[j] - 1));
        rctx.strokeStyle = 'hsla(' + hue + ',90%,65%,0.35)'; rctx.lineWidth = dpr * 2; rctx.beginPath(); rctx.arc(pts[j][0], pts[j][1], Math.max(rad, gl), 0, TAU); rctx.stroke();
        rctx.fillStyle = off ? 'rgba(255,255,255,0.12)' : 'hsla(' + hue + ',' + Math.round(90 - 50 * Math.max(0, C.pl[j])) + '%,' + Math.round(Math.max(28, Math.min(82, (35 + 25 * Math.min(1.2, C.br[j])) + 24 * C.pl[j]))) + '%,' + clamp(C.br[j], 0.15, 1) + ')';
        rctx.beginPath(); rctx.arc(pts[j][0], pts[j][1], rad, 0, TAU); rctx.fill();
      }
    });
  }

  function loop(now) {
    const tsec = (now - t0) / 1000;
    frame++;
    const R0 = lanes[0] ? lanes[0].R : 1;
    if (S.auto) {
      const u = tsec * 0.55;
      pushPoint(0.9 * R0 * Math.sin(u + 0.8 * Math.sin(0.23 * u)), 0.9 * R0 * Math.sin(1.618 * u + 0.5) * (0.6 + 0.4 * Math.cos(0.41 * u)));
    }
    chase();
    if (queue.length > 40) queue.splice(0, queue.length - 40);
    let n = 0;
    while (queue.length && n < 14) { const q = queue.shift(); lanes.forEach((L) => dab(L, q.ox, q.oy, q.s, tsec)); n++; }
    if (!S.drawing && !S.auto) speed *= 0.9;
    render(tsec);
    if (frame % 3 === 0) drawRing();
    if (frame % 6 === 0) updateLabels();
    requestAnimationFrame(loop);
  }

  // ----- input -----
  function toOffset(e) {
    const x = e.clientX * (canvas.width / window.innerWidth), y = e.clientY * (canvas.height / window.innerHeight);
    const L = lanes.length > 1 && x > W / 2 ? lanes[1] : lanes[0];
    return [x - L.cx, y - cy];
  }
  // the pen chases the mouse a little behind it, so corners and jitter in the hand movement are rounded off
  const pen = { x: 0, y: 0, tx: 0, ty: 0, on: false };
  function chase() {
    if (!pen.on) return;
    for (let i = 0; i < 4; i++) { pen.x += 0.1 * (pen.tx - pen.x); pen.y += 0.1 * (pen.ty - pen.y); pushPoint(pen.x, pen.y); }
    if (!S.drawing && Math.hypot(pen.tx - pen.x, pen.ty - pen.y) < 2 * dpr) { pen.on = false; last = null; lanes.forEach((L) => { L.lastDab = null; }); }
  }
  canvas.addEventListener('pointerdown', (e) => {
    S.drawing = true; if (S.auto) { S.auto = false; $('auto').setAttribute('aria-pressed', 'false'); }
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* not needed */ }
    last = null; lanes.forEach((L) => { L.lastDab = null; }); const [ox, oy] = toOffset(e);
    pen.x = pen.tx = ox; pen.y = pen.ty = oy; pen.on = true; pushPoint(ox, oy);
  });
  canvas.addEventListener('pointermove', (e) => { if (S.drawing) { const [ox, oy] = toOffset(e); pen.tx = ox; pen.ty = oy; } });
  const stop = () => { S.drawing = false; };
  canvas.addEventListener('pointerup', stop); canvas.addEventListener('pointercancel', stop);

  document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => { S.view = b.dataset.view; buildLanes(); }));
  document.querySelectorAll('.mach button').forEach((b) => {
    if ((b.dataset.m === 'ionq' && !IONS.length) || (b.dataset.m === 'sc' && !SCS.length)) { b.hidden = true; return; }
    b.addEventListener('click', () => { S.machine = b.dataset.m; buildLanes(); });
  });
  IONS.forEach((k) => { const o = document.createElement('option'); o.value = k; o.textContent = ERR[k].label + ' noise model'; $('ion').appendChild(o); });
  SCS.forEach((k) => { const o = document.createElement('option'); o.value = k; o.textContent = ERR[k].label + ' (simulated)'; $('sc').appendChild(o); });
  $('ion').value = S.ion; $('sc').value = S.sc;
  $('ion').addEventListener('change', (e) => { S.ion = e.target.value; buildLanes(); });
  $('sc').addEventListener('change', (e) => { S.sc = e.target.value; buildLanes(); });
  $('amp').addEventListener('input', (e) => { S.ampV = +e.target.value; $('ampval').textContent = '×' + ampOf(S.ampV).toFixed(ampOf(S.ampV) < 10 ? 1 : 0); });
  document.querySelectorAll('.seg button[data-mode]').forEach((b) => b.addEventListener('click', () => {
    S.mirror = b.dataset.mode === 'mir';
    const list = S.mirror ? FOLDS.mir : FOLDS.rot;
    S.folds = list.reduce((best, v) => (Math.abs(v - S.folds) < Math.abs(best - S.folds) ? v : best), list[0]);
    rebuild();
  }));
  $('folds').addEventListener('input', (e) => { S.folds = (S.mirror ? FOLDS.mir : FOLDS.rot)[+e.target.value]; rebuild(); });
  $('depth').addEventListener('input', (e) => { S.depth = +e.target.value; $('depthval').textContent = S.depth; });
  $('motion').addEventListener('input', (e) => { S.motion = +e.target.value; $('motionval').textContent = S.motion.toFixed(2); });
  $('life').addEventListener('input', (e) => { S.life = +e.target.value; $('lifeval').textContent = S.life + ' s'; });
  $('auto').addEventListener('click', () => { S.auto = !S.auto; pen.on = false; last = null; lanes.forEach((L) => { L.lastDab = null; }); $('auto').setAttribute('aria-pressed', String(S.auto)); });
  $('guides').addEventListener('click', () => { S.guides = !S.guides; $('guides').setAttribute('aria-pressed', String(S.guides)); });
  const clearAll = () => lanes.forEach((L) => { L.recs.length = 0; L.lastDab = null; L.sp.length = 0; L.cut = 0; L.shapes = []; L.dropUntil.fill(0); L.rc.fill(0); });
  $('clear').addEventListener('click', clearAll);
  $('save').addEventListener('click', () => {
    canvas.toBlob((blob) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'quantum-kaleidoscope.png'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); });
  });
  function setPanel(open) { $('panel').hidden = !open; $('showpanel').hidden = open; }
  $('hide').addEventListener('click', () => setPanel(false));
  $('showpanel').addEventListener('click', () => setPanel(true));
  window.addEventListener('keydown', (e) => { if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return; if (e.key === 'h') setPanel($('panel').hidden); if (e.key === 'c') clearAll(); });
  let rt = null;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { resize(); }, 150); });

  $('amp').value = S.ampV; $('ampval').textContent = '×' + ampOf(S.ampV).toFixed(1);
  $('depth').value = S.depth; $('depthval').textContent = S.depth;
  $('motion').value = S.motion; $('motionval').textContent = S.motion.toFixed(2);
  $('life').value = S.life; $('lifeval').textContent = S.life + ' s';
  $('auto').setAttribute('aria-pressed', String(S.auto));
  $('guides').setAttribute('aria-pressed', 'false');
  setPanel(window.innerWidth >= 700);
  resize(); rebuild(); placeLabels();
  window.__kal = { S, get lanes() { return lanes; }, EPS, makeSim, ERR, tauOf, shotsOf, LOOKS };
  requestAnimationFrame(loop);
}
