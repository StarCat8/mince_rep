/* ============================================================================
 * MINCE spectra — shared module (catalog page + visualizer popup)
 * ----------------------------------------------------------------------------
 * Data layout produced by tools/build_spectra_db.py:
 *   spectra/manifest.json
 *   spectra/data/<id>.ov.f32    overview: [x, ymin, ymax] float32 × ~2048 bins  (~24 KB)
 *   spectra/data/<id>.f32       flux float32 × n   [+ wave float64 × n if grid="array"]
 *   spectra/data/<id>.hdr.txt   FITS header text
 *
 * Speed strategy
 *   1. the overview (~24 KB) is drawn immediately;
 *   2. full resolution is fetched in the background as a raw typed array (no parsing);
 *   3. whatever the zoom, only ~2 points per screen pixel are sent to Plotly
 *      (min/max decimation of the visible window) → each redraw is a few ms.
 * Requires Plotly (basic bundle is enough).
 * ========================================================================== */
(function (global) {
  'use strict';

  const C_KMS = 299792.458;
  const COLORS = ['#4f9cf9', '#fbbf24', '#34d399', '#f472b6', '#a78bfa', '#f87171', '#22d3ee', '#a3e635'];

  // Diagnostic lines (air wavelengths, Å)
  const LINES = [
    [3933.66, 'Ca II K'], [3968.47, 'Ca II H'], [4061.08, 'Nd II'], [4077.71, 'Sr II'],
    [4086.71, 'La II'], [4101.74, 'Hδ'], [4129.72, 'Eu II'], [4215.52, 'Sr II'],
    [4340.47, 'Hγ'], [4554.03, 'Ba II'], [4562.36, 'Ce II'], [4861.33, 'Hβ'],
    [4883.68, 'Y II'], [5167.32, 'Mg b'], [5172.68, 'Mg b'], [5183.60, 'Mg b'],
    [5319.81, 'Nd II'], [5853.67, 'Ba II'], [5889.95, 'Na D2'], [5895.92, 'Na D1'],
    [6141.71, 'Ba II'], [6300.30, '[O I]'], [6496.90, 'Ba II'], [6562.80, 'Hα'],
    [6645.06, 'Eu II'], [6707.80, 'Li I'], [7771.94, 'O I'], [8498.02, 'Ca II'],
    [8542.09, 'Ca II'], [8662.14, 'Ca II'],
  ];

  // ── helpers ───────────────────────────────────────────────────────────────
  const normKey = s => (s == null ? '' : String(s).replace(/[\s_]+/g, '').toLowerCase());
  const fmtBytes = b => b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.round(b / 1024) + ' KB';
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const manifestCache = {};
  function loadManifest(base = 'spectra/') {
    if (!manifestCache[base]) {
      manifestCache[base] = fetch(base + 'manifest.json', { cache: 'no-cache' })
        .then(r => { if (!r.ok) throw new Error('manifest ' + r.status); return r.json(); })
        .then(m => { m.base = base; return m; });
    }
    return manifestCache[base];
  }

  /** files (array of {id, ...meta}) for any star name or alias list */
  function spectraFor(manifest, ...names) {
    for (const n of names) {
      const st = manifest.stars[normKey(n)];
      if (st) return { key: normKey(n), star: st, files: st.files.map(id => ({ id, ...manifest.files[id] })) };
    }
    return null;
  }

  function downloadUrl(manifest, f) {
    // relative "spectra/fits/..." must be resolved against the page, absolute URLs untouched
    return encodeURI(f.download);
  }

  // ── spectrum data store (LRU) ───────────────────────────────────────────────
  class Store {
    constructor(manifest, maxFull = 10) {
      this.m = manifest; this.maxFull = maxFull;
      this.ov = new Map(); this.full = new Map(); this.pending = new Map();
    }
    meta(id) { return this.m.files[id]; }
    url(id, ext) { return this.m.base + 'data/' + id + ext; }

    overview(id, signal) {
      if (this.ov.has(id)) return Promise.resolve(this.ov.get(id));
      return fetch(this.url(id, '.ov.f32'), { signal })
        .then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
        .then(buf => { const a = new Float32Array(buf); this.ov.set(id, a); return a; });
    }

    fullRes(id, signal) {
      if (this.full.has(id)) {                       // refresh LRU position
        const v = this.full.get(id); this.full.delete(id); this.full.set(id, v);
        return Promise.resolve(v);
      }
      if (this.pending.has(id)) return this.pending.get(id);
      const meta = this.meta(id);
      const p = fetch(this.url(id, '.f32'), { signal })
        .then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
        .then(buf => {
          const n = meta.n;
          const flux = new Float32Array(buf, 0, n);
          const wave = meta.grid === 'array' ? new Float64Array(buf.slice(n * 4)) : null;
          const v = { flux, wave };
          this.full.set(id, v);
          while (this.full.size > this.maxFull) this.full.delete(this.full.keys().next().value);
          return v;
        })
        .finally(() => this.pending.delete(id));
      this.pending.set(id, p);
      return p;
    }
  }

  // wavelength of pixel i / pixel index of wavelength w
  function xAt(meta, data, i) {
    if (meta.grid === 'linear') return meta.g0 + i * meta.dg;
    if (meta.grid === 'log') return Math.pow(10, meta.g0 + i * meta.dg);
    return data.wave[i];
  }
  function idxOf(meta, data, w) {            // first index with x >= w
    if (meta.grid === 'linear') return Math.ceil((w - meta.g0) / meta.dg);
    if (meta.grid === 'log') return Math.ceil((Math.log10(w) - meta.g0) / meta.dg);
    const a = data.wave; let lo = 0, hi = a.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < w) lo = mid + 1; else hi = mid; }
    return lo;
  }

  /** min/max decimation of full-res data over [a,b] (observed frame) */
  function decimateFull(meta, data, a, b, nb, k, s) {
    const n = meta.n, f = data.flux;
    const i0 = Math.max(0, idxOf(meta, data, a) - 1);
    const i1 = Math.min(n, idxOf(meta, data, b) + 1);
    const cnt = i1 - i0;
    if (cnt <= 0) return { x: [], y: [] };
    if (cnt <= nb * 2) {                                  // few points: raw pixels
      const x = new Float64Array(cnt), y = new Float32Array(cnt);
      for (let i = 0; i < cnt; i++) { x[i] = xAt(meta, data, i0 + i) * k; y[i] = f[i0 + i] * s; }
      return { x, y };
    }
    const x = new Float64Array(nb * 2), y = new Float32Array(nb * 2);
    let o = 0;
    for (let bi = 0; bi < nb; bi++) {
      const j0 = i0 + Math.floor(bi * cnt / nb), j1 = i0 + Math.floor((bi + 1) * cnt / nb);
      let mn = Infinity, mx = -Infinity, imn = -1, imx = -1;
      for (let j = j0; j < j1; j++) {
        const v = f[j];
        if (v < mn) { mn = v; imn = j; }               // NaN comparisons are false → skipped
        if (v > mx) { mx = v; imx = j; }
      }
      if (imn < 0) { x[o] = xAt(meta, data, j0) * k; y[o++] = NaN; continue; }   // gap
      const first = Math.min(imn, imx), second = Math.max(imn, imx);
      x[o] = xAt(meta, data, first) * k; y[o++] = f[first] * s;
      if (second !== first) { x[o] = xAt(meta, data, second) * k; y[o++] = f[second] * s; }
    }
    return { x: x.subarray(0, o), y: y.subarray(0, o) };
  }

  /** overview envelope over [a,b] */
  function decimateOverview(ov, a, b, k, s) {
    const nb = ov.length / 3, xs = [], ys = [];
    for (let i = 0; i < nb; i++) {
      const xv = ov[3 * i];
      if (xv < a || xv > b) continue;
      xs.push(xv * k, xv * k); ys.push(ov[3 * i + 1] * s, ov[3 * i + 2] * s);
    }
    return { x: xs, y: ys };
  }

  // ── Viewer ─────────────────────────────────────────────────────────────────
  const CSS = `
  .msv{display:flex;flex-direction:column;height:100%;min-height:0;font-family:'DM Sans',sans-serif;color:#e2e8f0}
  .msv-top{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:8px 12px;background:#111520;border-bottom:1px solid #232840;font-size:12px}
  .msv-top label{display:flex;align-items:center;gap:5px;cursor:pointer;color:#8892aa}
  .msv-top input[type=checkbox]{accent-color:#4f9cf9}
  .msv-top input[type=number]{width:76px;background:#181d2e;border:1px solid #232840;color:#e2e8f0;border-radius:4px;padding:3px 6px;font-family:'Space Mono',monospace;font-size:11px}
  .msv-btn{background:#181d2e;border:1px solid #232840;color:#8892aa;border-radius:4px;padding:4px 9px;cursor:pointer;font-family:'Space Mono',monospace;font-size:11px;text-decoration:none;white-space:nowrap}
  .msv-btn:hover{border-color:#4f9cf9;color:#4f9cf9}
  .msv-btn.primary{background:#4f9cf9;border-color:#4f9cf9;color:#fff}
  .msv-status{margin-left:auto;font-family:'Space Mono',monospace;font-size:10px;color:#4a5568}
  .msv-status.busy{color:#fbbf24}
  .msv-files{display:flex;flex-direction:column;gap:2px;padding:6px 12px;background:#0d1018;border-bottom:1px solid #232840;max-height:120px;overflow:auto}
  .msv-file{display:flex;align-items:center;gap:8px;font-size:11px;padding:2px 0}
  .msv-file .sw{width:10px;height:10px;border-radius:2px;flex-shrink:0}
  .msv-file .fn{font-family:'Space Mono',monospace;color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:330px}
  .msv-file .meta{color:#8892aa;font-family:'Space Mono',monospace;font-size:10px;white-space:nowrap}
  .msv-file .act{margin-left:auto;display:flex;gap:4px}
  .msv-plot{flex:1;min-height:260px}
  .msv-empty{display:flex;align-items:center;justify-content:center;height:100%;color:#4a5568;font-family:'Space Mono',monospace;font-size:12px}
  .msv-hdr{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:400;display:flex;align-items:center;justify-content:center}
  .msv-hdr>div{background:#111520;border:1px solid #232840;border-radius:8px;width:min(860px,92vw);height:80vh;display:flex;flex-direction:column}
  .msv-hdr header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #232840;font-family:'Space Mono',monospace;font-size:12px;color:#4f9cf9;position:static}
  .msv-hdr pre{flex:1;overflow:auto;padding:12px 14px;font-size:11px;color:#c7d0e0;font-family:'Space Mono',monospace;white-space:pre}
  `;
  function injectCss() {
    if (document.getElementById('msv-css')) return;
    const st = document.createElement('style'); st.id = 'msv-css'; st.textContent = CSS;
    document.head.appendChild(st);
  }

  class Viewer {
    /**
     * @param {HTMLElement} el      container (needs a height)
     * @param {object} manifest     from loadManifest()
     */
    constructor(el, manifest) {
      injectCss();
      this.el = el; this.m = manifest; this.store = new Store(manifest);
      this.files = []; this.visible = new Set(); this.range = null; this.yRange = null;
      this.opts = { norm: false, rest: false, rv: 0, lines: true };
      this.ctrl = null; this._busy = false; this._raf = 0;
      el.innerHTML = `
        <div class="msv">
          <div class="msv-top">
            <label title="Divide each spectrum by its median flux"><input type="checkbox" data-o="norm"> Normalize</label>
            <label title="Shift wavelengths to the stellar rest frame: λ/(1+v/c)"><input type="checkbox" data-o="rest"> Rest frame</label>
            <label>v<sub>r</sub> <input type="number" step="0.1" data-o="rv"> km/s</label>
            <label><input type="checkbox" data-o="lines" checked> Lines</label>
            <span style="color:#4a5568">|</span>
            <label>λ <input type="number" step="1" data-goto placeholder="4554"> ± <input type="number" step="1" value="5" data-gowidth style="width:52px"> Å</label>
            <button class="msv-btn" data-act="goto">Go</button>
            <button class="msv-btn" data-act="reset">Reset zoom</button>
            <span class="msv-status"></span>
          </div>
          <div class="msv-files"></div>
          <div class="msv-plot"><div class="msv-empty">Select a star to display its spectra</div></div>
        </div>`;
      this.$files = el.querySelector('.msv-files');
      this.$plot = el.querySelector('.msv-plot');
      this.$status = el.querySelector('.msv-status');
      el.querySelectorAll('[data-o]').forEach(inp => inp.addEventListener('change', () => {
        const k = inp.dataset.o;
        const prevK = this.k();
        this.opts[k] = inp.type === 'checkbox' ? inp.checked : (parseFloat(inp.value) || 0);
        if ((k === 'rest' || k === 'rv') && this.range) {          // keep same physical window
          const r = this.k() / prevK; this.range = [this.range[0] * r, this.range[1] * r];
        }
        if (k === 'norm') this.yRange = null;
        this.redraw();
      }));
      el.querySelector('[data-act=reset]').onclick = () => { this.range = null; this.yRange = null; this.redraw(); };
      const go = () => {
        const c = parseFloat(el.querySelector('[data-goto]').value), w = parseFloat(el.querySelector('[data-gowidth]').value) || 5;
        if (isFinite(c)) { this.range = [c - w, c + w]; this.yRange = null; this.redraw(); }
      };
      el.querySelector('[data-act=goto]').onclick = go;
      el.querySelector('[data-goto]').addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
      this._ro = new ResizeObserver(() => { if (this.$plot._fullLayout) { Plotly.Plots.resize(this.$plot); this.schedule(); } });
      this._ro.observe(this.$plot);
    }

    k() { return this.opts.rest ? 1 / (1 + this.opts.rv / C_KMS) : 1; }

    status(t, busy) { this.$status.textContent = t; this.$status.classList.toggle('busy', !!busy); }

    /** show a star.  files: [{id,...meta}] ; opts: {rv, only:[ids]} */
    async showStar(name, files, opt = {}) {
      if (this.ctrl) this.ctrl.abort();
      this.ctrl = new AbortController();
      const signal = this.ctrl.signal;
      this.name = name; this.files = files; this.range = null; this.yRange = null;
      this.opts.rv = opt.rv ?? 0;
      this.el.querySelector('[data-o=rv]').value = this.opts.rv ? this.opts.rv.toFixed(2) : '0';
      const only = opt.only && opt.only.length ? new Set(opt.only) : null;
      // default: show all files if ≤ 4, otherwise the first one
      this.visible = new Set(files.filter((f, i) => only ? only.has(f.id) : (files.length <= 4 || i === 0)).map(f => f.id));
      this.renderFileList();
      if (this.$plot._fullLayout) Plotly.purge(this.$plot);
      this.$plot.innerHTML = '';
      if (!files.length) { this.$plot.innerHTML = '<div class="msv-empty">No spectra available</div>'; return; }

      const t0 = performance.now();
      this.status('loading overview…', true);
      try {
        await Promise.all([...this.visible].map(id => this.store.overview(id, signal)));
      } catch (e) { if (signal.aborted) return; this.status('error: ' + e.message, true); return; }
      if (signal.aborted) return;
      this.redraw();
      this.status(`overview ${Math.round(performance.now() - t0)} ms · loading full resolution…`, true);
      this.loadFull(signal, t0);
    }

    async loadFull(signal, t0 = performance.now()) {
      const ids = [...this.visible].filter(id => !this.store.full.has(id));
      if (!ids.length) { this.status('full resolution'); return; }
      this.status('loading full resolution…', true);
      try {
        await Promise.all(ids.map(id => this.store.fullRes(id, signal).then(() => { if (!signal.aborted) this.schedule(); })));
        if (!signal.aborted) this.status(`full resolution · ${Math.round(performance.now() - t0)} ms`);
      } catch (e) { if (!signal.aborted) this.status('error: ' + e.message, true); }
    }

    renderFileList() {
      this.$files.innerHTML = this.files.map((f, i) => `
        <div class="msv-file">
          <input type="checkbox" data-id="${f.id}" ${this.visible.has(f.id) ? 'checked' : ''} style="accent-color:${COLORS[i % COLORS.length]}">
          <span class="sw" style="background:${COLORS[i % COLORS.length]}"></span>
          <span class="fn" title="${esc(f.fits)}">${esc(f.fits)}</span>
          <span class="meta">${esc(f.instrument)} · ${f.wmin.toFixed(0)}–${f.wmax.toFixed(0)} Å · ${(f.n / 1000).toFixed(0)}k px</span>
          <span class="act">
            <button class="msv-btn" data-hdr="${f.id}">Header</button>
            <a class="msv-btn" href="${downloadUrl(this.m, f)}" download="${esc(f.download.split('/').pop())}" target="_blank" rel="noopener">⬇ FITS · ${fmtBytes(f.size)}</a>
          </span>
        </div>`).join('');
      this.$files.querySelectorAll('input[data-id]').forEach(cb => cb.onchange = async () => {
        const id = cb.dataset.id;
        if (cb.checked) {
          this.visible.add(id);
          const signal = this.ctrl.signal;
          await this.store.overview(id, signal).catch(() => {});
          this.redraw(); this.loadFull(signal);
        } else { this.visible.delete(id); this.redraw(); }
      });
      this.$files.querySelectorAll('[data-hdr]').forEach(b => b.onclick = () => this.showHeader(b.dataset.hdr));
    }

    async showHeader(id) {
      const d = document.createElement('div'); d.className = 'msv-hdr';
      d.innerHTML = `<div><header><span>${esc(this.m.files[id].fits)}</span><button class="msv-btn">✕ Close</button></header><pre>loading…</pre></div>`;
      d.onclick = e => { if (e.target === d || e.target.tagName === 'BUTTON') d.remove(); };
      document.body.appendChild(d);
      try { d.querySelector('pre').textContent = await (await fetch(this.store.url(id, '.hdr.txt'))).text(); }
      catch (e) { d.querySelector('pre').textContent = 'Could not load header: ' + e.message; }
    }

    fullExtent() {
      let a = Infinity, b = -Infinity;
      for (const id of this.visible) { const f = this.m.files[id]; a = Math.min(a, f.wmin); b = Math.max(b, f.wmax); }
      const k = this.k();
      return [a * k, b * k];
    }

    schedule() { cancelAnimationFrame(this._raf); this._raf = requestAnimationFrame(() => this.redraw()); }

    traces(range) {
      const k = this.k(), a = range[0] / k, b = range[1] / k;   // observed-frame window
      const nb = Math.max(200, Math.round((this.$plot.clientWidth || 1000)));
      const out = [];
      this.files.forEach((f, i) => {
        if (!this.visible.has(f.id)) return;
        const s = this.opts.norm ? 1 / (f.median || 1) : 1;
        const full = this.store.full.get(f.id), ov = this.store.ov.get(f.id);
        let d;
        if (full) d = decimateFull(f, full, a, b, nb, k, s);
        else if (ov) d = decimateOverview(ov, a, b, k, s);
        else return;
        out.push({
          type: 'scatter', mode: 'lines', x: d.x, y: d.y, name: f.instrument + ' · ' + f.fits,
          line: { color: COLORS[i % COLORS.length], width: 1 }, connectgaps: false,
          hovertemplate: 'λ %{x:.3f} Å<br>F %{y:.4g}<extra>' + esc(f.instrument) + '</extra>',
        });
      });
      return out;
    }

    lineShapes(range) {
      if (!this.opts.lines) return { shapes: [], annotations: [] };
      const vis = LINES.filter(([w]) => w >= range[0] && w <= range[1]);
      if (vis.length > 40) return { shapes: [], annotations: [] };
      return {
        shapes: vis.map(([w]) => ({ type: 'line', xref: 'x', yref: 'paper', x0: w, x1: w, y0: 0, y1: 1,
          line: { color: 'rgba(167,139,250,.45)', width: 1, dash: 'dot' } })),
        annotations: vis.map(([w, n]) => ({ x: w, y: 1, xref: 'x', yref: 'paper', text: n, showarrow: false,
          textangle: -90, xanchor: 'right', yanchor: 'top', font: { size: 9, color: '#a78bfa' } })),
      };
    }

    redraw() {
      if (!this.files.length) return;
      const range = this.range || this.fullExtent();
      const { shapes, annotations } = this.lineShapes(range);
      const layout = {
        paper_bgcolor: '#0a0d14', plot_bgcolor: '#111520',
        font: { family: 'Space Mono, monospace', color: '#8892aa', size: 10 },
        margin: { t: 10, r: 14, b: 42, l: 64 }, showlegend: false, hovermode: 'closest',
        dragmode: 'zoom',
        xaxis: { range: range.slice(), autorange: false, gridcolor: '#232840', zeroline: false,
          title: { text: (this.opts.rest ? 'rest-frame ' : '') + 'λ (Å, air)', font: { color: '#e2e8f0', size: 11 } } },
        yaxis: { gridcolor: '#232840', zeroline: false, exponentformat: 'power',
          title: { text: this.opts.norm ? 'flux / median' : 'flux', font: { color: '#e2e8f0', size: 11 } },
          ...(this.yRange ? { range: this.yRange.slice(), autorange: false } : { autorange: true }) },
        shapes, annotations,
        hoverlabel: { bgcolor: '#181d2e', bordercolor: '#232840', font: { family: 'Space Mono', color: '#e2e8f0', size: 11 } },
      };
      const config = { responsive: false, displaylogo: false, scrollZoom: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
        toImageButtonOptions: { format: 'png', scale: 2, filename: (this.name || 'spectrum').replace(/\s+/g, '_') } };
      this._busy = true;
      const first = !this.$plot._fullLayout;
      Plotly.react(this.$plot, this.traces(range), layout, config).then(() => {
        this._busy = false;
        if (first) this.$plot.on('plotly_relayout', ev => this.onRelayout(ev));
      });
    }

    onRelayout(ev) {
      if (this._busy) return;
      let changed = false;
      if (ev['xaxis.autorange']) {
        this.range = null; this.yRange = null; changed = true;
      } else if ('xaxis.range[0]' in ev) {
        this.range = [ev['xaxis.range[0]'], ev['xaxis.range[1]']]; changed = true;
        this.yRange = 'yaxis.range[0]' in ev ? [ev['yaxis.range[0]'], ev['yaxis.range[1]']] : null;
      } else if (ev['xaxis.range']) {
        this.range = ev['xaxis.range'].slice(); changed = true;
        this.yRange = ev['yaxis.range'] ? ev['yaxis.range'].slice() : null;
      } else if ('yaxis.range[0]' in ev) {
        this.yRange = [ev['yaxis.range[0]'], ev['yaxis.range[1]']];      // y-only zoom: nothing to resample
      }
      if (changed) this.schedule();
    }

    destroy() {
      if (this.ctrl) this.ctrl.abort();
      this._ro.disconnect();
      if (this.$plot._fullLayout) Plotly.purge(this.$plot);
    }
  }

  global.MinceSpectra = { normKey, loadManifest, spectraFor, downloadUrl, fmtBytes, esc, Viewer, LINES };
})(window);
