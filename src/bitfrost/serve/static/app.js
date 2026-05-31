/* bitfrost dashboard — vanilla JS, no build step.
   Talks to the local /api/* endpoints, renders charts via the vendored
   Chart.js, and live-updates from /api/sse/live. Prompt/response content
   is masked by default in the detail drawer. */
"use strict";

const PALETTE = {
  accent: "#5eead4", accent2: "#a78bfa",
  success: "#4ade80", failed: "#f87171", pending: "#fbbf24",
  muted: "#8b9197",
};
const SERIES_COLORS = ["#5eead4", "#a78bfa", "#fbbf24", "#f87171", "#60a5fa", "#f472b6", "#34d399"];

const state = { filters: {}, charts: {}, knownAgents: new Set() };

// ---- helpers ----
function qs(extra) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters)) if (v) p.set(k, v);
  if (extra) for (const [k, v] of Object.entries(extra)) p.set(k, v);
  const s = p.toString();
  return s ? "?" + s : "";
}
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " → " + r.status);
  return r.json();
}
function fmtCost(c) {
  if (c == null) return "—";
  if (c === 0) return "$0";
  if (c < 1) return "$" + c.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  return "$" + c.toFixed(2);
}
function fmtNum(n) {
  if (n == null) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}
function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function fmtDur(ms) {
  if (!ms || ms <= 0) return "—";
  return ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : ms + "ms";
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// ---- tiles ----
function renderTiles(stats) {
  document.getElementById("t-calls").textContent = fmtNum(stats.totalCalls);
  document.getElementById("t-cost").textContent = fmtCost(stats.totalCost);
  const tt = stats.totalTokens || {};
  document.getElementById("t-tokens").textContent = fmtNum(tt.input) + " / " + fmtNum(tt.output);
  const failed = (stats.outcomes && stats.outcomes.failed) || 0;
  const rate = stats.totalCalls ? Math.round((failed / stats.totalCalls) * 100) : 0;
  document.getElementById("t-errors").textContent = rate + "%";
  const lat = stats.latency || {};
  document.getElementById("t-latency").textContent = fmtDur(lat.p50) + " / " + fmtDur(lat.p95);
}

// ---- charts ----
function upsertChart(key, ctxId, config) {
  if (state.charts[key]) {
    const c = state.charts[key];
    c.data = config.data;
    c.update();
    return;
  }
  const ctx = document.getElementById(ctxId);
  Chart.defaults.color = PALETTE.muted;
  Chart.defaults.borderColor = "rgba(35,40,45,0.6)";
  Chart.defaults.font.family = "ui-monospace, monospace";
  state.charts[key] = new Chart(ctx, config);
}
function renderCharts(stats) {
  const series = stats.series || [];
  upsertChart("cost", "chart-cost", {
    type: "line",
    data: {
      labels: series.map((b) => b.day),
      datasets: [{
        data: series.map((b) => b.cost),
        borderColor: PALETTE.accent,
        backgroundColor: "rgba(94,234,212,0.12)",
        fill: true, tension: 0.3, pointRadius: 3, borderWidth: 2,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { callback: (v) => "$" + v } } } },
  });
  const models = stats.models || [];
  upsertChart("models", "chart-models", {
    type: "bar",
    data: { labels: models.map((m) => m.model || "—"), datasets: [{ data: models.map((m) => m.calls), backgroundColor: PALETTE.accent2, borderRadius: 4 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
  });
  const provs = stats.providers || [];
  upsertChart("providers", "chart-providers", {
    type: "doughnut",
    data: { labels: provs.map((p) => p.provider), datasets: [{ data: provs.map((p) => p.calls), backgroundColor: SERIES_COLORS, borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, cutout: "62%" },
  });
  const oc = stats.outcomes || {};
  const ocKeys = Object.keys(oc);
  upsertChart("outcomes", "chart-outcomes", {
    type: "doughnut",
    data: { labels: ocKeys, datasets: [{ data: ocKeys.map((k) => oc[k]), backgroundColor: ocKeys.map((k) => PALETTE[k] || PALETTE.muted), borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, cutout: "62%" },
  });
}

// ---- filter dropdowns ----
function fillSelect(id, values, current) {
  const sel = document.getElementById(id);
  const first = sel.options[0];
  sel.innerHTML = "";
  sel.appendChild(first);
  for (const v of values) {
    if (!v) continue;
    const o = el("option", null, v);
    o.value = v;
    sel.appendChild(o);
  }
  if (current) sel.value = current;
}
function syncFilterOptions(stats) {
  fillSelect("f-model", (stats.models || []).map((m) => m.model).filter(Boolean), state.filters.model);
  fillSelect("f-provider", (stats.providers || []).map((p) => p.provider).filter((p) => p && p !== "unknown"), state.filters.provider);
  fillSelect("f-agent", [...state.knownAgents].sort(), state.filters.agent);
}

// ---- spans table ----
function rowFor(span, isNew) {
  const tr = el("tr", isNew ? "new" : null);
  tr.dataset.id = span.id;
  const tok = span.tokens || {};
  const cells = [
    fmtTime(span.timestamp), span.agentId || "—", span.model || "—", span.provider || "—",
    fmtNum(tok.input) + " → " + fmtNum(tok.output), fmtDur(span.durationMs), null, fmtCost(span.cost),
  ];
  cells.forEach((c, i) => {
    const td = el("td", i === 4 || i === 5 || i === 7 ? "num" : null);
    if (i === 6) {
      const oc = span.outcome || "pending";
      td.appendChild(el("span", "status " + oc, oc));
    } else {
      td.textContent = c;
    }
    tr.appendChild(td);
  });
  tr.addEventListener("click", () => openDrawer(span.id));
  return tr;
}
function renderSpans(spans) {
  const body = document.getElementById("spans-body");
  body.innerHTML = "";
  const empty = document.getElementById("empty");
  if (!spans.length) { empty.hidden = false; } else { empty.hidden = true; }
  for (const s of spans) {
    state.knownAgents.add(s.agentId);
    body.appendChild(rowFor(s, false));
  }
  document.getElementById("spans-count").textContent = spans.length ? "(" + spans.length + ")" : "";
}

// ---- detail drawer (masked by default) ----
function maskedBlock(label, text) {
  const wrap = el("div");
  const head = el("div", "section-h");
  head.appendChild(el("span", null, label));
  const hasText = text && text.length;
  const box = el("div", "content-box");
  let revealed = false;
  const render = () => {
    if (!hasText) { box.textContent = "—"; box.classList.remove("masked"); return; }
    box.textContent = revealed ? text : "•".repeat(Math.min(text.length, 80));
    box.classList.toggle("masked", !revealed);
  };
  if (hasText) {
    const btn = el("button", "reveal-btn", "reveal");
    btn.addEventListener("click", () => { revealed = !revealed; btn.textContent = revealed ? "hide" : "reveal"; render(); });
    head.appendChild(btn);
  }
  render();
  wrap.appendChild(head);
  wrap.appendChild(box);
  return wrap;
}
function promptText(input) {
  if (!input || !Array.isArray(input.messages)) return "";
  return input.messages.map((m) => (m.role ? m.role + ": " : "") + (m.content || "")).join("\n");
}
async function openDrawer(id) {
  const drawer = document.getElementById("drawer");
  const scrim = document.getElementById("drawer-scrim");
  const body = document.getElementById("drawer-body");
  body.innerHTML = "";
  drawer.hidden = false; scrim.hidden = false;
  let span;
  try { span = await api("/api/spans/" + encodeURIComponent(id)); }
  catch (e) { body.appendChild(el("div", "empty-sub", "could not load span")); return; }
  const tok = span.tokens || {};
  const kv = el("dl", "kv");
  const pairs = [
    ["agent", span.agentId], ["model", span.model], ["provider", span.provider],
    ["outcome", span.outcome], ["duration", fmtDur(span.durationMs)],
    ["tokens", fmtNum(tok.input) + " in / " + fmtNum(tok.output) + " out" + (tok.cache_read ? " · " + fmtNum(tok.cache_read) + " cached" : "")],
    ["cost", fmtCost(span.cost)], ["session", span.sessionId || "—"],
  ];
  for (const [k, v] of pairs) { kv.appendChild(el("dt", null, k)); kv.appendChild(el("dd", null, v == null ? "—" : String(v))); }
  body.appendChild(kv);
  body.appendChild(maskedBlock("prompt", promptText(span.input)));
  body.appendChild(maskedBlock("response", span.responseText || ""));
}
function closeDrawer() {
  document.getElementById("drawer").hidden = true;
  document.getElementById("drawer-scrim").hidden = true;
}

// ---- data load ----
async function reload() {
  try {
    const [stats, spansResp] = await Promise.all([api("/api/stats" + qs()), api("/api/spans" + qs({ limit: "200" }))]);
    renderTiles(stats);
    renderCharts(stats);
    renderSpans(spansResp.spans || []);
    syncFilterOptions(stats);
  } catch (e) {
    console.error(e);
  }
}

// ---- live stream ----
// New spans stream in over SSE and prepend to the table (with a brief
// flash); aggregates refresh on a short throttle. The browser reconnects
// automatically if the connection drops.
function startSSE() {
  let es;
  try { es = new EventSource("/api/sse/live"); }
  catch (e) { return; }
  es.onmessage = (ev) => {
    let span;
    try { span = JSON.parse(ev.data); } catch (e) { return; }
    state.knownAgents.add(span.agentId);
    const body = document.getElementById("spans-body");
    document.getElementById("empty").hidden = true;
    body.insertBefore(rowFor(span, true), body.firstChild);
    clearTimeout(state._t);
    state._t = setTimeout(() => api("/api/stats" + qs()).then((s) => { renderTiles(s); renderCharts(s); }).catch(() => {}), 600);
  };
}

// ---- wire up ----
function init() {
  for (const f of ["agent", "model", "provider", "outcome", "since"]) {
    document.getElementById("f-" + f).addEventListener("change", (e) => {
      state.filters[f] = e.target.value;
      reload();
    });
  }
  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  document.getElementById("drawer-scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  reload();
  startSSE();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
