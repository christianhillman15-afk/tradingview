"use strict";

const POLL_MS = 2000;
let lastState = null;
let tvSymbol = null;

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const fmtMoney = (v) =>
  (v < 0 ? "-$" : "$") +
  Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtNum = (v, d = 2) =>
  v === null || v === undefined ? "—" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(2)}%`);
const cls = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
    if (lastState) {
      drawEquity(lastState);
      drawDrawdown(lastState);
      if (btn.dataset.tab === "chart") ensureTradingView(lastState);
    }
  });
});

// ---------- polling ----------
async function poll() {
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    const state = await res.json();
    if (state && state.account) {
      lastState = state;
      render(state);
      setStatus(true, state);
    } else {
      setStatus(false, state);
    }
  } catch (e) {
    setStatus(false, null);
  }
}

function setStatus(ok, state) {
  const dot = $("statusDot");
  const txt = $("statusText");
  if (!ok) {
    dot.className = "dot stale";
    txt.textContent = state && state.message ? "waiting for data" : "disconnected";
    return;
  }
  dot.className = "dot live";
  const t = state.generated_at ? new Date(state.generated_at) : new Date();
  txt.textContent = "updated " + t.toLocaleTimeString();
}

// ---------- render ----------
function render(s) {
  const a = s.account, m = s.metrics;
  $("symbol").textContent = s.symbol || "—";
  $("contractName").textContent = s.contract_name || "";
  $("strategyBadge").textContent = s.strategy || "—";
  $("modeBadge").textContent = s.mode || "—";
  $("brandSub").textContent = `${s.exchange || ""} · ${s.mode || "paper"}`;
  $("hdrEquity").textContent = fmtMoney(a.equity);
  $("hdrRoi").textContent = pct(a.roi_pct);
  $("hdrRoi").className = "value " + cls(a.roi_pct);

  renderKpis(s);
  renderOverviewActivity(s);
  renderPosition(s);
  renderRisk(s);
  renderPortfolio(s);
  renderTrades(s);
  renderMetrics(s);
  renderStrategy(s);
  renderLogs(s);
  drawEquity(s);
  drawDrawdown(s);

  $("tvSymbolLabel").textContent = s.tradingview_symbol || s.symbol || "—";
  if ($("tab-chart").classList.contains("active")) ensureTradingView(s);
}

function renderKpis(s) {
  const a = s.account, m = s.metrics;
  const cards = [
    { label: "Equity", value: fmtMoney(a.equity), sub: `start ${fmtMoney(a.starting_equity)}` },
    { label: "Net P&L", value: fmtMoney(a.realized_pnl + a.unrealized_pnl), cls: cls(a.realized_pnl + a.unrealized_pnl), sub: `realized ${fmtMoney(a.realized_pnl)}` },
    { label: "ROI", value: pct(a.roi_pct), cls: cls(a.roi_pct), sub: `CAGR ${pct(m.cagr_pct)}` },
    { label: "Win Rate", value: pct(m.win_rate_pct), sub: `${m.wins}W / ${m.losses}L` },
    { label: "Profit Factor", value: m.profit_factor === "inf" ? "∞" : fmtNum(m.profit_factor, 2), sub: `expectancy ${fmtMoney(m.expectancy)}` },
    { label: "Max Drawdown", value: pct(m.max_drawdown_pct), cls: "neg", sub: fmtMoney(-Math.abs(m.max_drawdown_dollars)) },
    { label: "Sharpe", value: fmtNum(m.sharpe, 2), sub: `payoff ${fmtNum(m.payoff_ratio, 2)}` },
    { label: "Trades", value: String(m.num_trades), sub: `unrealized ${fmtMoney(a.unrealized_pnl)}` },
  ];
  $("kpis").innerHTML = cards
    .map(
      (c) => `<div class="kpi">
        <div class="k-label">${c.label}</div>
        <div class="k-value ${c.cls || ""}">${c.value}</div>
        <div class="k-sub">${c.sub || ""}</div>
      </div>`
    )
    .join("");
}

function renderOverviewActivity(s) {
  const items = (s.events || []).slice(0, 12);
  $("overviewActivity").innerHTML = items.length
    ? items.map(eventRow).join("")
    : `<li class="muted">No activity yet.</li>`;
}

function eventRow(e) {
  const when = e.time ? new Date(e.time).toLocaleString() : "";
  return `<li><span class="when">${when}</span><span class="kind ${e.kind}">${e.kind}</span><span>${escapeHtml(e.message || "")}</span></li>`;
}

function renderPosition(s) {
  const p = s.position;
  const box = $("positionBox");
  if (!p) {
    box.innerHTML = `<div class="empty">Flat — no open position.</div>`;
    return;
  }
  const sideCls = p.side === "long" ? "side-long" : "side-short";
  box.innerHTML = `
    <div class="pos-grid">
      <div class="pos-cell"><div class="pc-label">Side</div><div class="pc-value ${sideCls}">${p.side.toUpperCase()}</div></div>
      <div class="pos-cell"><div class="pc-label">Quantity</div><div class="pc-value">${p.quantity}</div></div>
      <div class="pos-cell"><div class="pc-label">Unrealized</div><div class="pc-value ${cls(p.unrealized)}">${fmtMoney(p.unrealized)}</div></div>
      <div class="pos-cell"><div class="pc-label">Entry</div><div class="pc-value">${fmtNum(p.entry_price)}</div></div>
      <div class="pos-cell"><div class="pc-label">Last</div><div class="pc-value">${fmtNum(p.price)}</div></div>
      <div class="pos-cell"><div class="pc-label">Stop</div><div class="pc-value">${p.stop !== null ? fmtNum(p.stop) : "—"}</div></div>
      <div class="pos-cell"><div class="pc-label">Target</div><div class="pc-value">${p.target !== null ? fmtNum(p.target) : "—"}</div></div>
      <div class="pos-cell"><div class="pc-label">Strategy</div><div class="pc-value" style="font-size:13px">${p.strategy || "—"}</div></div>
      <div class="pos-cell"><div class="pc-label">Since</div><div class="pc-value" style="font-size:13px">${p.entry_time ? new Date(p.entry_time).toLocaleString() : "—"}</div></div>
    </div>
    <div class="muted" style="margin-top:10px">${escapeHtml(p.reason || "")}</div>`;
}

function renderRisk(s) {
  const r = s.risk || {};
  const can = r.can_trade !== false && !r.halted;
  $("riskBox").innerHTML = `
    <div class="risk-row"><span>Trading enabled</span><span class="pill ${can ? "ok" : "bad"}">${can ? "ACTIVE" : "HALTED"}</span></div>
    <div class="risk-row"><span>Global kill switch</span><span class="pill ${r.halted ? "bad" : "ok"}">${r.halted ? "TRIPPED" : "armed"}</span></div>
    <div class="risk-row"><span>Today's realized P&L</span><span class="${cls(r.day_realized)}">${r.day_realized !== undefined ? fmtMoney(r.day_realized) : "—"}</span></div>
    <div class="risk-row"><span>Equity peak</span><span>${r.equity_peak !== undefined ? fmtMoney(r.equity_peak) : "—"}</span></div>
    <div class="risk-row"><span>Bars processed</span><span>${s.bars_processed ?? "—"}</span></div>`;
}

function renderPortfolio(s) {
  const p = s.portfolio;
  const empty = $("portfolioEmpty"), content = $("portfolioContent");
  if (!p || !p.sleeves || !p.sleeves.length) {
    empty.style.display = "";
    content.style.display = "none";
    return;
  }
  empty.style.display = "none";
  content.style.display = "";

  $("pfSummary").textContent = `${p.sleeves.length} markets · ${p.weighting}-weighted`;
  $("sleevesBody").innerHTML = p.sleeves.map((sl) => `
    <tr>
      <td style="text-align:left"><span class="tag long">${sl.symbol}</span></td>
      <td>${fmtMoney(sl.start_capital)}</td>
      <td class="${cls(sl.roi_pct)}">${pct(sl.roi_pct)}</td>
      <td class="${cls(sl.sharpe)}">${fmtNum(sl.sharpe, 2)}</td>
      <td>${pct(sl.win_rate_pct)}</td>
      <td>${sl.num_trades}</td>
      <td class="neg">${pct(sl.max_drawdown_pct)}</td>
    </tr>`).join("");

  const dr = p.diversification_ratio;
  $("divBox").innerHTML = `
    <div class="risk-row"><span>Portfolio Sharpe</span><span class="${cls(s.metrics.sharpe)}">${fmtNum(s.metrics.sharpe, 2)}</span></div>
    <div class="risk-row"><span>Mean single-market Sharpe</span><span>${fmtNum(p.mean_sleeve_sharpe, 2)}</span></div>
    <div class="risk-row"><span>Diversification ratio</span><span class="pill ${dr > 1.05 ? "ok" : "bad"}">${fmtNum(dr, 2)}×</span></div>
    <div class="risk-row"><span>Avg pairwise correlation</span><span>${fmtNum(p.avg_correlation, 3)}</span></div>
    <div class="risk-row"><span>Portfolio ROI</span><span class="${cls(s.metrics.roi_pct)}">${pct(s.metrics.roi_pct)}</span></div>
    <div class="risk-row"><span>Portfolio max drawdown</span><span class="neg">${pct(s.metrics.max_drawdown_pct)}</span></div>`;

  renderHeatmap(p.symbols, p.correlation_matrix);
}

function corrColor(v) {
  // +1 red, 0 dark, -1 blue
  if (v >= 0) {
    const a = Math.min(1, v);
    return `rgba(255,91,110,${0.12 + 0.6 * a})`;
  }
  const a = Math.min(1, -v);
  return `rgba(76,141,255,${0.12 + 0.6 * a})`;
}

function renderHeatmap(symbols, matrix) {
  if (!symbols || !matrix) { $("corrHeatmap").innerHTML = ""; return; }
  const n = symbols.length;
  let html = '<table class="corr-table"><thead><tr><th></th>';
  for (const s of symbols) html += `<th>${s}</th>`;
  html += "</tr></thead><tbody>";
  for (let i = 0; i < n; i++) {
    html += `<tr><th>${symbols[i]}</th>`;
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      html += `<td style="background:${corrColor(v)}" title="${symbols[i]}/${symbols[j]}: ${v}">${v.toFixed(2)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  $("corrHeatmap").innerHTML = html;
}

function renderTrades(s) {
  const trades = s.trades || [];
  $("tradeCount").textContent = `${trades.length} shown`;
  $("tradesBody").innerHTML = trades
    .map((t, i) => {
      const sideTag = `<span class="tag ${t.side}">${t.side}</span>`;
      return `<tr>
        <td>${trades.length - i}</td>
        <td>${sideTag}</td>
        <td>${t.quantity}</td>
        <td>${fmtNum(t.entry_price)}</td>
        <td>${fmtNum(t.exit_price)}</td>
        <td class="${cls(t.pnl)}">${fmtMoney(t.pnl)}</td>
        <td class="${cls(t.return_pct)}">${pct(t.return_pct)}</td>
        <td style="text-align:left">${escapeHtml(t.exit_reason || "")}</td>
        <td style="text-align:left">${t.entry_time ? new Date(t.entry_time).toLocaleString() : ""}</td>
      </tr>`;
    })
    .join("");
}

function renderMetrics(s) {
  const m = s.metrics;
  const rows = [
    ["Final Equity", fmtMoney(m.final_equity)],
    ["Net Profit", fmtMoney(m.net_profit)],
    ["ROI", pct(m.roi_pct)],
    ["CAGR", pct(m.cagr_pct)],
    ["Total Trades", m.num_trades],
    ["Win Rate", pct(m.win_rate_pct)],
    ["Profit Factor", m.profit_factor === "inf" ? "∞" : fmtNum(m.profit_factor, 2)],
    ["Expectancy / trade", fmtMoney(m.expectancy)],
    ["Avg Win", fmtMoney(m.avg_win)],
    ["Avg Loss", fmtMoney(-Math.abs(m.avg_loss))],
    ["Payoff Ratio", fmtNum(m.payoff_ratio, 2)],
    ["Max Drawdown", pct(m.max_drawdown_pct)],
    ["Max DD ($)", fmtMoney(-Math.abs(m.max_drawdown_dollars))],
    ["Sharpe", fmtNum(m.sharpe, 2)],
    ["Sortino", m.sortino === undefined ? "—" : fmtNum(m.sortino, 2)],
    ["Calmar", m.calmar === undefined ? "—" : fmtNum(m.calmar, 2)],
    ["Prob. Sharpe (P SR>0)", m.psr === undefined ? "—" : pct(m.psr * 100)],
    ["Exposure", m.exposure_pct === undefined ? "—" : pct(m.exposure_pct)],
    ["Avg Bars Held", m.avg_bars_held === undefined ? "—" : fmtNum(m.avg_bars_held, 1)],
    ["Avg MFE", m.avg_mfe === undefined ? "—" : fmtMoney(m.avg_mfe)],
    ["Avg MAE", m.avg_mae === undefined ? "—" : fmtMoney(m.avg_mae)],
    ["Largest Win", m.largest_win === undefined ? "—" : fmtMoney(m.largest_win)],
    ["Largest Loss", m.largest_loss === undefined ? "—" : fmtMoney(-Math.abs(m.largest_loss || 0))],
    ["Max Consec. Losses", m.max_consecutive_losses],
    ["Starting Equity", fmtMoney(m.starting_equity)],
  ];
  $("metricGrid").innerHTML = rows
    .map(([k, v]) => `<div class="metric"><div class="m-label">${k}</div><div class="m-value">${v}</div></div>`)
    .join("");
}

function renderStrategy(s) {
  $("strategyInfo").innerHTML = `
    <div class="si-row"><span class="si-key">Strategy</span><span>${s.strategy || "—"}</span></div>
    <div class="si-row"><span class="si-key">Contract</span><span>${s.symbol} — ${s.contract_name || ""}</span></div>
    <div class="si-row"><span class="si-key">Exchange</span><span>${s.exchange || "—"}</span></div>
    <div class="si-row"><span class="si-key">Mode</span><span>${s.mode || "—"}</span></div>
    <div class="si-row"><span class="si-key">Last price</span><span>${fmtNum(s.last_price)}</span></div>
    <div class="si-row"><span class="si-key">Last bar</span><span>${s.last_bar_time ? new Date(s.last_bar_time).toLocaleString() : "—"}</span></div>`;
}

function renderLogs(s) {
  const events = s.events || [];
  $("eventLog").innerHTML = events.length
    ? events.slice(0, 200).map(eventRow).join("")
    : `<li class="muted">No events.</li>`;
}

// ---------- charts (vanilla canvas) ----------
function drawEquity(s) {
  const canvas = $("equityCanvas");
  const curve = (s.equity_curve || []).map((p) => p[1]);
  drawLine(canvas, curve, s.account ? s.account.starting_equity : null);
  if (s.equity_curve && s.equity_curve.length) {
    const a = new Date(s.equity_curve[0][0]).toLocaleDateString();
    const b = new Date(s.equity_curve[s.equity_curve.length - 1][0]).toLocaleDateString();
    $("curveRange").textContent = `${a} → ${b}`;
  }
}

function drawDrawdown(s) {
  const canvas = $("drawdownCanvas");
  const curve = (s.equity_curve || []).map((p) => p[1]);
  let peak = -Infinity;
  const dd = curve.map((v) => {
    peak = Math.max(peak, v);
    return peak > 0 ? -((peak - v) / peak) * 100 : 0;
  });
  drawLine(canvas, dd, 0, { color: "#ff5b6e", fill: "rgba(255,91,110,0.12)" });
}

function drawLine(canvas, data, baseline, opts = {}) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  // Use the CSS box (clientWidth/Height) as the logical size; never read back
  // canvas.height after we mutate it, or the backing store grows each redraw.
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
  const h = canvas.clientHeight || 240;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!data || data.length < 2) {
    ctx.fillStyle = "#8a94a8";
    ctx.font = "13px Inter, sans-serif";
    ctx.fillText("Not enough data yet.", 12, 24);
    return;
  }
  const pad = 28;
  let min = Math.min(...data), max = Math.max(...data);
  if (baseline !== null && baseline !== undefined) {
    min = Math.min(min, baseline);
    max = Math.max(max, baseline);
  }
  if (min === max) { max += 1; min -= 1; }
  const x = (i) => pad + (i / (data.length - 1)) * (w - 2 * pad);
  const y = (v) => h - pad - ((v - min) / (max - min)) * (h - 2 * pad);

  // grid
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const gy = pad + (g / 4) * (h - 2 * pad);
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w - pad, gy); ctx.stroke();
  }
  // baseline
  if (baseline !== null && baseline !== undefined) {
    ctx.strokeStyle = "rgba(138,148,168,0.5)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad, y(baseline)); ctx.lineTo(w - pad, y(baseline)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // area fill
  const up = data[data.length - 1] >= (baseline ?? data[0]);
  const color = opts.color || (up ? "#21c982" : "#ff5b6e");
  const fill = opts.fill || (up ? "rgba(33,201,130,0.12)" : "rgba(255,91,110,0.12)");
  ctx.beginPath();
  ctx.moveTo(x(0), y(data[0]));
  data.forEach((v, i) => ctx.lineTo(x(i), y(v)));
  ctx.lineTo(x(data.length - 1), h - pad);
  ctx.lineTo(x(0), h - pad);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  // line
  ctx.beginPath();
  ctx.moveTo(x(0), y(data[0]));
  data.forEach((v, i) => ctx.lineTo(x(i), y(v)));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // y labels
  ctx.fillStyle = "#8a94a8";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText(max.toFixed(0), 4, y(max) + 4);
  ctx.fillText(min.toFixed(0), 4, y(min));
}

// ---------- TradingView widget ----------
function ensureTradingView(s) {
  const sym = s.tradingview_symbol || s.symbol;
  if (!sym || sym === tvSymbol) return;
  tvSymbol = sym;
  const container = $("tradingview-widget");
  container.innerHTML = "";
  const script = document.createElement("script");
  script.type = "text/javascript";
  script.async = true;
  script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
  script.innerHTML = JSON.stringify({
    autosize: true,
    symbol: sym,
    interval: "5",
    timezone: "Etc/UTC",
    theme: "dark",
    style: "1",
    locale: "en",
    hide_side_toolbar: false,
    allow_symbol_change: true,
    studies: ["STD;VWAP", "STD;Bollinger_Bands"],
    support_host: "https://www.tradingview.com",
  });
  container.appendChild(script);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ---------- boot ----------
poll();
setInterval(poll, POLL_MS);
window.addEventListener("resize", () => { if (lastState) { drawEquity(lastState); drawDrawdown(lastState); } });
