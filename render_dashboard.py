"""Build a single self-contained dashboard.html from random100_eval.json.

No external CSS/JS/fonts: everything is embedded so the page opens anywhere.
"""

import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "stratified150_eval.json")
if not os.path.exists(SRC):
    SRC = os.path.join(HERE, "random100_eval.json")
OUT = os.path.join(HERE, "dashboard.html")

CLASSES = ["positive", "negative", "neutral"]


def compute(records):
    n = len(records)
    correct = sum(1 for r in records if r["correct"])
    acc = correct / n if n else 0
    cm = {t: {c: 0 for c in CLASSES + ["unknown"]} for t in CLASSES + ["unknown"]}
    for r in records:
        t, p = r["star_label"], r["llm_label"]
        cm[t][p] = cm.get(t, {}).get(p, 0) + 1
    metrics = {}
    for cls in CLASSES:
        tp = cm[cls][cls]
        fp = sum(cm[t][cls] for t in CLASSES + ["unknown"] if t != cls)
        fn = sum(cm[cls][p] for p in CLASSES + ["unknown"] if p != cls)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        metrics[cls] = {"prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    star_dist = {c: sum(1 for r in records if r["star_label"] == c) for c in CLASSES}
    llm_dist = {c: sum(1 for r in records if r["llm_label"] == c) for c in CLASSES + ["unknown"]}
    mismatches = [r for r in records if not r["correct"]]
    return n, correct, acc, cm, metrics, star_dist, llm_dist, mismatches


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


_CLASS_COLORS = {"positive": "#2fbf71", "neutral": "#e0a33c", "negative": "#e5534b"}
_CLASS_ORDER = ["positive", "neutral", "negative"]


def chart_svg(star_dist, llm_dist):
    """Self-contained SVG grouped bar chart: actual (star-truth) vs predicted
    (LLM) count per sentiment class."""
    actual = [star_dist.get(c, 0) for c in _CLASS_ORDER]
    pred = [llm_dist.get(c, 0) for c in _CLASS_ORDER]
    nclass = len(_CLASS_ORDER)

    W, H, L, R, T, B = 720, 300, 50, 25, 46, 46
    pw, ph = W - L - R, H - T - B
    ytop = max((max(actual + pred) // 10) * 10, 10) or 10
    ytop = max(ytop, 10)
    ticks = sorted({0, ytop // 2 if ytop > 10 else 10, ytop})
    if ytop <= 10:
        ticks = [0, 5, 10]

    def barh(v):
        return v / ytop * ph

    parts = []
    # Legend (data series)
    ly = 26
    parts.append(
        f'<rect x="{L}" y="{ly-10}" width="12" height="12" rx="2" fill="#6ea8fe"/>'
        f'<text x="{L+18}" y="{ly}" font-size="11" fill="#e6ebf2">Actual (star truth)</text>'
        f'<rect x="{L+160}" y="{ly-10}" width="12" height="12" rx="2" fill="#6ea8fe" opacity="0.35" stroke="#6ea8fe" stroke-width="1.5"/>'
        f'<text x="{L+178}" y="{ly}" font-size="11" fill="#e6ebf2">Predicted (LLM)</text>'
    )
    # Gridlines + y labels
    for t in ticks:
        y = H - B - barh(t)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#2a3342" stroke-width="1"/>')
        parts.append(f'<text x="{L-6}" y="{y+4:.1f}" font-size="11" fill="#8b95a5" text-anchor="end">{t:g}</text>')
    # Bars + labels per class
    gw = pw / nclass
    for ci, c in enumerate(_CLASS_ORDER):
        col = _CLASS_COLORS[c]
        gx = L + ci * gw
        bw = min(gw * 0.30, 64)
        # actual bar (left, solid)
        xa = gx + gw * 0.16
        va, yha = actual[ci], barh(actual[ci])
        if va:
            parts.append(f'<rect x="{xa:.1f}" y="{H-B-yha:.1f}" width="{bw:.1f}" height="{yha:.1f}" rx="4" fill="{col}" opacity="0.95"/>')
        parts.append(f'<text x="{xa+bw/2:.1f}" y="{H-B-yha-6:.1f}" font-size="11" font-weight="700" fill="#e6ebf2" text-anchor="middle">{va}</text>')
        # predicted bar (right, translucent + outline)
        xp = gx + gw * 0.52
        vp, yhp = pred[ci], barh(pred[ci])
        if vp:
            parts.append(f'<rect x="{xp:.1f}" y="{H-B-yhp:.1f}" width="{bw:.1f}" height="{yhp:.1f}" rx="4" fill="{col}" opacity="0.35" stroke="{col}" stroke-width="1.5"/>')
        parts.append(f'<text x="{xp+bw/2:.1f}" y="{H-B-yhp-6:.1f}" font-size="11" font-weight="700" fill="#e6ebf2" text-anchor="middle">{vp}</text>')
        # class label
        parts.append(f'<text x="{gx+gw/2:.1f}" y="{H-B+24:.1f}" font-size="13" font-weight="600" fill="#e6ebf2" text-anchor="middle">{c}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:720px" '
            f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Predicted vs actual by class">'
            + "".join(parts) + "</svg>")


def main():
    with open(SRC, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Support both the metadata-wrapped format and a plain list.
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    records = data["records"] if isinstance(data, dict) else data
    n, correct, acc, cm, metrics, star_dist, llm_dist, mismatches = compute(records)

    data_js = json.dumps(records, ensure_ascii=False)

    # ---- Effectiveness chart (predicted vs actual) ----
    recalls = {}
    for cls in _CLASS_ORDER:
        n_true = sum(1 for r in records if r["star_label"] == cls)
        n_corr = sum(1 for r in records if r["star_label"] == cls and r["llm_label"] == cls)
        recalls[cls] = (n_corr / n_true * 100) if n_true else 0
    chart = chart_svg(star_dist, llm_dist)

    # ---- Helper fragments ----
    def cm_cell(t, p):
        count = cm.get(t, {}).get(p, 0)
        denom = max(sum(cm.get(t, {}).values()), 1)
        intensity = count / denom if count else 0
        invert = "cm-inv" if intensity > 0.65 else ""
        return (f'<div class="cm-cell {invert}" style="--iv:{intensity:.2f}">'
                f'<span class="cm-n">{count}</span></div>')

    metric_rows = ""
    for cls in CLASSES:
        m = metrics[cls]
        metric_rows += (
            f'<div class="mrow"><span class="cls {cls}">{cls}</span>'
            f'<span><b>{m["tp"] + m["fp"]}</b> pred</span>'
            f'<span><b>{m["tp"] + m["fn"]}</b> truth</span>'
            f'<span class="mval">{m["prec"]:.3f}</span>'
            f'<span class="mval">{m["rec"]:.3f}</span>'
            f'<span class="mval">{m["f1"]:.3f}</span></div>'
        )

    def dist_bar(dist):
        total = sum(dist.values()) or 1
        segs = ""
        for cls in CLASSES:
            v = dist.get(cls, 0)
            if v:
                segs += f'<div class="seg {cls}" style="flex:{v}"></div>'
        labels = "".join(f'<span class="dlabel {c}">{c} {dist.get(c, 0)}</span>' for c in CLASSES)
        return f'<div class="bar">{segs}</div><div class="dlabels">{labels}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amazon Gift Cards · LLM Sentiment Evaluation</title>
<style>
  :root {{
    --bg:#0e1218; --panel:#161c26; --panel2:#1b2330; --line:#2a3342;
    --fg:#e6ebf2; --mut:#8b95a5; --pos:#2fbf71; --neg:#e5534b;
    --neu:#e0a33c; --unk:#7b8794; --acc:#6ea8fe;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font-family:'Segoe UI', system-ui, -apple-system, sans-serif; }}
  header {{ position:sticky; top:0; z-index:10; display:flex; align-items:baseline; gap:16px;
    padding:16px 28px; background:rgba(14,18,24,.92); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--line); flex-wrap:wrap; }}
  header h1 {{ font-size:20px; margin:0; letter-spacing:.3px; }}
  header .tag {{ color:var(--mut); font-size:13px; }}
  main {{ padding:22px 28px 60px; max-width:1240px; margin:0 auto; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:14px; margin-bottom:26px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
    padding:16px 18px; }}
  .card .k {{ color:var(--mut); font-size:12px; text-transform:uppercase; letter-spacing:.8px; }}
  .card .v {{ font-size:30px; font-weight:700; margin-top:6px; }}
  .card .s {{ color:var(--mut); font-size:12px; margin-top:4px; }}
  .acc .v {{ color:var(--acc); }}
  .green .v {{ color:var(--pos); }} .red .v {{ color:var(--neg); }} .amber .v {{ color:var(--neu); }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-bottom:26px; }}
  @media(max-width:860px){{ .grid{{ grid-template-columns:1fr; }} }}
  .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:20px; }}
  h2 {{ font-size:15px; margin:0 0 14px; color:var(--fg); letter-spacing:.3px; }}
  .cm-wrap {{ display:inline-block; }}
  .cm-header {{ display:grid; grid-template-columns:110px repeat(4,1fr); gap:4px; margin-bottom:4px; }}
  .cm-header span {{ text-align:center; color:var(--mut); font-size:11px; }}
  .cm-row {{ display:grid; grid-template-columns:110px repeat(4,1fr); gap:4px; margin-bottom:4px; }}
  .cm-tl {{ display:flex; align-items:center; justify-content:center; color:var(--fg);
    font-size:11px; font-weight:600; border-radius:8px; }}
  .cm-tl.positive{{ background:rgba(47,191,113,.15); }} .cm-tl.negative{{ background:rgba(229,83,75,.15); }}
  .cm-tl.neutral{{ background:rgba(224,163,60,.15); }} .cm-tl.unknown{{ background:rgba(123,135,148,.15); }}
  .cm-cell {{ display:flex; align-items:center; justify-content:center; height:46px;
    border-radius:8px; background:rgba(110,168,254,calc(var(--iv)*.55)); }}
  .cm-cell.cm-inv .cm-n{{ color:#0e1218; }}
  .cm-n {{ font-weight:700; font-size:16px; }}
  .mrow {{ display:grid; grid-template-columns:1.2fr 1fr 1fr 1fr 1fr 1fr; gap:8px;
    align-items:center; padding:9px 4px; border-top:1px solid var(--line); font-size:13px; }}
  .mrow:first-of-type {{ border-top:none; }}
  .mrow span {{ color:var(--mut); }} .mrow .mval {{ color:var(--fg); font-variant-numeric:tabular-nums; }}
  .cls {{ padding:2px 9px; border-radius:999px; font-size:11px; font-weight:700; width:max-content; }}
  .cls.positive{{ background:rgba(47,191,113,.18); color:var(--pos); }}
  .cls.negative{{ background:rgba(229,83,75,.18); color:var(--neg); }}
  .cls.neutral{{ background:rgba(224,163,60,.18); color:var(--neu); }}
  .cls.unknown{{ background:rgba(123,135,148,.22); color:var(--unk); }}
  .bar {{ display:flex; height:26px; border-radius:8px; overflow:hidden; gap:2px; }}
  .seg {{ transition:flex .3s; }}
  .seg.positive{{ background:var(--pos); }} .seg.negative{{ background:var(--neg); }}
  .seg.neutral{{ background:var(--neu); }}
  .dlabels {{ display:flex; gap:18px; margin-top:8px; font-size:12px; color:var(--mut); }}
  .stock {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }}
  .ctl {{ background:var(--panel); border:1px solid var(--line); color:var(--fg);
    border-radius:10px; padding:9px 14px; font-size:13px; }}
  .search {{ flex:1; min-width:200px; }}
  input.search::placeholder{{ color:var(--mut); }}
  select.ctl, button.ctl {{ cursor:pointer; }}
  button.ctl.active {{ border-color:var(--acc); color:var(--acc); background:rgba(110,168,254,.12); }}
  .table-wrap {{ overflow:auto; background:var(--panel); border:1px solid var(--line);
    border-radius:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  thead th {{ text-align:left; padding:12px 14px; color:var(--mut); font-size:11px;
    text-transform:uppercase; letter-spacing:.6px; border-bottom:1px solid var(--line);
    cursor:pointer; user-select:none; position:sticky; top:57px; background:var(--panel); }}
  thead th:hover{{ color:var(--fg); }}
  tbody tr {{ border-bottom:1px solid var(--line); }}
  tbody tr:last-child{{ border-bottom:none; }}
  tbody tr:hover{{ background:var(--panel2); }}
  td {{ padding:10px 14px; vertical-align:top; }}
  td.num {{ font-variant-numeric:tabular-nums; color:var(--mut); }}
  .stars {{ color:var(--neu); letter-spacing:1px; }}
  .stars .off{{ color:#3a4552; }}
  .ttl {{ font-weight:600; }}
  .txt {{ color:var(--mut); }}
  .b {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:700; }}
  .b.pos{{ background:rgba(47,191,113,.18); color:var(--pos); }}
  .b.neg{{ background:rgba(229,83,75,.18); color:var(--neg); }}
  .b.neu{{ background:rgba(224,163,60,.18); color:var(--neu); }}
  .b.unk{{ background:rgba(123,135,148,.22); color:var(--unk); }}
  .ok {{ color:var(--pos); font-weight:700; }} .bad {{ color:var(--neg); font-weight:700; }}
  .eb {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:600;
    background:rgba(139,149,165,.16); color:var(--fg); }}
  .mut {{ color:var(--mut); font-size:12px; }}
  .mismrow {{ background:rgba(229,83,75,.06); }}
  .count {{ color:var(--mut); font-size:13px; margin:4px 2px 10px; }}
</style>
</head>
<body>
<header>
  <h1>🍀 Amazon Gift Cards · LLM Sentiment Evaluation</h1>
  <span class="tag">{esc(meta.get('description', f'{n} reviews'))} · model {esc(meta.get('model', 'DeepSeek-V4-Flash-0731'))} · {correct}/{n} = {acc:.1%} accurate · generated {time.strftime('%b %d %Y, %H:%M')}</span>
</header>
<main>

  <section class="cards">
    <div class="card acc"><div class="k">Accuracy</div><div class="v">{acc:.1%}</div><div class="s">{correct} / {n} correct</div></div>
    <div class="card"><div class="k">Reviews tested</div><div class="v">{n}</div><div class="s">{esc(meta.get('sample','sample'))}, seed {meta.get('seed','—')}</div></div>
    <div class="card green"><div class="k">Positive (4–5★)</div><div class="v">{star_dist['positive']}</div><div class="s">{metrics['positive']['f1']:.2f} F1</div></div>
    <div class="card red"><div class="k">Negative (1–2★)</div><div class="v">{star_dist['negative']}</div><div class="s">{metrics['negative']['f1']:.2f} F1</div></div>
    <div class="card amber"><div class="k">Neutral (3★)</div><div class="v">{star_dist['neutral']}</div><div class="s">{metrics['neutral']['f1']:.2f} F1</div></div>
    <div class="card red"><div class="k">Mismatches</div><div class="v">{len(mismatches)}</div><div class="s">LLM ≠ star</div></div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Confusion matrix</h2>
      <div class="cm-wrap">
        <div class="cm-header"><span>truth ↓</span><span>pos</span><span>neg</span><span>neu</span><span>unk</span></div>
        <div class="cm-row"><div class="cm-tl positive">positive</div>{cm_cell('positive','positive')}{cm_cell('positive','negative')}{cm_cell('positive','neutral')}{cm_cell('positive','unknown')}</div>
        <div class="cm-row"><div class="cm-tl negative">negative</div>{cm_cell('negative','positive')}{cm_cell('negative','negative')}{cm_cell('negative','neutral')}{cm_cell('negative','unknown')}</div>
        <div class="cm-row"><div class="cm-tl neutral">neutral</div>{cm_cell('neutral','positive')}{cm_cell('neutral','negative')}{cm_cell('neutral','neutral')}{cm_cell('neutral','unknown')}</div>
      </div>
    </div>

    <div class="panel">
      <h2>Per-class metrics</h2>
      <div class="mrow" style="grid-template-columns:1.2fr 1fr 1fr 1fr 1fr 1fr">
        <span style="grid:auto">class</span><span>pred</span><span>truth</span><span>prec</span><span>rec</span><span>F1</span>
      </div>
      {metric_rows}
      <h2 style="margin-top:22px">Class distribution</h2>
      <div class="bar">{''.join(f'<div class="seg {c}" style="flex:{star_dist[c] or 0}"></div>' for c in CLASSES)}</div>
      <div class="dlabels">
        {''.join(f'<span class="{c}">★truth {c}: {star_dist[c]}</span>' for c in CLASSES)}
      </div>
      <div class="dlabels">
        {''.join(f'<span class="{c}">LLM {c}: {llm_dist[c]}</span>' for c in CLASSES)}
      </div>
    </div>
  </section>

  <section class="panel" style="margin-bottom:26px">
    <h2>Model effectiveness: predicted vs actual per class</h2>
    <div style="display:flex; flex-wrap:wrap; gap:20px; align-items:flex-start;">
      {chart}
      <div style="min-width:220px; flex:1;">
        <table style="width:100%; font-size:13px; border-collapse:collapse;">
          <thead><tr>
            <th style="text-align:left; padding:8px 10px; color:var(--mut); font-size:11px; text-transform:uppercase; letter-spacing:.6px; border-bottom:1px solid var(--line);">class</th>
            <th style="text-align:center; padding:8px 10px; color:var(--mut); font-size:11px; border-bottom:1px solid var(--line);">actual (star)</th>
            <th style="text-align:center; padding:8px 10px; color:var(--mut); font-size:11px; border-bottom:1px solid var(--line);">predicted (LLM)</th>
            <th style="text-align:center; padding:8px 10px; color:var(--mut); font-size:11px; border-bottom:1px solid var(--line);">recall</th>
          </tr></thead>
          <tbody>
            {"".join(
                '<tr>'
                f'<td style="padding:8px 10px; border-bottom:1px solid var(--line);"><span class="cls {c}">{c}</span></td>'
                f'<td style="text-align:center; padding:8px 10px; border-bottom:1px solid var(--line); font-weight:700;">{star_dist[c]}</td>'
                f'<td style="text-align:center; padding:8px 10px; border-bottom:1px solid var(--line);">{llm_dist[c]}</td>'
                f'<td style="text-align:center; padding:8px 10px; border-bottom:1px solid var(--line); font-weight:700;">{recalls[c]:.0f}%</td>'
                '</tr>' for c in _CLASS_ORDER)}
          </tbody>
        </table>
        <div style="margin-top:12px; font-size:12px; color:var(--mut);">Gap between the two bars per class = where the model is over- or under-predicting that sentiment.</div>
      </div>
    </div>
  </section>

  <section>
    <div class="stock">
      <select id="fSent" class="ctl">
        <option value="all">All sentiments</option>
        <option value="positive">Positive</option>
        <option value="negative">Negative</option>
        <option value="neutral">Neutral</option>
      </select>
      <select id="fSource" class="ctl">
        <option value="all">Truth &amp; prediction</option>
        <option value="truth">By star-truth</option>
        <option value="llm">By LLM prediction</option>
      </select>
      <button id="fMis" class="ctl" onclick="toggleMis()">✖ Mismatches only</button>
      <button id="fSort" class="ctl" onclick="toggleSort()">Sort: line ↑</button>
      <input id="fSearch" class="ctl search" type="text" placeholder="Search title or text…" oninput="render()">
      <div class="count" id="count"></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th data-sort="index">#</th><th data-sort="rating">Rating</th>
          <th data-sort="title">Title</th><th data-sort="text">Review text</th>
          <th data-sort="star_label">Star → truth</th><th data-sort="llm_label">LLM</th><th data-sort="llm_emotion">LLM emo</th><th data-sort="nrc_emotion">NRC emo</th><th>Emo match</th><th>Status</th>

        </tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </section>

</main>
<script>
const DATA = {data_js};
let misOnly = false;
let sortKey = 'index', sortDesc = false;

function sentLabel(b){{
  const map={{positive:'pos',negative:'neg',neutral:'neu',unknown:'unk'}};
  return `<span class="b ${{map[b]}}">${{b}}</span>`;
}}
function stars(r){{
  const s='★'.repeat(Math.round(r))+'☆'.repeat(5-Math.round(r));
  return `<span class="stars">${{s}}</span>`;
}}
function emoCell(raw,map){{
  if(!raw) return '<span class="mut">—</span>';
  const shown = map || raw;
  const tag = map ? '' : ' <span class="mut">(unmapped)</span>';
  return '<span class="eb">'+esc(shown)+'</span>'+tag;
}}
function emoMatch(r){{
  return (r.emotion_match===null||r.emotion_match===undefined||r.emotion_match==='')
    ? '<span class="mut">—</span>'
    : (r.emotion_match===true ? '<span class="ok">✔</span>' : '<span class="bad">✖</span>');
}}
function esc(s){{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function render(){{
  const q=document.getElementById('fSearch').value.toLowerCase();
  const fs=document.getElementById('fSent').value;
  const fsrc=document.getElementById('fSource').value;
  let rows=DATA.filter(r=>{{
    if(misOnly && r.correct) return false;
    if(fs!=='all' && r.star_label!==fs && r.llm_label!==fs) return false;
    if(fsrc==='truth' && fs!=='all' && r.star_label!==fs) return false;
    if(fsrc==='llm' && fs!=='all' && r.llm_label!==fs) return false;
    if(q && !(esc(r.title).toLowerCase().includes(q)||esc(r.text).toLowerCase().includes(q))) return false;
    return true;
  }});
  rows.sort((a,b)=>{{
    let va=a[sortKey], vb=b[sortKey];
    if(sortKey==='index'||sortKey==='rating'){{va=+va;vb=+vb;}}
    const diff=va>vb?1:va<vb?-1:0;
    return sortDesc?-diff:diff;
  }});
  document.getElementById('count').textContent=`${{rows.length}} of ${{DATA.length}} reviews`;
  document.getElementById('rows').innerHTML=rows.map(r=>
    `<tr class="${{r.correct?'':'mismrow'}}">
      <td class="num">${{r.index}}</td>
      <td class="num">${{stars(r.rating)}}<div style="color:var(--mut);font-size:11px">${{r.rating.toFixed(1)}}</div></td>
      <td><span class="ttl">${{esc(r.title)||'—'}}</span></td>
      <td><span class="txt">${{esc(r.text)||'—'}}</span></td>
      <td>${{sentLabel(r.star_label)}}</td>
      <td>${{sentLabel(r.llm_label)}}</td>
      <td>${{emoCell(r.llm_emotion_raw, r.llm_emotion)}}</td>
      <td>${{emoCell(r.nrc_emotion)}}</td>
      <td>${{emoMatch(r)}}</td>
      <td>${{r.correct?'<span class="ok">✔ correct</span>':'<span class="bad">✖ mismatch</span>'}}</td>
    </tr>`
  ).join('');
}}
function toggleMis(){{ misOnly=!misOnly;
  document.getElementById('fMis').classList.toggle('active',misOnly); render(); }}
function toggleSort(){{ sortDesc=!sortDesc;
  document.getElementById('fSort').textContent='Sort: '+sortKey+(sortDesc?' ↓':' ↑'); render(); }}
document.querySelectorAll('th[data-sort]').forEach(th=>th.addEventListener('click',()=>{{
  sortKey=th.dataset.sort; sortDesc=!sortDesc;
  document.getElementById('fSort').textContent='Sort: '+sortKey+(sortDesc?' ↓':' ↑');
  render();
}}));
['fSent','fSource'].forEach(id=>document.getElementById(id).addEventListener('change',render));
render();
</script>
</body>
</html>
"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {OUT} ({os.path.getsize(OUT):,} bytes, {n} records)")


if __name__ == "__main__":
    main()
