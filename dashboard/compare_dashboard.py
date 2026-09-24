#!/usr/bin/env python3
"""
Live multi-tokenizer dashboard: type a sentence -> see a bar chart of token counts
per model, plus each tokenizer's resulting (colored) tokens side by side.

Run:  python3 compare_dashboard.py        then open http://localhost:8001
Gated models (Gemma/Llama) load from the local HF cache, or set HF_TOKEN first.
"""
import os
import argparse
from flask import Flask, request, jsonify

from compare_tokenizers import load_all, PALETTE  # reuse loaders + colors

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
TOKS = []      # loaded tokenizers
SKIPPED = []   # (name, reason)

PAGE = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tokenizer Comparison</title>
<style>
 :root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e9ef;--muted:#9aa4b2;--mine:#7ee2a8}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:22px 16px 80px}
 h1{font-size:20px;margin:0 0 4px} .sub{color:var(--muted);font-size:13px;margin:0 0 16px}
 textarea{width:100%;min-height:80px;background:var(--panel);color:var(--fg);border:1px solid var(--line);
   border-radius:10px;padding:12px;font-size:15px;resize:vertical}
 h2{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:24px 0 10px}
 /* bar chart */
 .bar{display:grid;grid-template-columns:190px 1fr 74px;align-items:center;gap:10px;margin:5px 0}
 .bar .name{font-size:13px;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .bar .track{background:var(--panel);border-radius:6px;overflow:hidden;height:22px}
 .bar .fill{height:100%;border-radius:6px;transition:width .18s}
 .bar .val{font-size:13px;font-variant-numeric:tabular-nums}
 .bar.mine .name{color:var(--mine);font-weight:600}
 /* token streams — each box hugs its content (grows/shrinks with the sentence) */
 .model{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0;
   width:fit-content;max-width:100%}
 .model.mine{border-color:var(--mine)}
 /* width:0 + min-width:100% => the header doesn't widen the box; it wraps to the token width */
 .model h3{font-size:14px;margin:0 0 8px;width:0;min-width:100%} .model.mine h3{color:var(--mine)}
 .model h3 small{color:var(--muted);font-weight:400}
 .stream{line-height:2.3;word-break:break-word;overflow-wrap:anywhere}
 .tok{padding:2px 3px;border-radius:4px;color:#0b0d10;white-space:pre;margin-right:2px;font-size:14px}
 .note{color:var(--muted);font-size:12px;margin-top:6px}
 .topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
 #export{background:#6ea8fe;color:#0b0d10;border:0;border-radius:8px;padding:9px 14px;font-size:14px;
   font-weight:600;cursor:pointer;white-space:nowrap} #export:disabled{opacity:.6;cursor:default}
 #report{background:var(--bg);padding:16px;border-radius:12px}
 .exportctl{display:flex;align-items:center;gap:10px}
 .exportctl label{font-size:13px;color:var(--muted);cursor:pointer;user-select:none}
 /* light theme: flip the CSS variables so the WHOLE page turns white (dark when off) */
 body.light{--bg:#ffffff;--panel:#f4f5f7;--line:#e2e6ec;--fg:#111111;--muted:#555555;--mine:#0a7f4f}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
</head><body><div class="wrap">
 <div class="topbar">
   <div><h1>Tokenizer Comparison</h1><p class="sub" id="sub">loading…</p></div>
   <div class="exportctl">
     <label><input type="checkbox" id="lightbg"> white background</label>
     <button id="export">⬇ Export PNG</button>
   </div>
 </div>
 <textarea id="in" placeholder="Bir cümle yazın...">Türkiye'nin başkenti Ankara'dır. Yapay zeka modelleri Türkçe gibi sondan eklemeli dillerde 2024 yılında çok gelişti.</textarea>

 <div id="report">
   <h2>Token count per model (fewer = better)</h2>
   <div id="chart"></div>
   <h2>Resulting tokens per model</h2>
   <div id="models"></div>
 </div>
 <p class="note" id="skipnote"></p>
</div>
<script>
const COLORS=["#ffd6a5","#caffbf","#9bf6ff","#bdb2ff","#ffc6ff","#fdffb6","#a0c4ff","#ffadad","#b9fbc0","#f1c0e8"];
const el=id=>document.getElementById(id);
const esc=s=>s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const mark=s=>esc(s).replaceAll("▁","␣").replaceAll("Ġ","␣").replaceAll("Ċ","⏎").replaceAll("&lt;/w&gt;","␣");
const barColor=(n,min)=> n===min ? "#7ee2a8" : "#6ea8fe";

async function run(){
  const text=el("in").value;
  const r=await fetch("/tokenize",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});
  const d=await r.json();
  const rows=d.results;                         // already sorted ascending by n_tokens server-side
  const counts=rows.map(x=>x.n_tokens);
  const min=Math.min(...counts), max=Math.max(...counts,1);

  // bar chart
  el("chart").innerHTML=rows.map(x=>{
    const w=Math.max(2,Math.round(100*x.n_tokens/max));
    return `<div class="bar ${x.mine?'mine':''}">
      <div class="name" title="${esc(x.name)}">${esc(x.name)}</div>
      <div class="track"><div class="fill" style="width:${w}%;background:${barColor(x.n_tokens,min)}"></div></div>
      <div class="val">${x.n_tokens}${x.mine?' ★':''}</div></div>`;
  }).join("");

  // token streams
  el("models").innerHTML=rows.map(x=>{
    const chips=x.tokens.map((t,i)=>`<span class="tok" style="background:${COLORS[i%COLORS.length]}" title="id ${x.ids[i]}">${mark(t)}</span>`).join("");
    return `<div class="model ${x.mine?'mine':''}"><h3>${esc(x.name)}
      <small>· ${x.n_tokens} tokens · ${x.tok_per_word} tok/word · vocab ${x.vocab.toLocaleString()}</small></h3>
      <div class="stream">${chips}</div></div>`;
  }).join("");
}
async function init(){
  const r=await fetch("/meta"); const m=await r.json();
  el("sub").textContent=`${m.n} tokenizers loaded · ␣=word-start/space, ⏎=newline`;
  el("skipnote").textContent = m.skipped.length ? ("Not loaded: "+m.skipped.map(s=>s[0]).join(", ")) : "";
  run();
}
el("lightbg").addEventListener("change",()=>{ document.body.classList.toggle("light", el("lightbg").checked); });
el("export").addEventListener("click", async ()=>{
  const btn=el("export"), old=btn.textContent; btn.disabled=true; btn.textContent="Rendering…";
  try{
    const canvas=await html2canvas(el("report"),
      {backgroundColor: getComputedStyle(document.body).backgroundColor, scale:2, useCORS:true});
    const a=document.createElement("a");
    a.download="tokenizer-comparison.png"; a.href=canvas.toDataURL("image/png"); a.click();
  }catch(e){ alert("Export failed: "+e); }
  finally{ btn.disabled=false; btn.textContent=old; }
});
let t; el("in").addEventListener("input",()=>{clearTimeout(t);t=setTimeout(run,140);});
init();
</script>
</body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/meta")
def meta():
    return jsonify({"n": len(TOKS), "skipped": SKIPPED})


@app.route("/tokenize", methods=["POST"])
def tokenize():
    text = (request.get_json(force=True) or {}).get("text", "")
    words = max(len(text.split()), 1)
    out = []
    for tk in TOKS:
        tokens, ids = tk.encode(text)
        out.append({
            "name": tk.name, "vocab": tk.vocab_size, "mine": tk.mine,
            "tokens": tokens, "ids": ids, "n_tokens": len(ids),
            "tok_per_word": round(len(ids) / words, 2),
        })
    out.sort(key=lambda x: x["n_tokens"])   # fewer tokens first
    return jsonify({"results": out})


def main():
    global TOKS, SKIPPED
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=os.path.join(HERE, "registry.json"))
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    print("Loading tokenizers (uses HF cache; set HF_TOKEN to fetch gated ones live)...")
    TOKS, SKIPPED = load_all(args.registry)
    print(f"\n{len(TOKS)} loaded, {len(SKIPPED)} skipped.  open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
