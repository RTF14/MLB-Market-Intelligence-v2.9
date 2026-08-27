from __future__ import annotations

import subprocess, sys, threading, webbrowser
from pathlib import Path
import pandas as pd
from flask import Flask, redirect, render_template_string, request, url_for

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "edge_picks.csv"
app = Flask(__name__)
state = {"message": "Choose the current inputs, then click Run predictions.", "error": ""}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLB Market Intelligence v2.9</title><style>
body{font-family:Segoe UI,Arial;margin:0;background:#f4f7fb;color:#172033}.wrap{max-width:1180px;margin:32px auto;padding:0 20px}.panel{background:#fff;border-radius:12px;padding:22px;box-shadow:0 4px 18px #14213d18;margin-bottom:18px}h1{margin:0 0 5px}p{color:#536077}.grid{display:grid;grid-template-columns:230px 1fr;gap:12px;align-items:center}input{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cbd4e1;border-radius:7px;font:inherit}button,.button{background:#1769e0;color:#fff;border:0;border-radius:8px;padding:11px 18px;font-weight:650;cursor:pointer;text-decoration:none}.secondary{background:#5b6678}.actions{margin-top:18px;display:flex;gap:10px}.status{padding:11px 14px;background:#eaf2ff}.error{background:#ffe9e9;color:#9d2020;white-space:pre-wrap}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #e4e9f0;text-align:left}th{background:#eef3fa}.empty{text-align:center;padding:35px;color:#667085}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap"><div class="panel"><h1>MLB Market Intelligence v2.9</h1><p>Fetch the next MLB slate and current odds, run the model, and display qualifying edge picks.</p><form method="post" action="{{url_for('run_predictions')}}"><div class="actions"><button>Run next-game predictions</button><a class="button secondary" href="{{url_for('open_output')}}">Open output folder</a></div></form></div>
{%if error%}<div class="panel status error">{{error}}</div>{%else%}<div class="panel status">{{message}}</div>{%endif%}<div class="panel"><h2>Edge picks</h2>{%if rows%}<table><thead><tr><th>Date</th><th>Game</th><th>Market</th><th>Pick</th><th>Line</th><th>Model probability</th><th>Edge</th><th>Tier</th></tr></thead><tbody>{%for r in rows%}<tr><td>{{r.date}}</td><td>{{r.game}}</td><td>{{r.market}}</td><td>{{r.pick}}</td><td>{{r.line}}</td><td>{{r.probability}}</td><td>{{r.edge}}</td><td>{{r.tier}}</td></tr>{%endfor%}</tbody></table>{%else%}<div class="empty">No output yet—or no selections passed the filters.</div>{%endif%}<p><small>Research output only. Verify sportsbook lines are current.</small></p></div></div></body></html>"""

def table_rows():
    if not OUTPUT.exists(): return []
    rows=[]
    for _,r in pd.read_csv(OUTPUT).iterrows():
        p=pd.to_numeric(pd.Series([r.get("model_probability")]),errors="coerce").iloc[0]; e=pd.to_numeric(pd.Series([r.get("edge")]),errors="coerce").iloc[0]
        rows.append({"date":r.get("date",""),"game":r.get("game",""),"market":r.get("market",""),"pick":r.get("pick",""),"line":r.get("line",""),"probability":f"{p:.1%}" if pd.notna(p) else "","edge":f"{e:.3f}" if pd.notna(e) else "","tier":r.get("tier","")})
    return rows

@app.get("/")
def index(): return render_template_string(PAGE,rows=table_rows(),**state)

@app.post("/run")
def run_predictions():
    command=[sys.executable,str(ROOT/"live_pipeline.py")]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    if result.returncode: state.update(message="",error=(result.stderr or result.stdout or "Unknown error")[-5000:])
    else: state.update(message=f"Finished successfully. {len(table_rows())} edge pick(s) passed the filters.",error="")
    return redirect(url_for("index"))

@app.get("/open-output")
def open_output():
    folder=ROOT/"output"; folder.mkdir(exist_ok=True); subprocess.Popen(["explorer",str(folder)]); return redirect(url_for("index"))

if __name__=="__main__":
    threading.Timer(1.2,lambda:webbrowser.open("http://127.0.0.1:8765")).start()
    app.run(host="127.0.0.1",port=8765,debug=False)
