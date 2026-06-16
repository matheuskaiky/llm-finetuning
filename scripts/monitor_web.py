#!/usr/bin/env python3
"""Live web monitor for a continued-pretraining run (standard library only).

Tails a training log written by ``scripts/train.py`` (HuggingFace Trainer
progress) and serves a small auto-refreshing web page with the current step,
loss, learning rate, throughput and ETA, plus the antes/depois evaluation lines
when present. No third-party dependencies; reads the log only (does not touch the
GPU), so it is safe to run alongside training.

Usage:
    python scripts/monitor_web.py --log /tmp/train_1p7b.log --port 8000
    # then open http://localhost:8000 (tunnel the port over SSH if remote)
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROGRESS_RE = re.compile(
    r"(\d+)/(\d+)\s*\[(\d+:\d+)<(\d+:\d+),\s*([\d.]+)s/it\]"
)
LOSS_RE = re.compile(r"(?:'loss':|loss=)\s*([\d.]+)")
TRAIN_LOSS_RE = re.compile(r"'train_loss':\s*([\d.]+)")
LR_RE = re.compile(r"'learning_rate':\s*([\d.eE+-]+)")
EPOCH_RE = re.compile(r"'epoch':\s*([\d.]+)")
GRADNORM_RE = re.compile(r"'grad_norm':\s*([\d.]+)")
EVAL_BEFORE_RE = re.compile(r"eval before:\s*(\{.*\})")
EVAL_AFTER_RE = re.compile(r"eval after:\s*(\{.*\})")


def _last(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def parse_log(path: Path, tail_bytes: int = 200_000) -> dict[str, object]:
    """Extract the latest training state from the tail of the log file."""
    if not path.exists():
        return {"status": "waiting", "message": f"log not found yet: {path}"}
    data = path.read_text(encoding="utf-8", errors="ignore")[-tail_bytes:]
    # tqdm uses carriage returns inside a single physical line.
    blob = data.replace("\r", "\n")

    state: dict[str, object] = {"status": "running"}
    prog = None
    for m in PROGRESS_RE.finditer(blob):
        prog = m
    if prog:
        step, total = int(prog.group(1)), int(prog.group(2))
        state.update(
            step=step,
            total=total,
            percent=round(100 * step / total, 1) if total else 0,
            elapsed=prog.group(3),
            eta=prog.group(4),
            s_per_it=float(prog.group(5)),
        )

    loss = _last(TRAIN_LOSS_RE, blob) or _last(LOSS_RE, blob)
    if loss is not None:
        state["loss"] = float(loss)
    lr = _last(LR_RE, blob)
    if lr is not None:
        state["learning_rate"] = float(lr)
    epoch = _last(EPOCH_RE, blob)
    if epoch is not None:
        state["epoch"] = float(epoch)
    grad = _last(GRADNORM_RE, blob)
    if grad is not None:
        state["grad_norm"] = float(grad)

    before = _last(EVAL_BEFORE_RE, blob)
    if before:
        try:
            state["eval_before"] = json.loads(before.replace("'", '"'))
        except json.JSONDecodeError:
            pass
    after = _last(EVAL_AFTER_RE, blob)
    if after:
        try:
            state["eval_after"] = json.loads(after.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    if "training done" in blob or "saved results" in blob:
        state["status"] = "done"
    if re.search(r"out of memory|CUDA error|Traceback", blob):
        state["status"] = "error"
    return state


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:760px;margin:40px auto;color:#1d1f23}}
 h1{{font-size:20px}} .sub{{color:#666;font-size:13px;margin-bottom:20px}}
 .bar{{height:26px;background:#eceef1;border-radius:6px;overflow:hidden}}
 .fill{{height:100%;background:#3b82f6;width:0%;transition:width .4s}}
 .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}
 .card{{background:#f6f7f9;border-radius:8px;padding:12px}}
 .k{{color:#666;font-size:12px}} .v{{font-size:19px;font-weight:600;margin-top:4px}}
 table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:14px}}
 td,th{{border:1px solid #e3e5e8;padding:6px 10px;text-align:left}}
 .st-running{{color:#1d4ed8}} .st-done{{color:#15803d}} .st-error{{color:#b91c1c}} .st-waiting{{color:#a16207}}
</style></head><body>
<h1>{title}</h1><div class="sub">log: {log} - atualiza a cada {refresh}s</div>
<div class="bar"><div class="fill" id="fill"></div></div>
<div id="pct" class="sub" style="margin-top:8px"></div>
<div class="grid" id="cards"></div>
<div id="evals"></div>
<script>
const REFRESH={refresh}*1000;
function card(k,v){{return `<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`}}
async function tick(){{
 try{{
  const r=await fetch('status'); const s=await r.json();
  const st=s.status||'?';
  document.getElementById('pct').innerHTML=
    `<span class="st-${{st}}">status: ${{st}}</span>`+(s.step!=null?` - passo ${{s.step}}/${{s.total}} (${{s.percent}}%)`:'')+(s.message?` - ${{s.message}}`:'');
  document.getElementById('fill').style.width=(s.percent||0)+'%';
  let c='';
  if(s.loss!=null)c+=card('loss',s.loss.toFixed(4));
  if(s.learning_rate!=null)c+=card('learning rate',s.learning_rate.toExponential(2));
  if(s.epoch!=null)c+=card('epoch',s.epoch.toFixed(3));
  if(s.s_per_it!=null)c+=card('s / passo',s.s_per_it.toFixed(2));
  if(s.elapsed)c+=card('decorrido',s.elapsed);
  if(s.eta)c+=card('ETA',s.eta);
  if(s.grad_norm!=null)c+=card('grad norm',s.grad_norm.toFixed(3));
  document.getElementById('cards').innerHTML=c;
  let e='';
  if(s.eval_before||s.eval_after){{
   e='<table><tr><th>metrica</th><th>antes</th><th>depois</th></tr>';
   const keys=new Set([...Object.keys(s.eval_before||{{}}),...Object.keys(s.eval_after||{{}})]);
   keys.forEach(k=>{{const b=s.eval_before?s.eval_before[k]:null,a=s.eval_after?s.eval_after[k]:null;
     e+=`<tr><td>${{k}}</td><td>${{b!=null?b.toFixed(4):'-'}}</td><td>${{a!=null?a.toFixed(4):'-'}}</td></tr>`}});
   e+='</table>';
  }}
  document.getElementById('evals').innerHTML=e;
 }}catch(err){{document.getElementById('pct').textContent='sem conexao com o monitor';}}
}}
tick(); setInterval(tick,REFRESH);
</script></body></html>"""


def make_handler(log_path: Path, title: str, refresh: int):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path.rstrip("/") in ("", "/index.html"):
                html = PAGE.format(
                    title=title, log=log_path, refresh=refresh
                ).encode("utf-8")
                self._send(html, "text/html; charset=utf-8")
            elif self.path.rstrip("/") == "/status":
                body = json.dumps(parse_log(log_path)).encode("utf-8")
                self._send(body, "application/json")
            else:
                self.send_error(404)

        def log_message(self, *args: object) -> None:
            pass  # silence per-request logging

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="training log path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--refresh", type=int, default=5, help="seconds between polls")
    parser.add_argument("--title", default="Fine-tuning ao vivo")
    args = parser.parse_args()

    handler = make_handler(args.log, args.title, args.refresh)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"monitor on http://{args.host}:{args.port}  (log={args.log})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
