"""Local web app: watch the trained model play itself, with an Elo slider.

    PYTHONPATH=. python scripts/selfplay_server.py            # then open http://localhost:5000
    PYTHONPATH=. python scripts/selfplay_server.py --ckpt checkpoints/base_300k_best.pt --port 5000
"""
import argparse
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root on path
import chess
import chess.svg
from flask import Flask, request, jsonify
from sahformer.training.loop import load_model
from sahformer.play import self_play

app = Flask(__name__)
MODEL = None
MCFG = None
START_CLOCK = 180.0


def make_game(elo, seed, temperature=1.0, top_p=1.0, max_plies=140, size=440):
    board = chess.Board()
    frames = [chess.svg.board(board, size=size, coordinates=True)]
    caps = [{"san": "", "think": 0.0, "white": START_CLOCK, "black": START_CLOCK, "mover": "", "n": 0}]
    result, reason = None, ""
    for rec in self_play(MODEL, max_plies=max_plies, elo=elo, temperature=temperature,
                         top_p=top_p, start_clock=START_CLOCK, seed=seed):
        board.push(rec["move"])
        chk = board.king(board.turn) if board.is_check() else None
        frames.append(chess.svg.board(board, size=size, lastmove=rec["move"], check=chk, coordinates=True))
        caps.append({
            "san": rec["san"], "think": round(rec["think"], 1),
            "white": round(rec["white_clock"], 1), "black": round(rec["black_clock"], 1),
            "mover": rec["mover"], "n": rec["ply"] // 2 + 1,
        })
        if rec["flagged"]:
            result = "0-1" if rec["mover"] == "white" else "1-0"
            reason = f"{rec['mover']} flagged"
            break
    if result is None:
        if board.is_game_over():
            result = board.result()
            reason = ("checkmate" if board.is_checkmate()
                      else "stalemate" if board.is_stalemate()
                      else "draw")
        else:
            result, reason = "*", "move cap reached"
    return frames, caps, result, reason


@app.route("/game")
def game():
    elo = max(600, min(3200, int(request.args.get("elo", 1500))))
    temp = max(0.0, min(1.5, float(request.args.get("temperature", 1.0))))
    top_p = max(0.3, min(1.0, float(request.args.get("top_p", 1.0))))
    seed = random.randint(0, 2**31 - 1)
    frames, caps, result, reason = make_game(elo, seed, temperature=temp, top_p=top_p)
    return jsonify({"frames": frames, "caps": caps, "result": result,
                    "reason": reason, "elo": elo, "seed": seed})


@app.route("/")
def index():
    return PAGE


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Watch it play itself</title>
<style>
  :root{--bg:#0f1115;--panel:#181b22;--line:#272b34;--txt:#e7e9ee;--dim:#9aa2b1;--accent:#6ea8fe;--good:#65d18a;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);
    font-family:-apple-system,Segoe UI,Roboto,sans-serif;display:flex;justify-content:center;padding:24px}
  .wrap{display:flex;gap:22px;max-width:1000px;width:100%;flex-wrap:wrap}
  .boardwrap{position:relative;flex:0 0 auto}
  #board{width:440px;height:440px;background:var(--panel);border-radius:10px;overflow:hidden}
  #board svg{display:block;width:100%;height:100%}
  #thinking{position:absolute;top:10px;left:10px;background:rgba(110,168,254,.92);color:#06101f;
    font-weight:700;font-size:13px;padding:5px 10px;border-radius:20px;opacity:0;transition:opacity .15s}
  #thinking.on{opacity:1}
  .panel{flex:1 1 320px;min-width:300px;background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:18px;display:flex;flex-direction:column;gap:16px}
  h1{font-size:17px;margin:0 0 2px} .sub{color:var(--dim);font-size:12.5px;margin:0}
  .clocks{display:flex;gap:10px}
  .clock{flex:1;background:#11141a;border:1px solid var(--line);border-radius:9px;padding:10px;text-align:center}
  .clock.active{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
  .clock .lbl{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
  .clock .t{font-size:26px;font-variant-numeric:tabular-nums;font-weight:700;margin-top:2px}
  .movebar{background:#11141a;border:1px solid var(--line);border-radius:9px;padding:10px 12px;
    display:flex;justify-content:space-between;align-items:center}
  .movebar .san{font-size:20px;font-weight:700} .movebar .think{color:var(--dim);font-size:13px}
  label{font-size:12.5px;color:var(--dim);display:block;margin-bottom:6px}
  .eloval{color:var(--accent);font-weight:700} input[type=range]{width:100%;accent-color:var(--accent)}
  .row{display:flex;gap:10px} button{flex:1;background:var(--accent);color:#06101f;border:0;
    border-radius:8px;padding:11px;font-weight:700;font-size:14px;cursor:pointer}
  button.ghost{background:#222733;color:var(--txt);border:1px solid var(--line)}
  button:disabled{opacity:.5;cursor:default}
  .moves{background:#11141a;border:1px solid var(--line);border-radius:9px;padding:10px;
    height:150px;overflow:auto;font-size:13px;line-height:1.7;font-variant-numeric:tabular-nums}
  .moves .cur{background:var(--accent);color:#06101f;border-radius:4px;padding:0 4px}
  .result{text-align:center;font-size:15px;font-weight:700;color:var(--good);min-height:20px}
</style></head><body>
<div class="wrap">
  <div class="boardwrap"><div id="board"></div><div id="thinking">thinking…</div></div>
  <div class="panel">
    <div><h1>♟ Watching the model play itself</h1><p class="sub" id="meta">clock-aware blitz · 3+0</p></div>
    <div class="clocks">
      <div class="clock" id="cw"><div class="lbl">White</div><div class="t" id="tw">3:00.0</div></div>
      <div class="clock" id="cb"><div class="lbl">Black</div><div class="t" id="tb">3:00.0</div></div>
    </div>
    <div class="movebar"><span class="san" id="san">—</span><span class="think" id="think"></span></div>
    <div>
      <label>Strength (Elo): <span class="eloval" id="eloval">1500</span></label>
      <input type="range" id="elo" min="800" max="2800" step="100" value="1500">
    </div>
    <div>
      <label>Move randomness: <span class="eloval" id="tval">1.0</span>
        <span style="color:var(--dim);font-weight:400">· 1.0 = full human · lower = cleaner/best</span></label>
      <input type="range" id="temp" min="0.1" max="1.2" step="0.1" value="1">
    </div>
    <div>
      <label>Move filter (top-p): <span class="eloval" id="tpval">0.90</span>
        <span style="color:var(--dim);font-weight:400">· lower = only sensible moves · 1.0 = allow anything</span></label>
      <input type="range" id="topp" min="0.5" max="1.0" step="0.05" value="0.9">
    </div>
    <div>
      <label>Playback speed: <span id="spdval">1.0×</span></label>
      <input type="range" id="spd" min="0.4" max="4" step="0.2" value="1">
    </div>
    <div class="row">
      <button id="new">New game</button>
      <button id="pp" class="ghost">Pause</button>
    </div>
    <div class="moves" id="moves"></div>
    <div class="result" id="res"></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
let G=null, idx=0, playing=false, timer=null;
function mmss(t){t=Math.max(0,t);const m=Math.floor(t/60);const s=(t%60).toFixed(1).padStart(4,'0');return m+':'+s;}
function renderMoves(){
  let h='';for(let i=1;i<G.caps.length;i++){const c=G.caps[i];
    if(c.mover==='white')h+=' '+c.n+'.';
    h+=' <span class="'+(i===idx?'cur':'')+'">'+c.san+'</span>';}
  $('moves').innerHTML=h;const cur=$('moves').querySelector('.cur');if(cur)cur.scrollIntoView({block:'nearest'});
}
function show(i){
  idx=i;const c=G.caps[i];
  $('board').innerHTML=G.frames[i];
  $('tw').textContent=mmss(c.white);$('tb').textContent=mmss(c.black);
  $('cw').classList.toggle('active',c.mover==='white');
  $('cb').classList.toggle('active',c.mover==='black');
  $('san').textContent=i===0?'—':(c.mover==='white'?c.n+'. ':c.n+'… ')+c.san;
  $('think').textContent=i===0?'':c.think.toFixed(1)+'s think';
  renderMoves();
}
function step(){
  if(idx>=G.frames.length-1){finish();return;}
  const next=G.caps[idx+1];
  const spd=parseFloat($('spd').value);
  // dwell scaled by the model's own think-time, clamped so it's watchable
  let dwell=Math.min(3.2,Math.max(0.28,next.think*0.16))/spd*1000;
  $('thinking').classList.toggle('on',next.think>=4.0);
  timer=setTimeout(()=>{$('thinking').classList.remove('on');show(idx+1);if(playing)step();},dwell);
}
function finish(){playing=false;$('pp').textContent='Replay';
  $('res').textContent=G.result+'  ·  '+G.reason;}
function play(){playing=true;$('pp').textContent='Pause';$('res').textContent='';step();}
function pause(){playing=false;clearTimeout(timer);$('pp').textContent='Play';}
async function newGame(){
  clearTimeout(timer);playing=false;
  $('new').disabled=true;$('new').textContent='thinking…';$('res').textContent='';
  $('san').textContent='…';$('think').textContent='';
  try{
    const elo=$('elo').value, temp=$('temp').value, tp=$('topp').value;
    const r=await fetch('/game?elo='+elo+'&temperature='+temp+'&top_p='+tp);G=await r.json();
    $('meta').textContent='clock-aware blitz · 3+0 · playing at Elo '+G.elo;
    idx=0;show(0);play();
  }catch(e){$('res').textContent='error: '+e;}
  finally{$('new').disabled=false;$('new').textContent='New game';}
}
$('elo').addEventListener('input',e=>$('eloval').textContent=e.target.value);
$('temp').addEventListener('input',e=>$('tval').textContent=parseFloat(e.target.value).toFixed(1));
$('topp').addEventListener('input',e=>$('tpval').textContent=parseFloat(e.target.value).toFixed(2));
$('spd').addEventListener('input',e=>$('spdval').textContent=parseFloat(e.target.value).toFixed(1)+'×');
$('new').addEventListener('click',newGame);
$('pp').addEventListener('click',()=>{
  if($('pp').textContent==='Replay'){idx=0;play();}
  else if(playing)pause();else play();
});
newGame();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print(f"loading {args.ckpt} ...")
    MODEL, MCFG = load_model(args.ckpt)
    print(f"loaded (dim={MCFG.dim_vit}, blocks={MCFG.num_blocks}). "
          f"open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, threaded=True)
