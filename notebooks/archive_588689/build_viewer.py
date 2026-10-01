#!/usr/bin/env python3
"""Assemble the interactive 3D docking viewer artifact (embeds aligned PDBs + meta)."""
import json
from pathlib import Path

R = Path("/global/scratch/users/sergiomar10/boltzaff/results/runs/588689")
VD = R / "top_poses/viewer_data"
OUT = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/ns5_docking_viewer.html")

meta = json.loads((VD / "meta.json").read_text())
poses = {f"{m['rank']}": (VD / f"pose_{m['rank']:02d}.pdb").read_text() for m in meta}
ref = (VD / "ref_3EVG.pdb").read_text()
data = {"meta": meta, "poses": poses, "ref": ref}
payload = json.dumps(data)

HTML = r"""<title>NS5 Docking Explorer</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#f4f5f7; --surface:#ffffff; --surface-2:#eceef1; --line:#d9dde3;
  --ink:#12151b; --ink-2:#5b6572; --accent:#2a78d6; --accent-soft:#dce8fa;
  --active:#0e9f6e; --active-soft:#d5f0e5; --inactive:#8a94a3; --inactive-soft:#e7e9ed;
  --shadow:0 1px 3px rgba(16,21,31,.08),0 4px 16px rgba(16,21,31,.06);
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,sans-serif;
}
:root:not([data-theme="light"]){ @media (prefers-color-scheme:dark){
  --bg:#0e1116; --surface:#161b22; --surface-2:#1e252f; --line:#2b333f;
  --ink:#e8ecf1; --ink-2:#9aa5b3; --accent:#3987e5; --accent-soft:#16283f;
  --active:#2ec98b; --active-soft:#123027; --inactive:#7d8896; --inactive-soft:#20262f;
  --shadow:0 1px 3px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.35); color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#0e1116; --surface:#161b22; --surface-2:#1e252f; --line:#2b333f;
  --ink:#e8ecf1; --ink-2:#9aa5b3; --accent:#3987e5; --accent-soft:#16283f;
  --active:#2ec98b; --active-soft:#123027; --inactive:#7d8896; --inactive-soft:#20262f;
  --shadow:0 1px 3px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.35); color-scheme:dark;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);}
.app{display:grid;grid-template-columns:300px 1fr;grid-template-rows:auto 1fr;height:100%;min-height:100vh;}
header{grid-column:1/3;display:flex;flex-wrap:wrap;align-items:center;gap:16px;
  padding:12px 20px;padding-top:calc(12px + env(safe-area-inset-top,0px));
  background:var(--surface);border-bottom:1px solid var(--line);}
.brand{display:flex;flex-direction:column;gap:1px;margin-right:auto}
.brand h1{font-size:16px;font-weight:600;margin:0;letter-spacing:.2px}
.brand p{margin:0;font-size:11.5px;color:var(--ink-2)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.toggle{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--ink-2);
  background:var(--surface-2);border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;user-select:none}
.toggle input{accent-color:var(--accent);margin:0}
.toggle[data-on="1"]{color:var(--ink);border-color:var(--accent);background:var(--accent-soft)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{font-family:var(--sans);font-size:12px;border:0;background:var(--surface-2);color:var(--ink-2);padding:6px 10px;cursor:pointer}
.seg button.on{background:var(--accent);color:#fff}
aside{grid-row:2;overflow-y:auto;background:var(--surface);border-right:1px solid var(--line);padding:10px}
.rail-head{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-2);padding:4px 6px 8px}
.item{display:grid;grid-template-columns:26px 1fr auto;gap:8px;align-items:center;
  padding:8px 8px;border-radius:9px;cursor:pointer;border:1px solid transparent}
.item:hover{background:var(--surface-2)}
.item.sel{background:var(--accent-soft);border-color:var(--accent)}
.rank{font-family:var(--mono);font-size:12px;color:var(--ink-2);text-align:right}
.cid{font-family:var(--mono);font-size:12.5px;font-weight:500}
.sub{font-size:11px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.pill{font-size:10px;font-weight:600;padding:2px 7px;border-radius:20px;letter-spacing:.02em;white-space:nowrap}
.pill.a{background:var(--active-soft);color:var(--active)}
.pill.i{background:var(--inactive-soft);color:var(--inactive)}
main{grid-row:2;position:relative;overflow:hidden}
#viewer{position:absolute;inset:0}
.detail{position:absolute;left:16px;bottom:16px;z-index:5;background:var(--surface);border:1px solid var(--line);
  border-radius:12px;box-shadow:var(--shadow);padding:12px 14px;max-width:min(340px,calc(100% - 32px));}
.detail .row{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline}
.detail .big{font-family:var(--mono);font-size:15px;font-weight:500}
.metrics{display:flex;gap:16px;margin-top:8px;flex-wrap:wrap}
.metric .k{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-2)}
.metric .v{font-family:var(--mono);font-size:14px;font-variant-numeric:tabular-nums}
.legend{position:absolute;right:16px;top:12px;z-index:5;font-size:11px;color:var(--ink-2);
  background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 10px;display:flex;flex-direction:column;gap:5px}
.legend b{color:var(--ink);font-weight:500}
.sw{display:inline-block;width:20px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}
@media (max-width:760px){
  .app{grid-template-columns:1fr;grid-template-rows:auto auto 1fr}
  aside{grid-row:2;grid-column:1;max-height:34vh;border-right:0;border-bottom:1px solid var(--line);
    display:flex;gap:8px;overflow-x:auto;overflow-y:hidden}
  .rail-head{display:none}
  .item{grid-template-columns:auto;min-width:132px;grid-auto-rows:min-content}
  main{grid-row:3}
}
</style>

<div class="app">
  <header>
    <div class="brand">
      <h1>NS5 Docking Explorer</h1>
      <p>Top-24 Boltz-2 poses · Dengue-2 NS5 methyltransferase · aligned to crystal 3EVG</p>
    </div>
    <div class="controls">
      <label class="toggle" id="t-overlay"><input type="checkbox"> 3EVG crystal overlay</label>
      <label class="toggle" data-on="1" id="t-pocket"><input type="checkbox" checked> Pocket residues</label>
      <label class="toggle" id="t-spin"><input type="checkbox"> Spin</label>
      <span class="seg" id="seg-color">
        <button data-m="spectrum" class="on">Spectrum</button>
        <button data-m="solid">Solid</button>
      </span>
    </div>
  </header>

  <aside id="rail"><div class="rail-head">Ranked by Boltz-2 score</div></aside>

  <main>
    <div id="viewer"></div>
    <div class="legend">
      <span><span class="sw" style="background:linear-gradient(90deg,#2a78d6,#eda100,#e34948)"></span>protein N→C</span>
      <span><b>green</b> = confirmed active</span>
      <span><b>grey</b> = crystal 3EVG (overlay)</span>
    </div>
    <div class="detail" id="detail"></div>
  </main>
</div>

<script src="https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js"></script>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const meta = DATA.meta.sort((a,b)=>a.rank-b.rank);
let viewer, current=null, colorMode='spectrum';
const state={overlay:false,pocket:true,spin:false};

function el(t,c,txt){const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e;}

// build rail
const rail=document.getElementById('rail');
meta.forEach(m=>{
  const it=el('div','item'); it.dataset.rank=m.rank;
  it.appendChild(el('div','rank','#'+m.rank));
  const mid=el('div'); mid.appendChild(el('div','cid',String(m.cid)));
  mid.appendChild(el('div','sub','p='+m.score.toFixed(3)+' · conf '+m.confidence.toFixed(2)));
  it.appendChild(mid);
  it.appendChild(el('span','pill '+(m.active===1?'a':'i'),m.active===1?'ACTIVE':'inactive'));
  it.addEventListener('click',()=>select(m.rank));
  rail.appendChild(it);
});

function initViewer(){
  const bg=getComputedStyle(document.body).getPropertyValue('--bg').trim();
  viewer=$3Dmol.createViewer('viewer',{backgroundColor:bg});
  select(1);
}
function styleModel(model){
  if(colorMode==='spectrum') model.setStyle({}, {}); // reset then per-selection below
}
function select(rank){
  current=rank;
  document.querySelectorAll('.item').forEach(n=>n.classList.toggle('sel',n.dataset.rank==rank));
  viewer.clear();
  const m=meta.find(x=>x.rank==rank);
  // reference crystal (optional)
  if(state.overlay){
    const r=viewer.addModel(DATA.ref,'pdb');
    r.setStyle({}, {cartoon:{color:'#9aa3b0',opacity:0.55}});
  }
  const mdl=viewer.addModel(DATA.poses[String(rank)],'pdb');
  const prot={cartoon: colorMode==='spectrum' ? {colorscheme:{prop:'resi',gradient:'roygb'}} : {color:'#2a78d6'}};
  mdl.setStyle({}, prot);
  // ligand
  mdl.setStyle({resn:'LIG'}, {stick:{colorscheme:'greenCarbon',radius:0.2}});
  // pocket residues within 4.5A of ligand
  if(state.pocket){
    const near=mdl.selectedAtoms({within:{distance:4.5,sel:{resn:'LIG'}}});
    const resis=[...new Set(near.filter(a=>a.resn!=='LIG').map(a=>a.resi))];
    mdl.setStyle({resi:resis,resn:'LIG',invert:true},{cartoon:prot.cartoon,stick:{radius:0.12,opacity:0.9}});
  }
  viewer.zoomTo({resn:'LIG'});
  viewer.zoom(0.55);
  viewer.spin(state.spin?'y':false);
  viewer.render();
  renderDetail(m);
}
function renderDetail(m){
  const d=document.getElementById('detail');
  d.innerHTML='';
  const row=el('div','row');
  row.appendChild(el('span','big','CID '+m.cid));
  row.appendChild(el('span','pill '+(m.active===1?'a':'i'),m.active===1?'CONFIRMED ACTIVE':'inactive'));
  d.appendChild(row);
  const met=el('div','metrics');
  [['Rank','#'+m.rank],['Boltz-2 p',m.score.toFixed(3)],['Confidence',m.confidence.toFixed(2)],
   ['pLDDT',m.plddt.toFixed(2)],['ligand ipTM',m.ligand_iptm.toFixed(2)]].forEach(([k,v])=>{
    const w=el('div','metric'); w.appendChild(el('div','k',k)); w.appendChild(el('div','v',v)); met.appendChild(w);
  });
  d.appendChild(met);
}
// controls
function wireToggle(id,key){const l=document.getElementById(id);l.addEventListener('change',e=>{
  state[key]=e.target.checked;l.dataset.on=e.target.checked?'1':'';if(current)select(current);});}
wireToggle('t-overlay','overlay'); wireToggle('t-pocket','pocket');
document.getElementById('t-spin').addEventListener('change',e=>{state.spin=e.target.checked;
  document.getElementById('t-spin').dataset.on=e.target.checked?'1':'';viewer.spin(state.spin?'y':false);});
document.getElementById('seg-color').addEventListener('click',e=>{
  const b=e.target.closest('button');if(!b)return;colorMode=b.dataset.m;
  document.querySelectorAll('#seg-color button').forEach(x=>x.classList.toggle('on',x===b));
  if(current)select(current);});
// re-tint background if theme changes
const mo=new MutationObserver(()=>{if(viewer){viewer.setBackgroundColor(getComputedStyle(document.body).getPropertyValue('--bg').trim());viewer.render();}});
mo.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});

if(window.$3Dmol) initViewer();
else window.addEventListener('load',initViewer);
</script>
"""

OUT.write_text(HTML.replace("__PAYLOAD__", payload))
kb = OUT.stat().st_size / 1024
print(f"wrote {OUT} ({kb:.0f} KB, {len(meta)} poses)")
