#!/usr/bin/env python3
from pathlib import Path
ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
payload = (ROOT/"notebooks/_density_payload.json").read_text()
HTML = r"""<title>NS5 Binding Density</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f4f5f7;--surface:#fff;--surface2:#eceef1;--line:#d9dde3;--ink:#12151b;--ink2:#5b6572;--accent:#2a78d6;--mono:"IBM Plex Mono",monospace;--sans:"IBM Plex Sans",system-ui,sans-serif;}
:root:not([data-theme="light"]){@media (prefers-color-scheme:dark){--bg:#0e1116;--surface:#161b22;--surface2:#1e252f;--line:#2b333f;--ink:#e8ecf1;--ink2:#9aa5b3;--accent:#3987e5;color-scheme:dark;}}
:root[data-theme="dark"]{--bg:#0e1116;--surface:#161b22;--surface2:#1e252f;--line:#2b333f;--ink:#e8ecf1;--ink2:#9aa5b3;--accent:#3987e5;color-scheme:dark;}
*{box-sizing:border-box} body{background:var(--bg);color:var(--ink);font-family:var(--sans)}
.app{display:grid;grid-template-rows:auto 1fr;height:100%;min-height:100vh}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 18px;padding-top:calc(12px + env(safe-area-inset-top,0px));background:var(--surface);border-bottom:1px solid var(--line)}
h1{font-size:16px;font-weight:600;margin:0}.sub{font-size:11.5px;color:var(--ink2);margin:0}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-left:auto}
.tg{display:inline-flex;align-items:center;gap:6px;font-size:12px;background:var(--surface2);border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;user-select:none}
.tg input{accent-color:var(--accent);margin:0}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
main{position:relative}#v{position:absolute;inset:0}
.stat{position:absolute;left:16px;bottom:16px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 14px;font-size:12px;max-width:380px;box-shadow:0 4px 16px rgba(0,0,0,.12)}
.stat b{font-family:var(--mono);font-weight:500}
</style>
<div class="app">
 <header>
  <div><h1>NS5 Binding Density</h1><p class="sub">~35k Boltz-2 poses, common frame — two pockets, colored by site</p></div>
  <div class="controls">
   <label class="tg"><input type="checkbox" id="c-prot" checked> Protein</label>
   <label class="tg"><span class="dot" style="background:#7b8494"></span><input type="checkbox" id="c-A" checked> Site A (major)</label>
   <label class="tg"><span class="dot" style="background:#2a78d6"></span><input type="checkbox" id="c-B" checked> Site B (SAH)</label>
   <label class="tg"><span class="dot" style="background:#0e9f6e"></span><input type="checkbox" id="c-act" checked> Actives</label>
   <label class="tg"><span class="dot" style="background:#eda100"></span><input type="checkbox" id="c-seen"> Held-out seen</label>
   <label class="tg"><span class="dot" style="background:#4a3aa7"></span><input type="checkbox" id="c-uns"> Held-out unseen</label>
   <label class="tg"><input type="checkbox" id="c-surf"> Surface</label>
  </div>
 </header>
 <main><div id="v"></div><div class="stat" id="stat"></div></main>
</div>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js"></script>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent), M=D.meta;
const bg=()=>getComputedStyle(document.body).getPropertyValue('--bg').trim();
const v=$3Dmol.createViewer('v',{backgroundColor:bg()});
const prot=v.addModel(D.protein,'pdb');
const cl={};
function add(key,color,r,op){cl[key]={m:v.addModel(D.layers[key],'pdb'),color,r,op};}
add('siteA','#7b8494',0.35,0.22); add('siteB','#2a78d6',0.4,0.5);
add('active_all','#0e9f6e',0.6,0.95); add('seen','#eda100',0.75,1.0); add('unseen','#4a3aa7',0.75,1.0);
v.addSphere({center:{x:M.pocket[0],y:M.pocket[1],z:M.pocket[2]},radius:1.6,color:'#e34948'});
v.addLabel('SAH cofactor',{position:{x:M.pocket[0],y:M.pocket[1],z:M.pocket[2]},fontSize:11,backgroundOpacity:.6});
const vis={prot:true,siteA:true,siteB:true,active_all:true,seen:false,unseen:false,surf:false};
function draw(){
  prot.setStyle({}, vis.prot?{cartoon:{color:'spectrum'}}:{});
  v.removeAllSurfaces();
  if(vis.surf) v.addSurface($3Dmol.SurfaceType.VDW,{opacity:.5,color:'#c9ced6'},{model:prot});
  for(const k in cl){cl[k].m.setStyle({}, vis[k]?{sphere:{radius:cl[k].r,color:cl[k].color,opacity:cl[k].op}}:{});}
  v.render();
}
[['c-prot','prot'],['c-A','siteA'],['c-B','siteB'],['c-act','active_all'],['c-seen','seen'],['c-uns','unseen'],['c-surf','surf']]
 .forEach(([id,k])=>document.getElementById(id).onchange=e=>{vis[k]=e.target.checked;draw();});
const A=M.siteA,B=M.siteB;
document.getElementById('stat').innerHTML=
 `<b>${M.n_total.toLocaleString()}</b> poses split across <b>two</b> pockets ${M.sep.toFixed(0)} Å apart:<br>`+
 `Site A (major): <b>${A.pct}%</b>, active rate ${A.active_rate}% · Site B = SAH pocket: <b>${B.pct}%</b> (${M.distB_sah.toFixed(1)} Å from SAH), active rate ${B.active_rate}%<br>`+
 `neither site is enriched for actives; held-out seen & unseen scaffolds split the same way (${M.seen_pctA.toFixed(0)}% vs ${M.unseen_pctA.toFixed(0)}% into Site A)`;
draw(); v.zoomTo(); v.zoom(1.1); v.render();
new MutationObserver(()=>{v.setBackgroundColor(bg());v.render();}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script>
"""
out = ROOT/"notebooks/ns5_density_viewer.html"
out.write_text(HTML.replace("__PAYLOAD__", payload))
print("wrote", out, out.stat().st_size//1024, "KB")
