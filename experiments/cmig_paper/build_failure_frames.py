"""Failure-mechanism figure: for each unrescued / instructive case, three hand-selected frames
(indices chosen from dense review strips) with the reviewed mechanism. CPU-only."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import cv2, numpy as np, subprocess, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; from figstyle import apply_style, OKABE; apply_style()
ROWS=[("20-V2",[908,1136,2499],"Collapsing lumen — round → slit → puckered star; pale, ring-poor mucosa (dynamic wall motion)"),
      ("14-V1",[714,816,1224],"Posterior wall bulges into a crescent lumen along the whole ringed segment (dynamic wall motion)"),
      ("31-V1",[541,974,1299],"Clean subglottic rings, then a smooth wall bulge dominates the field for most of the pass"),
      ("16-V2",[708,816,876],"Scope nearly static at the carina while the lumen changes shape → SfM splits into interleaved rigid models"),
      ("26-V2",[1895,1912,1925],"Transient posterior-wall collapse (f1912–1918) breaks the chain; both sides reconstruct (RESCUED)"),
      ("32-V2",[608,1064,1520],"Inflamed, irregular mucosa + intruding wall fold + glare from wall contact"),
      ("24-V2",[5818,6492,8851],"Compressed crescent lumen, froth/secretions, oedematous cobblestoned arytenoids; prolonged laryngeal manipulation"),
      ("24-V1",[608,2253,2619],"No lumen: wet tissue against the lens with glare and froth; rounded mass and irregular tissue (lesion)"),
      ("10-V1",[533,853,1386],"Suction catheter in the lumen for most of the pass; secretions; compliant bulging wall"),
      ("1-V3",[133,466,733],"Lens haze (low contrast) with an instrument at the cords; short traversal"),
      ("23-V2",[630,1892,2522],"Not an airway traversal: inside the endotracheal tube (printed markings), drapes, operating room")]
def find_video(c):
    u=c.replace("-","_")
    for p in [f"{c}.mp4",f"{c} - *.mp4",f"{u}.mp4",f"*{c}*.mp4"]:
        r=[x for x in subprocess.run(["find","/home/mi3dr/dataset","-type","f","-iname",p,"!","-iname","*alib*"],capture_output=True,text=True).stdout.split("\n") if x]
        if r: return r[0]
tiles={}
for c,idx,_ in ROWS:
    cap=cv2.VideoCapture(find_video(c)); fi=0; want=set(idx)
    while True:
        ok,f=cap.read()
        if not ok or fi>max(idx): break
        if fi in want: tiles[(c,fi)]=cv2.cvtColor(cv2.resize(f,(480,270)),cv2.COLOR_BGR2RGB)
        fi+=1
    cap.release(); print(c,"frames extracted",flush=True)
for tag, rows in (("p1", ROWS[:6]), ("p2", ROWS[6:])):
    fig=plt.figure(figsize=(13.5,2.35*len(rows)+0.3)); gs=fig.add_gridspec(len(rows),3,hspace=0.55,wspace=0.03)
    for i,(c,idx,label) in enumerate(rows):
        for j,fi in enumerate(idx):
            ax=fig.add_subplot(gs[i,j]); ax.imshow(tiles[(c,fi)]); ax.set_axis_off(); ax.set_title(f"f{fi}",fontsize=9.5,color="0.4",pad=1)
        col=OKABE["green"] if "RESCUED" in label else ("0.3")
        fig.text(0.5,1-(i+0.02)/len(rows)*0.985,f"{c} — {label}",ha="center",va="top",fontsize=11,fontweight="bold",color=col)
    fig.subplots_adjust(left=0.01,right=0.99,top=0.975,bottom=0.01); fig.savefig(f"runs/own_data/renders/new/fig_failure_frames_{tag}.png",dpi=125); plt.close(fig); print(f"saved fig_failure_frames_{tag}.png")
