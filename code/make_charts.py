import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
for f in os.listdir(os.path.expanduser("~/Library/Fonts"))+os.listdir("/Library/Fonts"):
    if f.startswith("SF-Pro-Display"):
        for d in (os.path.expanduser("~/Library/Fonts"),"/Library/Fonts"):
            if os.path.exists(os.path.join(d,f)): fm.fontManager.addfont(os.path.join(d,f))
plt.rcParams.update({"font.family":"SF Pro Display","font.size":13,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#c9ccd3","axes.labelcolor":"#3a3f4b","xtick.color":"#5b6170","ytick.color":"#5b6170"})
ME="#2563eb"; OT="#9aa1ae"; INK="#111827"; MUTED="#6b7280"
def title(ax,t,s):
    ax.set_title(t,loc="left",fontsize=19,fontweight="bold",color=INK,pad=34)
    ax.text(0,1.035,s,transform=ax.transAxes,fontsize=12.5,color=MUTED)
def save(fig,n): fig.savefig(f"assets/{n}.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close(fig)

# 1. Accuracy vs context (the hero)
pts=[("Papr, gpt-5-mini",15,94.4,True,(14,-34)),("Papr High (top 5)",16.5,95.4,True,(14,18)),("Papr High (top 20)",60,96.0,True,(-14,14)),
     ("Hindsight",43.6,94.6,False,(14,-30))]
fig,ax=plt.subplots(figsize=(10,6))
ax.axvspan(0,22,color=ME,alpha=0.06,lw=0)
ax.text(1.5,92.65,"Papr top-5 configs:\na third of Hindsight's context",color=ME,fontsize=11.5,va="bottom")
for n,x,y,me,o in pts:
    ax.scatter(x,y,s=190 if me else 130,color=ME if me else OT,zorder=3,edgecolor="white",linewidth=1.5)
    ax.annotate(f"{n}\n{y:.1f}%  ·  {x:g}k tokens",(x,y),xytext=o,textcoords="offset points",fontsize=11.5,arrowprops=dict(arrowstyle="-",color="#c9ccd3",lw=0.8),
        color=INK if me else MUTED,fontweight="bold" if me else "normal",ha="right" if o[0]<0 else "left",va="bottom" if o[1]>0 else "top")
ax.annotate("Full history in the prompt:\n60.2% at 112k tokens (off chart)",(70,92.6),xytext=(-8,8),textcoords="offset points",ha="right",va="bottom",fontsize=11,color=MUTED)
ax.set_xlim(0,70); ax.set_ylim(92.5,96.8)
ax.set_xlabel("Context sent to the answer model per question (thousand tokens)"); ax.set_ylabel("LongMemEval-S accuracy (%)")
ax.grid(axis="y",color="#eef0f3")
title(ax,"Top accuracy at a fraction of the context","Accuracy vs tokens per question, all 500 questions. Up and to the left is better.")
save(fig,"accuracy-vs-tokens")

# 2. Leaderboard bars
rows=[("Papr High (top 20)",96.0,1),("Chronos High",95.6,0),("Papr High (top 5)",95.4,1),("Mastra OM (gpt-5-mini)",94.9,0),("Hindsight*",94.6,0),
      ("Papr (gpt-5-mini)",94.4,1),("Chronos Low",92.6,0),("Honcho",90.4,0),("Zep",71.2,0),("Full history in prompt",60.2,0)][::-1]
fig,ax=plt.subplots(figsize=(10,6.4))
b=ax.barh([r[0] for r in rows],[r[1] for r in rows],color=[ME if r[2] else "#d5d9e0" for r in rows],height=0.62)
for r,bar in zip(rows,b): ax.text(r[1]+0.6,bar.get_y()+bar.get_height()/2,f"{r[1]:.1f}%",va="center",fontsize=12,color=INK if r[2] else MUTED,fontweight="bold" if r[2] else "normal")
for t,r in zip(ax.get_yticklabels(),rows): t.set_color(INK if r[2] else MUTED); t.set_fontweight("bold" if r[2] else "normal")
ax.set_xlim(50,100); ax.set_xlabel("Accuracy (%)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
title(ax,"LongMemEval-S: published results","500 questions. Each team's reported number; judge gpt-4o except *Hindsight (Gemini 2.5 Flash Lite).")
save(fig,"leaderboard")

# 3. Tokens per question
tk=[("Papr (gpt-5-mini)",15,1),("Papr High (top 5)",16.5,1),("Hindsight",43.6,0),("Papr High (top 20)",60,1),("Full history in prompt",112,0)][::-1]
fig,ax=plt.subplots(figsize=(10,4.6))
b=ax.barh([r[0] for r in tk],[r[1] for r in tk],color=[ME if r[2] else "#d5d9e0" for r in tk],height=0.6)
for r,bar in zip(tk,b): ax.text(r[1]+1.2,bar.get_y()+bar.get_height()/2,f"{r[1]:g}k",va="center",fontsize=12,color=INK if r[2] else MUTED,fontweight="bold" if r[2] else "normal")
for t,r in zip(ax.get_yticklabels(),tk): t.set_color(INK if r[2] else MUTED)
ax.set_xlim(0,125); ax.set_xlabel("Tokens sent to the answer model per question (thousands)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
title(ax,"Tokens per question","Fewer tokens is lower cost and faster answers on every call. Average over all 500 questions.")
save(fig,"tokens")

# 4. Top 5 vs top 20 by category
cats=["Temporal\nreasoning","Multi-\nsession","Knowledge\nupdate","Preference","Single-session\nuser","Single-session\nassistant"]
k5=[123,122,76,30,70,56]; k20=[130,121,74,29,70,56]; tot=[133,133,78,30,70,56]
import numpy as np
x=np.arange(len(cats)); w=0.38
fig,ax=plt.subplots(figsize=(10,5.4))
a=ax.bar(x-w/2,[100*a/t for a,t in zip(k5,tot)],w,color="#93b4f5",label="Top 5 sessions  (16.5k tokens)")
c=ax.bar(x+w/2,[100*a/t for a,t in zip(k20,tot)],w,color=ME,label="Top 20 sessions  (60k tokens)")
for i in range(len(cats)):
    d=k20[i]-k5[i]
    if d: ax.text(x[i]+w/2,100*k20[i]/tot[i]+0.4,f"{d:+d}",ha="center",fontsize=12,fontweight="bold",color=ME if d>0 else "#b91c1c")
ax.set_xticks(x,cats,fontsize=11.5); ax.set_ylim(85,101.5); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y",color="#eef0f3")
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2,fontsize=11.5)
title(ax,"Top 5 vs top 20 sessions","More sessions helps date math (+7) and costs a little elsewhere. Net +3 of 500.")
save(fig,"top5-vs-top20")

# 5. Retrieval pipeline latency
fig,ax=plt.subplots(figsize=(10,3.1))
ax.barh([0],[0.17],color=ME,height=0.5); ax.barh([0],[0.01],left=0.17,color="#1e40af",height=0.5)
ax.barh([0],[1.8],left=0.18,color="#c7d6fb",height=0.5)
ax.text(1.08,0,"Gemini 2.5 Flash rerank  ·  1.8 s median",ha="center",va="center",fontsize=11.5,color=INK)
ax.annotate("papr-embed 170 ms + search <1 ms",(0.09,-0.25),xytext=(0,-14),textcoords="offset points",ha="left",va="top",fontsize=11.5,color=ME,fontweight="bold",arrowprops=dict(arrowstyle="-",color=ME,lw=0.8))
ax.text(0.0,0.4,"embedding only: 91.0% with gpt-5-mini",fontsize=11,color=ME)
ax.text(1.98,0.4,"with rerank: 94.4% to 96.0%",fontsize=11,color=INK,ha="right")
ax.set_xlim(0,2.05); ax.set_ylim(-0.75,0.7); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("Seconds per question (median)")
title(ax,"Retrieval time per question","Papr embedding + search, then an optional reranker.")
save(fig,"latency")
print(os.listdir("img"))
