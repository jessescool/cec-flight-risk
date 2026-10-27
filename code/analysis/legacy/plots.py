import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
OUT='/Users/jcool/Desktop/cec-flight-risk/reports/figs/'
A3=pd.read_parquet('data/audit_lead3.parquet'); A4=pd.read_parquet('data/audit_lead4.parquet')
# 1 ROC
fig,ax=plt.subplots(figsize=(4.2,4.2))
for A,lab,c in [(A3,'Friday lead (3 d)','#2a78d6'),(A4,'Thursday lead (4 d)','#eb6834')]:
    f,t,_=roc_curve(A.y2,A.p2c); ax.plot(f,t,c,lw=2,label=f'{lab}  AUC {roc_auc_score(A.y2,A.p2c):.2f}')
f,t,_=roc_curve(A3.y2,A3.c2); ax.plot(f,t,'#9aa3a9',lw=1.5,ls='--',label=f'Monthly climatology  AUC {roc_auc_score(A3.y2,A3.c2):.2f}')
ax.plot([0,1],[0,1],color='#d9dee1',lw=1); ax.set_xlabel('False-alarm rate (good evenings flagged)'); ax.set_ylabel('Hit rate (below-minimums evenings flagged)'); ax.legend(frameon=False,fontsize=8,loc='lower right'); ax.set_title('Out-of-sample ROC, 2025–2026 (n=440, 34 events)',fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'roc.svg'); fig.savefig(OUT+'roc.png',dpi=110); plt.close()
# 2 reliability
fig,ax=plt.subplots(figsize=(4.2,4.2))
for A,lab,c in [(A3,'Friday lead',"#2a78d6"),(A4,'Thursday lead','#eb6834')]:
    b=pd.cut(A.p2c,[0,0.02,0.05,0.1,0.2,0.4,1]); g=A.groupby(b,observed=True).agg(p=('p2c','mean'),o=('y2','mean'),n=('y2','size'))
    ax.plot(g.p,g.o,'o-',color=c,lw=1.5,ms=5,label=lab)
    for _,r in g.iterrows(): ax.annotate(int(r.n),(r.p,r.o),textcoords='offset points',xytext=(5,-10),fontsize=7,color='#52514e')
ax.plot([0,0.7],[0,0.7],color='#9aa3a9',lw=1,ls='--'); ax.set_xlabel('Forecast P(below minimums)'); ax.set_ylabel('Observed frequency'); ax.set_xlim(0,0.7); ax.set_ylim(0,0.7); ax.legend(frameon=False,fontsize=8); ax.set_title('Reliability (bin counts beside points)',fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'reliability.svg'); fig.savefig(OUT+'reliability.png',dpi=110); plt.close()
# 3 tradeoff curve: weekends skipped vs P(below | flew), Friday lead, Oct-May vs all
fig,ax=plt.subplots(figsize=(4.6,3.6))
for sel,lab,c in [(A3.index==A3.index,'All months','#2a78d6'),(A3.month.isin([10,11,12,1,2,3,4,5]),'Oct–May','#1baf7a')]:
    Z=A3[sel]; xs=[];ys=[]
    for thr in np.linspace(0.02,0.8,60):
        g=Z.p1<thr; xs.append((~g).mean()*100); ys.append(Z.y2[g].mean()*100 if g.sum() else np.nan)
    ax.plot(xs,ys,color=c,lw=2,label=lab)
    for thr,mk in [(0.15,'Y'),(0.30,'R')]:
        g=Z.p1<thr; ax.plot((~g).mean()*100,Z.y2[g].mean()*100,'o',color=c,ms=5); ax.annotate(f'P≥{thr}',((~g).mean()*100,Z.y2[g].mean()*100),textcoords='offset points',xytext=(6,4),fontsize=7)
ax.set_xlabel('Weekends skipped (%)'); ax.set_ylabel('P(below minimums | you flew) (%)'); ax.legend(frameon=False,fontsize=8); ax.set_title('Friday-lead tradeoff, rolling validation 2025–2026',fontsize=10); ax.set_xlim(0,70)
fig.tight_layout(); fig.savefig(OUT+'tradeoff.svg'); fig.savefig(OUT+'tradeoff.png',dpi=110); plt.close()
print('ok')
