"""Does NBM (Fri 12Z IFC/CIG/VIS) add to the 4-model ensemble at the Friday lead? Rolling-origin CV 2025-2026."""
import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'code'); 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
A=pd.read_parquet('data/audit_lead3.parquet')   # has p1,p2c from ensemble, y1,y2
E=pd.read_parquet('data/nbm_eval.parquet'); E=E[(E.lead_h>48)&(E.lead_h<=72)].set_index('date')[['ifc_max','ifv_max','cig_min','vis_min']]
# rebuild design to stack: use ensemble out-of-sample p1 as a feature + NBM features (stacking with time-ordered CV)
D=A.join(E,how='inner').dropna(); D['lp1']=np.log(D.p1/(1-D.p1)); D['lcig']=np.log(D.cig_min.clip(100,20000)); D['lvis']=np.log(D.vis_min.clip(0.1,10))
print("overlap days:",len(D),"below:",D.y2.sum(),"marginal+:",D.y1.sum())
def cv(cols,y):
    P=pd.Series(index=D.index,dtype=float)
    for q in pd.date_range('2025-04-01','2026-10-01',freq='QS'):
        tr=D.index<q; te=(D.index>=q)&(D.index<q+pd.offsets.QuarterBegin(1))
        if te.sum()==0 or y[tr].sum()<5: continue
        P[te]=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000)).fit(D.loc[tr,cols],y[tr]).predict_proba(D.loc[te,cols])[:,1]
    return P.dropna()
for yname in ['y1','y2']:
    y=D[yname]
    for lab,cols in [('ensemble only (recalibrated)',['lp1']),('NBM only',['ifc_max','ifv_max','lcig','lvis']),('ensemble + NBM',['lp1','ifc_max','ifv_max','lcig','lvis'])]:
        P=cv(cols,y); yy=y[P.index]
        print(f"{yname} {lab:32s} n={len(P)} AUC={roc_auc_score(yy,P):.3f} Brier={brier_score_loss(yy,P):.4f}")
    yy=y[cv(['lp1'],y).index]; print(f"{yname} raw ensemble p (no refit)          AUC={roc_auc_score(yy,D.loc[yy.index,'p1' if yname=='y1' else 'p2c']):.3f}")
