"""Part 1 - non-circular benchmark analysis.
Experimental release points digitised from Fig. 5 of Vueba et al. 2004 (see digitize.py):
  F7 = 20% w/w HPMC K100M (training / fitting set)
  F5 = 20% w/w HPMC K15M  (independent cross-grade test set)
Both formulations contain lactose; drug = ketoprofen."""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, tensorflow as tf
from scipy.optimize import curve_fit
tf.keras.utils.set_random_seed(42); tf.config.experimental.enable_op_determinism()

d = pd.read_csv('vueba_fig5_digitised.csv')
t = d.t_h.values.astype(np.float32); F7 = d.F7_K100M.values.astype(np.float32); F5 = d.F5_K15M.values.astype(np.float32)

def metrics(obs, pred, k):
    r = np.asarray(pred)-np.asarray(obs); n = len(obs); rss = float((r**2).sum())
    aic = n*np.log(rss/n)+2*k
    aicc = aic + 2*k*(k+1)/(n-k-1) if n-k-1 > 0 else np.nan
    return dict(RMSE_pp=round(float(np.sqrt(rss/n)),3),
                R2=round(1-rss/float(((obs-obs.mean())**2).sum()),4),
                AIC=round(aic,1), AICc=(round(aicc,1) if np.isfinite(aicc) else None), k=k)

print('### 1. Classical release models fitted to digitised F7 (HPMC K100M)')
M = {'Zero-order  Q = k0 t':                (lambda x,k0: k0*x, [3]),
     'First-order Q = 100(1-e^-k1 t)':      (lambda x,k1: 100*(1-np.exp(-k1*x)), [0.05]),
     'Higuchi     Q = kH t^0.5':            (lambda x,kh: kh*np.sqrt(x), [14]),
     'Korsmeyer-Peppas Q = k t^n':          (lambda x,k,n: k*x**n, [10,0.6]),
     'Peppas-Sahlin Q = kd t^0.45+kr t^0.89':(lambda x,kd,kr: kd*x**0.45+kr*x**0.89, [8,2])}
rows=[]; fits={}
for nm,(f,p0) in M.items():
    p,_ = curve_fit(f, t.astype(float), F7.astype(float), p0=p0, maxfev=200000); fits[nm]=(f,p)
    rows.append(dict(Model=nm, **metrics(F7, f(t.astype(float),*p), len(p0)), params=np.round(p,4).tolist()))
kKP, nKP = fits['Korsmeyer-Peppas Q = k t^n'][1]
print(pd.DataFrame(rows).to_string(index=False))
print('\n  K-P fitted to digitised data : k = %.3f, n = %.3f   (published Table 4: k = 9.926, n = 0.621)'%(kKP,nKP))

def build():
    ini = tf.keras.initializers.GlorotUniform(seed=42)
    return tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(32,activation='tanh',kernel_initializer=ini),
        tf.keras.layers.Dense(32,activation='tanh',kernel_initializer=ini),
        tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer=ini)])
def train(td,yd,epochs=1000,lam=0.5):
    m=build(); opt=tf.keras.optimizers.Adam(0.01)
    kk=tf.constant(kKP/100,tf.float32); nn=tf.constant(nKP,tf.float32)
    tc=tf.convert_to_tensor(np.linspace(0.1,20,60,dtype=np.float32).reshape(-1,1))
    TD=tf.convert_to_tensor(td.reshape(-1,1)); YD=tf.convert_to_tensor((yd/100).reshape(-1,1)); log=[]
    for e in range(epochs):
        with tf.GradientTape() as g:
            with tf.GradientTape() as gi:
                gi.watch(tc); yc=m(tc)
            dy=gi.gradient(yc,tc)
            ld=tf.reduce_mean(tf.square(m(TD)-YD)); lp=tf.reduce_mean(tf.square(dy-kk*nn*tc**(nn-1.0)))
            L=ld+lam*lp
        opt.apply_gradients(zip(g.gradient(L,m.trainable_variables),m.trainable_variables))
        log.append((e+1,float(L),float(ld),float(lp)))
    return m,pd.DataFrame(log,columns=['epoch','total','data','physics'])
pred=lambda m,x: 100*m(tf.convert_to_tensor(x.reshape(-1,1))).numpy().ravel()

print('\n### 2. PINN trained on all 14 digitised F7 points')
m,log=train(t,F7); log.to_csv('pinn_log_F7_digitised.csv',index=False)
print('  final  L_data %.3e   L_physics %.3e'%(log.data.iloc[-1],log.physics.iloc[-1]))
mp=metrics(F7,pred(m,t),1153); print('  fit to F7 :',mp)
rows.append(dict(Model='PINN (physics-informed NN)', **mp, params='1153 weights'))
pd.DataFrame(rows).to_csv('table7_model_comparison.csv',index=False)

print('\n### 3. Physics-constrained temporal extrapolation (train 0.5-8 h -> predict 10-12 h)')
tr=t<=8; te=(t>=10)&(t<=12)
m2,_=train(t[tr],F7[tr]); pe=pred(m2,t[te])
print('  predicted %s   observed %s'%(np.round(pe,2),F7[te]))
print('  held-out 10 & 12 h :',metrics(F7[te],pe,1153))
kp2,_=curve_fit(lambda x,k,n:k*x**n,t[tr].astype(float),F7[tr].astype(float),p0=[10,.6],maxfev=200000)
print('  K-P trained on same window, held-out :',metrics(F7[te],kp2[0]*t[te].astype(float)**kp2[1],2))

print('\n### 4. Cross-grade transfer (PINN trained on F7 K100M -> F5 K15M, no retraining)')
print('  ',metrics(F5,pred(m,t),1153),' mean signed error %.2f pp'%(pred(m,t)-F5).mean())

print('\n### 5. Ablation - does the physics constraint actually buy anything?')
for lam,tag in ((0.5,'PINN  (lambda_phys = 0.5)'),(0.0,'plain NN (lambda_phys = 0, identical architecture)')):
    ms,_=train(t[tr],F7[tr],lam=lam); p=pred(ms,t[te])
    print('  %-48s held-out 10&12 h: pred %s  %s'%(tag,np.round(p,2),metrics(F7[te],p,1153)))
    pf=pred(ms,np.array([16.,20.],dtype=np.float32))
    print('  %-48s far extrapolation 16 & 20 h: pred %s  observed %s'%('',np.round(pf,2),F7[(t==16)|(t==20)]))
