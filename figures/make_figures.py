# -*- coding: utf-8 -*-
"""Regenerates Figures 4, 6, 7 and 8 from the digitised benchmark and the fitted
RSM models.  Output sizes match the display sizes used in the manuscript.
Run from the directory holding vueba_fig5_digitised.csv and pinn_log_F7_digitised.csv:
    python3 -I make_figures.py
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, tensorflow as tf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

tf.keras.utils.set_random_seed(42); tf.config.experimental.enable_op_determinism()
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.linewidth': 0.8, 'savefig.dpi': 300})
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv('vueba_fig5_digitised.csv')
t = d.t_h.values.astype(np.float32)
F7 = d.F7_K100M.values.astype(np.float32)
F5 = d.F5_K15M.values.astype(np.float32)

PS = lambda x, kd, kr: kd * x ** 0.45 + kr * x ** 0.89
ps7, _ = curve_fit(PS, t.astype(float), F7.astype(float), p0=[8, 2], maxfev=200000)
ps5, _ = curve_fit(PS, t.astype(float), F5.astype(float), p0=[8, 2], maxfev=200000)
kKP, nKP = curve_fit(lambda x, k, n: k * x ** n, t.astype(float), F7.astype(float),
                     p0=[10, .6], maxfev=200000)[0]

# ---------------------------------------------------------------- Figure 4
fig, ax = plt.subplots(figsize=(7.5, 5.0))
tt = np.linspace(0.3, 20, 400)
ax.plot(t, F7, 'o', ms=5, mfc='#1f77b4', mec='k', mew=.5, zorder=5,
        label='Digitised, HPMC K100M 20% (F7) [44]')
ax.plot(t, F5, 's', ms=5, mfc='#d62728', mec='k', mew=.5, zorder=5,
        label='Digitised, HPMC K15M 20% (F5) [44]')
ax.plot(tt, PS(tt, *ps7), '-', c='#1f77b4', lw=1.6,
        label='Peppas$-$Sahlin fit, F7 (R$^2$ = 0.9991)')
ax.plot(tt, PS(tt, *ps5), '-', c='#d62728', lw=1.6,
        label='Peppas$-$Sahlin fit, F5 (R$^2$ = 0.9991)')
# Levina K-P profiles, plotted only over their 5-60% fitting range (t in minutes)
for (k, n, lbl, c) in [(3.5188, .5822, '4 kN', '#2ca02c'),
                       (1.2356, .7268, '10 kN', '#9467bd'),
                       (1.2152, .7367, '14 kN', '#8c564b')]:
    tm = np.linspace((5 / k) ** (1 / n), (60 / k) ** (1 / n), 200)
    ax.plot(tm / 60.0, k * tm ** n, '--', c=c, lw=1.2,
            label=f'HPMC K4M 20%, {lbl} [49]')
for tmin, lbl, c in [(246, 'PEO WSR 303', '#ff7f0e'), (357, 'HPMC K100M', '#17becf')]:
    ax.plot(tmin / 60.0, 75, 'v', ms=8, mfc=c, mec='k', mew=.6, zorder=6)
    ax.annotate(f'$t_{{75}}$, {lbl} [50]', (tmin / 60.0, 75), textcoords='offset points',
                xytext=(6, 6), fontsize=7.5)
ax.set_xlabel('Time (h)'); ax.set_ylabel('Cumulative release (%)')
ax.set_xlim(0, 20); ax.set_ylim(0, 90)
ax.grid(alpha=.25, lw=.5)
ax.legend(fontsize=7, loc='lower right', framealpha=.95)
fig.tight_layout(); fig.savefig(f'{OUT}/figure4.png'); plt.close(fig)
print('figure4 done')

# ---------------------------------------------------------------- Figure 6
def build():
    ini = tf.keras.initializers.GlorotUniform(seed=42)
    return tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=ini),
        tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=ini),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=ini)])

def train(td, yd, epochs=1000, lam=0.5):
    m = build(); opt = tf.keras.optimizers.Adam(0.01)
    kk = tf.constant(kKP / 100, tf.float32); nn = tf.constant(nKP, tf.float32)
    tc = tf.convert_to_tensor(np.linspace(0.1, 20, 60, dtype=np.float32).reshape(-1, 1))
    TD = tf.convert_to_tensor(td.reshape(-1, 1)); YD = tf.convert_to_tensor((yd / 100).reshape(-1, 1))
    for e in range(epochs):
        with tf.GradientTape() as g:
            with tf.GradientTape() as gi:
                gi.watch(tc); yc = m(tc)
            dy = gi.gradient(yc, tc)
            ld = tf.reduce_mean(tf.square(m(TD) - YD))
            lp = tf.reduce_mean(tf.square(dy - kk * nn * tc ** (nn - 1.0)))
            L = ld + lam * lp
        opt.apply_gradients(zip(g.gradient(L, m.trainable_variables), m.trainable_variables))
    return m
pred = lambda m, x: 100 * m(tf.convert_to_tensor(x.reshape(-1, 1))).numpy().ravel()

log = pd.read_csv('pinn_log_F7_digitised.csv')
m_full = train(t, F7)
tr = t <= 8
m_phys = train(t[tr], F7[tr], lam=0.5)
m_plain = train(t[tr], F7[tr], lam=0.0)

fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.02))
a = axes[0]
a.semilogy(log.epoch, log.total, c='k', lw=1.1, label=r'Total: $L_{data}+0.5\,L_{phys}$')
a.semilogy(log.epoch, log.data, c='#1f77b4', lw=1.1, label=r'Data fidelity, $L_{data}$')
a.semilogy(log.epoch, log.physics, c='#ff7f0e', lw=1.1, label=r'Physics residual, $L_{phys}$')
a.set_xlabel('Epoch'); a.set_ylabel('Loss (log scale)'); a.set_title('(a) PINN training losses', fontsize=10)
a.grid(alpha=.25, lw=.5, which='both'); a.legend(fontsize=7.5)

b = axes[1]
tt = np.linspace(0.3, 20, 300).astype(np.float32)
b.plot(t, F7, 'o', ms=5.5, mfc='#1f77b4', mec='k', mew=.5, zorder=5, label='Digitised points (n = 14) [44]')
b.plot(tt, pred(m_full, tt), '-', c='k', lw=1.5, label='PINN prediction')
b.plot(tt, PS(tt.astype(float), *ps7), '--', c='#d62728', lw=1.2, label='Peppas$-$Sahlin fit')
b.set_xlabel('Time (h)'); b.set_ylabel('Cumulative release (%)')
b.set_title('(b) PINN vs digitised benchmark (HPMC K100M)', fontsize=10)
b.set_xlim(0, 20); b.set_ylim(0, 80); b.grid(alpha=.25, lw=.5)
b.text(.04, .93, 'RMSE = 0.41 pp\nR$^2$ = 0.9995', transform=b.transAxes, fontsize=8, va='top')
b.legend(fontsize=7.5, loc='lower right')

c = axes[2]
c.axvspan(0, 8, color='#dddddd', alpha=.55, lw=0)
c.plot(t, F7, 'o', ms=5.5, mfc='#1f77b4', mec='k', mew=.5, zorder=5, label='Observed (digitised)')
c.plot(tt, pred(m_phys, tt), '-', c='#2ca02c', lw=1.6, label=r'PINN, $\lambda_{phys}=0.5$')
c.plot(tt, pred(m_plain, tt), '-', c='#d62728', lw=1.6, label=r'Same network, $\lambda_{phys}=0$')
c.axvline(8, color='k', ls=':', lw=1)
c.text(4, 74, 'training window\n0.5$-$8 h', ha='center', fontsize=7.5)
c.annotate('', xy=(19.6, 51.0), xytext=(19.6, 64.6),
           arrowprops=dict(arrowstyle='<->', lw=.9, color='k'))
c.text(18.9, 69.5, 'extrapolation\ngap at 20 h', ha='right', fontsize=7.5)
c.set_xlabel('Time (h)'); c.set_ylabel('Cumulative release (%)')
c.set_title('(c) Physics ablation: extrapolation beyond training data', fontsize=10)
c.set_xlim(0, 20); c.set_ylim(0, 80); c.grid(alpha=.25, lw=.5)
c.legend(fontsize=7.5, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/figure6.png'); plt.close(fig)
print('figure6 done')

# ---------------------------------------------------------------- Figure 7
S1 = [0.0494, 0.9506]; S1c = [0.0202, 0.0631]
ST = [0.0494, 0.9506]; STc = [0.0039, 0.0582]
fig, axes = plt.subplots(1, 2, figsize=(9.75, 4.8))
x = np.arange(2); w = .35
a = axes[0]
a.bar(x - w/2, S1, w, yerr=S1c, capsize=4, color='#1f77b4', edgecolor='k', lw=.6, label='First-order, $S_i$')
a.bar(x + w/2, ST, w, yerr=STc, capsize=4, color='#ff7f0e', edgecolor='k', lw=.6, label='Total-effect, $S_{Ti}$')
a.set_xticks(x); a.set_xticklabels(['$k_d$\n(diffusional)', '$k_r$\n(relaxational)'])
a.set_ylabel('Sobol index'); a.set_ylim(0, 1.15)
a.set_title('(a) Variance of $Q_{12h}$ across HPMC grades', fontsize=10)
for xi, v in zip(x - w/2, S1): a.text(xi, v + .09, f'{v:.3f}', ha='center', fontsize=8)
for xi, v in zip(x + w/2, ST): a.text(xi, v + .09, f'{v:.3f}', ha='center', fontsize=8)
a.grid(axis='y', alpha=.25, lw=.5); a.legend(fontsize=8, loc='upper left')
b = axes[1]
b.bar(['Diffusional\n$k_d t^{0.45}$', 'Relaxational\n$k_r t^{0.89}$'], [61.9, 38.1],
      color=['#1f77b4', '#ff7f0e'], edgecolor='k', lw=.6, width=.5)
for xi, v in enumerate([61.9, 38.1]): b.text(xi, v + 2, f'{v:.1f}%', ha='center', fontsize=8)
b.set_ylabel('Contribution to drug released at 12 h (%)'); b.set_ylim(0, 80)
b.set_title('(b) Mechanistic decomposition of release at 12 h', fontsize=10)
b.grid(axis='y', alpha=.25, lw=.5)
fig.tight_layout(); fig.savefig(f'{OUT}/figure7.png'); plt.close(fig)
print('figure7 done')

# ---------------------------------------------------------------- Figure 8
code = lambda H, S: ((H - 25.) / 10., (S - 100.) / 50.)
def R1(H, S):
    A, B = code(H, S)
    return 46.2200019188103 - 19.6376*A - 1.5954*B + 0.95*A*B + 2.2302*A**2 - 0.2198*B**2
def R2(H, S):
    A, B = code(H, S)
    return 3.74 + 0.3091*A + 1.8014*B - 0.05*A*B - 0.5275*A**2 - 0.5275*B**2
H = np.linspace(15, 35, 240); S = np.linspace(50, 150, 240)
HH, SS = np.meshgrid(H, S)
DES = np.array([[17.93, 64.64], [32.07, 64.64], [17.93, 135.36], [32.07, 135.36],
                [15, 100], [35, 100], [25, 50], [25, 150], [25, 100]])

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
for ax, Z, ttl, unit, cmap in [(axes[0], R1(HH, SS), '(a) R1: cumulative release at 12 h', '%', 'viridis'),
                               (axes[1], R2(HH, SS), '(b) R2: enzymatic protection window', 'h', 'plasma')]:
    cf = ax.contourf(HH, SS, Z, levels=18, cmap=cmap, alpha=.9)
    cs = ax.contour(HH, SS, Z, levels=9, colors='k', linewidths=.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f' if unit == '%' else '%.1f')
    ax.plot(DES[:, 0], DES[:, 1], 'o', mfc='none', mec='w', mew=1.3, ms=6)
    ax.plot(22.94, 150, '*', ms=16, mfc='w', mec='k', mew=.9, zorder=6)
    ax.set_xlabel('A: HPMC K100M (% w/w)'); ax.set_ylabel('B: SNAC (mg/tablet)')
    ax.set_title(ttl, fontsize=10)
    fig.colorbar(cf, ax=ax, label=unit)
fig.tight_layout(); fig.savefig(f'{OUT}/figure8ab.png'); plt.close(fig)

def desir(H, S):
    T, L, U = 48.35, 28.8, 68.1
    y = R1(H, S)
    d1 = np.where(y <= T, (y - L) / (T - L), (U - y) / (U - T))
    d1 = np.clip(d1, 0, 1)
    d2 = np.clip((R2(H, S) - 1.4) / (5.0 - 1.4), 0, 1)
    return np.sqrt(d1 * d2)
fig, ax = plt.subplots(figsize=(6.0, 4.6))
D = desir(HH, SS)
cf = ax.contourf(HH, SS, D, levels=20, cmap='YlGnBu', alpha=.92)
cs = ax.contour(HH, SS, D, levels=[.2, .4, .6, .8, .9, .95, .99], colors='k', linewidths=.6)
ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
ax.plot(DES[:, 0], DES[:, 1], 'o', mfc='none', mec='w', mew=1.3, ms=6, label='Design points')
ax.plot(22.94, 150, '*', ms=18, mfc='w', mec='k', mew=1.0, zorder=6, label='Optimum, D = 0.991')
ax.annotate('HPMC 22.94% w/w\nSNAC 150 mg\n$Q_{12h}$ = 48.35%, window = 4.94 h',
            (22.94, 150), textcoords='offset points', xytext=(-150, -78), fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='w', ec='k', lw=.6, alpha=.92))
ax.set_xlabel('A: HPMC K100M (% w/w)'); ax.set_ylabel('B: SNAC (mg/tablet)')
ax.set_title('(c) Overall desirability', fontsize=10)
fig.colorbar(cf, ax=ax, label='Desirability, D')
ax.legend(fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig(f'{OUT}/figure8c.png'); plt.close(fig)
print('figure8 done')
print('optimum check: R1=%.4f  R2=%.5f  D=%.4f' % (R1(22.9396, 150), R2(22.9396, 150), float(desir(np.array(22.9396), np.array(150.)))))
