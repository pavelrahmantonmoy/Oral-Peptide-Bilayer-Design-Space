"""Digitisation of Fig. 5 of Vueba ML, Batista de Carvalho LAE, Veiga F, Sousa JJ, Pina ME.
Eur J Pharm Biopharm 2004;58:51-59 -- ketoprofen release from lactose-containing HPMC matrices.
Curves used here: F5 (20% w/w HPMC K15M) and F7 (20% w/w HPMC K100M).
Marker centroids are detected automatically from the figure bitmap extracted from the publisher PDF;
axis calibration uses the printed tick marks (0-20 h, 0-100%)."""
import numpy as np, pandas as pd
from PIL import Image
im = np.array(Image.open('p6-002.png').convert('L')); dark = im < 128
X0, X1, T1 = 87.5, 637.5, 20.0          # x ticks 0 h and 20 h
Y0, Y1, P1 = 363.5, 17.5, 100.0         # y ticks 0 % and 100 %
to_t = lambda c: (c - X0) / (X1 - X0) * T1
to_p = lambda r: (Y0 - r) / (Y0 - Y1) * P1

# solid markers = pixels with a dense dark neighbourhood (lines are thin)
k = 2
dens = np.zeros_like(dark, dtype=int)
for dy in range(-k, k+1):
    for dx in range(-k, k+1):
        dens += np.roll(np.roll(dark, dy, 0), dx, 1)
soli = dens >= 20                                  # >=20 of 25 px dark
soli[:, :90] = False; soli[366:, :] = False        # axes
soli[280:360, 415:650] = False                     # legend box

lab = -np.ones(soli.shape, int); blobs = []
for r, c in zip(*np.where(soli)):
    if lab[r, c] >= 0: continue
    i = len(blobs); st = [(r, c)]; pts = []
    lab[r, c] = i
    while st:
        y, x = st.pop(); pts.append((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y2, x2 = y+dy, x+dx
                if 0 <= y2 < soli.shape[0] and 0 <= x2 < soli.shape[1] and soli[y2, x2] and lab[y2, x2] < 0:
                    lab[y2, x2] = i; st.append((y2, x2))
    if len(pts) >= 6:
        a = np.array(pts); blobs.append((a[:, 1].mean(), a[:, 0].mean(), len(pts)))
blobs.sort()
df = pd.DataFrame(blobs, columns=['col', 'row', 'px'])
df['t_h'] = to_t(df.col).round(2); df['pct'] = to_p(df.row).round(1)
df = df[(df.t_h > -0.2) & (df.t_h < 20.6)]
df.to_csv('vueba_fig5_allblobs.csv', index=False)

# at each nominal sampling time take the two LOWEST markers: F7 (lowest) then F5
times = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
rows = []
for t in times:
    sel = df[(df.t_h - t).abs() < 0.30].sort_values('pct')
    if len(sel) >= 2:
        rows.append(dict(t_h=t, F7_K100M_pct=sel.pct.iloc[0], F5_K15M_pct=sel.pct.iloc[1], n_blobs=len(sel)))
out = pd.DataFrame(rows); out.to_csv('vueba_fig5_digitised.csv', index=False)
print(out.to_string(index=False))
