"""
Sobol global sensitivity analysis and Monte Carlo uncertainty quantification
for the 12-hour cumulative release (Q12h) predicted by the Peppas-Sahlin model.

PROVENANCE
----------
The two uncertain inputs are the Peppas-Sahlin diffusional constant (k_d) and
relaxational constant (k_r).  Their bounds are NOT assumed: they are obtained by
fitting the Peppas-Sahlin equation to release profiles DIGITISED from Figure 5
of Vueba et al., Eur J Pharm Biopharm 2004;58:51-59 (doi:10.1016/j.ejpb.2004.03.006)
for two HPMC grades at the same 20% w/w loading:
    F7 = HPMC K100M 20% w/w   (slower grade -> lower k_d)
    F5 = HPMC K15M  20% w/w   (faster grade -> higher k_d)
The interval spanned by the two fitted values defines the uncertainty range.
Digitisation is performed by digitize.py; the digitised points are in
vueba_fig5_digitised.csv.

Q is expressed in PERCENT released, so k_d and k_r are on the percent scale.

Run:  python3 -I sobol_mc_digitised.py
"""
import numpy as np
from scipy.optimize import curve_fit
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

SEED = 42
M_FICK, M_CASEII = 0.45, 0.89          # cylindrical limits (Peppas & Sahlin 1989)


def peppas_sahlin(t, kd, kr):
    return kd * t ** M_FICK + kr * t ** M_CASEII


# ----------------------------------------------------------------- 1. load data
raw = np.genfromtxt("vueba_fig5_digitised.csv", delimiter=",", names=True)
t = raw["t_h"]
f7 = raw["F7_K100M"]      # HPMC K100M 20% w/w
f5 = raw["F5_K15M"]       # HPMC K15M  20% w/w

print("=" * 72)
print("1. PEPPAS-SAHLIN FITS TO DIGITISED VUEBA FIG. 5 DATA")
print("=" * 72)

fits = {}
for name, y in (("F7 (HPMC K100M 20%)", f7), ("F5 (HPMC K15M 20%)", f5)):
    p, _ = curve_fit(peppas_sahlin, t, y, p0=[9.0, 2.0], maxfev=20000)
    pred = peppas_sahlin(t, *p)
    rss = float(np.sum((y - pred) ** 2))
    rmse = float(np.sqrt(rss / len(y)))
    r2 = 1.0 - rss / float(np.sum((y - y.mean()) ** 2))
    q12 = float(peppas_sahlin(12.0, *p))
    fits[name] = dict(kd=p[0], kr=p[1], rmse=rmse, r2=r2, q12=q12)
    print(f"{name:24s} k_d = {p[0]:7.4f}   k_r = {p[1]:7.4f}   "
          f"R2 = {r2:.4f}   RMSE = {rmse:.3f} pp   Q12h = {q12:.2f}%")

kd_lo = min(v["kd"] for v in fits.values())
kd_hi = max(v["kd"] for v in fits.values())
kr_lo = min(v["kr"] for v in fits.values())
kr_hi = max(v["kr"] for v in fits.values())
print(f"\nSobol bounds from the two grades: k_d [{kd_lo:.4f}, {kd_hi:.4f}]   "
      f"k_r [{kr_lo:.4f}, {kr_hi:.4f}]")

# nominal = the K100M matrix actually used in the CR layer
kd_nom = fits["F7 (HPMC K100M 20%)"]["kd"]
kr_nom = fits["F7 (HPMC K100M 20%)"]["kr"]
q12_nom = fits["F7 (HPMC K100M 20%)"]["q12"]

# ------------------------------------------------------- 2. Sobol GSA on Q12h
print()
print("=" * 72)
print("2. SOBOL GLOBAL SENSITIVITY ANALYSIS OF Q12h")
print("=" * 72)

problem = {
    "num_vars": 2,
    "names": ["k_d (diffusional)", "k_r (relaxational)"],
    "bounds": [[kd_lo, kd_hi], [kr_lo, kr_hi]],
}
N = 1024
X = sobol_sample.sample(problem, N, calc_second_order=True, seed=SEED)
Y = np.array([peppas_sahlin(12.0, xi[0], xi[1]) for xi in X])
print(f"Saltelli sample: N = {N}, second order enabled -> {X.shape[0]} model evaluations")

Si = sobol_analyze.analyze(problem, Y, calc_second_order=True,
                           print_to_console=False, seed=SEED)
for i, nm in enumerate(problem["names"]):
    print(f"{nm:22s} S1 = {Si['S1'][i]:.4f} +/- {Si['S1_conf'][i]:.4f}   "
          f"ST = {Si['ST'][i]:.4f} +/- {Si['ST_conf'][i]:.4f}")
print(f"{'S12 (interaction)':22s} {Si['S2'][0, 1]:.5f} +/- {Si['S2_conf'][0, 1]:.5f}")
print(f"Q12h over the Sobol sample: mean {Y.mean():.2f}%  min {Y.min():.2f}%  max {Y.max():.2f}%")

# --------------------------------------------- 3. Monte Carlo (+/-10% on k_d,k_r)
print()
print("=" * 72)
print("3. MONTE CARLO UNCERTAINTY QUANTIFICATION (+/-10% ON k_d AND k_r)")
print("=" * 72)

rng = np.random.default_rng(SEED)
N_MC = 1000
kd_s = rng.uniform(kd_nom * 0.9, kd_nom * 1.1, N_MC)
kr_s = rng.uniform(kr_nom * 0.9, kr_nom * 1.1, N_MC)
q = peppas_sahlin(12.0, kd_s, kr_s)

lo, hi = np.percentile(q, [2.5, 97.5])
spread = (hi - lo) / 2.0 / q.mean() * 100.0
print(f"nominal Q12h (K100M fit)   : {q12_nom:.2f}%")
print(f"MC mean                    : {q.mean():.2f}%")
print(f"MC standard deviation      : {q.std(ddof=1):.2f} pp")
print(f"95% interval               : {lo:.2f}% - {hi:.2f}%")
print(f"relative 95% spread        : +/-{spread:.2f}%")

print()
print("=" * 72)
print("SUMMARY FOR MANUSCRIPT")
print("=" * 72)
print(f"k_d = {kd_nom:.4f}, k_r = {kr_nom:.4f} (HPMC K100M, digitised)")
print(f"Sobol  S1/ST  k_d = {Si['S1'][0]:.3f}/{Si['ST'][0]:.3f}   "
      f"k_r = {Si['S1'][1]:.3f}/{Si['ST'][1]:.3f}")
print(f"Monte Carlo Q12h = {q.mean():.1f}% +/- {q.std(ddof=1):.1f} pp "
      f"(95% CI {lo:.1f}-{hi:.1f}%, +/-{spread:.1f}%)")
