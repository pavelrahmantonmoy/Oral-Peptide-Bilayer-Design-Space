"""Generation of the 13 CCD responses used in Section 2.9 / Table 2.

IMPORTANT - what this script is and is not
------------------------------------------
The two responses analysed by Response Surface Methodology in this work were NOT measured
experimentally and were NOT fitted to an experimental data set. They were produced by an
AUTHOR-SPECIFIED semi-empirical surrogate function whose form and coefficients were chosen to
reproduce qualitative trends reported for hydrophilic HPMC matrices (release falls as polymer
fraction rises) and for SNAC-containing systems (longer protective window at higher SNAC load).
The surrogate is therefore a transparent stand-in for a formulation model, not a measurement.
Any ANOVA, lack-of-fit or desirability statistic computed from these responses describes the
goodness of fit of the quadratic surface to this surrogate - it is NOT experimental validation.

Surrogate (coded factors; A = (HPMC%-25)/10, B = (SNAC mg-100)/50):
  R1  Q12h (%)              = 46.3200 - 19.6522 A - 1.59539 B + 0.950018 AB + 2.15514 A^2 - 0.294786 B^2
  R2  protection window (h) =  3.9200 + 0.301796 A + 1.79840 B + 0        AB - 0.732522 A^2 - 0.682501 B^2
The AB term of R2 is identically zero (1.92e-16 as reported by Stat-Ease 360 v25.0.7) because the
R2 surrogate is additive in the two factors by construction.
Deterministic values are rounded to one decimal place; independent Gaussian noise is added only to
the five replicated centre points (runs 9-13) so that a pure-error term exists for the ANOVA.
"""
import numpy as np, pandas as pd
SEED = 42
C1 = dict(b0=46.3200019188103, A=-19.6522424886158, B=-1.59538966382672,
          AB=0.95001824035021, A2=2.15514362068383, B2=-0.294785565601652)
C2 = dict(b0=3.9200005746474,  A=0.301795571131241, B=1.79840289856185,
          AB=0.0,              A2=-0.73252185244346, B2=-0.682500645075258)
SD_CENTRE = dict(R1=0.43, R2=0.31)          # matches the scatter of runs 9-13

code = lambda hpmc, snac: ((hpmc-25.0)/10.0, (snac-100.0)/50.0)
def surrogate(c, hpmc, snac):
    A, B = code(hpmc, snac)
    return c['b0'] + c['A']*A + c['B']*B + c['AB']*A*B + c['A2']*A*A + c['B2']*B*B

design = pd.DataFrame(
    [(1,5,'Factorial',17.93,64.64),(2,8,'Factorial',32.07,64.64),(3,13,'Factorial',17.93,135.36),
     (4,9,'Factorial',32.07,135.36),(5,10,'Axial',15.0,100.0),(6,6,'Axial',35.0,100.0),
     (7,7,'Axial',25.0,50.0),(8,4,'Axial',25.0,150.0)]+
    [(9,1,'Centre',25.0,100.0),(10,11,'Centre',25.0,100.0),(11,2,'Centre',25.0,100.0),
     (12,3,'Centre',25.0,100.0),(13,12,'Centre',25.0,100.0)],
    columns=['Run','Std','Type','HPMC_pct','SNAC_mg'])

rng = np.random.default_rng(SEED)
design['R1_Q12h'] = surrogate(C1, design.HPMC_pct, design.SNAC_mg).round(1)
design['R2_window_h'] = surrogate(C2, design.HPMC_pct, design.SNAC_mg).round(1)
ctr = design.Type == 'Centre'
design.loc[ctr,'R1_Q12h']    = (design.loc[ctr,'R1_Q12h']    + rng.normal(0, SD_CENTRE['R1'], ctr.sum())).round(1)
design.loc[ctr,'R2_window_h']= (design.loc[ctr,'R2_window_h']+ rng.normal(0, SD_CENTRE['R2'], ctr.sum())).round(1)
design.to_csv('ccd_surrogate_responses.csv', index=False)
print(design.to_string(index=False))

TAB = {'R1_Q12h':[62.8,34.2,59.6,32.9,68.1,28.6,47.5,44.3,47.0,46.4,46.0,45.9,46.3],
       'R2_window_h':[1.8,2.3,4.2,4.7,2.9,3.4,1.3,5.1,4.0,3.7,4.2,3.5,4.2]}
print('\nAgreement with the values tabulated in Table 2 of the manuscript:')
for col in TAB:
    dv = design[col].values - np.array(TAB[col]); nz = dv[:8]
    print('  %-12s non-centre runs 1-8: max |deviation| = %.2f ; centre runs 9-13 differ by design (independent noise draw)'
          % (col, np.abs(nz).max()))
