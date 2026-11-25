# Design-Space Resolution of Oral Peptide Bilayer Tablets

Scripts and data for *An API-Sparing In-Silico Framework for Formulation Design Space
Resolution of Oral Peptide Bilayer Tablets: A Literature-Benchmarked Computational
Prototype* (Future Pharmacology, manuscript futurepharmacol-4673176).

Every number, table and figure in the manuscript can be regenerated from this repository.

## Scope of this README

The five folders listed below — `descriptors/`, `benchmark/`, `pinn/`, `rsm/` and `figures/` —
together with this README are the material accompanying the revised manuscript (November 2026
revision). They are the authoritative source for every value reported in it.

Any other files present in this repository are earlier working material retained for the
authors' own records and are **superseded**. In particular, the earlier release benchmark
reconstructed dissolution curves from published Korsmeyer-Peppas parameters and then fitted
kinetic models of the same family to those reconstructions; that approach is circular, it was
identified as such during peer review, and it has been replaced in this revision by direct
digitisation of the published dissolution figure (see `benchmark/`). Earlier response-surface
inputs are likewise superseded by `rsm/ccd_surrogate.py` and its output.

## Requirements

    python >= 3.11
    numpy scipy pandas matplotlib tensorflow SALib pillow
    poppler-utils   (pdfimages, for the digitisation step only)

All scripts fix their random seeds and enable deterministic TensorFlow operations,
so repeated runs reproduce the published values exactly.

## descriptors/

| file | what it does |
|---|---|
| `tirzepatide.smi` | canonical SMILES for the complete 39-residue sequence including both Aib residues and the gamma-Glu-bis(AEEA)-linked C20 fatty diacid |
| `snac_free_acid.smi`, `snac_sodium.smi` | SNAC as the free acid and as salcaprozate sodium |
| `compute_descriptors.py` | recomputes every value in Table 3 |

The script checks the parsed Tirzepatide structure against the published molecular
formula C225H348N48O68 and aborts if it does not match, so a truncated or
mis-transcribed SMILES cannot silently produce descriptors.

## benchmark/

| file | what it does |
|---|---|
| `digitize.py` | extracts the release points from Figure 5 of Vueba et al. 2004 (publisher PDF bitmap), calibrates the axes against the printed ticks, locates markers by connected-component analysis and separates overlapping early-time markers by 1-D 2-means clustering |
| `vueba_fig5_source.png` | the extracted source bitmap |
| `vueba_fig5_digitised.csv` | 14 release points, 0.5-20 h, for HPMC K100M 20% w/w (F7) and HPMC K15M 20% w/w (F5) |
| `analysis.py` | fits zero-order, first-order, Higuchi, Korsmeyer-Peppas and Peppas-Sahlin models to the digitised points; trains the PINN; runs the temporal-extrapolation test, the cross-grade transfer test and the physics ablation |
| `table7_model_comparison.csv` | the release-model comparison reported as Table 6 |
| `sobol_mc_digitised.py` | Peppas-Sahlin fits to both grades, Sobol global sensitivity analysis of Q12h (N = 1024, second order, 6144 evaluations) and the +/-10% Monte Carlo uncertainty quantification |

The benchmark is built by digitising the published figure rather than by reconstructing
curves from published model parameters. Reconstructing curves from Korsmeyer-Peppas
parameters and then fitting a model of the same family to them is circular; digitisation
avoids that. Digitisation fidelity is checked against the authors' own reported fit
(refitted k = 10.745, n = 0.593 against published k = 9.926, n = 0.621; mean absolute
deviation 0.79 pp for F7 and 0.61 pp for F5).

## pinn/

`pinn_log_F7_digitised.csv` is the per-epoch loss log (total, data, physics) for the
1,000-epoch run reported in Figure 6(a). The training code itself is in
`benchmark/analysis.py` (`build()` and `train()`), which also runs the lambda_phys = 0
ablation.

Architecture: 2 x 32 tanh, sigmoid output, Glorot uniform (seed 42), Adam lr = 0.01,
L_total = L_data + 0.5 * L_physics, physics residual = dQ/dt - k*n*t^(n-1) evaluated by
nested `tf.GradientTape` at 60 collocation points over 0.1-20 h.

## rsm/

| file | what it does |
|---|---|
| `ccd_surrogate.py` | generates the 13 CCD response values of Table 2 |
| `ccd_surrogate_responses.csv` | the generated responses |
| `MDPI_Future_CCD.dxpx` | the Stat-Ease 360 (v25.0.7) design file |

**Read this before using the RSM results.** The response values were *not* experimentally
measured and were *not* fitted to an experimental dataset. They were produced by an
author-specified semi-empirical surrogate function reflecting known qualitative HPMC/SNAC
matrix trends. Deterministic values are rounded to one decimal place and Gaussian noise
(SD 0.43% for R1, 0.31 h for R2) is added to the five centre points only, so that a
pure-error estimate and hence a lack-of-fit test exist.

The ANOVA, lack-of-fit tests and desirability optimum reported in the manuscript therefore
measure how well a quadratic model describes this surrogate surface. They are an internal
consistency check on the optimisation workflow, not experimental validation. The script is
published so that this is verifiable rather than merely asserted.

## figures/

`make_figures.py` regenerates Figures 4, 6, 7 and 8; `make_figures_1_9.py` regenerates
Figures 1 and 9. Run both from the `benchmark/` directory, or copy
`vueba_fig5_digitised.csv` and `pinn_log_F7_digitised.csv` alongside them.

## Key results

| quantity | value |
|---|---|
| Peppas-Sahlin, digitised HPMC K100M | k_d = 9.417, k_r = 1.944, R2 = 0.9991, RMSE 0.55 pp |
| PINN on 14 digitised points | RMSE 0.41 pp, R2 = 0.9995, AIC +2280.9 (1,153 weights) |
| Peppas-Sahlin, same points | RMSE 0.55 pp, AIC -12.8 (2 parameters) - preferred on parsimony |
| Extrapolation 0.5-8 h -> 10-12 h, PINN | RMSE 0.95 pp |
| same, lambda_phys = 0 (ablation) | RMSE 2.50 pp |
| same, Korsmeyer-Peppas refit | RMSE 0.87 pp |
| Sobol S_T, k_r / k_d | 0.951 / 0.049 |
| Monte Carlo Q12h (+/-10%) | 46.6%, 95% CI 42.8-50.3% (+/-8.1%) |
| RSM optimum (surrogate surface) | HPMC K100M 22.94% w/w, SNAC 150 mg, D = 0.991 |
| Tirzepatide descriptors | MW 4813.53, Crippen logP -12.10, TPSA 1789.63 A^2, HBD 58, HBA 65, RotB 163 |

The PINN does not outperform the analytical Korsmeyer-Peppas model on these data. Its
measurable contribution is confined to retaining physically plausible long-horizon
behaviour that an otherwise identical unconstrained network loses.

## Data source

Vueba ML, Batista de Carvalho LAE, Veiga F, Sousa JJ, Pina ME. Influence of cellulose-ether
polymers on ketoprofen release from hydrophilic matrix tablets. Eur J Pharm Biopharm.
2004;58(1):51-59. doi:10.1016/j.ejpb.2004.03.006

Secondary benchmarks: Levina & Rajabi-Siahboomi, J Pharm Sci 2004;93(11):2746-2754;
Maggi, Bruni & Conte, Int J Pharm 2000;195(1-2):229-238.
