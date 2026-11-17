# Within-student domain contrasts in PISA 2025: replication code

Code and results for the paper *Within-Student Domain Contrasts in International
Large-Scale Assessments: An Identification Strategy, a Test of Its Assumption, and an
Application to Students' Use of AI Chatbots in PISA 2025*.

The analysis estimates how the composition of students' AI chatbot use (delegating tasks
versus obtaining support, at fixed intensity) relates to the difference between a
student's conventional-domain scores (reading, mathematics, science) and computational
scores, using partialling-out double machine learning with plausible values, Fay balanced
repeated replication and senate weights. It then tests the equal-loading assumption on
which the within-student contrast rests, with anchor covariates, recalibrated contrasts,
negative controls and a simulation study.

## Data

The PISA 2025 international database is distributed by the OECD
(<https://www.oecd.org/pisa/data/>) and is not included here. Download the student
questionnaire file in SPSS format (`CY09_MS_STU_PUF.zip`) and place it, or the extracted
`.sav` file, in `data/raw/`.

## Requirements

Python 3.13 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

`00_prepare_data.py` calls the system `unzip` command, because the OECD archives use
Deflate64 compression, which Python's `zipfile` module cannot extract.

## Running the analysis

Run the scripts in order from the repository root. Each writes its output to `results/`.

| Script | Produces | In the paper |
|---|---|---|
| `code/00_prepare_data.py` | `data/processed/CY09_MS_STU_PUF.parquet` | — |
| `code/01_build_sample.py` | `data/cache/aiuse_sample.parquet` | Sample construction (Table 1), treatment definition |
| `code/02_main_estimates.py` | `results/main_estimates.json` | Uncalibrated contrasts and scale-by-scale coefficients (Table 3); variance shares |
| `code/03_descriptives_pv1.py` | `results/descriptives_pv1.json` | Treatment descriptives; cross-fitted nuisance R² |
| `code/04_robustness_pv1.py` | `results/robustness_pv1.json` | Positivity check; categorical specification |
| `code/05_heterogeneity_pv1.py` | `results/heterogeneity_pv1.json` | Moderation slope of the control on the first plausible value (Section 3.5) |
| `code/06_diagnostics.py` | `results/diagnostics_stats.json`, `results/diagnostics_tables.json` | Negative controls (Table 4), anchor test and recalibrated contrasts (Table 5), moderation (Section 3.5), further checks (Section 3.6) |
| `code/07_simulation.py` | `results/simulation.json` | Simulation study (Table 2) |

Scripts whose names end in `_pv1` use the first plausible value with cluster-robust
standard errors. All results in Tables 3–5 use the ten plausible values and the eighty
replicate weights.

Run times are long because every nuisance model is refitted under each replicate weight.
On the machine used for the paper, `02_main_estimates.py` took about eleven hours and
`06_diagnostics.py` about twelve. `06_diagnostics.py` saves its statistics after every replicate weight and
resumes where it stopped. `python3 code/06_diagnostics.py --tables` recomputes all its
tables from the stored statistics in a few seconds, and `--smoke` runs the full pipeline
on a small subsample as a quick check.

`07_simulation.py` takes about two minutes. Run it single-threaded; multithreaded BLAS is
much slower on these small problems:

```
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 code/07_simulation.py
```

Random seeds are fixed. Very small differences can arise across platforms and library
versions.

## Results

`results/` holds the outputs reported in the paper, so the tables can be checked without
rerunning the analysis.

## License

MIT. See `LICENSE`.
