# LLM-based A/B Testing: Reproducibility Code

Code for the simulations and empirical application in "Statistical Foundations
of LLM-based A/B Testing: A Surrogacy Framework for Human Causal Inference"
(Persson, Schultzberg, and Ankargren).

- Paper: [arXiv:2606.17165](https://arxiv.org/abs/2606.17165)
- Workshop paper: [AI Measurement Science Workshop @ COLM 2026](https://openreview.net/forum?id=ly0x0V2NcP)

The framework calibrates an LLM surrogate against a small human-outcome
sample, then evaluates new variants against the calibrated function. This
repository reproduces every figure, table, and inline number in the paper. No
API key or network access is required — the LLM completions are shipped as a
cache.

## Install

```bash
pip install -r Code/requirements.txt
```

Pin numpy to 1.x. The committed simulation pickles were written under numpy
1.26.4 and cannot be loaded under numpy 2.x.

## Reproduce

```bash
# Simulation (Figures 1-6, ~35 min)
cd Code/simulation && python3 run_all.py

# Empirical application (Upworthy headline A/B tests)
cd Code/empirical/04_upworthy
python3 run_llm.py --K 10        # rebuilds the surrogate from the cache; run first
python3 question_treatment.py    # headline ATE results, K-sweep figure
python3 question_diagnostics.py  # falsification test, overlap bound, positive control
python3 contamination_probe.py   # memorization probe
python3 replot_paper_figs.py     # final figure styling
```

All randomized procedures are seeded: `seed=42` for the simulation scenarios,
`seed=2026` for the empirical pipeline. See `DATA.md` for the data source.

## Structure

```
Code/
  simulation/       Monte Carlo scenarios, estimators, figures
  empirical/
    04_upworthy/    Upworthy headline A/B test analysis
    common/         shared estimation and LLM-parsing code
  requirements.txt
```

## Citation

```bibtex
@article{persson2026statistical,
  title   = {Statistical Foundations of LLM-based A/B Testing: A Surrogacy Framework for Human Causal Inference},
  author  = {Persson, Joel and Schultzberg, M{\aa}rten and Ankargren, Sebastian},
  journal = {arXiv preprint arXiv:2606.17165},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
