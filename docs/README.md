# Documentation index

| Document | Purpose |
|----------|---------|
| [../pipeline/README.md](../pipeline/README.md) | **Riccio runbook:** numbered steps 0–7 (setup → NPZs → BiLSTM → eval / Colab / optional EgoExo) |
| [CAPSTONE_PIPELINE.md](CAPSTONE_PIPELINE.md) | Maps thesis steps §2–§9 to code paths, data folders, and command order |
| [CAPSTONE_REPORT.md](CAPSTONE_REPORT.md) | Draft capstone paper (abstract through appendix) — fill tables with your metrics |
| [export_evaluation_plots.py](export_evaluation_plots.py) | Bar charts from `evaluate_exercise_models.py` JSON → `figures/*.png` |

Place evaluation plots in **`figures/`** after running:

```bash
./venv/bin/python docs/export_evaluation_plots.py --json results/eval_test.json --out-dir docs/figures
```
