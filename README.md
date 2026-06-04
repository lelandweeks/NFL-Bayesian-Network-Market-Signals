# NFL Bayesian Network: Game Stats & Market Signals

**CS610 Final Project**

**Leland Weeks · Johnny Belichev · Ishant Somal**

---

## Start Here

The two primary documents for grading are in the `docs/` folder:

- **`docs/cs610_final_report.pdf`** — the final paper
- **`docs/cs610_final_report_proposal_comparison.pdf`** — comparison of final results against the original proposal

---

## Running the Code

Install dependencies:

```bash
pip install -r requirements.txt
```

> `pgmpy` must stay pinned to `1.0.0`. Do not upgrade it — newer versions have breaking API changes.

Run the full ablation (all models, all feature configurations):

```bash
python src/main.py --model all --config all
```

Results are written to `output/ablation_summary.csv`.

---

## Research Question

Does line movement (market signals) contain predictive information about NFL game outcomes beyond what game statistics already encode? Framed as a three-class classification problem (cover / loss / push) against the closing spread, using 3,990 NFL games across 15 seasons (2007–2022).

---

## Key Results

| Config | NB Accuracy | HC Accuracy | PC Accuracy | NB Log Loss | HC Log Loss | PC Log Loss |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|
| Majority baseline | 58.0% | — | — | — | — | — |
| Stats only | **84.1%** | 50.4% | 50.4% | **0.439** | 0.375 | 0.368 |
| Market only | 56.4% | 58.6% | 58.6% | 0.840 | 0.776 | 0.760 |
| Combined | 81.0% | 50.4% | 47.3% | 0.538 | 0.375 | **0.345** |

Market signals do not improve NB classification accuracy. Combined PC achieves the best probability calibration (log loss).

---

## Data Sources

- **Game stats:** [nflverse](https://nflverse.nflverse.com) — EPA/play, turnovers, touchdowns, yards, completion pct, 2007–2022
- **Market data:** [SportsBookReviewsOnline](https://sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm) — opening/closing spreads and totals, moneylines

---

## File Structure

```
├── data/raw/              # unmodified source CSVs (nflverse + SBR)
├── src/
│   ├── main.py            # entry point — --model [nb|hc|pc|all] --config [stats|market|combined|all]
│   ├── features.py        # feature engineering and discretization
│   ├── ablation.py        # ablation configurations and run logic
│   ├── evaluate.py        # metrics
│   └── models/
│       ├── naive_bayes.py
│       ├── hill_climbing.py
│       └── pc_algorithm.py
├── output/                # ablation_summary.csv, learned DAG edge lists
├── docs/                  # final report, proposal comparison, proposal, presentation
└── requirements.txt
```

---

## AI Disclosure

The following files were produced with AI assistance and reviewed by the project team.

| File | Reason |
|------|--------|
| `scripts/fetch_nfl_game_stats.py` | Data acquisition scripting |
| `scripts/scrape_sbr_nfl_odds.py` | Data acquisition scripting |
| `README.md` | Documentation |
