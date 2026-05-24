# NFL Bayesian Network: Game Stats & Market Signals

**CS610 Final Project — Drexel MSAIML**
Leland Weeks · Johnny Belichev · Ishant Somal

## Research Question

Does line movement (market signals) contain predictive information about game outcomes beyond what game statistics already encode? We model this as a three-class classification problem (cover / loss / push) using a Bayesian Network trained on 3,990 NFL games across 15 seasons.

---

## File Structure

```
project/
├── data/
│   ├── raw/                    # nflverse game stats + SBR odds CSVs (unmodified)
│   └── processed/              # merged, cleaned, discretized data
├── src/
│   ├── data_loader.py          # data loading and merging
│   ├── features.py             # feature engineering (get_cont) and discretization (get_cat)
│   ├── evaluate.py             # accuracy, log loss, classification report, confusion matrix
│   ├── ablation.py             # ablation logic and feature group definitions
│   ├── models/
│   │   ├── naive_bayes.py      # Gaussian Naive Bayes baseline
│   │   ├── hill_climbing.py    # BN structure learning via Hill Climbing
│   │   └── pc_algorithm.py     # BN structure learning via PC Algorithm
│   └── main.py                 # pipeline orchestrator
├── output/                     # learned DAG edge lists, ablation_summary.csv
├── docs/                       # proposal, diagrams, learnings log
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `pgmpy` is pinned to `1.0.0`. Newer versions have breaking API changes that are incompatible with this codebase. Do not upgrade it.

---

## Usage

### Run all models, all configs (full ablation)

```bash
python src/main.py --model all --config all
```

### Run a specific model

```bash
python src/main.py --model nb
python src/main.py --model hc
python src/main.py --model pc
```

### Run a specific feature config

```bash
python src/main.py --config stats
python src/main.py --config market
python src/main.py --config combined
```

### Combine flags

```bash
python src/main.py --model nb --config all
python src/main.py --model all --config combined
```

Default behavior (`python src/main.py`) runs all models on the combined feature config.

---

## Feature Configs

| Config | Features |
|--------|----------|
| `stats` | EPA diff, yards diff, turnover diff, TD diff, completion diff, temp, wind, roof, surface, rolling ATS |
| `market` | Close spread, close total, spread move, total move, home win probability |
| `combined` | All of the above |

---

## Outputs

All outputs are written to `output/`:

- `ablation_summary.csv` — accuracy and log loss for every model/config combination
- `hc_{config}_dag.txt` — learned HC DAG edge list per config
- `pc_{config}_dag.txt` — learned PC DAG edge list per config

---

## Key Results

| Config | NB Accuracy | HC Accuracy | PC Accuracy | NB Log Loss | PC Log Loss |
|--------|-------------|-------------|-------------|-------------|-------------|
| Stats only | **84.1%** | 50.4% | 50.4% | **0.439** | 0.368 |
| Market only | 56.4% | 58.6% | 58.6% | 0.840 | 0.760 |
| Combined | 81.0% | 50.4% | 47.3% | 0.538 | **0.345** |
| Majority baseline | 58.0% | — | — | — | — |

Market signals do not improve NB classification accuracy — stats-only NB outperforms combined. BN models outperform NB on market-only accuracy. Combined PC achieves the best probability calibration (log loss).

---

## Deliverables

| Due | Deliverable | Status |
|-----|-------------|--------|
| Apr 26 | Data pipeline complete | ✅ Done |
| May 3 | Project Proposal | ✅ Done |
| May 7 | Proposal Presentation | ✅ Done |
| May 14 | DAG validation via PC algorithm | ✅ Done |
| May 17 | Naive Bayes baseline implemented and evaluated | ✅ Done |
| May 21 | BN with Hill Climbing implemented | ✅ Done |
| May 24 | Ablation runs (stats-only / market-only / combined) | ✅ Done |
| May 26 | Results interpreted and written up | — |
| May 28 | Paper draft complete | — |
| May 31 | Final revisions and artifact submitted | — |
| Jun 4 | Final Presentation | — |

---

## Data Sources

- **Game stats:** [nflverse](https://nflverse.nflverse.com) — EPA/play, turnovers, scoring, 2007–2022
- **Market data:** [SportsBookReviewsOnline](https://sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm) — opening/closing spreads and totals, moneylines

**Note:** Findings reflect a specific historical window. The 2018 PASPA repeal substantially changed market participation; results should not be interpreted as a prescriptive betting system.

---

## AI Disclosure

The following files were produced with AI assistance (Claude, Anthropic) and reviewed by the project team.

| File | Reason |
|------|--------|
| `src/data/fetch_nflverse.py` | Data acquisition is a laborious, mechanical task unsuited for manual scripting |
| `src/data/fetch_sbr.py` | Data acquisition is a laborious, mechanical task unsuited for manual scripting |
| `README.md` | Documentation structure and organization is better quality with AI assistance than manual authoring |
