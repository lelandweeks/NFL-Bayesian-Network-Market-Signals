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
│   ├── data.py                 # data loading and preprocessing
│   ├── features.py             # feature engineering (rolling ATS, discretization)
│   ├── models/
│   │   ├── naive_bayes.py      # Gaussian Naive Bayes baseline
│   │   ├── hill_climbing.py    # BN structure learning via Hill Climbing
│   │   └── pc_algorithm.py     # DAG validation via PC algorithm
│   ├── evaluate.py             # accuracy, log loss, brier score
│   └── ablation.py             # stats-only / market-only / combined runs
├── results/                    # saved metrics, learned DAG exports
├── main.py                     # full pipeline orchestrator
└── README.md
```

---

## Setup

```bash
pip install pgmpy scikit-learn pandas numpy
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

Runs preprocessing → feature engineering → all three models → ablation → evaluation. Outputs metrics to `results/`.

### Run individual scripts

**Preprocess data**
```bash
python src/data.py
```
Merges nflverse and SBR datasets by season and matchup. Outputs cleaned data to `data/processed/`.

**Feature engineering**
```bash
python src/features.py
```
Computes rolling ATS (using only prior games), discretizes continuous features for BN models.

**Naive Bayes baseline**
```bash
python src/models/naive_bayes.py
```
Trains and evaluates a Gaussian Naive Bayes classifier. Saves metrics to `results/naive_bayes.json`.

**Hill Climbing (BN structure learning)**
```bash
python src/models/hill_climbing.py
```
Learns DAG structure via Hill Climbing with BIC scoring. Saves learned structure to `results/hc_dag.json`.

**PC Algorithm (DAG validation)**
```bash
python src/models/pc_algorithm.py
```
Runs constraint-based structure learning as a validation check on the Hill Climbing DAG. Saves result to `results/pc_dag.json`.

**Ablation runs**
```bash
python src/ablation.py
```
Runs all three models under three feature configurations: stats-only, market-only, and combined. Outputs comparison table to `results/ablation.csv`.

---

## Deliverables

| Due | Deliverable | Status |
|-----|-------------|--------|
| Apr 26 | Data pipeline complete | ✅ Done |
| May 3 | Project Proposal | ✅ Done |
| May 7 | Proposal Presentation | ✅ Done |
| May 14 | DAG validation via PC algorithm | — |
| May 17 | Naive Bayes baseline implemented and evaluated | — |
| May 21 | BN with Hill Climbing implemented | — |
| May 24 | Ablation runs (stats-only / market-only / combined) | — |
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
