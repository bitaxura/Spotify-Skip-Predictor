# What Drives My Skips? Modeling Personal Spotify Listening Behavior

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A data science project analyzing 5 years of my personal Spotify streaming history (193,464 streams) to predict song skipping behavior. The core question: **how much do intrinsic audio features (tempo, energy, valence) explain skipping compared to situational context (device, time, session momentum) and listening habits (track familiarity, prior skip rates)?**

### Key Finding
**Context and session momentum dominate skipping behavior (~40% of predictive power), while audio features provide incremental lift (+1.09% PR-AUC) for familiar tracks but become significantly more important for novel/unseen tracks (ROC-AUC 0.6040 vs 0.5350).**

The best model (Gradient Boosted Decision Tree) achieves **0.8871 PR-AUC** and **81.4% accuracy** on a chronological test split with zero data leakage.

---

## Research Questions

| Question | Finding |
| :--- | :--- |
| **How predictable is skipping?** | Highly predictable — GBDT achieves 0.8949 ROC-AUC, far above baseline (0.50) |
| **Context vs. audio vs. familiarity?** | Context dominates (39.6% relative impact), habit (2.1%), audio (1.8%) |
| **Do audio features matter for new songs?** | Yes — Audio-Only ROC-AUC jumps from 0.5350 → 0.6040 for cold-start tracks |
| **When and where do I skip most?** | Late-night, Android mobile, shuffle mode, session beginnings |

---

## Dataset

- **Source**: 13 chronological JSON exports (March 2021 – February 2026) + Kaggle Spotify catalog parquets (audio features + track metadata)
- **Scale**: 193,464 streams across 7,705 unique tracks
- **Match Rate**: 96.81% of streams matched to audio features with no systematic selection bias

![Figure 1: Match Rate Analysis](results/figures/01_match_rate_analysis.png)

---

## Target Definition

**Primary target (`skipped_fwd`)**: Binary — did the user press the forward skip button (`reason_end == 'fwdbtn'`)? This captures deliberate user intent, excluding passive exits. Results are robust across alternative definitions (short-play < 30s, telemetry flag).

---

## Model Benchmark

All models evaluated on identical **chronological splits** (70/15/15) with validation-tuned thresholds and bootstrap 95% CIs.

| Model | PR-AUC | ROC-AUC | Accuracy | Log Loss |
| :--- | :---: | :---: | :---: | :---: |
| Majority Baseline | 0.5088 | 0.5000 | 0.5088 | 0.6980 |
| Track Skip Rate Heuristic | 0.5841 | 0.6082 | 0.5571 | 0.6863 |
| Logistic Regression (L2) | 0.8239 | 0.8314 | 0.7519 | 0.5055 |
| Neural MLP | 0.8301 | 0.8509 | 0.7696 | 0.4909 |
| Random Forest | 0.8775 | 0.8793 | 0.7977 | 0.4328 |
| **Gradient Boosting (GBDT)** | **0.8871** | **0.8949** | **0.8143** | **0.4088** |

![Figure 7: Model Performance Curves](results/figures/07_model_roc_pr_curves.png)

---

## Key Experiments

### Feature-Group Ablation

| Feature Group | PR-AUC | ROC-AUC |
| :--- | :---: | :---: |
| Audio Only | 0.5423 | 0.5373 |
| Context Only | 0.8681 | 0.8642 |
| Context + History | 0.8775 | 0.8791 |
| **All Features** | **0.8871** | **0.8949** |

> Adding audio features to Context + History delivers **+1.09% PR-AUC lift**.

![Figure 8: Feature Ablation](results/figures/08_feature_group_ablation.png)

### Seen vs. Unseen (Cold-Start) Tracks

For familiar tracks, habit overshadows audio. But for **novel tracks** where familiarity is zero, audio-only ROC-AUC jumps from 0.5350 → **0.6040** — audio features serve as the primary content filter before user habits form.

![Figure 9: Seen vs Unseen](results/figures/09_seen_vs_unseen_comparison.png)

### Feature Importance

Top predictors via permutation importance:
1. `time_since_last_play_min` — rapid track changes signal skip-surfing
2. `prev_track_skipped` — "skipping mood" carries forward
3. `reason_start` — clicked songs skip far less than autoplay
4. `skips_in_last_5` — rolling measure of session dissatisfaction
5. `duration_sec` — longer songs get skipped more

![Figure 12: Feature Importance](results/figures/12_shap_summary.png)

---

## Limitations

1. **Single-user dataset** — findings reflect one person's habits; feature weights may not generalize
2. **Skip ≠ dislike** — forward button press can signal situational needs, not musical aversion
3. **Audio features are estimates** — Spotify's danceability/valence are black-box model outputs, not ground-truth measurements

---

## Tech Stack

| Tool | Use |
| :--- | :--- |
| **DuckDB** | High-performance SQL for ingesting and joining 14+ GB parquet files |
| **pandas / NumPy** | Feature engineering, data manipulation |
| **scikit-learn** | Model training (HistGBDT, Random Forest, Logistic Regression, MLP) |
| **matplotlib / seaborn** | Publication-quality visualizations |

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Results are saved to `results/figures/` and `results/tables/`.

---

## Project Structure

```
├── src/
│   ├── config.py           # Paths, parameters, feature definitions
│   ├── ingest.py           # Data ingestion, parquet joins, match-rate audit
│   ├── eda.py              # Exploratory data analysis & visualizations
│   ├── features.py         # Causal feature engineering (zero leakage)
│   ├── models.py           # Baselines and ML model definitions
│   ├── evaluate.py         # Chronological splits, metrics, bootstrap CIs
│   ├── experiments.py      # Feature ablation, cold-start, drift experiments
│   └── interpretation.py   # Permutation importance & error case studies
├── data/                   # (Gitignored) Cleaned parquet files
├── results/
│   ├── figures/            # Generated plots
│   └── tables/             # CSV summary tables
├── main.py                 # Run the full pipeline
├── requirements.txt        # Dependencies
└── README.md
```
