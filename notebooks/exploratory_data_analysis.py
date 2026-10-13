import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    CLEANED_HISTORY_PATH,
    ENGINEERED_FEATURES_PATH,
    TABLES_DIR,
    FIGURES_DIR,
)

def inspect_summary_tables():
    print("=== 1. Match Rate Summary ===")
    match_df = pd.read_csv(TABLES_DIR / "match_rate_summary.csv")
    print(match_df.to_string(index=False))

    print("\n=== 2. Model Ladder Comparison (Test Set) ===")
    ladder_df = pd.read_csv(TABLES_DIR / "model_ladder_comparison.csv")
    print(ladder_df.to_string(index=False))

    print("\n=== 3. Feature-Group Ablation (Audio Lift) ===")
    ablation_df = pd.read_csv(TABLES_DIR / "feature_ablation_results.csv")
    print(ablation_df.to_string(index=False))

    print("\n=== 4. Seen vs. Unseen Tracks ===")
    seen_df = pd.read_csv(TABLES_DIR / "seen_vs_unseen_results.csv")
    print(seen_df.to_string(index=False))

    print("\n=== 5. Continuous Target Modeling (fraction_played) ===")
    cont_df = pd.read_csv(TABLES_DIR / "continuous_target_results.csv")
    print(cont_df.to_string(index=False))

    print("\n=== 6. Grouped & Individual Permutation Importance (Top 10) ===")
    perm_df = pd.read_csv(TABLES_DIR / "permutation_importance.csv")
    print(perm_df.head(10).to_string(index=False))

    print("\n=== 7. Confident Error Case Studies (Sample) ===")
    cases_df = pd.read_csv(TABLES_DIR / "confident_error_case_studies.csv")
    print(cases_df[["Error Type", "Track Name", "Artist Name", "Predicted Skip Prob", "Actual End Reason"]].to_string(index=False))

if __name__ == "__main__":
    inspect_summary_tables()
