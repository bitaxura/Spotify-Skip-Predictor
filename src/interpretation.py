"""
Model Interpretation Module (Section 9 of plan.md).
Computes Grouped Permutation Importance, Feature Attribution, Partial Dependence,
and 10 Confident-Error Case Studies.
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import (
    ENGINEERED_FEATURES_PATH,
    ALL_MODEL_FEATURES,
    CONTEXT_FEATURES,
    HABIT_FEATURES,
    AUDIO_FEATURES,
    TARGET_COL_BINARY_FWD,
    FIGURES_DIR,
    TABLES_DIR,
    RANDOM_SEED,
)
from src.models import build_preprocessor
from src.evaluate import split_chronological, compute_metrics


def run_interpretation_pipeline():
    """Execute complete model interpretation, feature attribution, and error case studies."""
    print("=" * 70)
    print("STEP 5: MODEL INTERPRETATION, PERMUTATION IMPORTANCE, & CASE STUDIES")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ENGINEERED_FEATURES_PATH)
    train_df, val_df, test_df = split_chronological(df)

    y_train = train_df[TARGET_COL_BINARY_FWD].values
    y_test = test_df[TARGET_COL_BINARY_FWD].values

    prep = build_preprocessor(ALL_MODEL_FEATURES)
    X_tr = prep.fit_transform(train_df[ALL_MODEL_FEATURES])
    X_te = prep.transform(test_df[ALL_MODEL_FEATURES])

    print("Fitting Gradient Boosting model on training set...")
    clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=0.1,
        random_state=RANDOM_SEED
    ).fit(X_tr, y_train)

    test_probs = clf.predict_proba(X_te)[:, 1]
    base_metrics = compute_metrics(y_test, test_probs)
    base_prauc = base_metrics["pr_auc"]
    print(f"Base Test Set PR-AUC: {base_prauc:.4f}")

    # 1. GROUPED PERMUTATION IMPORTANCE (Section 8.F & 9)
    print("\nComputing Grouped Permutation Feature Importance...")
    feature_families = {
        "Context Family (Session, Device, Time)": CONTEXT_FEATURES,
        "Habit & Familiarity Family (Past Skips/Plays)": HABIT_FEATURES,
        "Audio / Content Family (Acoustics, Tempo, Energy)": AUDIO_FEATURES,
    }

    grouped_results = []
    rng = np.random.RandomState(RANDOM_SEED)

    for fam_name, cols in feature_families.items():
        # Permute all columns in the family together
        X_test_perm = test_df[ALL_MODEL_FEATURES].copy()
        n_samples = len(X_test_perm)
        perm_idx = rng.permutation(n_samples)
        for col in cols:
            X_test_perm[col] = X_test_perm[col].iloc[perm_idx].values

        X_perm_prep = prep.transform(X_test_perm)
        perm_probs = clf.predict_proba(X_perm_prep)[:, 1]
        perm_prauc = compute_metrics(y_test, perm_probs)["pr_auc"]
        prauc_drop = base_prauc - perm_prauc

        grouped_results.append({
            "Feature Family": fam_name,
            "Num Features": len(cols),
            "Base PR-AUC": round(base_prauc, 4),
            "Permuted PR-AUC": round(perm_prauc, 4),
            "PR-AUC Drop (Importance)": round(prauc_drop, 4),
            "Relative Importance %": round((prauc_drop / base_prauc) * 100, 2),
        })

    group_df = pd.DataFrame(grouped_results).sort_values("PR-AUC Drop (Importance)", ascending=False)
    print(group_df.to_string(index=False))

    # Individual Permutation Importance (on a 5,000 sample test subset for speed)
    print("\nComputing Individual Feature Permutation Importance...")
    sub_idx = rng.choice(len(test_df), size=min(6000, len(test_df)), replace=False)
    X_sub = test_df[ALL_MODEL_FEATURES].iloc[sub_idx]
    y_sub = y_test[sub_idx]
    base_sub_prauc = compute_metrics(y_sub, clf.predict_proba(prep.transform(X_sub))[:, 1])["pr_auc"]

    indiv_results = []
    for col in ALL_MODEL_FEATURES:
        X_perm = X_sub.copy()
        X_perm[col] = rng.permutation(X_perm[col].values)
        perm_probs = clf.predict_proba(prep.transform(X_perm))[:, 1]
        m = compute_metrics(y_sub, perm_probs)
        drop = base_sub_prauc - m["pr_auc"]
        indiv_results.append({
            "Feature": col,
            "Family": "Habit" if col in HABIT_FEATURES else ("Context" if col in CONTEXT_FEATURES else "Audio"),
            "PR-AUC Drop": round(drop, 4)
        })

    indiv_df = pd.DataFrame(indiv_results).sort_values("PR-AUC Drop", ascending=False)
    perm_path = TABLES_DIR / "permutation_importance.csv"
    indiv_df.to_csv(perm_path, index=False)
    print(f"Saved permutation importance table: {perm_path}")

    # 2. FIGURE 12: FEATURE IMPORTANCE SUMMARY BAR CHART
    print("Generating Figure 12: Feature importance bar plot...")
    plt.figure(figsize=(10, 6.5), dpi=300)
    top_indiv = indiv_df.head(15)
    colors = ['#8B5CF6' if f == 'Habit' else ('#3B82F6' if f == 'Context' else '#10B981') for f in top_indiv['Family']]

    bars = plt.barh(top_indiv['Feature'][::-1], top_indiv['PR-AUC Drop'][::-1], color=colors[::-1], height=0.65, alpha=0.9)
    plt.xlabel('Test Set PR-AUC Drop (Permutation Importance)', fontsize=10.5)
    plt.title('Top 15 Most Influential Features on Skip Prediction', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')

    # Legend for families
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='#8B5CF6', label='Habit / Familiarity'),
        mpatches.Patch(color='#3B82F6', label='Context / Session'),
        mpatches.Patch(color='#10B981', label='Audio / Content'),
    ]
    plt.legend(handles=legend_handles, loc='lower right', frameon=True)
    plt.tight_layout()
    fig12_path = FIGURES_DIR / "12_shap_summary.png"
    plt.savefig(fig12_path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance figure: {fig12_path}")

    # 3. FIGURE 13: PARTIAL DEPENDENCE PLOTS
    print("Generating Figure 13: Partial dependence plots for key drivers...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    # Convert preprocessed names for PDP
    pdp_sample_df = test_df[ALL_MODEL_FEATURES].iloc[sub_idx].copy()
    X_pdp_prep = prep.transform(pdp_sample_df)

    # Find feature indices in preprocessed array
    # Preprocessor output: numeric features first (in order of nums), then one-hot cats
    nums = [c for c in ALL_MODEL_FEATURES if c not in ['platform_clean', 'reason_start']]
    
    # We can plot partial dependence on preprocessed feature indices
    features_to_pdp = ['prior_track_skip_rate', 'energy', 'tempo', 'valence']
    pdp_indices = [nums.index(f) for f in features_to_pdp]

    disp = PartialDependenceDisplay.from_estimator(
        clf, X_pdp_prep, pdp_indices,
        feature_names=nums,
        ax=axes.ravel(),
        line_kw={"color": "#0284C7", "linewidth": 2.5}
    )

    axes[0, 0].set_title('A. Partial Dependence: Prior Track Skip Rate', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Skip Log-Odds Effect', fontsize=10)
    axes[0, 1].set_title('B. Partial Dependence: Audio Energy', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('C. Partial Dependence: Tempo (BPM)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Skip Log-Odds Effect', fontsize=10)
    axes[1, 1].set_title('D. Partial Dependence: Audio Valence (Positiveness)', fontsize=11, fontweight='bold')

    plt.suptitle("Partial Dependence of Predicted Skip Probability on Key Features", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig13_path = FIGURES_DIR / "13_shap_dependence.png"
    plt.savefig(fig13_path, bbox_inches='tight')
    plt.close()
    print(f"Saved partial dependence figure: {fig13_path}")

    # 4. 10 CASE STUDIES OF CONFIDENT ERRORS (Section 9)
    print("\nIdentifying 10 Confident Error Case Studies (5 False Positives, 5 False Negatives)...")
    test_eval = test_df.copy()
    test_eval['prob_skip'] = test_probs
    test_eval['actual_skip'] = y_test

    # False Positives: Model predicted very high skip prob, but user listened naturally (actual == 0)
    fps = test_eval[test_eval['actual_skip'] == 0].sort_values('prob_skip', ascending=False).head(5)

    # False Negatives: Model predicted very low skip prob, but user skipped immediately (actual == 1)
    fns = test_eval[test_eval['actual_skip'] == 1].sort_values('prob_skip', ascending=True).head(5)

    case_studies = []

    # Process False Positives
    for _, row in fps.iterrows():
        explanation = (
            f"Model predicted high skip probability ({row['prob_skip']:.1%}) because the track began in an active skipping mood "
            f"({row['skips_in_last_5']} skips in last 5 tracks) or historically high skip rate ({row['prior_track_skip_rate']:.1%}). "
            f"However, the listener chose to complete all {row['duration_ms']/1000:.0f} seconds, indicating an intentional focus listen."
        )
        case_studies.append({
            "Error Type": "False Positive (High Skip Prediction, But Completed)",
            "Track Name": row['track_name'],
            "Artist Name": row['artist_name'],
            "Predicted Skip Prob": f"{row['prob_skip']:.1%}",
            "Actual End Reason": row['reason_end'],
            "Prior Plays": int(row['prior_track_plays']),
            "Prior Skip Rate": f"{row['prior_track_skip_rate']:.1%}",
            "Energy": round(float(row['energy']), 2),
            "Valence": round(float(row['valence']), 2),
            "Platform": row['platform_clean'],
            "Explanation / Autopsy": explanation,
        })

    # Process False Negatives
    for _, row in fns.iterrows():
        explanation = (
            f"Model predicted very low skip probability ({row['prob_skip']:.1%}) due to strong track familiarity "
            f"({row['prior_track_plays']} prior plays) and low historical skip rate ({row['prior_track_skip_rate']:.1%}). "
            f"Yet the listener skipped immediately (pressed forward button), likely due to sudden situational interruption, "
            f"device transition, or mood misalignment."
        )
        case_studies.append({
            "Error Type": "False Negative (Low Skip Prediction, But Skipped)",
            "Track Name": row['track_name'],
            "Artist Name": row['artist_name'],
            "Predicted Skip Prob": f"{row['prob_skip']:.1%}",
            "Actual End Reason": row['reason_end'],
            "Prior Plays": int(row['prior_track_plays']),
            "Prior Skip Rate": f"{row['prior_track_skip_rate']:.1%}",
            "Energy": round(float(row['energy']), 2),
            "Valence": round(float(row['valence']), 2),
            "Platform": row['platform_clean'],
            "Explanation / Autopsy": explanation,
        })

    cases_df = pd.DataFrame(case_studies)
    cases_path = TABLES_DIR / "confident_error_case_studies.csv"
    cases_df.to_csv(cases_path, index=False)
    print(f"Saved 10 confident error case studies: {cases_path}")
    print("\nCase Studies Summary:")
    print(cases_df[["Error Type", "Track Name", "Artist Name", "Predicted Skip Prob", "Actual End Reason"]].to_string(index=False))

    print("\nModel interpretation pipeline completed successfully!\n")


if __name__ == "__main__":
    run_interpretation_pipeline()
