"""
Core Research Experiments Module (Section 8 of plan.md).
Executes Model Ladder, Feature-Group Ablations, Seen vs Unseen analysis, Target Robustness,
Cold-Start Curve, and Temporal Drift analysis.
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.config import (
    ENGINEERED_FEATURES_PATH,
    ALL_MODEL_FEATURES,
    CONTEXT_FEATURES,
    HABIT_FEATURES,
    AUDIO_FEATURES,
    TARGET_COL_BINARY_FWD,
    TARGET_COL_BINARY_30S,
    TARGET_COL_BINARY_FLAG,
    TARGET_COL_CONTINUOUS,
    FIGURES_DIR,
    TABLES_DIR,
    RANDOM_SEED,
)
from src.models import get_model_ladder, build_preprocessor
from src.evaluate import (
    split_chronological,
    compute_metrics,
    find_optimal_threshold,
    compute_bootstrap_ci,
    compute_regression_metrics,
    plot_model_curves,
)


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def run_all_experiments():
    """Run all benchmarks and experiments A through E."""
    print("=" * 70)
    print("STEP 4: MODEL LADDER AND EXPERIMENTS (A, B, C, D, E)")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading engineered feature dataset from: {ENGINEERED_FEATURES_PATH}")
    df = pd.read_parquet(ENGINEERED_FEATURES_PATH)
    print(f"Total plays loaded: {len(df):,}")

    train_df, val_df, test_df = split_chronological(df)

    y_train = train_df[TARGET_COL_BINARY_FWD].values
    y_val = val_df[TARGET_COL_BINARY_FWD].values
    y_test = test_df[TARGET_COL_BINARY_FWD].values

    # 1. MODEL LADDER BENCHMARK
    print("\n" + "=" * 50)
    print("1. RUNNING MODEL LADDER BENCHMARK")
    print("=" * 50)
    
    ladder_results = []
    test_predictions = {}
    models = get_model_ladder(ALL_MODEL_FEATURES)

    for name, model in models.items():
        print(f"--> Training {name}...")
        model.fit(train_df[ALL_MODEL_FEATURES], y_train)

        # Predict probabilities
        val_probs = model.predict_proba(val_df[ALL_MODEL_FEATURES])[:, 1]
        test_probs = model.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 1]
        test_predictions[name] = test_probs

        # Tune threshold on validation set
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, val_probs)
        
        # Test metrics
        metrics = compute_metrics(y_test, test_probs, threshold=opt_thresh)

        # Bootstrap 95% CI on Test set for PR-AUC and ROC-AUC
        pr_low, pr_high = compute_bootstrap_ci(
            y_test, test_probs,
            metric_func=lambda yt, yp: compute_metrics(yt, yp)["pr_auc"],
            n_bootstraps=300
        )
        roc_low, roc_high = compute_bootstrap_ci(
            y_test, test_probs,
            metric_func=lambda yt, yp: compute_metrics(yt, yp)["roc_auc"],
            n_bootstraps=300
        )

        res = {
            "Model": name,
            "PR-AUC": round(metrics["pr_auc"], 4),
            "PR-AUC 95% CI": f"[{pr_low:.3f}, {pr_high:.3f}]",
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "ROC-AUC 95% CI": f"[{roc_low:.3f}, {roc_high:.3f}]",
            "F1 Score": round(metrics["f1"], 4),
            "Optimal Thresh": round(opt_thresh, 3),
            "Accuracy": round(metrics["accuracy"], 4),
            "Log Loss": round(metrics["log_loss"], 4),
            "Brier Score": round(metrics["brier_score"], 4),
        }
        ladder_results.append(res)
        print(f"    PR-AUC: {res['PR-AUC']} {res['PR-AUC 95% CI']} | ROC-AUC: {res['ROC-AUC']} | F1: {res['F1 Score']} | Log Loss: {res['Log Loss']}")

    ladder_df = pd.DataFrame(ladder_results)
    ladder_path = TABLES_DIR / "model_ladder_comparison.csv"
    ladder_df.to_csv(ladder_path, index=False)
    print(f"\nSaved model ladder comparison table to: {ladder_path}")

    # Generate Figure 07: Curves
    plot_model_curves(test_predictions, y_test)

    # Continuous Target Benchmark (fraction_played)
    print("\nEvaluating Continuous Target (fraction_played)...")
    y_train_cont = train_df[TARGET_COL_CONTINUOUS].values
    y_val_cont = val_df[TARGET_COL_CONTINUOUS].values
    y_test_cont = test_df[TARGET_COL_CONTINUOUS].values

    prep = build_preprocessor(ALL_MODEL_FEATURES)
    X_train_prep = prep.fit_transform(train_df[ALL_MODEL_FEATURES])
    X_test_prep = prep.transform(test_df[ALL_MODEL_FEATURES])

    mean_reg = MeanBaselineRegressor().fit(X_train_prep, y_train_cont)
    ridge_reg = Ridge(alpha=1.0).fit(X_train_prep, y_train_cont)
    lgb_reg = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=31, random_state=RANDOM_SEED
    ).fit(X_train_prep, y_train_cont)

    cont_res = [
        {"Model": "Mean Baseline", **compute_regression_metrics(y_test_cont, mean_reg.predict(X_test_prep))},
        {"Model": "Ridge Regression", **compute_regression_metrics(y_test_cont, ridge_reg.predict(X_test_prep))},
        {"Model": "Gradient Boosting Regressor", **compute_regression_metrics(y_test_cont, lgb_reg.predict(X_test_prep))},
    ]
    cont_df = pd.DataFrame(cont_res)
    cont_path = TABLES_DIR / "continuous_target_results.csv"
    cont_df.to_csv(cont_path, index=False)
    print(f"Continuous Target Results:\n{cont_df.to_string(index=False)}")

    # 2. EXPERIMENT A: FEATURE-GROUP ABLATION
    print("\n" + "=" * 50)
    print("2. EXPERIMENT A: FEATURE-GROUP ABLATION")
    print("=" * 50)
    run_experiment_a_ablation(train_df, val_df, test_df, y_train, y_val, y_test)

    # 3. EXPERIMENT B: SEEN VS UNSEEN TRACKS
    print("\n" + "=" * 50)
    print("3. EXPERIMENT B: SEEN VS. UNSEEN TRACKS")
    print("=" * 50)
    run_experiment_b_seen_vs_unseen(train_df, val_df, test_df, y_train, y_val, y_test)

    # 4. EXPERIMENT C: TARGET DEFINITION ROBUSTNESS
    print("\n" + "=" * 50)
    print("4. EXPERIMENT C: TARGET DEFINITION ROBUSTNESS")
    print("=" * 50)
    run_experiment_c_robustness(train_df, val_df, test_df)

    # 5. EXPERIMENT D: COLD-START FAMILIARITY CURVE
    print("\n" + "=" * 50)
    print("5. EXPERIMENT D: COLD-START FAMILIARITY CURVE")
    print("=" * 50)
    run_experiment_d_cold_start(train_df, val_df, test_df, y_train, y_val, y_test)

    # 6. EXPERIMENT E: TEMPORAL DRIFT ANALYSIS
    print("\n" + "=" * 50)
    print("6. EXPERIMENT E: TEMPORAL DRIFT ANALYSIS")
    print("=" * 50)
    run_experiment_e_temporal_drift(train_df, val_df, test_df, y_train, y_val, y_test)

    print("\nAll experiments completed successfully!\n")


def run_experiment_a_ablation(train_df, val_df, test_df, y_train, y_val, y_test):
    """
    Feature-group ablation:
    1. Context only
    2. History/Habit only
    3. Audio only
    4. Context + History
    5. All (Context + History + Audio)
    Measures the incremental lift provided by audio features.
    """
    ablation_groups = {
        "Audio Only": AUDIO_FEATURES,
        "Context Only": CONTEXT_FEATURES,
        "History Only": HABIT_FEATURES,
        "Context + History": CONTEXT_FEATURES + HABIT_FEATURES,
        "All Features (Context + History + Audio)": ALL_MODEL_FEATURES,
    }

    results = []
    trained_models = {}

    for grp_name, feat_cols in ablation_groups.items():
        print(f"--> Training LightGBM on: {grp_name} ({len(feat_cols)} features)...")
        prep = build_preprocessor(feat_cols)
        X_tr = prep.fit_transform(train_df[feat_cols])
        X_va = prep.transform(val_df[feat_cols])
        X_te = prep.transform(test_df[feat_cols])

        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=0.1,
            random_state=RANDOM_SEED
        )
        clf.fit(X_tr, y_train)

        val_probs = clf.predict_proba(X_va)[:, 1]
        test_probs = clf.predict_proba(X_te)[:, 1]
        trained_models[grp_name] = (clf, prep, test_probs)

        opt_th, _ = find_optimal_threshold(y_val, val_probs)
        m = compute_metrics(y_test, test_probs, threshold=opt_th)

        results.append({
            "Feature Group": grp_name,
            "Num Features": len(feat_cols),
            "PR-AUC": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "F1 Score": round(m["f1"], 4),
            "Log Loss": round(m["log_loss"], 4),
            "Brier Score": round(m["brier_score"], 4),
        })

    ablation_df = pd.DataFrame(results)

    # Compute lifts
    ctx_hist_pr = ablation_df.loc[ablation_df["Feature Group"] == "Context + History", "PR-AUC"].values[0]
    all_pr = ablation_df.loc[ablation_df["Feature Group"] == "All Features (Context + History + Audio)", "PR-AUC"].values[0]
    audio_lift_pr = all_pr - ctx_hist_pr
    audio_lift_pct = (audio_lift_pr / ctx_hist_pr) * 100

    print(f"\nIncremental Audio Lift over (Context + History):")
    print(f"  PR-AUC Lift: +{audio_lift_pr:.4f} (+{audio_lift_pct:.2f}%)")

    ablation_path = TABLES_DIR / "feature_ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Saved ablation results table: {ablation_path}")

    # Plot Figure 08: Ablation Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    colors = ['#0284C7', '#F59E0B', '#8B5CF6', '#3B82F6', '#10B981']
    bars = plt.bar(ablation_df['Feature Group'], ablation_df['PR-AUC'], color=colors, width=0.55, alpha=0.9)
    plt.axhline(ctx_hist_pr, color='#3B82F6', linestyle='--', alpha=0.7, label=f'Context+History Baseline ({ctx_hist_pr:.4f})')
    plt.ylabel('Test Set PR-AUC', fontsize=11)
    plt.title(f'Feature-Group Ablation: Audio Features Lift = +{audio_lift_pr:.4f} (+{audio_lift_pct:.2f}%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontsize=9.5)
    plt.ylim(0.50, max(ablation_df['PR-AUC']) + 0.05)
    plt.legend(frameon=True, loc='upper left')

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_path = FIGURES_DIR / "08_feature_group_ablation.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation figure: {fig_path}")


def run_experiment_b_seen_vs_unseen(train_df, val_df, test_df, y_train, y_val, y_test):
    """
    Experiment B: Seen vs. Unseen Tracks.
    Evaluates whether audio features become the primary signal for tracks the user has never heard before.
    """
    # Identify seen vs unseen in test set based strictly on prior_track_plays (computed causally)
    is_unseen = (test_df['prior_track_plays'] == 0).values
    is_seen = (test_df['prior_track_plays'] > 0).values

    print(f"Test split breakdown:")
    print(f"  Seen tracks:   {is_seen.sum():,} plays ({is_seen.mean()*100:.1f}%)")
    print(f"  Unseen tracks: {is_unseen.sum():,} plays ({is_unseen.mean()*100:.1f}%)")

    # Fit 3 models on training data: Audio Only, Context+History, All Features
    models_configs = {
        "Audio Only": AUDIO_FEATURES,
        "Context + History": CONTEXT_FEATURES + HABIT_FEATURES,
        "All Features": ALL_MODEL_FEATURES,
    }

    seen_unseen_results = []

    for name, cols in models_configs.items():
        prep = build_preprocessor(cols)
        X_tr = prep.fit_transform(train_df[cols])
        X_te = prep.transform(test_df[cols])

        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=0.1,
            random_state=RANDOM_SEED
        ).fit(X_tr, y_train)

        test_probs = clf.predict_proba(X_te)[:, 1]

        # Evaluate on Seen
        m_seen = compute_metrics(y_test[is_seen], test_probs[is_seen])
        # Evaluate on Unseen
        m_unseen = compute_metrics(y_test[is_unseen], test_probs[is_unseen])

        seen_unseen_results.append({
            "Model": name,
            "Cohort": "Seen Tracks (Familiar)",
            "N": int(is_seen.sum()),
            "PR-AUC": round(m_seen["pr_auc"], 4),
            "ROC-AUC": round(m_seen["roc_auc"], 4),
            "F1 Score": round(m_seen["f1"], 4),
            "Log Loss": round(m_seen["log_loss"], 4),
        })
        seen_unseen_results.append({
            "Model": name,
            "Cohort": "Unseen Tracks (Cold-Start)",
            "N": int(is_unseen.sum()),
            "PR-AUC": round(m_unseen["pr_auc"], 4),
            "ROC-AUC": round(m_unseen["roc_auc"], 4),
            "F1 Score": round(m_unseen["f1"], 4),
            "Log Loss": round(m_unseen["log_loss"], 4),
        })

    su_df = pd.DataFrame(seen_unseen_results)
    su_path = TABLES_DIR / "seen_vs_unseen_results.csv"
    su_df.to_csv(su_path, index=False)
    print(f"\nSeen vs Unseen Evaluation:\n{su_df.to_string(index=False)}")

    # Plot Figure 09
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    seen_df = su_df[su_df['Cohort'].str.startswith('Seen')].set_index('Model')['PR-AUC']
    unseen_df = su_df[su_df['Cohort'].str.startswith('Unseen')].set_index('Model')['PR-AUC']

    x = np.arange(len(seen_df))
    width = 0.35
    ax.bar(x - width/2, seen_df, width, label='Seen Tracks (Familiar)', color='#3B82F6', alpha=0.85)
    ax.bar(x + width/2, unseen_df, width, label='Unseen Tracks (Cold Start)', color='#F59E0B', alpha=0.85)

    ax.set_ylabel('PR-AUC on Cohort', fontsize=11)
    ax.set_title('Sub-Question 3: For Unseen Tracks, Do Audio Features Dominate?', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seen_df.index, fontsize=10)
    ax.set_ylim(0.40, max(seen_df.max(), unseen_df.max()) + 0.08)
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    for i in x:
        ax.text(i - width/2, seen_df.iloc[i] + 0.008, f'{seen_df.iloc[i]:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, unseen_df.iloc[i] + 0.008, f'{unseen_df.iloc[i]:.4f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_path = FIGURES_DIR / "09_seen_vs_unseen_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved seen vs unseen figure: {fig_path}")


def run_experiment_c_robustness(train_df, val_df, test_df):
    """
    Experiment C: Target Definition Robustness.
    Compares 3 skip definitions:
    1. Forward-button skip: reason_end == 'fwdbtn' (primary)
    2. Short-play skip: ms_played < 30000 (industry standard)
    3. Telemetry skip: skipped == True
    """
    target_defs = [
        ("Forward-Button Skip (reason_end='fwdbtn')", TARGET_COL_BINARY_FWD),
        ("Short-Play Skip (played < 30 seconds)", TARGET_COL_BINARY_30S),
        ("Spotify Telemetry Flag (skipped=True)", TARGET_COL_BINARY_FLAG),
    ]

    robust_res = []

    prep = build_preprocessor(ALL_MODEL_FEATURES)
    X_tr = prep.fit_transform(train_df[ALL_MODEL_FEATURES])
    X_va = prep.transform(val_df[ALL_MODEL_FEATURES])
    X_te = prep.transform(test_df[ALL_MODEL_FEATURES])

    for label, target_col in target_defs:
        y_tr = train_df[target_col].values
        y_va = val_df[target_col].values
        y_te = test_df[target_col].values

        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=0.1,
            random_state=RANDOM_SEED
        ).fit(X_tr, y_tr)

        val_probs = clf.predict_proba(X_va)[:, 1]
        test_probs = clf.predict_proba(X_te)[:, 1]

        opt_th, _ = find_optimal_threshold(y_va, val_probs)
        m = compute_metrics(y_te, test_probs, threshold=opt_th)

        robust_res.append({
            "Target Definition": label,
            "Positive Class %": round(np.mean(y_te) * 100, 2),
            "PR-AUC": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "F1 Score": round(m["f1"], 4),
            "Accuracy": round(m["accuracy"], 4),
            "Log Loss": round(m["log_loss"], 4),
        })

    rob_df = pd.DataFrame(robust_res)
    rob_path = TABLES_DIR / "target_robustness_results.csv"
    rob_df.to_csv(rob_path, index=False)
    print(f"\nTarget Definition Robustness Results:\n{rob_df.to_string(index=False)}")


def run_experiment_d_cold_start(train_df, val_df, test_df, y_train, y_val, y_test):
    """
    Experiment D: Cold-Start Familiarity Curve.
    Tracks test set performance as a function of prior play count (0, 1, 2-5, 6-10, 11-25, 25+).
    """
    prep = build_preprocessor(ALL_MODEL_FEATURES)
    X_tr = prep.fit_transform(train_df[ALL_MODEL_FEATURES])
    X_te = prep.transform(test_df[ALL_MODEL_FEATURES])

    clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=0.1,
        random_state=RANDOM_SEED
    ).fit(X_tr, y_train)

    test_probs = clf.predict_proba(X_te)[:, 1]

    bins = [-1, 0, 1, 5, 10, 25, 10000]
    labels = ['0 (Cold)', '1', '2-5', '6-10', '11-25', '25+']
    test_df_copy = test_df.copy()
    test_df_copy['fam_bracket'] = pd.cut(test_df_copy['prior_track_plays'], bins=bins, labels=labels)
    test_df_copy['prob'] = test_probs

    bracket_stats = []
    for br in labels:
        subset = test_df_copy[test_df_copy['fam_bracket'] == br]
        y_sub = subset[TARGET_COL_BINARY_FWD].values
        p_sub = subset['prob'].values
        m = compute_metrics(y_sub, p_sub)
        bracket_stats.append({
            "Familiarity Bracket": br,
            "Count": len(subset),
            "Observed Skip %": round(np.mean(y_sub) * 100, 2),
            "PR-AUC": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Brier Score": round(m["brier_score"], 4),
        })

    b_df = pd.DataFrame(bracket_stats)

    # Plot Figure 10
    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=300)
    color_prauc = '#10B981'
    color_brier = '#EF4444'

    x = np.arange(len(b_df))
    ax1.plot(x, b_df['PR-AUC'], color=color_prauc, marker='o', linewidth=2.5, markersize=6, label='PR-AUC')
    ax1.set_xlabel('Prior Play Count Bracket (Familiarity)', fontsize=11)
    ax1.set_ylabel('PR-AUC', color=color_prauc, fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(b_df['Familiarity Bracket'], fontsize=10)
    ax1.set_ylim(0.50, 0.95)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, b_df['Brier Score'], color=color_brier, marker='s', linewidth=2, linestyle='--', markersize=5, label='Brier Score (Error)')
    ax2.set_ylabel('Brier Score (Lower is Better)', color=color_brier, fontsize=11)
    ax2.set_ylim(0.10, 0.30)

    plt.title('Cold-Start Curve: Model Performance vs. Track Familiarity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path = FIGURES_DIR / "10_cold_start_curve.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved cold-start curve figure: {fig_path}")


def run_experiment_e_temporal_drift(train_df, val_df, test_df, y_train, y_val, y_test):
    """
    Experiment E: Temporal Habit Drift.
    Evaluates fixed model performance month-by-month over the test period to detect habit drift.
    """
    prep = build_preprocessor(ALL_MODEL_FEATURES)
    X_tr = prep.fit_transform(train_df[ALL_MODEL_FEATURES])
    X_te = prep.transform(test_df[ALL_MODEL_FEATURES])

    clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=0.1,
        random_state=RANDOM_SEED
    ).fit(X_tr, y_train)

    test_probs = clf.predict_proba(X_te)[:, 1]

    test_df_copy = test_df.copy()
    test_df_copy['year_month'] = test_df_copy['ts'].dt.to_period('M')
    test_df_copy['prob'] = test_probs

    monthly_stats = []
    for ym, subset in test_df_copy.groupby('year_month'):
        y_sub = subset[TARGET_COL_BINARY_FWD].values
        p_sub = subset['prob'].values
        if len(subset) < 50 or len(np.unique(y_sub)) < 2:
            continue
        m = compute_metrics(y_sub, p_sub)
        monthly_stats.append({
            "Month": str(ym),
            "Plays": len(subset),
            "Observed Skip %": round(np.mean(y_sub) * 100, 2),
            "PR-AUC": round(m["pr_auc"], 4),
            "Log Loss": round(m["log_loss"], 4),
        })

    m_df = pd.DataFrame(monthly_stats)

    # Plot Figure 11
    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=300)
    color_metric = '#3B82F6'
    color_rate = '#10B981'

    x = np.arange(len(m_df))
    ax1.plot(x, m_df['PR-AUC'], color=color_metric, marker='o', linewidth=2.5, markersize=6, label='Test PR-AUC')
    ax1.set_xlabel('Test Set Timeline (Monthly)', fontsize=11)
    ax1.set_ylabel('PR-AUC Metric', color=color_metric, fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_df['Month'], rotation=35, ha='right', fontsize=9)
    ax1.set_ylim(0.55, 0.90)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, m_df['Observed Skip %'], color=color_rate, marker='^', linestyle='--', linewidth=2, markersize=5, label='Observed Skip %')
    ax2.set_ylabel('Observed Skip Rate (%)', color=color_rate, fontsize=11)
    ax2.set_ylim(40, 75)

    plt.title('Temporal Stability Audit: Fixed Model PR-AUC Over Test Period', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path = FIGURES_DIR / "11_temporal_drift.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal drift figure: {fig_path}")


if __name__ == "__main__":
    run_all_experiments()
