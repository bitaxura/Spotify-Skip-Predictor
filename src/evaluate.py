"""
Evaluation framework, chronological splitting, metric computation, bootstrap CIs, and curve plotting.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve

from src.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    FIGURES_DIR,
    TABLES_DIR,
)


def split_chronological(df):
    """
    Splits DataFrame chronologically into Train (70%), Validation (15%), and Test (15%).
    Never shuffles or leaks future information.
    """
    df_sorted = df.sort_values('ts').reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    print(f"Chronological split completed:")
    print(f"  Train:      {len(train_df):,} plays ({train_df['ts'].min().date()} to {train_df['ts'].max().date()})")
    print(f"  Validation: {len(val_df):,} plays ({val_df['ts'].min().date()} to {val_df['ts'].max().date()})")
    print(f"  Test:       {len(test_df):,} plays ({test_df['ts'].min().date()} to {test_df['ts'].max().date()})")

    return train_df, val_df, test_df


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score on validation set."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return best_thresh, best_f1


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Calculate comprehensive classification metrics."""
    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, y_prob_clipped, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_prob_clipped),
        "threshold": threshold,
    }
    return metrics


def compute_bootstrap_ci(y_true, y_prob, metric_func, n_bootstraps=500, random_state=RANDOM_SEED):
    """Computes 95% non-parametric bootstrap confidence interval for a given metric."""
    rng = np.random.RandomState(random_state)
    boot_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_true = y_true[idx]
        sample_prob = y_prob[idx]
        # Skip if sample contains only 1 class
        if len(np.unique(sample_true)) < 2:
            continue
        boot_scores.append(metric_func(sample_true, sample_prob))

    low = np.percentile(boot_scores, 2.5)
    high = np.percentile(boot_scores, 97.5)
    return low, high


def compute_regression_metrics(y_true, y_pred):
    """Computes evaluation metrics for continuous target fraction_played."""
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = root_mean_squared_error(y_true, y_pred_clipped)
    r2 = r2_score(y_true, y_pred_clipped)
    spearman_corr, _ = spearmanr(y_true, y_pred_clipped)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_rho": spearman_corr
    }


def plot_model_curves(model_predictions, y_test, output_path=None):
    """
    Figure 07: Precision-Recall and ROC curves across all models in ladder.
    model_predictions: dict mapping model_name -> y_prob_test
    """
    fig, (ax_roc, ax_pr, ax_cal) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    colors = ['#94A3B8', '#F59E0B', '#EF4444', '#10B981', '#3B82F6', '#8B5CF6', '#059669', '#EC4899']

    for (name, y_prob), col in zip(model_predictions.items(), colors):
        if name == "Majority Baseline":
            continue
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=col, linewidth=2)

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax_pr.plot(rec, prec, label=f"{name} ({pr_auc:.3f})", color=col, linewidth=2)

        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker='s', label=f"{name}", color=col, linewidth=1.8, markersize=4)

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (0.500)')
    ax_roc.set_title('A. Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate', fontsize=10)
    ax_roc.set_ylabel('True Positive Rate', fontsize=10)
    ax_roc.legend(loc='lower right', fontsize=8, frameon=True)
    ax_roc.grid(True, linestyle='--', alpha=0.5)

    # PR formatting
    baseline_rate = np.mean(y_test)
    ax_pr.plot([0, 1], [baseline_rate, baseline_rate], 'k--', alpha=0.5, label=f'Baseline ({baseline_rate:.3f})')
    ax_pr.set_title('B. Precision-Recall Curve (PR-AUC)', fontsize=12, fontweight='bold')
    ax_pr.set_xlabel('Recall', fontsize=10)
    ax_pr.set_ylabel('Precision', fontsize=10)
    ax_pr.legend(loc='lower left', fontsize=8, frameon=True)
    ax_pr.grid(True, linestyle='--', alpha=0.5)

    # Calibration formatting
    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax_cal.set_title('C. Calibration Curve (Reliability)', fontsize=12, fontweight='bold')
    ax_cal.set_xlabel('Mean Predicted Probability', fontsize=10)
    ax_cal.set_ylabel('Fraction of Positives', fontsize=10)
    ax_cal.legend(loc='upper left', fontsize=8, frameon=True)
    ax_cal.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Model Ladder Performance Comparison on Test Set (N=25,810)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path is None:
        output_path = FIGURES_DIR / "07_model_roc_pr_curves.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved model performance curves: {output_path}")
