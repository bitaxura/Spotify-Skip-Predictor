"""
Exploratory Data Analysis (EDA) module.
Generates comprehensive figures and summary tables according to Section 4 of plan.md.
"""
import os
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    CLEANED_HISTORY_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    SESSION_GAP_MINUTES,
)


def run_eda_pipeline():
    """Execute complete exploratory data analysis."""
    print("=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Load cleaned streaming dataset where user made a decision and audio features exist
    # Focus on matched cohort for audio analyses, and full user-decision cohort for context
    print("Loading cleaned dataset from DuckDB...")
    df = con.execute(f"""
        SELECT 
            ts,
            track_id,
            track_name,
            artist_name,
            platform_clean,
            reason_start,
            reason_end,
            shuffle,
            offline,
            skipped_fwd,
            skipped_30s,
            fraction_played,
            is_user_decision,
            danceability,
            energy,
            key,
            loudness,
            mode,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo,
            popularity,
            explicit,
            duration_ms
        FROM '{CLEANED_HISTORY_PATH}'
        WHERE is_user_decision = 1
        ORDER BY ts ASC
    """).df()

    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.hour
    df['day_of_week'] = df['ts'].dt.day_name()
    df['year_month'] = df['ts'].dt.to_period('M')

    print(f"Loaded {len(df):,} user-decision plays for analysis.")

    # 1. Figure 02: Skip Rate by Context (Hour, Day, Platform, Shuffle, Reason Start)
    print("Generating Figure 02: Skip rate by context...")
    plot_skip_rate_by_context(df)

    # 2. Figure 03: Skip Rate by Session Position & Track Familiarity
    print("Generating Figure 03: Skip rate by session position & familiarity...")
    plot_session_and_familiarity(df)

    # 3. Figure 04: Skip Rate by Audio Feature Bins with 95% Confidence Intervals
    print("Generating Figure 04: Skip rate by audio feature bins with 95% CIs...")
    plot_audio_feature_bins_ci(df)

    # 4. Figure 05: Correlation Heatmap of Audio Features
    print("Generating Figure 05: Audio feature correlation heatmap...")
    plot_audio_correlations(df)

    # 5. Figure 06: Monthly Listening Volume & Skip Rate Trends
    print("Generating Figure 06: Monthly listening volume and skip trends...")
    plot_monthly_listening_volume(df)

    print("Exploratory Data Analysis completed successfully!\n")


def compute_ci(k, n):
    """Compute 95% normal approximation confidence interval for proportions."""
    p = k / n
    se = np.sqrt((p * (1 - p)) / n)
    return p, 1.96 * se


def plot_skip_rate_by_context(df):
    """Figure 02: Contextual drivers of skips."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # A: Hour of Day
    hour_stats = df.groupby('hour')['skipped_fwd'].agg(['count', 'sum']).reset_index()
    hour_stats['rate'], hour_stats['ci'] = compute_ci(hour_stats['sum'], hour_stats['count'])
    hour_stats['rate'] *= 100
    hour_stats['ci'] *= 100

    axes[0, 0].errorbar(
        hour_stats['hour'], hour_stats['rate'], yerr=hour_stats['ci'],
        fmt='-o', color='#1DB954', ecolor='#17873E', capsize=3, elinewidth=1.5, linewidth=2, markersize=5
    )
    axes[0, 0].set_title('A. Skip Rate by Hour of Day (24h Clock)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Hour of Day (UTC)', fontsize=10)
    axes[0, 0].set_ylabel('Skip Rate (%)', fontsize=10)
    axes[0, 0].set_xticks(range(0, 24, 2))
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # B: Day of Week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = df.groupby('day_of_week')['skipped_fwd'].agg(['count', 'sum']).reindex(days_order).reset_index()
    day_stats['rate'], day_stats['ci'] = compute_ci(day_stats['sum'], day_stats['count'])
    day_stats['rate'] *= 100
    day_stats['ci'] *= 100

    axes[0, 1].bar(
        range(len(day_stats)), day_stats['rate'], yerr=day_stats['ci'],
        color='#2E77D0', capsize=4, alpha=0.85, width=0.55
    )
    axes[0, 1].set_title('B. Skip Rate by Day of Week', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Skip Rate (%)', fontsize=10)
    axes[0, 1].set_xticks(range(len(day_stats)))
    axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=10)
    axes[0, 1].set_ylim(40, 60)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # C: Platform / Client
    plat_stats = df.groupby('platform_clean')['skipped_fwd'].agg(['count', 'sum']).reset_index()
    plat_stats = plat_stats[plat_stats['count'] > 100].sort_values('count', ascending=False)
    plat_stats['rate'], plat_stats['ci'] = compute_ci(plat_stats['sum'], plat_stats['count'])
    plat_stats['rate'] *= 100
    plat_stats['ci'] *= 100

    bars = axes[1, 0].bar(
        plat_stats['platform_clean'], plat_stats['rate'], yerr=plat_stats['ci'],
        color='#E91429', capsize=4, alpha=0.85, width=0.5
    )
    axes[1, 0].set_title('C. Skip Rate by Playback Client/Platform', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Skip Rate (%)', fontsize=10)
    axes[1, 0].set_ylim(40, 70)
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # D: reason_start
    start_stats = df.groupby('reason_start')['skipped_fwd'].agg(['count', 'sum']).reset_index()
    start_stats = start_stats[start_stats['count'] > 500].sort_values('sum', ascending=False)
    start_stats['rate'], start_stats['ci'] = compute_ci(start_stats['sum'], start_stats['count'])
    start_stats['rate'] *= 100
    start_stats['ci'] *= 100

    axes[1, 1].barh(
        start_stats['reason_start'], start_stats['rate'], xerr=start_stats['ci'],
        color='#FF9900', capsize=4, alpha=0.85, height=0.55
    )
    axes[1, 1].set_title('D. Skip Rate by How Song Started (reason_start)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Skip Rate (%)', fontsize=10)
    axes[1, 1].set_xlim(0, 80)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Contextual Predictors of Song Skipping", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "02_skip_rate_by_context.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot_session_and_familiarity(df):
    """Figure 03: Session progression and cumulative play count dynamics."""
    # Compute session index
    df_sorted = df.sort_values('ts').copy()
    time_diff = df_sorted['ts'].diff().dt.total_seconds().fillna(99999)
    is_new_session = (time_diff > (SESSION_GAP_MINUTES * 60)).astype(int)
    df_sorted['session_id'] = is_new_session.cumsum()
    df_sorted['session_pos'] = df_sorted.groupby('session_id').cumcount() + 1

    # Compute prior track plays strictly causally
    track_counts = {}
    cum_plays = []
    for tid in df_sorted['track_id']:
        cum_plays.append(track_counts.get(tid, 0))
        track_counts[tid] = track_counts.get(tid, 0) + 1
    df_sorted['prior_plays'] = cum_plays

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Panel A: Session Position (cap at 20)
    df_sorted['pos_binned'] = df_sorted['session_pos'].clip(upper=20)
    pos_stats = df_sorted.groupby('pos_binned')['skipped_fwd'].agg(['count', 'sum']).reset_index()
    pos_stats['rate'], pos_stats['ci'] = compute_ci(pos_stats['sum'], pos_stats['count'])
    pos_stats['rate'] *= 100
    pos_stats['ci'] *= 100

    axes[0].plot(pos_stats['pos_binned'], pos_stats['rate'], marker='o', color='#1DB954', linewidth=2, markersize=6)
    axes[0].fill_between(
        pos_stats['pos_binned'],
        pos_stats['rate'] - pos_stats['ci'],
        pos_stats['rate'] + pos_stats['ci'],
        color='#1DB954', alpha=0.2
    )
    axes[0].set_title('A. Skip Rate vs. Track Position Within Listening Session', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Track Sequence in Session (20 = 20+)', fontsize=10)
    axes[0].set_ylabel('Skip Rate (%)', fontsize=10)
    axes[0].set_xticks(range(1, 21, 2))
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel B: Familiarity (Prior play count bins)
    bins = [-1, 0, 1, 5, 10, 25, 50, 100, 10000]
    labels = ['First Listen (0)', '1 Prior', '2-5', '6-10', '11-25', '26-50', '51-100', '100+']
    df_sorted['fam_bin'] = pd.cut(df_sorted['prior_plays'], bins=bins, labels=labels)

    fam_stats = df_sorted.groupby('fam_bin', observed=False)['skipped_fwd'].agg(['count', 'sum']).reset_index()
    fam_stats['rate'], fam_stats['ci'] = compute_ci(fam_stats['sum'], fam_stats['count'])
    fam_stats['rate'] *= 100
    fam_stats['ci'] *= 100

    axes[1].bar(
        range(len(fam_stats)), fam_stats['rate'], yerr=fam_stats['ci'],
        color='#8B5CF6', capsize=4, alpha=0.85, width=0.6
    )
    axes[1].set_title('B. Skip Rate vs. Track Familiarity (Prior Plays)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Track Prior Play Count', fontsize=10)
    axes[1].set_ylabel('Skip Rate (%)', fontsize=10)
    axes[1].set_xticks(range(len(fam_stats)))
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylim(20, 75)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Session Progression and Familiarity Dynamics", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "03_skip_rate_by_session_and_familiarity.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot_audio_feature_bins_ci(df):
    """Figure 04: Binned audio features with 95% confidence intervals."""
    # Filter to tracks with non-null audio features
    matched = df.dropna(subset=['energy', 'tempo', 'valence', 'acousticness', 'danceability', 'loudness']).copy()

    features_to_plot = [
        ('energy', 'Energy (0 to 1)', 8),
        ('valence', 'Valence / Musical Positiveness (0 to 1)', 8),
        ('danceability', 'Danceability (0 to 1)', 8),
        ('acousticness', 'Acousticness (0 to 1)', 8),
        ('loudness', 'Loudness (dB)', 8),
        ('tempo', 'Tempo (BPM)', 8),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    axes = axes.flatten()

    for idx, (col, label, num_bins) in enumerate(features_to_plot):
        matched['qbin'] = pd.qcut(matched[col], q=num_bins, duplicates='drop')
        b_stats = matched.groupby('qbin', observed=False)['skipped_fwd'].agg(['count', 'sum']).reset_index()
        b_stats['rate'], b_stats['ci'] = compute_ci(b_stats['sum'], b_stats['count'])
        b_stats['rate'] *= 100
        b_stats['ci'] *= 100
        
        # Midpoint of intervals for x-axis
        midpoints = [interval.mid for interval in b_stats['qbin']]

        ax = axes[idx]
        ax.errorbar(
            midpoints, b_stats['rate'], yerr=b_stats['ci'],
            fmt='-s', color='#0284C7', ecolor='#0369A1', capsize=3, elinewidth=1.5, linewidth=2, markersize=5
        )
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Skip Rate (%)', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(40, 65)

    plt.suptitle("Nonlinear Skip Probability Across Audio Feature Quantiles (95% CIs)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "04_audio_feature_bins_ci.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot_audio_correlations(df):
    """Figure 05: Audio feature correlation matrix heatmap."""
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    matched = df[feature_cols].dropna().copy()
    matched['duration_sec'] = matched['duration_ms'] / 1000.0
    matched = matched.drop(columns=['duration_ms'])

    corr = matched.corr(method='spearman')

    plt.figure(figsize=(9, 7.5), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-0.8, vmax=0.8, center=0,
        annot=True, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title('Spearman Correlation Matrix of Audio Features\n(Motivates Multicollinearity Handling & Tree Ensembles)', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "05_audio_feature_correlations.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot_monthly_listening_volume(df):
    """Figure 06: Monthly listening volume and skip rate over 5 years."""
    monthly = df.groupby('year_month').agg(
        total_plays=('skipped_fwd', 'count'),
        skips=('skipped_fwd', 'sum')
    ).reset_index()
    monthly['skip_rate'] = (monthly['skips'] / monthly['total_plays']) * 100
    monthly['month_str'] = monthly['year_month'].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 5), dpi=300)

    color_vol = '#64748B'
    color_skip = '#10B981'

    ax1.set_xlabel('Time (Month)', fontsize=11)
    ax1.set_ylabel('Total Monthly Plays', color=color_vol, fontsize=11)
    ax1.bar(monthly['month_str'], monthly['total_plays'], color=color_vol, alpha=0.5, width=0.6, label='Monthly Volume')
    ax1.tick_params(axis='y', labelcolor=color_vol)
    ax1.set_xticks(range(0, len(monthly), 3))
    ax1.set_xticklabels(monthly['month_str'].iloc[::3], rotation=45, ha='right', fontsize=9)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Monthly Skip Rate (%)', color=color_skip, fontsize=11)
    ax2.plot(monthly['month_str'], monthly['skip_rate'], color=color_skip, linewidth=2.5, marker='o', markersize=4, label='Skip Rate %')
    ax2.tick_params(axis='y', labelcolor=color_skip)
    ax2.set_ylim(30, 75)
    ax2.grid(False)

    plt.title('Listening Volume and Skip Rate Over 5 Years (March 2021 – February 2026)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = FIGURES_DIR / "06_monthly_listening_volume.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_eda_pipeline()
