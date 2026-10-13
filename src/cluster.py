"""
Add-on 1: Listening Modes Clustering (Section 10 of plan.md).
Discovers semantic listening clusters via K-Means on audio features and maps temporal distributions.
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import (
    CLEANED_HISTORY_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    RANDOM_SEED,
)


def run_clustering_pipeline():
    """Discover semantic listening modes and temporal patterns."""
    print("=" * 70)
    print("STEP 6: ADD-ON 1 - LISTENING MODES CLUSTERING")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Load unique completed/liked tracks to cluster genuine listening modes
    tracks_df = con.execute(f"""
        SELECT 
            track_id,
            track_name,
            artist_name,
            danceability,
            energy,
            loudness,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo
        FROM '{CLEANED_HISTORY_PATH}'
        WHERE danceability IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    """).df()

    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(tracks_df[feature_cols])

    # K-Means with k=4 clusters
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    tracks_df['cluster'] = clusters

    # Semantic Cluster Labeling based on centroids
    cluster_means = tracks_df.groupby('cluster')[feature_cols].mean()
    print("Cluster Audio Feature Centroids:")
    print(cluster_means.to_string())

    cluster_labels = {}
    for c_id, row in cluster_means.iterrows():
        if row['energy'] > 0.70 or row['loudness'] > -7.0:
            cluster_labels[c_id] = "High-Energy Upbeat & Workout"
        elif row['acousticness'] > 0.40 or row['energy'] < 0.45:
            cluster_labels[c_id] = "Acoustic & Low-Fi Focus"
        elif row['danceability'] > 0.65 and row['valence'] > 0.50:
            cluster_labels[c_id] = "Joyful Melodic & Dance"
        else:
            cluster_labels[c_id] = "Mellow / Atmospheric Chill"

    # Ensure unique labels
    seen_labels = set()
    for c_id in range(4):
        lbl = cluster_labels.get(c_id, f"Mode {c_id+1}")
        if lbl in seen_labels:
            lbl = f"{lbl} (Variant B)"
        seen_labels.add(lbl)
        cluster_labels[c_id] = lbl

    tracks_df['cluster_name'] = tracks_df['cluster'].map(cluster_labels)

    # Save cluster summary
    summary_df = tracks_df.groupby(['cluster', 'cluster_name'])[feature_cols].mean().reset_index()
    summary_df['num_tracks'] = tracks_df.groupby('cluster')['track_id'].count().values
    summary_path = TABLES_DIR / "listening_modes_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved listening modes summary to: {summary_path}")

    # Map clusters back to user streaming history for temporal occurrence
    history_df = con.execute(f"""
        SELECT 
            ts,
            track_id,
            reason_end
        FROM '{CLEANED_HISTORY_PATH}'
        WHERE is_user_decision = 1
    """).df()
    history_df['ts'] = pd.to_datetime(history_df['ts'])
    history_df['hour'] = history_df['ts'].dt.hour
    history_df = history_df.merge(tracks_df[['track_id', 'cluster', 'cluster_name']], on='track_id', how='inner')

    # Figure 14: 2-Panel Listening Modes (PCA Scatter + Temporal Hour Distribution)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    tracks_df['pca_1'] = X_pca[:, 0]
    tracks_df['pca_2'] = X_pca[:, 1]

    fig, (ax_pca, ax_time) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    palette = ['#10B981', '#3B82F6', '#F59E0B', '#8B5CF6']

    for c_id, color in zip(range(4), palette):
        sub = tracks_df[tracks_df['cluster'] == c_id]
        ax_pca.scatter(sub['pca_1'], sub['pca_2'], label=cluster_labels[c_id], color=color, alpha=0.5, s=20)
    ax_pca.set_title('A. PCA Space of Discovered Listening Modes (N=7,367 Tracks)', fontsize=11.5, fontweight='bold')
    ax_pca.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
    ax_pca.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
    ax_pca.legend(loc='lower right', fontsize=8.5, frameon=True)
    ax_pca.grid(True, linestyle='--', alpha=0.5)

    # Panel B: Temporal Distribution across 24 hours
    hourly_cluster = history_df.groupby(['hour', 'cluster_name']).size().unstack(fill_value=0)
    hourly_pct = hourly_cluster.div(hourly_cluster.sum(axis=1), axis=0) * 100

    for c_name, color in zip(hourly_pct.columns, palette):
        ax_time.plot(hourly_pct.index, hourly_pct[c_name], label=c_name, color=color, linewidth=2.2, marker='o', markersize=4)

    ax_time.set_title('B. Diurnal Rhythm: Listening Mode Share by Hour (UTC)', fontsize=11.5, fontweight='bold')
    ax_time.set_xlabel('Hour of Day', fontsize=10)
    ax_time.set_ylabel('Share of Listening (%)', fontsize=10)
    ax_time.set_xticks(range(0, 24, 2))
    ax_time.legend(loc='upper right', fontsize=8.5, frameon=True)
    ax_time.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Add-On 1: Semantic Listening Modes & Temporal Diurnal Dynamics", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "14_listening_modes_clusters.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved listening modes figure: {fig_path}\n")


if __name__ == "__main__":
    run_clustering_pipeline()
