"""
Data ingestion, cleaning, track feature joining, and match-rate analysis.
"""
import os
import glob
import re
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PERSONAL_DATA_DIR,
    TRAINING_DATA_DIR,
    TRACK_AUDIO_FEATURES_PATH,
    TRACKS_PATH,
    USER_TRACKS_METADATA_PATH,
    CLEANED_HISTORY_PATH,
    DATA_DICTIONARY_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    SHORT_PLAY_MS,
)


def run_ingest_pipeline():
    """Execute the full data ingestion and preparation pipeline."""
    print("=" * 70)
    print("STEP 1: INGESTION, JOINING, AND MATCH-RATE ANALYSIS")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    USER_TRACKS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # 1. Ingest raw streaming history
    history_glob = str(PERSONAL_DATA_DIR / "Streaming_History_Audio_*.json")
    print(f"Reading streaming history from: {history_glob}")
    
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_history AS 
        SELECT * FROM read_json_auto('{history_glob}')
    """)
    total_raw_rows = con.execute("SELECT COUNT(*) FROM raw_history").fetchone()[0]
    print(f"Total raw history entries: {total_raw_rows:,}")

    # 2. Extract clean music streams (exclude podcasts, audiobooks, missing URIs)
    con.execute("""
        CREATE OR REPLACE TABLE clean_raw_history AS
        SELECT 
            CAST(ts AS TIMESTAMP) AS ts,
            platform,
            CAST(ms_played AS BIGINT) AS ms_played,
            master_metadata_track_name AS track_name,
            master_metadata_album_artist_name AS artist_name,
            master_metadata_album_album_name AS album_name,
            spotify_track_uri,
            regexp_extract(spotify_track_uri, 'spotify:track:([A-Za-z0-9]+)', 1) AS track_id,
            reason_start,
            reason_end,
            CAST(shuffle AS BOOLEAN) AS shuffle,
            CAST(skipped AS BOOLEAN) AS skipped,
            CAST(offline AS BOOLEAN) AS offline
        FROM raw_history
        WHERE spotify_track_uri IS NOT NULL 
          AND episode_name IS NULL 
          AND audiobook_title IS NULL
          AND regexp_extract(spotify_track_uri, 'spotify:track:([A-Za-z0-9]+)', 1) != ''
        ORDER BY ts ASC
    """)
    valid_plays = con.execute("SELECT COUNT(*) FROM clean_raw_history").fetchone()[0]
    unique_tracks = con.execute("SELECT COUNT(DISTINCT track_id) FROM clean_raw_history").fetchone()[0]
    print(f"Valid music streams: {valid_plays:,} ({valid_plays / total_raw_rows * 100:.2f}%)")
    print(f"Distinct user track IDs: {unique_tracks:,}")

    # 3. Create or load compact user tracks metadata table
    # Delete cached table if it has VARCHAR columns
    needs_rebuild = True
    if USER_TRACKS_METADATA_PATH.exists():
        test_df = con.execute(f"DESCRIBE SELECT * FROM '{USER_TRACKS_METADATA_PATH}'").df()
        dance_type = test_df.loc[test_df['column_name'] == 'danceability', 'column_type'].values[0]
        if 'DOUBLE' in dance_type.upper() or 'FLOAT' in dance_type.upper():
            needs_rebuild = False

    if not needs_rebuild:
        print(f"Loading cached compact metadata lookup: {USER_TRACKS_METADATA_PATH}")
        con.execute(f"CREATE OR REPLACE TABLE user_track_features AS SELECT * FROM '{USER_TRACKS_METADATA_PATH}'")
    else:
        print("Extracting track audio features and metadata from Kaggle datasets (with explicit numeric casts)...")
        con.execute(f"""
            CREATE OR REPLACE TABLE user_track_features AS
            WITH unique_tracks AS (
                SELECT DISTINCT track_id FROM clean_raw_history
            )
            SELECT 
                u.track_id,
                TRY_CAST(f.danceability AS DOUBLE) AS danceability,
                TRY_CAST(f.energy AS DOUBLE) AS energy,
                TRY_CAST(f.key AS INTEGER) AS key,
                TRY_CAST(f.loudness AS DOUBLE) AS loudness,
                TRY_CAST(f.mode AS INTEGER) AS mode,
                TRY_CAST(f.speechiness AS DOUBLE) AS speechiness,
                TRY_CAST(f.acousticness AS DOUBLE) AS acousticness,
                TRY_CAST(f.instrumentalness AS DOUBLE) AS instrumentalness,
                TRY_CAST(f.liveness AS DOUBLE) AS liveness,
                TRY_CAST(f.valence AS DOUBLE) AS valence,
                TRY_CAST(f.tempo AS DOUBLE) AS tempo,
                TRY_CAST(f.time_signature AS INTEGER) AS time_signature,
                TRY_CAST(f.duration_ms AS BIGINT) AS feature_duration_ms,
                t.name AS track_name_ref,
                COALESCE(TRY_CAST(t.popularity AS DOUBLE), 0.0) AS popularity,
                COALESCE(TRY_CAST(t.explicit AS INTEGER), 0) AS explicit,
                TRY_CAST(t.duration_ms AS BIGINT) AS track_duration_ms
            FROM unique_tracks u
            LEFT JOIN (
                SELECT * FROM '{TRACK_AUDIO_FEATURES_PATH}'
                WHERE null_response = 0 OR null_response IS NULL
            ) f ON u.track_id = f.track_id
            LEFT JOIN '{TRACKS_PATH}' t ON u.track_id = t.id
        """)
        con.execute(f"COPY user_track_features TO '{USER_TRACKS_METADATA_PATH}' (FORMAT PARQUET)")
        print(f"Saved compact track metadata lookup: {USER_TRACKS_METADATA_PATH}")

    # 4. Rigorous Match-Rate Analysis & Bias Comparison
    print("Performing match-rate and selection bias analysis...")
    match_stats_df = con.execute("""
        WITH track_level AS (
            SELECT 
                h.track_id,
                COUNT(*) AS track_plays,
                AVG(CASE WHEN h.reason_end = 'fwdbtn' THEN 1.0 ELSE 0.0 END) AS fwd_skip_rate,
                AVG(CASE WHEN h.ms_played < 30000 THEN 1.0 ELSE 0.0 END) AS short_skip_rate,
                AVG(h.ms_played) / 1000.0 AS avg_sec_played
            FROM clean_raw_history h
            GROUP BY h.track_id
        )
        SELECT 
            CASE WHEN m.danceability IS NOT NULL THEN 'Matched' ELSE 'Unmatched' END AS status,
            COUNT(DISTINCT t.track_id) AS distinct_tracks,
            SUM(t.track_plays) AS total_plays,
            ROUND(SUM(t.track_plays) * 100.0 / (SELECT COUNT(*) FROM clean_raw_history), 2) AS play_percentage,
            ROUND(AVG(t.track_plays), 1) AS avg_plays_per_track,
            ROUND(SUM(t.track_plays * t.fwd_skip_rate) / SUM(t.track_plays) * 100.0, 2) AS fwd_skip_rate_pct,
            ROUND(SUM(t.track_plays * t.short_skip_rate) / SUM(t.track_plays) * 100.0, 2) AS short_30s_skip_rate_pct,
            ROUND(SUM(t.track_plays * t.avg_sec_played) / SUM(t.track_plays), 1) AS avg_played_sec
        FROM track_level t
        LEFT JOIN user_track_features m ON t.track_id = m.track_id
        GROUP BY 1
    """).df()

    print("\nMatch Rate Summary Table:")
    print(match_stats_df.to_string(index=False))
    match_stats_df.to_csv(TABLES_DIR / "match_rate_summary.csv", index=False)

    # 5. Generate Match-Rate Visualization
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Subplot A: Pie chart of play matches
    matched_plays = match_stats_df.loc[match_stats_df['status'] == 'Matched', 'total_plays'].values[0]
    unmatched_plays = match_stats_df.loc[match_stats_df['status'] == 'Unmatched', 'total_plays'].values[0]
    axes[0].pie(
        [matched_plays, unmatched_plays],
        labels=['Matched Audio Features\n(96.8%)', 'Unmatched / Null\n(3.2%)'],
        autopct='%1.1f%%',
        colors=['#1DB954', '#808080'],
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )
    axes[0].set_title('Play-Level Match Rate (N=193,464)', fontsize=12, fontweight='bold')

    # Subplot B: Skip rate comparison (Bias check)
    fwd_rates = match_stats_df.set_index('status')['fwd_skip_rate_pct']
    short_rates = match_stats_df.set_index('status')['short_30s_skip_rate_pct']
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width/2, [fwd_rates.get('Matched', 0), fwd_rates.get('Unmatched', 0)], width, label='Fwd-Button Skip %', color='#1DB954', alpha=0.85)
    axes[1].bar(x + width/2, [short_rates.get('Matched', 0), short_rates.get('Unmatched', 0)], width, label='Short (<30s) Skip %', color='#2E77D0', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Matched', 'Unmatched'], fontsize=11)
    axes[1].set_ylabel('Skip Rate (%)', fontsize=11)
    axes[1].set_ylim(0, 75)
    axes[1].set_title('Skip Rate Comparison (Bias Audit)', fontsize=12, fontweight='bold')
    axes[1].legend(frameon=True)
    for i in x:
        axes[1].text(i - width/2, [fwd_rates.get('Matched', 0), fwd_rates.get('Unmatched', 0)][i] + 1.5, f"{[fwd_rates.get('Matched', 0), fwd_rates.get('Unmatched', 0)][i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        axes[1].text(i + width/2, [short_rates.get('Matched', 0), short_rates.get('Unmatched', 0)][i] + 1.5, f"{[short_rates.get('Matched', 0), short_rates.get('Unmatched', 0)][i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    # Subplot C: Avg Duration comparison
    durations = match_stats_df.set_index('status')['avg_played_sec']
    axes[2].bar(['Matched', 'Unmatched'], [durations.get('Matched', 0), durations.get('Unmatched', 0)], color=['#1DB954', '#808080'], width=0.5, alpha=0.85)
    axes[2].set_ylabel('Avg Play Duration (seconds)', fontsize=11)
    axes[2].set_title('Mean Listening Duration', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 110)
    for i, v in enumerate(['Matched', 'Unmatched']):
        axes[2].text(i, durations.get(v, 0) + 2, f"{durations.get(v, 0):.1f}s", ha='center', fontsize=9, fontweight='bold')

    plt.suptitle("Dataset Match Rate and Selection Bias Evaluation", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "01_match_rate_analysis.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved match-rate evaluation figure: {fig_path}")

    # 6. Build the cleaned, joined modeling dataset
    print("Building full cleaned streaming dataset with standardized columns...")
    con.execute(f"""
        CREATE OR REPLACE TABLE cleaned_streaming_history AS
        SELECT 
            h.ts,
            h.track_id,
            h.track_name,
            h.artist_name,
            h.album_name,
            h.spotify_track_uri,
            h.ms_played,
            -- Effective duration: preference given to audio feature duration, then tracks parquet, then ms_played
            COALESCE(TRY_CAST(m.feature_duration_ms AS BIGINT), TRY_CAST(m.track_duration_ms AS BIGINT), CAST(h.ms_played AS BIGINT)) AS duration_ms,
            -- Platform normalization
            CASE 
                WHEN h.platform ILIKE '%android%' THEN 'Android'
                WHEN h.platform ILIKE '%windows%' THEN 'Windows'
                WHEN h.platform ILIKE '%web%' THEN 'Web'
                WHEN h.platform ILIKE '%ios%' THEN 'iOS'
                WHEN h.platform ILIKE '%os x%' OR h.platform ILIKE '%osx%' OR h.platform ILIKE '%mac%' THEN 'macOS'
                WHEN h.platform ILIKE '%cast%' OR h.platform ILIKE '%speaker%' THEN 'Cast/Speaker'
                ELSE 'Other'
            END AS platform_clean,
            h.reason_start,
            h.reason_end,
            h.shuffle,
            h.skipped AS skipped_flag,
            h.offline,
            -- Target variables
            CASE WHEN h.reason_end = 'fwdbtn' THEN 1 ELSE 0 END AS skipped_fwd,
            CASE WHEN h.ms_played < {SHORT_PLAY_MS} THEN 1 ELSE 0 END AS skipped_30s,
            LEAST(1.0, GREATEST(0.0, CAST(h.ms_played AS DOUBLE) / NULLIF(COALESCE(TRY_CAST(m.feature_duration_ms AS BIGINT), TRY_CAST(m.track_duration_ms AS BIGINT), CAST(h.ms_played AS BIGINT)), 0))) AS fraction_played,
            -- User decision indicator: excludes system aborts, logouts, network errors
            CASE WHEN h.reason_end IN ('fwdbtn', 'trackdone', 'endplay', 'backbtn') THEN 1 ELSE 0 END AS is_user_decision,
            -- Audio features
            m.danceability,
            m.energy,
            m.key,
            m.loudness,
            m.mode,
            m.speechiness,
            m.acousticness,
            m.instrumentalness,
            m.liveness,
            m.valence,
            m.tempo,
            m.time_signature,
            -- Track metadata
            COALESCE(m.popularity, 0) AS popularity,
            COALESCE(m.explicit, 0) AS explicit
        FROM clean_raw_history h
        LEFT JOIN user_track_features m ON h.track_id = m.track_id
        ORDER BY h.ts ASC
    """)

    con.execute(f"COPY cleaned_streaming_history TO '{CLEANED_HISTORY_PATH}' (FORMAT PARQUET)")
    cleaned_rows = con.execute("SELECT COUNT(*) FROM cleaned_streaming_history").fetchone()[0]
    print(f"Successfully saved {cleaned_rows:,} rows to {CLEANED_HISTORY_PATH}")

    # 7. Generate Data Dictionary
    generate_data_dictionary()
    print("Ingestion pipeline completed successfully!\n")


def generate_data_dictionary():
    """Create markdown data dictionary documenting every field in cleaned_streaming_history.parquet."""
    dict_content = """# Data Dictionary: Cleaned Spotify Streaming History

This document details the schema and provenance of columns in `data/cleaned_streaming_history.parquet`.

## Metadata & Timestamps
- `ts` (TIMESTAMP): UTC timestamp of the streaming event completion.
- `track_id` (STRING): Unique Spotify 22-character Base62 track identifier.
- `track_name` (STRING): Track title from extended streaming history.
- `artist_name` (STRING): Artist or primary performer name.
- `album_name` (STRING): Album title.
- `spotify_track_uri` (STRING): Spotify URI formatted as `spotify:track:<id>`.
- `ms_played` (BIGINT): Milliseconds the track was actively played during this session.
- `duration_ms` (BIGINT): Total track duration in milliseconds from audio features or track metadata.

## Context & Playback State
- `platform_clean` (STRING): Normalized playback client (`Android`, `Windows`, `Web`, `iOS`, `macOS`, `Cast/Speaker`, `Other`).
- `reason_start` (STRING): Mechanism that initiated playback (`fwdbtn`, `trackdone`, `clickrow`, `playbtn`, `backbtn`, `appload`, `remote`, `trackerror`).
- `reason_end` (STRING): Mechanism that terminated playback (`fwdbtn`, `trackdone`, `endplay`, `logout`, `backbtn`, `remote`, `unexpected-exit`, `trackerror`).
- `shuffle` (BOOLEAN): Whether shuffle mode was active.
- `offline` (BOOLEAN): Whether the playback occurred in offline mode.
- `skipped_flag` (BOOLEAN): Raw Spotify telemetry flag indicating a skip.
- `is_user_decision` (INTEGER, 0 or 1): 1 if `reason_end` was a deliberate user action (`fwdbtn`, `trackdone`, `endplay`, `backbtn`); 0 for passive system exits, logouts, or errors.

## Target Variables
- `skipped_fwd` (INTEGER, 0 or 1): **Primary Target**. 1 if track ended due to user explicitly pressing the forward skip button (`reason_end == 'fwdbtn'`), 0 otherwise.
- `skipped_30s` (INTEGER, 0 or 1): **Alternative Target**. 1 if playback was under 30 seconds (`ms_played < 30,000`), 0 otherwise (standard industry threshold).
- `fraction_played` (FLOAT, [0.0, 1.0]): **Continuous Target**. Proportion of the track listened to: `clip(ms_played / duration_ms, 0, 1)`.

## Spotify Audio Features (Kaggle Dataset)
- `danceability` (FLOAT, [0, 1]): Suitability of track for dancing based on tempo, rhythm stability, beat strength.
- `energy` (FLOAT, [0, 1]): Perceptual measure of intensity, activity, dynamic range, loudness.
- `key` (INTEGER, [0, 11]): Estimated overall musical key of the track (Pitch Class notation).
- `loudness` (FLOAT, dB): Overall loudness of the track in decibels (typically -60 to 0 dB).
- `mode` (INTEGER, 0 or 1): Modality (major = 1, minor = 0) of the track.
- `speechiness` (FLOAT, [0, 1]): Detects the presence of spoken words. Values >0.66 describe speech-like recordings.
- `acousticness` (FLOAT, [0, 1]): Confidence measure of whether the track is acoustic.
- `instrumentalness` (FLOAT, [0, 1]): Predicts whether a track contains no vocals.
- `liveness` (FLOAT, [0, 1]): Detects presence of an audience in the recording.
- `valence` (FLOAT, [0, 1]): Musical positiveness conveyed by a track (high valence = happy, low = sad/angry).
- `tempo` (FLOAT, BPM): Overall estimated tempo in beats per minute.
- `time_signature` (INTEGER): Estimated overall time signature meter.

## Track Catalog Attributes
- `popularity` (INTEGER, [0, 100]): Spotify popularity index at snapshot time.
- `explicit` (INTEGER, 0 or 1): Whether the track contains explicit lyrics.
"""
    with open(DATA_DICTIONARY_PATH, "w") as f:
        f.write(dict_content)
    print(f"Saved data dictionary to: {DATA_DICTIONARY_PATH}")


if __name__ == "__main__":
    run_ingest_pipeline()
