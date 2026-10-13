"""
Causal Feature Engineering Pipeline (Zero Data Leakage).
Computes context, habit/familiarity, session dynamics, and audio profile features.
"""
import duckdb
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import (
    CLEANED_HISTORY_PATH,
    ENGINEERED_FEATURES_PATH,
    SESSION_GAP_MINUTES,
    BAYESIAN_ALPHA_TRACK,
    BAYESIAN_BETA_ARTIST,
    TRAIN_RATIO,
)


def run_feature_engineering():
    """Execute feature engineering pipeline strictly causally without data leakage."""
    print("=" * 70)
    print("STEP 3: CAUSAL FEATURE ENGINEERING (ZERO DATA LEAKAGE)")
    print("=" * 70)

    con = duckdb.connect()

    # Load cleaned user-decision records with audio features
    print("Loading cleaned streaming history...")
    df = con.execute(f"""
        SELECT 
            ts,
            track_id,
            track_name,
            artist_name,
            album_name,
            platform_clean,
            reason_start,
            reason_end,
            shuffle,
            offline,
            skipped_fwd,
            skipped_30s,
            skipped_flag,
            fraction_played,
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
            time_signature,
            popularity,
            explicit,
            duration_ms
        FROM '{CLEANED_HISTORY_PATH}'
        WHERE is_user_decision = 1 AND danceability IS NOT NULL
        ORDER BY ts ASC
    """).df()

    total_records = len(df)
    print(f"Engineering features for {total_records:,} chronological plays...")

    df['ts'] = pd.to_datetime(df['ts'])

    # 1. Temporal & Cyclical Context Features
    hours = df['ts'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df['day_of_week'] = df['ts'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['shuffle'] = df['shuffle'].astype(int)
    df['offline'] = df['offline'].astype(int)
    df['duration_sec'] = df['duration_ms'] / 1000.0

    # 2. Session Dynamics & In-Session Skipping Mood
    print("Computing sessionization and in-session mood features...")
    time_diff_sec = df['ts'].diff().dt.total_seconds().fillna(99999).values
    gap_threshold = SESSION_GAP_MINUTES * 60

    session_ids = []
    curr_session = 0
    for diff in time_diff_sec:
        if diff > gap_threshold:
            curr_session += 1
        session_ids.append(curr_session)
    df['session_id'] = session_ids

    # Vectorized in-session features
    df['time_since_last_play_min'] = np.clip(time_diff_sec / 60.0, 0, 180.0)
    df['session_track_index'] = df.groupby('session_id').cumcount() + 1
    
    # Session duration so far (cumulative duration in minutes of prior tracks in session)
    df['prior_track_duration_min'] = df.groupby('session_id')['duration_sec'].shift(1).fillna(0) / 60.0
    df['session_duration_min'] = df.groupby('session_id')['prior_track_duration_min'].cumsum()
    df = df.drop(columns=['prior_track_duration_min'])

    # Previous track skipped (within same session)
    df['prev_track_skipped'] = df.groupby('session_id')['skipped_fwd'].shift(1).fillna(0).astype(int)

    # Skips in last 5 tracks in current session (rolling sum of shifts 1..5)
    skips_last_5 = np.zeros(total_records, dtype=int)
    for shift_i in range(1, 6):
        shifted = df.groupby('session_id')['skipped_fwd'].shift(shift_i).fillna(0).values
        skips_last_5 += shifted.astype(int)
    df['skips_in_last_5'] = skips_last_5

    # 3. Habit & Familiarity Features (Strictly Causal - Iterative Pass)
    print("Computing causal habit, familiarity, and Bayesian smoothed skip rates...")
    
    # Determine train split cutoff to compute global training skip rate prior
    train_size = int(total_records * TRAIN_RATIO)
    global_train_prior = df['skipped_fwd'].iloc[:train_size].mean()
    print(f"Global training skip rate prior: {global_train_prior:.4f} (from first {train_size:,} plays)")

    track_play_counts = {}
    track_skip_counts = {}
    track_last_timestamps = {}

    artist_play_counts = {}
    artist_skip_counts = {}

    prior_track_plays = np.zeros(total_records, dtype=int)
    prior_track_skip_rate = np.zeros(total_records, dtype=float)
    days_since_last = np.full(total_records, 365.0, dtype=float)

    prior_artist_plays = np.zeros(total_records, dtype=int)
    prior_artist_skip_rate = np.zeros(total_records, dtype=float)

    # Rolling liked centroid profile tracking (for distance to liked profile)
    # Features: danceability, energy, acousticness, valence, loudness_norm, tempo_norm
    norm_loudness = ((df['loudness'] - (-60.0)) / 60.0).clip(0, 1).values
    norm_tempo = ((df['tempo'] - 40.0) / 180.0).clip(0, 1).values
    dance_arr = df['danceability'].values
    energy_arr = df['energy'].values
    acoust_arr = df['acousticness'].values
    valence_arr = df['valence'].values

    audio_dist_arr = np.zeros(total_records, dtype=float)
    running_liked_sum = np.zeros(6, dtype=float)
    running_liked_count = 0

    track_ids = df['track_id'].values
    artist_names = df['artist_name'].fillna('Unknown').values
    ts_array = df['ts'].values
    skips_array = df['skipped_fwd'].values
    reasons_end = df['reason_end'].values

    alpha = BAYESIAN_ALPHA_TRACK
    beta = BAYESIAN_BETA_ARTIST

    for i in range(total_records):
        tid = track_ids[i]
        art = artist_names[i]
        curr_ts = ts_array[i]

        # Prior Track Stats
        n_tp = track_play_counts.get(tid, 0)
        n_ts = track_skip_counts.get(tid, 0)
        prior_track_plays[i] = n_tp
        prior_track_skip_rate[i] = (n_ts + alpha * global_train_prior) / (n_tp + alpha)

        # Days since track last played
        last_t = track_last_timestamps.get(tid, None)
        if last_t is not None:
            delta_days = (curr_ts - last_t) / np.timedelta64(1, 'D')
            days_since_last[i] = min(float(delta_days), 365.0)

        # Prior Artist Stats
        n_ap = artist_play_counts.get(art, 0)
        n_as = artist_skip_counts.get(art, 0)
        prior_artist_plays[i] = n_ap
        prior_artist_skip_rate[i] = (n_as + beta * global_train_prior) / (n_ap + beta)

        # Distance to Liked Profile
        curr_audio_vec = np.array([
            dance_arr[i], energy_arr[i], acoust_arr[i], valence_arr[i],
            norm_loudness[i], norm_tempo[i]
        ], dtype=float)

        if running_liked_count > 0:
            centroid = running_liked_sum / running_liked_count
            audio_dist_arr[i] = np.linalg.norm(curr_audio_vec - centroid)
        else:
            audio_dist_arr[i] = 0.5  # Neutral default before any liked songs observed

        # Causal State Update (happens strictly AFTER computing features for row i)
        track_play_counts[tid] = n_tp + 1
        track_last_timestamps[tid] = curr_ts
        if skips_array[i] == 1:
            track_skip_counts[tid] = n_ts + 1

        artist_play_counts[art] = n_ap + 1
        if skips_array[i] == 1:
            artist_skip_counts[art] = n_as + 1

        # If user finished song naturally without skipping, update liked profile
        if reasons_end[i] == 'trackdone':
            running_liked_sum += curr_audio_vec
            running_liked_count += 1

    df['prior_track_plays'] = prior_track_plays
    df['prior_track_skip_rate'] = prior_track_skip_rate
    df['days_since_track_last_played'] = days_since_last
    df['is_first_listen'] = (prior_track_plays == 0).astype(int)
    df['is_top_track'] = (prior_track_plays >= 10).astype(int)

    df['prior_artist_plays'] = prior_artist_plays
    df['prior_artist_skip_rate'] = prior_artist_skip_rate
    df['audio_dist_to_liked_profile'] = audio_dist_arr

    # Save engineered feature dataset
    print(f"Saving engineered feature dataset to: {ENGINEERED_FEATURES_PATH}")
    ENGINEERED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ENGINEERED_FEATURES_PATH, index=False)
    print(f"Features successfully generated and saved! Total rows: {len(df):,}\n")

    return df


if __name__ == "__main__":
    run_feature_engineering()
