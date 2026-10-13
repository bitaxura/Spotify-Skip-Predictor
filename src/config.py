import os
from pathlib import Path

# Set Matplotlib cache dir to prevent permission warning
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAL_DATA_DIR = PROJECT_ROOT / "personal data" / "Spotify Extended Streaming History"
TRAINING_DATA_DIR = PROJECT_ROOT / "training data"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Raw input files
TRACK_AUDIO_FEATURES_PATH = TRAINING_DATA_DIR / "track_audio_features.parquet"
TRACKS_PATH = TRAINING_DATA_DIR / "tracks.parquet"

# Intermediate and processed files
USER_TRACKS_METADATA_PATH = DATA_DIR / "user_tracks_metadata.parquet"
CLEANED_HISTORY_PATH = DATA_DIR / "cleaned_streaming_history.parquet"
ENGINEERED_FEATURES_PATH = DATA_DIR / "engineered_features.parquet"
DATA_DICTIONARY_PATH = DATA_DIR / "data_dictionary.md"

# Random Seed
RANDOM_SEED = 42

# Thresholds & Parameters
SESSION_GAP_MINUTES = 30  # New session if gap > 30 mins
SHORT_PLAY_MS = 30000     # 30 seconds threshold for short-play skip definition
BAYESIAN_ALPHA_TRACK = 3.0   # Smoothing pseudo-counts for track skip rate
BAYESIAN_BETA_ARTIST = 5.0   # Smoothing pseudo-counts for artist skip rate

# Train / Validation / Test split fractions (Chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature Families
CONTEXT_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "platform_clean",
    "shuffle",
    "offline",
    "reason_start",
    "session_track_index",
    "session_duration_min",
    "time_since_last_play_min",
    "prev_track_skipped",
    "skips_in_last_5",
]

HABIT_FEATURES = [
    "prior_track_plays",
    "prior_artist_plays",
    "prior_track_skip_rate",
    "prior_artist_skip_rate",
    "days_since_track_last_played",
    "is_first_listen",
    "is_top_track",
]

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
    "time_signature",
    "popularity",
    "explicit",
    "duration_sec",
    "audio_dist_to_liked_profile",
]

ALL_MODEL_FEATURES = CONTEXT_FEATURES + HABIT_FEATURES + AUDIO_FEATURES

CATEGORICAL_FEATURES = [
    "platform_clean",
    "reason_start",
]

TARGET_COL_BINARY_FWD = "skipped_fwd"
TARGET_COL_BINARY_30S = "skipped_30s"
TARGET_COL_BINARY_FLAG = "skipped_flag"
TARGET_COL_CONTINUOUS = "fraction_played"
