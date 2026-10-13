"""
Model implementations, preprocessors, and baselines for the Spotify Skip Prediction project.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES,
    CONTEXT_FEATURES,
    HABIT_FEATURES,
    AUDIO_FEATURES,
    ALL_MODEL_FEATURES,
    RANDOM_SEED,
)


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the training set class prior probability unconditionally."""
    def __init__(self):
        self.prior_ = 0.5

    def fit(self, X, y):
        self.prior_ = np.mean(y)
        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.prior_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (np.full(len(X), self.prior_) >= 0.5).astype(int)


class HeuristicColumnBaseline(BaseEstimator, ClassifierMixin):
    """Predicts a precomputed feature directly as a probability estimate."""
    def __init__(self, column_name="prior_track_skip_rate"):
        self.column_name = column_name
        self.default_prior_ = 0.5

    def fit(self, X, y):
        self.default_prior_ = np.mean(y)
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame) and self.column_name in X.columns:
            probs = X[self.column_name].fillna(self.default_prior_).clip(0.001, 0.999).values
        else:
            probs = np.full(len(X), self.default_prior_)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class HourHeuristicBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the empirical skip rate of the hour of the day."""
    def __init__(self):
        self.hour_map_ = {}
        self.default_ = 0.5

    def fit(self, X, y):
        self.default_ = np.mean(y)
        if isinstance(X, pd.DataFrame) and 'hour_sin' in X.columns and 'hour_cos' in X.columns:
            # Reconstruct hour from sin and cos
            hours = np.round(np.arctan2(X['hour_sin'], X['hour_cos']) * 24.0 / (2 * np.pi)) % 24
            temp_df = pd.DataFrame({'hour': hours.astype(int), 'y': y.values if hasattr(y, 'values') else y})
            self.hour_map_ = temp_df.groupby('hour')['y'].mean().to_dict()
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame) and 'hour_sin' in X.columns and 'hour_cos' in X.columns:
            hours = np.round(np.arctan2(X['hour_sin'], X['hour_cos']) * 24.0 / (2 * np.pi)) % 24
            probs = np.array([self.hour_map_.get(h, self.default_) for h in hours.astype(int)])
        else:
            probs = np.full(len(X), self.default_)
        probs = np.clip(probs, 0.001, 0.999)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_preprocessor(feature_list):
    """Constructs sklearn ColumnTransformer pipeline fitted strictly on training data."""
    cats = [c for c in CATEGORICAL_FEATURES if c in feature_list]
    nums = [c for c in feature_list if c not in CATEGORICAL_FEATURES]

    transformers = []
    if nums:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', num_pipe, nums))

    if cats:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat_pipe, cats))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor


def get_model_ladder(feature_list):
    """
    Returns a dictionary of all models in the evaluation ladder for a given feature set.
    """
    preprocessor = build_preprocessor(feature_list)

    models = {
        "Majority Baseline": MajorityBaseline(),
        "Track Skip Rate Heuristic": HeuristicColumnBaseline("prior_track_skip_rate"),
        "Artist Skip Rate Heuristic": HeuristicColumnBaseline("prior_artist_skip_rate"),
        "Hour Skip Rate Heuristic": HourHeuristicBaseline(),
        
        "Logistic Regression (L2)": Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED))
        ]),
        
        "Random Forest": Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=15,
                random_state=RANDOM_SEED, n_jobs=-1
            ))
        ]),
        
        "Gradient Boosting (GBDT / LightGBM)": Pipeline([
            ('prep', preprocessor),
            ('clf', HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                min_samples_leaf=30, l2_regularization=0.1,
                random_state=RANDOM_SEED
            ))
        ]),

        "Neural MLP": Pipeline([
            ('prep', preprocessor),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=30, alpha=0.01,
                early_stopping=True, random_state=RANDOM_SEED
            ))
        ]),
    }

    return models
