# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------
# Small containers (UI needs)
# ---------------------------

@dataclass
class ModelBest:
    name: str
    prediction: float
    mae: float
    mse: float


# ---------------------------
# Helpers
# ---------------------------

def _to_num_df(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric DataFrame; impute with feature medians; zero-fill leftovers.
    Why: robust against occasional NaNs/Inf in engineered features."""
    X = pd.DataFrame(X).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X = X.fillna(0.0)
    return X


def _rolling_splits(n: int, n_folds: int = 6, min_train: int = 24, val_size: int = 6) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits along time. Returns list of (train_idx, val_idx)."""
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = max(min_train, val_size)
    if n < (min_train + val_size + 1):
        return splits
    # roughly distribute folds across remaining range
    step = max(1, (n - start - val_size) // max(1, (n_folds - 1)))
    while start + val_size <= n and len(splits) < n_folds:
        tr_end = start
        va_end = tr_end + val_size
        tr_idx = np.arange(0, tr_end, dtype=int)
        va_idx = np.arange(tr_end, va_end, dtype=int)
        if len(va_idx) >= 2 and len(tr_idx) >= min_train:
            splits.append((tr_idx, va_idx))
        start += step
    return splits


def _mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(y_true - y_pred)))


def _mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean((y_true - y_pred) ** 2))


# ---------------------------
# Model registry
# ---------------------------

def _pipe_enet() -> Pipeline:
    # ElasticNetCV is strong on small-N, wide-X. Scaled.
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("enet", ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 1.0],
                              alphas=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                              max_iter=5000))
    ])


def _pipe_lasso() -> Pipeline:
    # Kept for backward compatibility with "lasso" whitelist.
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lasso", LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], max_iter=5000))
    ])


def _pipe_ridge() -> Pipeline:
    # RidgeCV for legacy "ridge" key; quick and stable.
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=[0.1, 0.5, 1.0, 2.0, 5.0]))
    ])


def _pipe_rf(random_state: int = 42) -> RandomForestRegressor:
    # RF kept for legacy "random_forest"/"rf" whitelist buckets.
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )


def _pipe_hgb(random_state: int = 42) -> HistGradientBoostingRegressor:
    # Captures non-linearities; early_stopping keeps it fast.
    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.06,
        max_iter=400,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=random_state
    )


def _pipe_stack(random_state: int = 42) -> StackingRegressor:
    # ENet + HGB with Ridge meta-learner. Strong 2-3 model combo.
    enet = _pipe_enet()
    hgb = _pipe_hgb(random_state=random_state)
    return StackingRegressor(
        estimators=[("enet", enet), ("hgb", hgb)],
        final_estimator=Ridge(alpha=0.5),
        passthrough=False,
        n_jobs=None
    )


# ---------------------------
# Baseline (tiny data safety)
# ---------------------------

class _MeanBaseline:
    """10-game rolling mean fallback. Not for accuracy — for stability when data is very small."""
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self._mean_ = float(np.nanmean(y[-10:])) if y.size else 0.0
        return self

    def predict(self, X):
        n = len(X) if isinstance(X, (pd.DataFrame, pd.Series)) else 1
        return np.full(n, self._mean_, dtype=float)


# ---------------------------
# ModelManager
# ---------------------------

class ModelManager:
    """
    Drop-in manager used by the app:
      - train(X, y): walk-forward CV per candidate, refit on all data
      - predict(X_next): get predictions, pick best by MAE
      - best_model(): ModelBest(name, prediction, mae, mse)
      - set_model_whitelist(keys): restrict candidates for speed

    Supported keys:
      "elasticnet", "hgb", "stack", "lasso", "ridge",
      "random_forest", "rf"
    """
    def __init__(self, random_state: int = 42,
                 n_folds: int = 6,
                 min_train: int = 24,
                 val_size: int = 6):
        self.random_state = random_state
        self.n_folds = n_folds
        self.min_train = min_train
        self.val_size = val_size

        self._whitelist: Optional[List[str]] = None
        self._trained: Dict[str, Dict] = {}  # name -> {est, mae, mse}
        self._best: Optional[ModelBest] = None

        # exposed for compatibility with callers that introspect
        self.available_models: List[str] = [
            "elasticnet", "hgb", "stack", "lasso", "ridge", "random_forest", "rf"
        ]
        self.models = self.available_models  # alias

    # -------- Whitelist --------

    def set_model_whitelist(self, keys: List[str]) -> None:
        """Restrict candidates to a subset by key. Unknown keys ignored.
        Why: speed control from the UI."""
        if not keys:
            self._whitelist = None
            return
        keys = [str(k).lower().strip() for k in keys]
        self._whitelist = [k for k in keys if k in self.available_models]

    # -------- Candidates --------

    def _make_estimator(self, key: str):
        k = key.lower()
        if k == "elasticnet":
            return _pipe_enet()
        if k == "hgb":
            return _pipe_hgb(self.random_state)
        if k == "stack":
            return _pipe_stack(self.random_state)
        if k in ("lasso",):
            return _pipe_lasso()
        if k in ("ridge",):
            return _pipe_ridge()
        if k in ("random_forest", "rf"):
            return _pipe_rf(self.random_state)
        # fallback
        return _pipe_enet()

    def _candidate_keys(self) -> List[str]:
        if self._whitelist:
            return [k for k in self._whitelist if k in self.available_models]
        # Default strong trio
        return ["elasticnet", "hgb", "stack"]

    # -------- Core API --------

    def train(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit all candidates; compute walk-forward MAE/MSE; refit on all data.
        Why: best-on-time splits correlates with next-game error."""
        X = _to_num_df(X)
        y = np.asarray(y, dtype=float)
        n = len(X)

        # Tiny data: store only baseline
        splits = _rolling_splits(n, self.n_folds, self.min_train, self.val_size)
        if len(splits) < 1:
            base = _MeanBaseline().fit(X, y)
            self._trained = {"baseline": {"est": base, "mae": np.nan, "mse": np.nan}}
            self._best = None
            return

        results: Dict[str, Dict] = {}
        for key in self._candidate_keys():
            est = self._make_estimator(key)
            fold_mae, fold_mse = [], []
            for tr, va in splits:
                try:
                    est_fold = clone(est)
                    est_fold.fit(X.iloc[tr], y[tr])
                    pred = est_fold.predict(X.iloc[va])
                    fold_mae.append(_mae(y[va], pred))
                    fold_mse.append(_mse(y[va], pred))
                except Exception:
                    # defensive: treat failure as poor score
                    fold_mae.append(np.inf)
                    fold_mse.append(np.inf)
            mae = float(np.nanmean(fold_mae))
            mse = float(np.nanmean(fold_mse))
            # refit on all data for final prediction
            try:
                est.fit(X, y)
            except Exception:
                est = _MeanBaseline().fit(X, y)
                mae, mse = (np.nan, np.nan)
            results[key] = {"est": est, "mae": mae, "mse": mse}

        self._trained = results
        self._best = None  # will be determined at predict-time

    def predict(self, X_next: pd.DataFrame) -> Dict[str, float]:
        """Predict with all trained models; select best by MAE (tie-break with MSE)."""
        if not self._trained:
            raise RuntimeError("Call train(X, y) before predict(X_next).")

        Xn = _to_num_df(pd.DataFrame(X_next))
        preds: Dict[str, float] = {}

        # compute predictions
        for name, pack in self._trained.items():
            est = pack["est"]
            try:
                preds[name] = float(est.predict(Xn)[0])
            except Exception:
                preds[name] = float("nan")

        # choose best by MAE then MSE
        ordered = sorted(
            self._trained.items(),
            key=lambda kv: (kv[1]["mae"], kv[1]["mse"])
        )
        best_name, best_pack = ordered[0]
        best_pred = preds.get(best_name, float("nan"))
        self._best = ModelBest(
            name=_canonical_display_name(best_name),
            prediction=best_pred,
            mae=float(best_pack["mae"]),
            mse=float(best_pack["mse"]),
        )
        return preds

    def best_model(self) -> ModelBest:
        """Return ModelBest(name, prediction, mae, mse)."""
        if self._best is not None:
            return self._best
        # If predict() not called, select by MAE and return without prediction
        if not self._trained:
            raise RuntimeError("No models trained.")
        ordered = sorted(
            self._trained.items(),
            key=lambda kv: (kv[1]["mae"], kv[1]["mse"])
        )
        best_name, best_pack = ordered[0]
        return ModelBest(
            name=_canonical_display_name(best_name),
            prediction=float("nan"),
            mae=float(best_pack["mae"]),
            mse=float(best_pack["mse"]),
        )


def _canonical_display_name(key: str) -> str:
    k = key.lower()
    if k == "elasticnet": return "Elastic Net"
    if k == "hgb": return "HistGradientBoosting"
    if k == "stack": return "Stack(ENet+HGB)"
    if k == "lasso": return "Lasso"
    if k == "ridge": return "Ridge"
    if k in ("random_forest", "rf"): return "RandomForest"
    if k == "baseline": return "Baseline"
    return key
