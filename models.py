"""
models.py — Multi-Model Engine (KNN Safe Version)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
except:
    xgb = None


@dataclass
class ModelInfo:
    name: str
    model: Any
    mae: float
    mse: float
    prediction: Optional[float] = None


class ModelManager:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models: Dict[str, ModelInfo] = {}

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.models.clear()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )

        def _eval(name, model):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            self.models[name] = ModelInfo(name, model, float(mae), float(mse))

        # Core linear models
        _eval("LinearRegression", LinearRegression())
        _eval("Ridge", Ridge(random_state=self.random_state))
        _eval("Lasso", Lasso(random_state=self.random_state))

        # Tree models
        _eval("DecisionTree", DecisionTreeRegressor(random_state=self.random_state))
        _eval("RandomForest", RandomForestRegressor(
            n_estimators=300, random_state=self.random_state, n_jobs=-1))
        _eval("GradientBoosting", GradientBoostingRegressor(random_state=self.random_state))

        # KNN allowed only if dataset is large enough
        if len(X_train) >= 10:
            _eval("KNN", KNeighborsRegressor())

        # SVR model
        _eval("SVR", SVR(kernel="rbf"))

        # XGBoost
        if xgb is not None:
            _eval("XGBoost",
                xgb.XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.07,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    tree_method="hist"
                )
            )

        return self.models

    def predict(self, X_new):
        preds = {}
        for name, info in self.models.items():
            try:
                pred = float(info.model.predict(X_new)[0])
                info.prediction = pred
                preds[name] = pred
            except:
                pass
        return preds

    def best_model(self):
        return min(self.models.values(), key=lambda m: m.mae)
