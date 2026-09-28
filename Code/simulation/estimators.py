"""
Calibration estimators and ATE computation.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def fit_calibration(Ystar, X, Y, method="ols"):
    """
    Fit the calibration function mu_hat: (Y*, X) -> Y on the P=0 sample.

    Returns a fitted model with a .predict(features) method.
    """
    features = np.column_stack([Ystar, X])
    if method == "ols":
        model = LinearRegression()
    elif method == "rf":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=0
        )
    elif method == "gbt":
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=0
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    model.fit(features, Y)
    return model


def predict_calibrated(model, Ystar, X):
    """Apply a fitted calibration model to new data."""
    features = np.column_stack([Ystar, X])
    return model.predict(features)


def calibrated_ate(model, Ystar, X, W):
    """Compute the calibrated ATE on a sample."""
    Y_hat = predict_calibrated(model, Ystar, X)
    return Y_hat[W == 1].mean() - Y_hat[W == 0].mean()


def raw_llm_ate(Ystar, W):
    """Compute the naive (uncalibrated) ATE from LLM outputs."""
    return Ystar[W == 1].mean() - Ystar[W == 0].mean()


def human_ate(Y, W):
    """Compute the sample ATE from human outcomes."""
    return Y[W == 1].mean() - Y[W == 0].mean()
