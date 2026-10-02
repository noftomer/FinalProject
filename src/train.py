"""Fit the Ridge model, choosing the regularization strength on the validation split."""
import numpy as np
from sklearn.linear_model import Ridge

from src.models import build_model


def fit(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
        alphas: list[float]) -> tuple[Ridge, float]:
    """Return the model whose alpha gives the lowest validation MSE, and that alpha."""
    best_model, best_alpha, best_mse = None, None, float("inf")
    for alpha in alphas:
        model = build_model(alpha).fit(X_train, y_train)
        mse = float(np.mean((model.predict(X_val) - y_val) ** 2))
        print(f"alpha {alpha:>9g} | val MSE {mse:.3e}")
        if mse < best_mse:
            best_model, best_alpha, best_mse = model, alpha, mse
    print(f"Chosen alpha: {best_alpha:g}")
    return best_model, best_alpha
