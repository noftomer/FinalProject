"""Linear model that maps the last few daily returns to the next-day return."""
from sklearn.linear_model import Ridge


def build_model(alpha: float) -> Ridge:
    """Ridge regression with L2 strength ``alpha``."""
    return Ridge(alpha=alpha)
