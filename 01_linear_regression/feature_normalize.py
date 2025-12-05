import numpy as np

def feature_normalize(X):
    """
    Perform feature normalization

    Transformation applied:
        X_norm = (X - μ) / σ

    Returns:
        X_norm: normalized feature matrix
        μ:     per-feature mean
        σ:  per-feature std deviation
    """

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma