"""Model definitions for baseline and SVM classifiers."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def create_baseline(config: dict, seed: int) -> LogisticRegression:
    """Logistic regression baseline from config."""
    baseline_cfg = config.get("baseline", {})
    return LogisticRegression(
        class_weight=baseline_cfg.get("class_weight", "balanced"),
        max_iter=baseline_cfg.get("max_iter", 1000),
        solver=baseline_cfg.get("solver", "lbfgs"),
        penalty=baseline_cfg.get("penalty", "l2"),
        C=baseline_cfg.get("C", 1.0),
        random_state=seed,
        n_jobs=-1,
    )


def create_svm(config: dict, seed: int) -> SVC:
    """SVM classifier from config (supports linear and RBF kernels)."""
    model_cfg = config.get("model", {})
    return SVC(
        kernel=model_cfg.get("kernel", "linear"),
        C=model_cfg.get("C", 1.0),
        gamma=model_cfg.get("gamma", "scale"),
        class_weight=model_cfg.get("class_weight", "balanced"),
        max_iter=model_cfg.get("max_iter", 1000000),
        tol=model_cfg.get("tol", 0.001),
        random_state=seed,
    )
