"""Model definitions for Random Forest classifiers."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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


def create_random_forest(config: dict, seed: int) -> RandomForestClassifier:
    """Random Forest classifier from config."""
    model_cfg = config.get("model", {})
    return RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", None),
        min_samples_split=model_cfg.get("min_samples_split", 2),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
        max_features=model_cfg.get("max_features", "sqrt"),
        class_weight=model_cfg.get("class_weight", "balanced_subsample"),
        bootstrap=model_cfg.get("bootstrap", True),
        oob_score=model_cfg.get("oob_score", False),
        n_jobs=model_cfg.get("n_jobs", -1),
        random_state=seed,
        verbose=0,
    )
