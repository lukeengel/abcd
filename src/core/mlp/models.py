"""Model definitions for baseline and MLP classifiers."""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


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


def create_mlp(config: dict, seed: int) -> MLPClassifier:
    """MLP classifier from config."""
    model_cfg = config.get("model", {})
    return MLPClassifier(
        hidden_layer_sizes=model_cfg.get("hidden_layer_sizes", (128, 64)),
        activation=model_cfg.get("activation", "relu"),
        alpha=model_cfg.get("alpha", 0.001),
        learning_rate=model_cfg.get("learning_rate", "adaptive"),
        max_iter=model_cfg.get("max_iter", 1000),
        early_stopping=model_cfg.get("early_stopping", True),
        validation_fraction=model_cfg.get("validation_fraction", 0.1),
        n_iter_no_change=model_cfg.get("n_iter_no_change", 10),
        random_state=seed,
    )
