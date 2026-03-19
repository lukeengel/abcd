"""Model definitions for regression."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


# ---------------------------------------------------------------------------
# Sample-weight support registry
# ---------------------------------------------------------------------------

# sklearn support status for sample_weight in .fit():
#   SVR          — YES: rescales C per sample (per-sample margin penalty)
#   Ridge        — YES
#   ElasticNet   — YES
#   LinearRegression — YES
#   RandomForest — YES
#   MLPRegressor — NO: not supported in sklearn
_DEFAULT_SAMPLE_WEIGHT_SUPPORT: dict[str, bool] = {
    "linear": True,
    "ridge": True,
    "elastic_net": True,
    "svr": True,
    "random_forest": True,
    "mlp": False,
}


def model_supports_sample_weight(model_name: str, config: dict) -> bool:
    """Return True if this model accepts sample_weight in fit().

    Reads 'supports_sample_weight' from the model's config block first,
    then falls back to sklearn defaults. This allows per-run overrides via
    regression.yaml without code changes.

    Note: SVR support is enabled by default (rescales C per sample) but can be
    disabled in config (supports_sample_weight: false) to reproduce results
    from runs before this fix.
    """
    model_cfg = config.get("models", {}).get(model_name, {})
    if "supports_sample_weight" in model_cfg:
        return bool(model_cfg["supports_sample_weight"])
    return _DEFAULT_SAMPLE_WEIGHT_SUPPORT.get(model_name, False)


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def create_baseline(config: dict, seed: int) -> Ridge:
    """Ridge regression baseline from config."""
    baseline_cfg = config.get("baseline", {})
    return Ridge(alpha=baseline_cfg.get("alpha", 1.0), random_state=seed)


def create_linear(config: dict, seed: int) -> LinearRegression:
    """Ordinary Least Squares linear regression."""
    return LinearRegression()


def create_ridge(config: dict, seed: int) -> Ridge:
    """Ridge regression from config."""
    model_cfg = config.get("models", {}).get("ridge", {})
    return Ridge(alpha=model_cfg.get("alpha", 1.0), random_state=seed)


def create_elastic_net(config: dict, seed: int) -> ElasticNet:
    """Elastic Net regression from config."""
    model_cfg = config.get("models", {}).get("elastic_net", {})
    return ElasticNet(
        alpha=model_cfg.get("alpha", 0.01),
        l1_ratio=model_cfg.get("l1_ratio", 0.5),
        max_iter=10000,
        random_state=seed,
    )


def create_svr(config: dict, seed: int) -> SVR:
    """Support Vector Regression from config.

    SVR.fit() accepts sample_weight, which rescales C per sample
    (higher weight = larger margin penalty = forced support vector).
    Enable via supports_sample_weight: true in regression.yaml.
    """
    model_cfg = config.get("models", {}).get("svr", {})
    return SVR(
        kernel=model_cfg.get("kernel", "rbf"),
        C=model_cfg.get("C", 10.0),
        epsilon=model_cfg.get("epsilon", 0.01),
        gamma=model_cfg.get("gamma", "scale"),
    )


def create_random_forest(config: dict, seed: int) -> RandomForestRegressor:
    """Random Forest regression from config."""
    model_cfg = config.get("models", {}).get("random_forest", {})
    return RandomForestRegressor(
        n_estimators=model_cfg.get("n_estimators", 400),
        max_depth=model_cfg.get("max_depth", 15),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 5),
        random_state=seed,
        n_jobs=1,  # Disable parallelization to avoid Jupyter deadlock
    )


def create_mlp(config: dict, seed: int) -> MLPRegressor:
    """MLP regression from config. Does NOT support sample_weight."""
    model_cfg = config.get("models", {}).get("mlp", {})
    hidden_layers = model_cfg.get("hidden_layer_sizes", (128, 64, 32))
    if isinstance(hidden_layers, list):
        hidden_layers = tuple(hidden_layers)

    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=model_cfg.get("activation", "relu"),
        alpha=model_cfg.get("alpha", 0.001),
        learning_rate=model_cfg.get("learning_rate", "adaptive"),
        learning_rate_init=model_cfg.get("learning_rate_init", 0.0005),
        max_iter=model_cfg.get("max_iter", 10000),
        early_stopping=model_cfg.get("early_stopping", True),
        validation_fraction=0.15,
        n_iter_no_change=model_cfg.get("n_iter_no_change", 50),
        batch_size=model_cfg.get("batch_size", 32),
        random_state=seed,
    )


MODEL_REGISTRY = {
    "linear": create_linear,
    "ridge": create_ridge,
    "elastic_net": create_elastic_net,
    "svr": create_svr,
    "random_forest": create_random_forest,
    "mlp": create_mlp,
}
