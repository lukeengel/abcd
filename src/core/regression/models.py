"""Model definitions for regression."""

from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def create_baseline(config: dict, seed: int) -> Ridge:
    """Ridge regression baseline from config."""
    baseline_cfg = config.get("baseline", {})
    return Ridge(
        alpha=baseline_cfg.get("alpha", 1.0),
        random_state=seed,
    )


def create_linear(config: dict, seed: int) -> LinearRegression:
    """Ordinary Least Squares (OLS) linear regression."""
    return LinearRegression()


def create_ridge(config: dict, seed: int) -> Ridge:
    """Ridge regression model."""
    model_cfg = config.get("models", {}).get("ridge", {})
    return Ridge(
        alpha=model_cfg.get("alpha", 1.0),
        random_state=seed,
    )


def create_elastic_net(config: dict, seed: int) -> ElasticNet:
    """Elastic Net regression model."""
    model_cfg = config.get("models", {}).get("elastic_net", {})
    return ElasticNet(
        alpha=model_cfg.get("alpha", 0.01),
        l1_ratio=model_cfg.get("l1_ratio", 0.5),
        max_iter=10000,
        random_state=seed,
    )


def create_svr(config: dict, seed: int) -> SVR:
    """Support Vector Regression model."""
    model_cfg = config.get("models", {}).get("svr", {})
    return SVR(
        kernel=model_cfg.get("kernel", "rbf"),
        C=model_cfg.get("C", 1.0),
        epsilon=model_cfg.get("epsilon", 0.1),
    )


def create_random_forest(config: dict, seed: int) -> RandomForestRegressor:
    """Random Forest regression model."""
    model_cfg = config.get("models", {}).get("random_forest", {})
    return RandomForestRegressor(
        n_estimators=model_cfg.get("n_estimators", 400),
        max_depth=model_cfg.get("max_depth", 15),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 5),
        random_state=seed,
        n_jobs=1,  # Disable parallelization to avoid Jupyter deadlock
    )


def create_mlp(config: dict, seed: int) -> MLPRegressor:
    """MLP regression model."""
    model_cfg = config.get("models", {}).get("mlp", {})

    hidden_layers = model_cfg.get("hidden_layer_sizes", (64, 32))
    if isinstance(hidden_layers, list):
        hidden_layers = tuple(hidden_layers)

    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=model_cfg.get("activation", "relu"),
        alpha=model_cfg.get("alpha", 0.001),
        learning_rate=model_cfg.get("learning_rate", "adaptive"),
        max_iter=model_cfg.get("max_iter", 3000),
        early_stopping=model_cfg.get("early_stopping", True),
        validation_fraction=0.1,
        n_iter_no_change=10,
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
