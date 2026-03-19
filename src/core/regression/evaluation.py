"""Evaluation metrics and statistical tests for regression."""

import numpy as np
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

    if len(y_true) > 1:
        # Constant predictions → r is undefined; return 0 instead of warning
        if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
            metrics["pearson_r"] = 0.0
            metrics["pearson_p"] = 1.0
            metrics["spearman_r"] = 0.0
            metrics["spearman_p"] = 1.0
        else:
            r, p = pearsonr(y_true, y_pred)
            metrics["pearson_r"] = float(r)
            metrics["pearson_p"] = float(p)
            rho, p_rho = spearmanr(y_true, y_pred)
            metrics["spearman_r"] = float(rho)
            metrics["spearman_p"] = float(p_rho)
    else:
        metrics["pearson_r"] = 0.0
        metrics["pearson_p"] = 1.0
        metrics["spearman_r"] = 0.0
        metrics["spearman_p"] = 1.0

    return metrics


def aggregate_cv_results(folds: list[dict]) -> dict:
    """Aggregate regression results across CV folds."""
    all_y_true = np.concatenate([fold["y_test"] for fold in folds])
    all_y_pred = np.concatenate([fold["y_pred"] for fold in folds])
    overall_metrics = compute_regression_metrics(all_y_true, all_y_pred)

    # Parametric p-value on concatenated CV predictions is anti-conservative
    # (assumes independent observations, but they come from overlapping training sets).
    # Use permutation_test() + compute_permutation_pvalue() for reporting.
    if "pearson_p" in overall_metrics:
        overall_metrics["pearson_p_parametric"] = overall_metrics.pop("pearson_p")

    per_fold_metrics = {}
    metric_names = ["r2", "mae", "mse", "rmse", "pearson_r", "spearman_r"]

    for metric_name in metric_names:
        fold_values = [fold["metrics"][metric_name] for fold in folds]
        per_fold_metrics[f"{metric_name}_mean"] = float(np.mean(fold_values))
        per_fold_metrics[f"{metric_name}_std"] = float(np.std(fold_values))
        per_fold_metrics[f"{metric_name}_min"] = float(np.min(fold_values))
        per_fold_metrics[f"{metric_name}_max"] = float(np.max(fold_values))

    return {
        "overall": overall_metrics,
        "per_fold": per_fold_metrics,
        "n_folds": len(folds),
        "n_samples": len(all_y_true),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = (
        (n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)
    ) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    return float((group1.mean() - group2.mean()) / pooled_std) if pooled_std > 0 else 0.0


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 10000,
    metrics: list[str] | None = None,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals on held-out CV predictions.

    Args:
        y_true: Observed values (concatenated across CV folds).
        y_pred: Predicted values (concatenated across CV folds).
        n_bootstrap: Number of bootstrap resamples.
        metrics: Metrics to bootstrap. Defaults to [pearson_r, spearman_r, r2].
        seed: Random seed for reproducibility.
        ci: Confidence interval width (0.95 → 95% CI).

    Returns:
        Dict keyed by metric name, each with:
            observed (float), lower (float), upper (float),
            boot_dist (ndarray), ci_level (float), n_bootstrap (int).
    """
    if metrics is None:
        metrics = ["pearson_r", "spearman_r", "r2"]

    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (1 - ci) / 2
    boot_samples: dict[str, list] = {m: [] for m in metrics}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        m = compute_regression_metrics(y_true[idx], y_pred[idx])
        for metric in metrics:
            boot_samples[metric].append(m.get(metric, np.nan))

    observed = compute_regression_metrics(y_true, y_pred)
    results = {}
    for metric in metrics:
        dist = np.array(boot_samples[metric])
        results[metric] = {
            "observed": float(observed.get(metric, np.nan)),
            "lower": float(np.nanpercentile(dist, 100 * alpha)),
            "upper": float(np.nanpercentile(dist, 100 * (1 - alpha))),
            "boot_dist": dist,
            "ci_level": ci,
            "n_bootstrap": n_bootstrap,
        }
    return results


def fisher_z_compare(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Fisher z-test comparing two independent Pearson correlations.

    Tests H0: r1 == r2 (two-tailed).

    Args:
        r1, r2: Pearson correlation coefficients to compare.
        n1, n2: Sample sizes for each correlation.

    Returns:
        (z, p) — z-statistic and two-tailed p-value.
    """
    # Clip to avoid atanh blowing up at ±1
    r1c = np.clip(r1, -0.9999, 0.9999)
    r2c = np.clip(r2, -0.9999, 0.9999)
    z1 = np.arctanh(r1c)
    z2 = np.arctanh(r2c)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    p = float(2 * norm.sf(abs(z)))
    return float(z), p


def three_way_lateralization_interaction(
    groups: dict,
    target_col: str,
    left_col: str,
    right_col: str,
    structure_name: str = "pallidum",
) -> dict:
    """Test sex × timepoint × hemisphere interaction on brain-symptom correlations.

    Computes Pearson r for left and right hemisphere vs target, separately for
    each (sex, timepoint) group, then applies Fisher z-tests to test pairwise
    differences and the 3-way interaction.

    Args:
        groups: Dict of label → dict with keys 'X_left', 'X_right', 'y' (all ndarrays).
                Expected keys: 'male_bl', 'female_bl', 'male_y2', 'female_y2'.
        target_col: Name of target column (for display only).
        left_col: Name of left hemisphere feature (for display only).
        right_col: Name of right hemisphere feature (for display only).
        structure_name: Structure label for printing.

    Returns:
        Dict with correlation table and Fisher z-test results.
    """
    from scipy.stats import pearsonr

    required = ["male_bl", "female_bl", "male_y2", "female_y2"]
    for k in required:
        if k not in groups:
            raise ValueError(f"Missing group '{k}' in groups dict. Need: {required}")

    corr_table = {}
    for label, grp in groups.items():
        xl, xr, y = grp["X_left"], grp["X_right"], grp["y"]
        valid_l = np.isfinite(xl) & np.isfinite(y)
        valid_r = np.isfinite(xr) & np.isfinite(y)
        r_l, p_l = pearsonr(xl[valid_l], y[valid_l]) if valid_l.sum() > 5 else (np.nan, np.nan)
        r_r, p_r = pearsonr(xr[valid_r], y[valid_r]) if valid_r.sum() > 5 else (np.nan, np.nan)
        corr_table[label] = {
            "r_left": r_l, "p_left": p_l, "n_left": int(valid_l.sum()),
            "r_right": r_r, "p_right": p_r, "n_right": int(valid_r.sum()),
        }

    # Print correlation table
    print(f"\n  {structure_name.upper()} L/R × Sex × Timepoint — {target_col}")
    print(f"  {'Group':<14} {'r(Left)':>9} {'p(L)':>8} {'r(Right)':>10} {'p(R)':>8} {'n':>6}")
    print(f"  {'-' * 60}")
    for label, row in corr_table.items():
        sig_l = "*" if row["p_left"] < 0.05 else ""
        sig_r = "*" if row["p_right"] < 0.05 else ""
        print(f"  {label:<14} {row['r_left']:>+9.3f}{sig_l:<1}  {row['p_left']:>7.4f} "
              f"  {row['r_right']:>+9.3f}{sig_r:<1}  {row['p_right']:>7.4f} "
              f"  {row['n_left']:>5}")

    # Pairwise Fisher z-tests (key comparisons)
    comparisons = [
        # Sex effect at each timepoint
        ("male_bl",   "female_bl",  "Sex diff @ BL    (left)"),
        ("male_y2",   "female_y2",  "Sex diff @ Y2    (left)"),
        # Timepoint effect by sex
        ("male_bl",   "male_y2",    "BL→Y2 (male,    left)"),
        ("female_bl", "female_y2",  "BL→Y2 (female,  left)"),
        # 3-way: sex×timepoint change in lateralization
        # Approximate: (male_bl_L - male_y2_L) vs (female_bl_L - female_y2_L) on left r
    ]

    print(f"\n  Fisher z-tests (left hemisphere r comparisons):")
    print(f"  {'Comparison':<35} {'z':>8} {'p':>8}")
    print(f"  {'-' * 55}")
    fisher_results = {}
    for g1, g2, desc in comparisons:
        r1 = corr_table[g1]["r_left"]; n1 = corr_table[g1]["n_left"]
        r2 = corr_table[g2]["r_left"]; n2 = corr_table[g2]["n_left"]
        if np.isnan(r1) or np.isnan(r2) or n1 < 4 or n2 < 4:
            print(f"  {desc:<35} {'n/a':>8} {'n/a':>8}")
            continue
        z, p = fisher_z_compare(r1, n1, r2, n2)
        sig = "*" if p < 0.05 else ""
        print(f"  {desc:<35} {z:>+8.3f} {p:>8.4f}{sig}")
        fisher_results[desc] = {"z": z, "p": p, "r1": r1, "n1": n1, "r2": r2, "n2": n2}

    # 3-way interaction approximation: compare sex difference at BL vs Y2
    if all(k in fisher_results for k in ["Sex diff @ BL    (left)", "Sex diff @ Y2    (left)"]):
        bl_sex = fisher_results["Sex diff @ BL    (left)"]
        y2_sex = fisher_results["Sex diff @ Y2    (left)"]
        # Fisher z on the z-statistics themselves (test if sex effect changes over time)
        z_bl = np.arctanh(np.clip(bl_sex["r1"] - bl_sex["r2"], -0.9999, 0.9999))
        z_y2 = np.arctanh(np.clip(y2_sex["r1"] - y2_sex["r2"], -0.9999, 0.9999))
        se_3way = np.sqrt(
            1/(bl_sex["n1"] - 3) + 1/(bl_sex["n2"] - 3) +
            1/(y2_sex["n1"] - 3) + 1/(y2_sex["n2"] - 3)
        )
        z_3way = (z_bl - z_y2) / se_3way
        p_3way = float(2 * norm.sf(abs(z_3way)))
        sig = "*" if p_3way < 0.05 else ""
        print(f"\n  3-WAY (Sex × Timepoint × Hemisphere approx): z={z_3way:+.3f}, p={p_3way:.4f}{sig}")
        fisher_results["3way_sex_x_timepoint"] = {"z": z_3way, "p": p_3way}

    return {"corr_table": corr_table, "fisher_results": fisher_results}


def compute_permutation_pvalue(observed_r: float, null_distribution: np.ndarray) -> float:
    """Two-tailed empirical p-value from a permutation null distribution.

    Uses |r| comparison so it handles both positive and negative observed r.
    Adds 1 to numerator and denominator (conservative, avoids p=0).
    """
    n_exceed = int(np.sum(np.abs(null_distribution) >= abs(observed_r)))
    n = len(null_distribution)
    return (n_exceed + 1) / (n + 1)


def permutation_test(
    env,
    full_df,
    target_config: dict,
    model_name: str,
    n_permutations: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Pipeline-matched permutation test: shuffles labels, runs full nested CV.

    Matches the main pipeline exactly — per-fold ComBat harmonization,
    per-fold residualization, site stratification, and family-aware CV.
    This ensures the null distribution accounts for any inflation from
    preprocessing.

    Args:
        env: Environment with configs.
        full_df: Full dataset (same as passed to run_target_with_nested_cv).
        target_config: Target configuration dict (name, column).
        model_name: Model to permute-test (e.g., "svr").
        n_permutations: Overrides regression.yaml permutation.n_permutations.
        seed: Overrides run.yaml seed (used to seed the label shuffler,
              NOT the CV splits — CV seed is always env.configs.run["seed"]).
        verbose: Show tqdm progress bar.

    Returns:
        Dict with:
            null_distribution (ndarray): Pearson r values from permuted runs.
            n_permutations (int): Number completed.
            null_mean (float): Mean of null distribution.
            null_std (float): Std of null distribution.

        Use compute_permutation_pvalue(observed_r, result["null_distribution"])
        to obtain the empirical p-value.
    """
    # Lazy import to avoid circular dependency (pipeline.py imports from here)
    from .pipeline import run_target_with_nested_cv

    reg_config = env.configs.regression
    if n_permutations is None:
        n_permutations = reg_config.get("permutation", {}).get("n_permutations", 1000)
    if seed is None:
        seed = env.configs.run.get("seed", 42)

    target_col = target_config["column"]
    target_name = target_config["name"]

    if verbose:
        print(f"Pipeline-matched permutation test: {target_name} / {model_name}")
        print(f"  n_permutations={n_permutations} | shuffle_seed={seed}")
        print("  Per-fold ComBat + residualization + family-aware CV — matches main pipeline")

    null_rs: list[float] = []
    rng = np.random.RandomState(seed)

    perm_iter: range | object = range(n_permutations)
    if verbose:
        from tqdm import tqdm
        perm_iter = tqdm(perm_iter, desc="Permutations")

    for _ in perm_iter:
        perm_df = full_df.copy()
        # Shuffle ONLY the target column — features, covariates, family IDs unchanged
        perm_df[target_col] = rng.permutation(perm_df[target_col].values)
        result = run_target_with_nested_cv(
            env, perm_df, target_config, model_name, verbose=False
        )
        null_rs.append(result[model_name]["overall"]["pearson_r"])

    null_arr = np.array(null_rs)
    return {
        "null_distribution": null_arr,
        "n_permutations": n_permutations,
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
    }
