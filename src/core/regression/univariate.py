"""Univariate analyses: feature correlations, asymmetry, sex differences, circuit interactions."""

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn
from scipy.stats import pearsonr, ttest_ind
from statsmodels.stats.multitest import multipletests

from ..tsne.embeddings import get_roi_columns_from_config
from .pipeline import fit_residualize, apply_residualize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bilateral pair extraction
# ---------------------------------------------------------------------------

def extract_bilateral_pairs(data_config, network_names):
    """Extract L/R bilateral pairs from data.yaml network definitions.

    Returns:
        bilateral_pairs: list of (short_name, left_col, right_col)
        unilateral: list of columns without a matched pair
    """
    roi_features = data_config.get("roi_features", {})
    all_features = []
    for net in network_names:
        net_def = roi_features.get(net, {})
        for feat_type in ("structural", "connectivity"):
            all_features.extend(net_def.get(feat_type) or [])
    # deduplicate preserving order
    all_features = list(dict.fromkeys(all_features))

    bilateral_pairs = []
    used = set()

    suffix_pairs = [("lh", "rh"), ("l", "r")]

    for f in all_features:
        if f in used:
            continue
        matched = False

        # Try standard lateral suffixes
        for lsuf, rsuf in suffix_pairs:
            if f.endswith(lsuf):
                candidate = f[: -len(lsuf)] + rsuf
                if candidate in all_features and candidate not in used:
                    short = _make_short_name(f[: -len(lsuf)])
                    bilateral_pairs.append((short, f, candidate))
                    used.update([f, candidate])
                    matched = True
                    break
            elif f.endswith(rsuf):
                candidate = f[: -len(rsuf)] + lsuf
                if candidate in all_features and candidate not in used:
                    short = _make_short_name(f[: -len(rsuf)])
                    bilateral_pairs.append((short, candidate, f))
                    used.update([f, candidate])
                    matched = True
                    break

        # Accumbens special case (aal / aar)
        if not matched and f not in used:
            if f.endswith("aal"):
                candidate = f[:-1] + "r"
                if candidate in all_features and candidate not in used:
                    bilateral_pairs.append(("accumbens", f, candidate))
                    used.update([f, candidate])
                    matched = True
            elif f.endswith("aar"):
                candidate = f[:-1] + "l"
                if candidate in all_features and candidate not in used:
                    bilateral_pairs.append(("accumbens", candidate, f))
                    used.update([f, candidate])
                    matched = True

    unilateral = [f for f in all_features if f not in used]
    return bilateral_pairs, unilateral


def _make_short_name(base):
    """Derive a short display name from the feature base string."""
    parts = base.split("_")
    short = parts[-1] if parts[-1] else parts[-2]
    if "dtifa" in base:
        short += "_FA"
    elif "dtimd" in base:
        short += "_MD"
    elif "thick" in base:
        short += "_thick"
    return short


# ---------------------------------------------------------------------------
# Asymmetry computation — ENIGMA convention: AI = (L - R) / (L + R)
# ---------------------------------------------------------------------------

def compute_asymmetry_features(X, feature_cols, bilateral_pairs):
    """Compute asymmetry index and total for each bilateral pair.

    Convention: AI = (L - R) / (L + R)  (positive = leftward, ENIGMA standard).

    Args:
        X: ndarray (n_subjects, n_features)
        feature_cols: list of column names matching X columns
        bilateral_pairs: list of (short_name, left_col, right_col)

    Returns:
        dict with '{name}_AI' and '{name}_total' arrays
    """
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    result = {}
    for name, lcol, rcol in bilateral_pairs:
        L = X[:, col_to_idx[lcol]]
        R = X[:, col_to_idx[rcol]]
        total = L + R
        total_safe = np.where(np.abs(total) < 1e-6, np.nan, total)
        result[f"{name}_AI"] = (L - R) / total_safe
        result[f"{name}_total"] = total
    return result


# ---------------------------------------------------------------------------
# Data preparation with full-sample ComBat
# ---------------------------------------------------------------------------

def prepare_harmonized_data(
    df,
    feature_cols,
    harmonize_config,
    regression_config,
    target_col,
    target_name=None,
    residualize_age_sex=True,
):
    """Full-sample ComBat harmonization + optional target residualisation.

    Appropriate for univariate tests (no model training ⇒ no data-leakage risk).

    Returns:
        X_harm: harmonised feature matrix
        y: target array (optionally residualised)
        df_filtered: filtered dataframe
        feature_names: list of feature column names in X_harm
    """
    # ---- filter NaN targets ----
    mask = df[target_col].notna()
    df_f = df[mask].copy()
    y = df_f[target_col].values.astype(float)

    # ---- bin filter ----
    if target_name is not None:
        weighting_cfg = regression_config.get("sample_weighting", {})
        custom_bins = weighting_cfg.get("custom_bins", {})
        if target_name in custom_bins:
            edges = custom_bins[target_name]
            keep = (y >= edges[0]) & (y < edges[-1])
            df_f = df_f[keep].reset_index(drop=True)
            y = y[keep]

    # ---- residualise ----
    if residualize_age_sex:
        cov_cfg = regression_config.get("covariates", {})
        if cov_cfg.get("residualize", False):
            is_raw = target_name is not None and target_name.endswith("_raw")
            if not cov_cfg.get("apply_to_raw_scores_only", True) or is_raw:
                cov_cols = cov_cfg.get("columns", [])
                resid_model = fit_residualize(y, df_f, cov_cols)
                y = apply_residualize(y, df_f, cov_cols, resid_model)

    # ---- ensure features present ----
    valid_cols = [c for c in feature_cols if c in df_f.columns]
    feat_valid = df_f[valid_cols].notna().all(axis=1) & ~np.isnan(y)
    df_f = df_f[feat_valid].reset_index(drop=True)
    y = y[feat_valid.values]

    # ---- site filter (< n_splits samples) ----
    site_col = harmonize_config.get("site_column", "mri_info_manufacturer")
    n_splits = regression_config.get("cv", {}).get("n_outer_splits", 5)
    if site_col in df_f.columns:
        site_counts = df_f[site_col].value_counts()
        small = site_counts[site_counts < n_splits].index.tolist()
        if small:
            keep = ~df_f[site_col].isin(small)
            df_f = df_f[keep].reset_index(drop=True)
            y = y[keep.values]

    # ---- ComBat ----
    # Only use covariates that exist AND have at least some data
    harm_cov_cols = [
        c for c in harmonize_config.get("covariates", [])
        if c in df_f.columns and df_f[c].notna().sum() > 0
    ]
    # Filter subjects with NaN in site or covariate columns (NaN covariates
    # silently corrupt the ComBat OLS fit, producing all-NaN output)
    all_cov_for_check = [c for c in [site_col] + harm_cov_cols if c in df_f.columns]
    cov_nan_mask = df_f[all_cov_for_check].isna().any(axis=1)
    if cov_nan_mask.any():
        n_drop = int(cov_nan_mask.sum())
        print(f"  ComBat: dropping {n_drop} subjects with NaN in site/covariate columns")
        df_f = df_f[~cov_nan_mask].reset_index(drop=True)
        y = y[~cov_nan_mask.values]
    X_raw = df_f[valid_cols].values.astype(float)
    covars = df_f[[site_col] + harm_cov_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})
    # Encode string covariates as float64 (neuroHarmonize requires numeric input;
    # int8 from pd.Categorical can silently cause OLS failures in some versions)
    for col in list(covars.columns):
        if col == "SITE":
            continue
        if not pd.api.types.is_numeric_dtype(covars[col]):
            covars[col] = pd.Categorical(covars[col]).codes.astype(float)
        else:
            covars[col] = covars[col].astype(float)
    # Drop constant covariates (e.g. sex when data is sex-stratified)
    for col in list(covars.columns):
        if col == "SITE":
            continue
        if covars[col].nunique() <= 1:
            covars = covars.drop(columns=col)
    eb = harmonize_config.get("empirical_bayes", True)
    smooth_terms = harmonize_config.get("smooth_terms", [])
    _, X_harm = harmonizationLearn(X_raw, covars, eb=eb, smooth_terms=smooth_terms)

    return X_harm, y, df_f, valid_cols


# ---------------------------------------------------------------------------
# Univariate correlations
# ---------------------------------------------------------------------------

def univariate_correlations(
    X, y, feature_names, corrections=("bonferroni", "fdr_bh"), partial_covariates=None
):
    """Pearson r for each feature vs target with multiple-comparison correction.

    Returns DataFrame sorted by |r| descending.
    """
    rows = []
    for i, name in enumerate(feature_names):
        x_i = X[:, i] if X.ndim == 2 else X[name] if isinstance(X, dict) else X
        r, p = pearsonr(x_i, y)
        rows.append({"feature": name, "r": r, "p_raw": p, "n": len(y)})

    df = pd.DataFrame(rows)
    p_raw = df["p_raw"].values

    for method in corrections:
        reject, p_corr, _, _ = multipletests(p_raw, method=method)
        df[f"p_{method}"] = p_corr
        df[f"sig_{method}"] = reject

    df["abs_r"] = df["r"].abs()
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Volume vs Asymmetry comparison
# ---------------------------------------------------------------------------

def _steiger_z(r_ai, r_tot, r_ai_tot, n):
    """Meng, Rosenthal, Rubin (1992) Z-test for two dependent correlations sharing one variable."""
    from scipy.stats import norm
    r_mean = (r_ai**2 + r_tot**2) / 2
    r_ai_z = np.arctanh(r_ai)
    r_tot_z = np.arctanh(r_tot)
    f = (1 - r_ai_tot) / (2 * (1 - r_mean))
    f = np.clip(f, 0.0, 1.0)  # Meng et al. (1992) bounds f to [0, 1]
    h = (1 - f * r_mean) / (1 - r_mean)
    z = (r_ai_z - r_tot_z) * np.sqrt((n - 3) / (2 * (1 - r_ai_tot) * h))
    p = 2 * norm.sf(abs(z))
    return z, p


def volume_vs_asymmetry_tests(X_harm, y, feature_cols, bilateral_pairs):
    """Compare r(AI, target) vs r(Total, target) for each structure.

    Returns DataFrame with Steiger's test results and FDR correction.
    """
    asym = compute_asymmetry_features(X_harm, feature_cols, bilateral_pairs)
    n = len(y)
    rows = []
    for name, lcol, rcol in bilateral_pairs:
        ai = asym[f"{name}_AI"]
        tot = asym[f"{name}_total"]
        r_ai, p_ai = pearsonr(ai, y)
        r_tot, p_tot = pearsonr(tot, y)
        r_ai_tot, _ = pearsonr(ai, tot)
        z, p_steiger = _steiger_z(r_ai, r_tot, r_ai_tot, n)
        rows.append({
            "structure": name,
            "r_AI": r_ai,
            "p_AI": p_ai,
            "r_total": r_tot,
            "p_total": p_tot,
            "steiger_z": z,
            "p_steiger": p_steiger,
        })
    df = pd.DataFrame(rows)
    _, p_fdr, _, _ = multipletests(df["p_steiger"].values, method="fdr_bh")
    df["p_steiger_fdr"] = p_fdr
    return df


# ---------------------------------------------------------------------------
# Compare univariate to SVR coefficients
# ---------------------------------------------------------------------------

def compare_univariate_to_svr(univariate_df, svr_coefficients, feature_names, save_path=None):
    """Side-by-side scatter: univariate r vs SVR coefficient + rank correlation.

    Returns: (rank_r, rank_p, fig) or (rank_r, rank_p, None) if save_path is None.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    coef_df = pd.DataFrame({"feature": feature_names, "svr_coef": svr_coefficients})
    merged = univariate_df.merge(coef_df, on="feature", how="inner")

    rank_r, rank_p = spearmanr(merged["r"], merged["svr_coef"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(merged["r"], merged["svr_coef"], s=60, alpha=0.7,
               edgecolors="black", linewidth=0.5)
    for _, row in merged.iterrows():
        ax.annotate(row["feature"], (row["r"], row["svr_coef"]),
                     fontsize=7, alpha=0.7, textcoords="offset points",
                     xytext=(4, 4))
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Univariate Pearson r", fontweight="bold", fontsize=12)
    ax.set_ylabel("SVR Coefficient", fontweight="bold", fontsize=12)
    ax.set_title(f"Univariate vs Multivariate (rank r={rank_r:.3f}, p={rank_p:.4f})",
                 fontweight="bold", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rank_r, rank_p


# ---------------------------------------------------------------------------
# Sex differences
# ---------------------------------------------------------------------------

def sex_differences_anova(asymmetry_data, sex_labels):
    """Independent t-test per AI feature: male vs female.

    Args:
        asymmetry_data: dict of {name_AI: array}
        sex_labels: array-like of 'male'/'female' (or 0/1)

    Returns DataFrame with t, p, Cohen's d, FDR-corrected p.
    """
    sex = np.asarray(sex_labels)
    is_female = (sex == "female") | (sex == 1)
    rows = []
    for name, vals in asymmetry_data.items():
        if not name.endswith("_AI"):
            continue
        male_vals = vals[~is_female]
        female_vals = vals[is_female]
        t_stat, p_val = ttest_ind(male_vals, female_vals)
        pooled_std = np.sqrt(
            ((len(male_vals) - 1) * male_vals.std(ddof=1) ** 2
             + (len(female_vals) - 1) * female_vals.std(ddof=1) ** 2)
            / (len(male_vals) + len(female_vals) - 2)
        )
        d = (male_vals.mean() - female_vals.mean()) / pooled_std if pooled_std > 0 else 0
        rows.append({
            "feature": name,
            "mean_male": male_vals.mean(),
            "mean_female": female_vals.mean(),
            "t": t_stat,
            "p_raw": p_val,
            "cohens_d": d,
            "n_male": len(male_vals),
            "n_female": len(female_vals),
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        _, p_fdr, _, _ = multipletests(df["p_raw"].values, method="fdr_bh")
        df["p_fdr"] = p_fdr
    return df


def sex_interaction_test(asymmetry_data, y, sex_labels):
    """OLS: y ~ AI + Sex + AI*Sex per feature.

    Returns DataFrame with interaction p-values and per-sex correlations.
    """
    import statsmodels.api as sm

    sex = np.asarray(sex_labels)
    is_female = ((sex == "female") | (sex == 1)).astype(float)
    rows = []
    for name, vals in asymmetry_data.items():
        if not name.endswith("_AI"):
            continue
        X_design = np.column_stack([vals, is_female, vals * is_female])
        X_design = sm.add_constant(X_design)
        try:
            model = sm.OLS(y, X_design).fit()
            p_interaction = model.pvalues[3]
            beta_interaction = model.params[3]
        except Exception:
            p_interaction = np.nan
            beta_interaction = np.nan

        r_male, p_male = pearsonr(vals[is_female == 0], y[is_female == 0])
        r_female, p_female = pearsonr(vals[is_female == 1], y[is_female == 1])
        rows.append({
            "feature": name,
            "beta_interaction": beta_interaction,
            "p_interaction": p_interaction,
            "r_male": r_male,
            "p_male": p_male,
            "r_female": r_female,
            "p_female": p_female,
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        _, p_fdr, _, _ = multipletests(df["p_interaction"].dropna().values, method="fdr_bh")
        df.loc[df["p_interaction"].notna(), "p_interaction_fdr"] = p_fdr
    return df


# ---------------------------------------------------------------------------
# Circuit interactions
# ---------------------------------------------------------------------------

def concordant_asymmetry_score(asymmetry_data):
    """Mean |AI| across structures (global asymmetry burden) + sign concordance.

    Returns: (mean_abs_ai, sign_concordance) arrays.
    """
    ai_names = [k for k in asymmetry_data if k.endswith("_AI")]
    ai_mat = np.column_stack([asymmetry_data[k] for k in ai_names])
    mean_abs = np.mean(np.abs(ai_mat), axis=1)
    sign_concordance = np.abs(np.mean(np.sign(ai_mat), axis=1))
    return mean_abs, sign_concordance


def interaction_terms(asymmetry_data, y, pairs_to_test=None):
    """OLS: y ~ AI_A + AI_B + AI_A*AI_B for all/selected pairs.

    Returns DataFrame with interaction p-values, FDR-corrected.
    """
    import statsmodels.api as sm

    ai_names = sorted(k for k in asymmetry_data if k.endswith("_AI"))
    if pairs_to_test is None:
        pairs_to_test = list(combinations(ai_names, 2))

    rows = []
    for a, b in pairs_to_test:
        va = asymmetry_data[a]
        vb = asymmetry_data[b]
        X_design = np.column_stack([va, vb, va * vb])
        X_design = sm.add_constant(X_design)
        try:
            model = sm.OLS(y, X_design).fit()
            rows.append({
                "feature_A": a,
                "feature_B": b,
                "beta_interaction": model.params[3],
                "p_interaction": model.pvalues[3],
                "r2_full": model.rsquared,
            })
        except Exception:
            rows.append({
                "feature_A": a,
                "feature_B": b,
                "beta_interaction": np.nan,
                "p_interaction": np.nan,
                "r2_full": np.nan,
            })

    df = pd.DataFrame(rows)
    valid = df["p_interaction"].notna()
    if valid.sum() > 0:
        _, p_fdr, _, _ = multipletests(df.loc[valid, "p_interaction"].values, method="fdr_bh")
        df.loc[valid, "p_interaction_fdr"] = p_fdr
    return df


def group_correlation_matrix(asymmetry_data, y, threshold_percentile=75):
    """Split high vs low target groups, compute AI correlation heatmaps.

    Returns: (corr_high, corr_low, fisher_z_df)
    """
    ai_names = sorted(k for k in asymmetry_data if k.endswith("_AI"))
    ai_mat = pd.DataFrame({k: asymmetry_data[k] for k in ai_names})

    cutoff = np.percentile(y, threshold_percentile)
    high = y >= cutoff
    low = y < np.percentile(y, 100 - threshold_percentile)

    corr_high = ai_mat[high].corr()
    corr_low = ai_mat[low].corr()

    # Fisher z-test on pairwise differences
    n_high = high.sum()
    n_low = low.sum()
    rows = []
    for a, b in combinations(ai_names, 2):
        r_h = corr_high.loc[a, b]
        r_l = corr_low.loc[a, b]
        z_h = np.arctanh(r_h)
        z_l = np.arctanh(r_l)
        se = np.sqrt(1 / (n_high - 3) + 1 / (n_low - 3))
        z_diff = (z_h - z_l) / se
        from scipy.stats import norm
        p_diff = 2 * norm.sf(abs(z_diff))
        rows.append({
            "feature_A": a, "feature_B": b,
            "r_high": r_h, "r_low": r_l,
            "z_diff": z_diff, "p_diff": p_diff,
        })
    fisher_df = pd.DataFrame(rows)
    return corr_high, corr_low, fisher_df


# ---------------------------------------------------------------------------
# Lateralization feature set builder
# ---------------------------------------------------------------------------

def build_lateralization_feature_sets(
    X_h: np.ndarray,
    valid_feature_cols: list,
    valid_pairs: list,
) -> dict:
    """Build four feature representations from a harmonized bilateral feature matrix.

    Args:
        X_h: Harmonized feature matrix (n_subjects, n_features).
        valid_feature_cols: Column names corresponding to X_h columns.
        valid_pairs: List of (short_name, left_col, right_col) for bilateral structures.

    Returns:
        dict with keys:
            "Asymmetry only (AI)": AI features only (n_pairs columns)
            "Total volume only": sum L+R per pair (n_pairs columns)
            "AI + Total": concatenation of AI and total (2*n_pairs columns)
            "Original L/R": raw L/R features from X_h
    """
    asym = compute_asymmetry_features(X_h, valid_feature_cols, valid_pairs)
    ai_names = sorted(k for k in asym if k.endswith("_AI"))
    tot_names = sorted(k for k in asym if k.endswith("_total"))

    X_ai = np.column_stack([asym[k] for k in ai_names]) if ai_names else X_h[:, :0]
    X_tot = np.column_stack([asym[k] for k in tot_names]) if tot_names else X_h[:, :0]

    # Original L/R: just the raw harmonized columns for valid pairs
    lr_cols = []
    for _, lcol, rcol in valid_pairs:
        for col in [lcol, rcol]:
            if col in valid_feature_cols:
                lr_cols.append(valid_feature_cols.index(col))
    X_lr = X_h[:, sorted(set(lr_cols))] if lr_cols else X_h

    return {
        "Asymmetry only (AI)": X_ai,
        "Total volume only": X_tot,
        "AI + Total": np.column_stack([X_ai, X_tot]) if X_ai.shape[1] and X_tot.shape[1] else X_ai,
        "Original L/R": X_lr,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_univariate_results(results_df, title, save_path, top_n=20):
    """Forest plot of univariate correlations with CIs."""
    import matplotlib.pyplot as plt

    df = results_df.head(top_n).sort_values("r")
    n = df["n"].values
    ci = 1.96 / np.sqrt(n - 3)  # approximate CI on Fisher-z → r

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.35)))
    y_pos = np.arange(len(df))
    colors = ["#d62728" if r > 0 else "#1f77b4" for r in df["r"]]
    ax.barh(y_pos, df["r"].values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.errorbar(df["r"].values, y_pos, xerr=ci, fmt="none", ecolor="black",
                capsize=3, linewidth=1)

    # significance markers
    for i, row in enumerate(df.itertuples()):
        p = row.p_raw
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if star:
            x = row.r + 0.005 * np.sign(row.r)
            ax.text(x, i, star, va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].values, fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Pearson r", fontweight="bold", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ai_vs_total_comparison(vol_asym_df, save_path, target_name="target"):
    """Paired bar chart: AI vs Total correlations per structure."""
    import matplotlib.pyplot as plt

    df = vol_asym_df.copy()
    names = df["structure"].values
    x_pos = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - width / 2, df["r_AI"].values, width, label="Asymmetry Index",
           color="#d62728", alpha=0.7)
    ax.bar(x_pos + width / 2, df["r_total"].values, width, label="Total Volume",
           color="#1f77b4", alpha=0.7)

    for i, row in df.iterrows():
        p = row["p_AI"]
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if star:
            y_off = row["r_AI"] + 0.005 * np.sign(row["r_AI"])
            ax.text(i - width / 2, y_off, star, ha="center", va="bottom" if row["r_AI"] > 0 else "top",
                    fontsize=9, fontweight="bold", color="#d62728")

    # Display names
    name_map = {
        "caudate": "Caudate", "putamen": "Putamen", "pallidum": "Pallidum",
        "vedc": "VEDC/VTA", "aa": "Accumbens", "accumbens": "Accumbens",
        "tp": "Thalamus", "scs_MD": "SCS MD",
    }
    display = [name_map.get(n, n) for n in names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(f"Pearson r with {target_name}", fontweight="bold")
    ax.set_title("Asymmetry vs Total Volume (ComBat-harmonised)", fontweight="bold", fontsize=13)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sex_interaction(asymmetry_data, y, sex_labels, structure_name, save_path):
    """Scatter plot coloured by sex with separate regression lines."""
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    sex = np.asarray(sex_labels)
    is_female = (sex == "female") | (sex == 1)
    ai = asymmetry_data[structure_name]

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, mask, color in [("Male", ~is_female, "#1f77b4"), ("Female", is_female, "#d62728")]:
        ax.scatter(ai[mask], y[mask], alpha=0.2, s=15, color=color, label=label, edgecolors="none")
        sl, ic, r, p, _ = linregress(ai[mask], y[mask])
        xs = np.linspace(ai[mask].min(), ai[mask].max(), 50)
        ax.plot(xs, sl * xs + ic, color=color, lw=2,
                label=f"{label}: r={r:.3f} (p={p:.4f})")

    ax.set_xlabel(f"{structure_name} (L-R)/(L+R)", fontweight="bold")
    ax.set_ylabel("PQ-BC Severity", fontweight="bold")
    ax.set_title(f"Sex Interaction: {structure_name}", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
