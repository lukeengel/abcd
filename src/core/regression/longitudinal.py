"""Longitudinal analyses: cross-timepoint prediction, change scores, stability."""

import logging

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn
from scipy.stats import pearsonr, ttest_ind
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..preprocessing.ingest import load_and_merge
from ..preprocessing.qc import quality_control
from ..preprocessing.transforms import recode
from ..tsne.embeddings import get_roi_columns_from_config
from .models import MODEL_REGISTRY
from .pipeline import fit_residualize, apply_residualize
from .univariate import compute_asymmetry_features, extract_bilateral_pairs

logger = logging.getLogger(__name__)


def _get_feature_transform(env):
    """Read feature_transform from regression config. Returns 'raw' or 'asymmetry'."""
    return env.configs.regression.get("feature_transform", "asymmetry")


def _prepare_features(X_raw, present_cols, bilateral_pairs, feature_transform):
    """Prepare features based on transform setting.

    Returns (X, feature_names) where X is the feature matrix and feature_names
    are the corresponding column names.
    """
    if feature_transform == "asymmetry":
        asym = compute_asymmetry_features(X_raw, present_cols, bilateral_pairs)
        ai_names = sorted(k for k in asym if k.endswith("_AI"))
        X = np.column_stack([asym[k] for k in ai_names])
        return X, ai_names
    else:
        # Raw L/R features
        return X_raw, list(present_cols)


def _filter_to_severity_range(df, y, X_raw, target_col, env, y_baseline=None):
    """Filter to subjects within custom bin range, matching regression pipeline.

    Args:
        df: DataFrame to filter.
        y: Target values (follow-up or current timepoint).
        X_raw: Feature matrix to filter (aligned with df/y).
        target_col: Target column name (used to look up bin config).
        env: Environment with configs.
        y_baseline: If provided, apply the range filter on baseline severity instead
            of y. Important for cross-timepoint prediction: we want subjects who were
            high-severity at baseline (the imaging timepoint), not at follow-up.

    Returns (df, y, X_raw) filtered, or originals if no bins configured.
    """
    reg_config = env.configs.regression
    bins_config = reg_config.get("sample_weighting", {}).get("custom_bins", {})

    # Find the target name that matches this target_col
    target_name = None
    for t in reg_config.get("targets", []):
        if t["column"] == target_col:
            target_name = t["name"]
            break

    if target_name is None or target_name not in bins_config:
        return df, y, X_raw

    bin_edges = bins_config[target_name]
    lo, hi = bin_edges[0], bin_edges[-1]
    # Filter on baseline severity when available (cross-timepoint prediction),
    # otherwise fall back to current y (stability analysis, etc.)
    y_filter = y_baseline if y_baseline is not None else y
    mask = (y_filter >= lo) & (y_filter < hi)

    n_before = len(y)
    df = df[mask].reset_index(drop=True)
    y = y[mask]
    X_raw = X_raw[mask]

    n_after = len(y)
    severity_src = "baseline" if y_baseline is not None else "followup"
    print(f"  Severity filter [{lo}, {hi}) on {severity_src}: {n_before} -> {n_after} subjects")
    return df, y, X_raw


def _combat_harmonize(X_raw, df, harm_config, min_site_n=5):
    """Run ComBat harmonization with proper covariate handling.

    Filters for complete covariates and removes small sites before harmonization.
    Gracefully drops covariates that are entirely missing (e.g., demographics
    not collected at follow-up timepoints).

    Returns:
        X_harm: harmonized feature matrix
        keep_mask: boolean mask of rows that were kept (relative to input)
        combat_model: fitted ComBat model (for apply if needed)
    """
    site_col = harm_config.get("site_column", "mri_info_manufacturer")
    cov_cols = harm_config.get("covariates", [])
    eb = harm_config.get("empirical_bayes", True)

    # Start with all rows valid
    keep = np.ones(len(df), dtype=bool)

    # Remove rows with any NaN in the feature matrix — neuroHarmonize requires
    # complete feature data; NaN rows cause the entire harmonization to fail.
    finite_rows = np.all(np.isfinite(X_raw), axis=1)
    if not np.all(finite_rows):
        n_nan = (~finite_rows).sum()
        logger.info(f"ComBat: dropping {n_nan} rows with NaN/Inf features before harmonization")
        keep &= finite_rows

    # Only use covariates that actually have data at this timepoint
    available_cov_cols = []
    for col in cov_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            available_cov_cols.append(col)
            keep &= df[col].notna().values

    # Site column is required
    if site_col in df.columns:
        keep &= df[site_col].notna().values
    else:
        return X_raw, np.ones(len(df), dtype=bool), None

    if keep.sum() < 30:
        return X_raw[keep], keep, None

    df_f = df[keep].reset_index(drop=True)
    X_f = X_raw[keep]

    # Remove small sites
    site_counts = df_f[site_col].value_counts()
    small_sites = site_counts[site_counts < min_site_n].index.tolist()
    if small_sites:
        site_keep = ~df_f[site_col].isin(small_sites)
        df_f = df_f[site_keep].reset_index(drop=True)
        X_f = X_f[site_keep.values]
        # Update the original keep mask
        keep_indices = np.where(keep)[0]
        keep[keep_indices[~site_keep.values]] = False

    if len(df_f) < 30:
        return X_f, keep, None

    covars = df_f[[site_col] + available_cov_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})
    # Encode string covariates as float64 (neuroHarmonize requires numeric input;
    # int8 from pd.Categorical can cause OLS failures in some neuroHarmonize versions)
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

    if available_cov_cols != cov_cols:
        dropped = set(cov_cols) - set(available_cov_cols)
        print(f"  ComBat: dropped unavailable covariates {dropped} (site-only harmonization)")

    try:
        combat_model, X_harm = harmonizationLearn(X_f, covars, eb=eb)
    except Exception as e:
        logger.warning(f"ComBat failed: {e}")
        print(f"  ComBat failed: {e}")
        return X_f, keep, None

    # Check for NaN in harmonized data (safety)
    if np.any(np.isnan(X_harm)):
        nan_count = np.isnan(X_harm).sum()
        logger.warning(f"ComBat returned {nan_count} NaN values, using raw data")
        print(f"  Warning: ComBat returned NaN values, using raw data")
        return X_f, keep, None

    return X_harm, keep, combat_model


def load_longitudinal_data(env, target_col):
    """Load all timepoints, apply QC, require baseline + >= 1 follow-up.

    Returns:
        long_df: long-format dataframe (one row per subject-timepoint)
        wide_df: wide-format (one row per subject, baseline/followup columns)
        subject_ids: array of subject IDs with longitudinal data
    """
    config = env.configs.data
    id_col = config["columns"]["mapping"]["id"]
    tp_col = config["columns"]["mapping"]["timepoint"]
    timepoints = config["timepoints"]

    # Load raw merged data (all timepoints)
    raw_df = load_and_merge(env)
    raw_df = recode(env, raw_df)

    # QC
    raw_df, _ = quality_control(env, raw_df)
    raw_df = raw_df[raw_df["qc_pass"]].copy()

    # Keep only subjects with non-null target and valid timepoints
    valid_tp = list(timepoints.values())
    raw_df = raw_df[raw_df[tp_col].isin(valid_tp)].copy()
    raw_df = raw_df[raw_df[target_col].notna()].copy()

    # Label timepoints by index (0 = baseline, 1 = year2, etc.)
    tp_order = {v: i for i, v in enumerate(timepoints.values())}
    raw_df["tp_idx"] = raw_df[tp_col].map(tp_order)

    # Require baseline
    baseline_tp = timepoints["baseline"]
    has_baseline = raw_df[raw_df[tp_col] == baseline_tp][id_col].unique()

    # Require at least one follow-up
    followup_tps = [v for k, v in timepoints.items() if k != "baseline"]
    has_followup = raw_df[raw_df[tp_col].isin(followup_tps)][id_col].unique()

    longitudinal_ids = np.intersect1d(has_baseline, has_followup)
    long_df = raw_df[raw_df[id_col].isin(longitudinal_ids)].copy()
    long_df = long_df.sort_values([id_col, "tp_idx"]).reset_index(drop=True)

    # Carry forward baseline demographics to follow-up timepoints.
    # ABCD only collects demographics (age, sex) at baseline — follow-up rows
    # have NaN for these columns. Forward-fill within each subject so ComBat
    # can use them as covariates at every timepoint.
    harm_cov_cols = env.configs.harmonize.get("covariates", [])
    for col in harm_cov_cols:
        if col in long_df.columns:
            n_missing_before = long_df[col].isna().sum()
            long_df[col] = long_df.groupby(id_col)[col].ffill()
            n_filled = n_missing_before - long_df[col].isna().sum()
            if n_filled > 0:
                logger.info(f"Carried forward {n_filled} baseline {col} values to follow-up")

    # Build wide format — pivot target and a timepoint label
    wide_parts = []
    for tp_name, tp_val in timepoints.items():
        tp_data = long_df[long_df[tp_col] == tp_val][[id_col, target_col]].copy()
        tp_data = tp_data.rename(columns={target_col: f"{target_col}_{tp_name}"})
        wide_parts.append(tp_data)

    wide_df = wide_parts[0]
    for part in wide_parts[1:]:
        wide_df = wide_df.merge(part, on=id_col, how="outer")

    # Also attach baseline imaging features + metadata
    baseline_data = long_df[long_df[tp_col] == baseline_tp].copy()
    baseline_data = baseline_data.drop(columns=[tp_col, "tp_idx", "qc_pass", "qc_reason"], errors="ignore")
    wide_df = wide_df.merge(baseline_data, on=id_col, how="left", suffixes=("", "_bl"))

    return long_df, wide_df, longitudinal_ids


def compute_change_scores(wide_df, target_col, baseline_name="baseline", followup_name="year2",
                          threshold=None):
    """Compute change scores: followup - baseline.

    Args:
        threshold: If None, worsened = delta > 0 (any increase).
                   If numeric, worsened = delta >= threshold (e.g. 0.5 × SD for MCID).

    Returns wide_df with added columns: delta_target, worsened.
    """
    bl_col = f"{target_col}_{baseline_name}"
    fu_col = f"{target_col}_{followup_name}"

    df = wide_df.copy()
    valid = df[bl_col].notna() & df[fu_col].notna()
    df = df[valid].copy()
    df["delta_target"] = df[fu_col] - df[bl_col]
    if threshold is None:
        df["worsened"] = (df["delta_target"] > 0).astype(int)
    else:
        df["worsened"] = (df["delta_target"] >= threshold).astype(int)
    return df


def cross_timepoint_prediction(
    env, wide_df, target_col, bilateral_pairs, feature_cols,
    followup_name="year2", n_splits=5, min_severity=None,
    combat_mode: str = "full",
):
    """Baseline features -> follow-up target prediction with CV.

    Uses full-sample ComBat harmonization (standard for small longitudinal samples —
    per-fold ComBat is unreliable with <200 subjects across 8 scanner models).
    Set combat_mode="full" (default) for standard full-sample ComBat.

    Severity filtering uses BASELINE severity (y_baseline), not follow-up severity.
    This matches the cross-sectional pipeline: subjects are selected based on
    their imaging-timepoint (baseline) severity, not their future state.

    Args:
        env: Environment with configs.
        wide_df: Wide-format longitudinal dataframe from load_longitudinal_data().
        target_col: Target column (e.g. "pps_y_ss_severity_score").
        bilateral_pairs: Bilateral ROI pairs for asymmetry computation.
        feature_cols: Raw feature column names.
        followup_name: Timepoint name for follow-up target (default "year2").
        n_splits: Number of CV folds (reduced automatically for small n).
        min_severity: Manual baseline severity threshold. If None, uses regression.yaml bins.
        combat_mode: "full" (full-sample ComBat before CV) or reserved for future modes.

    Returns dict with aggregated r, p, all_true, all_pred, n.
    """
    feature_transform = _get_feature_transform(env)
    harm_config = env.configs.harmonize
    bl_col = f"{target_col}_baseline"
    fu_col = f"{target_col}_{followup_name}"

    # Require both baseline and follow-up target
    valid = wide_df[fu_col].notna()
    if bl_col in wide_df.columns:
        valid = valid & wide_df[bl_col].notna()
    df = wide_df[valid].copy()
    y_followup = df[fu_col].values.astype(float)

    # Get available feature columns from baseline data
    present_cols = [c for c in feature_cols if c in df.columns]
    X_raw = df[present_cols].values.astype(float)
    valid_rows = np.all(np.isfinite(X_raw), axis=1) & np.isfinite(y_followup)
    X_raw = X_raw[valid_rows]
    y_followup = y_followup[valid_rows]
    df = df[valid_rows].reset_index(drop=True)

    # Filter to high-severity subjects (matching regression pipeline)
    if min_severity is not None:
        y_baseline = df[bl_col].values.astype(float) if bl_col in df.columns else y_followup
        mask = y_baseline >= min_severity
        df = df[mask].reset_index(drop=True)
        X_raw = X_raw[mask]
        y_followup = y_followup[mask]
        print(f"  Severity filter (baseline >= {min_severity}): {mask.sum()}/{len(mask)} subjects")
    else:
        # Use regression config bins — filter on BASELINE severity so subject
        # selection matches the cross-sectional imaging pipeline.
        y_bl_filter = df[bl_col].values.astype(float) if bl_col in df.columns else None
        df, y_followup, X_raw = _filter_to_severity_range(
            df, y_followup, X_raw, target_col, env, y_baseline=y_bl_filter
        )

    if len(y_followup) < 30:
        print(f"  Insufficient subjects after filtering: n={len(y_followup)}")
        return {"r": np.nan, "p": np.nan, "all_true": [], "all_pred": [], "n": len(y_followup)}

    # Full-sample ComBat harmonization (standard for small longitudinal samples)
    # Per-fold ComBat is unreliable with <200 subjects across 8 scanner models
    X_harm, keep_mask, _ = _combat_harmonize(X_raw, df, harm_config, min_site_n=3)
    df = df[keep_mask].reset_index(drop=True)
    y_followup = y_followup[keep_mask]

    if len(y_followup) < 30:
        print(f"  Insufficient subjects after ComBat filtering: n={len(y_followup)}")
        return {"r": np.nan, "p": np.nan, "all_true": [], "all_pred": [], "n": len(y_followup)}

    # Prepare features (raw or asymmetry) from harmonized data
    X_feat, feat_names = _prepare_features(X_harm, present_cols, bilateral_pairs, feature_transform)

    print(f"  Cross-timepoint prediction: n={len(y_followup)}, features={X_feat.shape[1]}, "
          f"transform={feature_transform}")

    # Residualization config
    reg_config = env.configs.regression
    cov_cfg = reg_config.get("covariates", {})
    covariate_cols = None
    if cov_cfg.get("residualize", False):
        target_name = None
        for t in reg_config.get("targets", []):
            if t["column"] == target_col:
                target_name = t["name"]
                break
        is_raw = target_name and target_name.endswith("_raw")
        raw_only = cov_cfg.get("apply_to_raw_scores_only", True)
        if not raw_only or is_raw:
            covariate_cols = cov_cfg.get("columns", [])

    # CV setup — per-fold SVR on already-harmonized features
    seed = env.configs.run.get("seed", 42)
    n_bins = min(5, max(2, len(y_followup) // 20))
    y_binned = pd.qcut(y_followup, q=n_bins, labels=False, duplicates="drop")

    # Use fewer folds for small samples
    actual_splits = min(n_splits, max(2, len(y_followup) // 20))

    # Family-aware CV
    family_groups = None
    id_col = env.configs.data["columns"]["mapping"]["id"]
    if id_col in df.columns and "rel_family_id" in df.columns:
        family_groups = pd.to_numeric(df["rel_family_id"], errors="coerce").values
        missing = np.isnan(family_groups)
        if missing.any():
            max_id = np.nanmax(family_groups) if (~missing).any() else 0
            family_groups[missing] = np.arange(max_id + 1, max_id + 1 + missing.sum())
        cv = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)

    all_true, all_pred = [], []

    split_args = (X_feat, y_binned, family_groups) if family_groups is not None else (X_feat, y_binned)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(*split_args)):
        X_tr = X_feat[train_idx]
        X_te = X_feat[test_idx]
        y_tr = y_followup[train_idx].copy()
        y_te_true = y_followup[test_idx].copy()

        # Per-fold target residualization
        if covariate_cols:
            resid_model = fit_residualize(y_tr, df.iloc[train_idx], covariate_cols)
            y_tr = apply_residualize(y_tr, df.iloc[train_idx], covariate_cols, resid_model)
            y_te_true_resid = apply_residualize(y_te_true, df.iloc[test_idx], covariate_cols, resid_model)
        else:
            y_te_true_resid = y_te_true

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Scale target
        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

        # Fit and predict — use config-driven SVR so kernel/C/epsilon match regression pipeline
        svr = MODEL_REGISTRY["svr"](env.configs.regression, seed)
        svr.fit(X_tr_s, y_tr_s)
        pred = y_scaler.inverse_transform(svr.predict(X_te_s).reshape(-1, 1)).ravel()

        all_true.extend(y_te_true_resid)
        all_pred.extend(pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    r, p = pearsonr(all_true, all_pred)
    return {"r": r, "p": p, "all_true": all_true, "all_pred": all_pred, "n": len(y_followup)}


def run_cross_sectional_svr(
    df: pd.DataFrame,
    y: np.ndarray,
    env,
    feature_cols: list,
    bilateral_pairs: list,
    label: str = "",
    n_splits: int | None = None,
) -> dict:
    """Cross-sectional nested CV SVR with per-fold ComBat.

    Mirrors the NB07 regression pipeline for use in longitudinal comparisons
    (e.g., applying the SVR to a time-point subset instead of the full sample).

    Uses config-driven n_splits, seed, model (SVR from MODEL_REGISTRY).

    Args:
        df: Subject-level dataframe with feature columns and metadata.
        y: Target array (already aligned with df).
        env: Environment namespace with configs.
        feature_cols: List of raw imaging feature column names.
        bilateral_pairs: Bilateral ROI pairs for asymmetry computation.
        label: Human-readable label for progress messages.
        n_splits: Number of CV folds. Falls back to regression.yaml cv.n_outer_splits.

    Returns:
        dict with keys: r, p, all_true, all_pred, n, fold_data.
            fold_data is a list of dicts (X_train, X_test, y_train, y_test) that
            can be passed to run_svr_on_saved_folds() for fast permutation tests.
    """
    reg_config = env.configs.regression
    harm_config = env.configs.harmonize
    feature_transform = _get_feature_transform(env)
    seed = env.configs.run.get("seed", 42)

    if n_splits is None:
        n_splits = reg_config.get("cv", {}).get("n_outer_splits", 5)

    # Filter to rows with valid features
    present_cols = [c for c in feature_cols if c in df.columns]
    X_raw = df[present_cols].values.astype(float)
    valid = np.all(np.isfinite(X_raw), axis=1) & np.isfinite(y)
    df = df[valid].reset_index(drop=True)
    y = y[valid]
    X_raw = X_raw[valid]

    if len(y) < 20:
        print(f"  [{label}] Too few subjects after filtering: n={len(y)}")
        return {"r": np.nan, "p": np.nan, "all_true": np.array([]),
                "all_pred": np.array([]), "n": len(y), "fold_data": []}

    # Residualization config
    cov_cfg = reg_config.get("covariates", {})
    covariate_cols = None
    if cov_cfg.get("residualize", False):
        covariate_cols = cov_cfg.get("columns", [])

    # Stratified CV setup
    y_binned = pd.qcut(y, q=min(5, max(2, len(y) // 20)), labels=False, duplicates="drop")
    site_col = harm_config.get("site_column", "mri_info_manufacturer")

    family_groups = None
    if "rel_family_id" in df.columns:
        family_groups = pd.to_numeric(df["rel_family_id"], errors="coerce").values
        missing = np.isnan(family_groups)
        if missing.any():
            max_id = np.nanmax(family_groups) if (~missing).any() else 0
            family_groups[missing] = np.arange(max_id + 1, max_id + 1 + missing.sum())

    if family_groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (df, y_binned, family_groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (df, y_binned)

    all_true, all_pred, fold_data = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(*split_args)):
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_te = df.iloc[test_idx].reset_index(drop=True)
        X_tr_raw = X_raw[train_idx]
        X_te_raw = X_raw[test_idx]
        y_tr = y[train_idx].copy()
        y_te = y[test_idx].copy()

        # Per-fold target residualization
        if covariate_cols:
            resid_model = fit_residualize(y_tr, df_tr, covariate_cols)
            y_tr = apply_residualize(y_tr, df_tr, covariate_cols, resid_model)
            y_te = apply_residualize(y_te, df_te, covariate_cols, resid_model)

        # Per-fold ComBat: fit on train only, apply to test (no leakage).
        # Mirrors main pipeline (harmonizationLearn on train + harmonizationApply on test).
        try:
            from neuroHarmonize import harmonizationLearn, harmonizationApply

            site_col = harm_config.get("site_column", "mri_info_manufacturer")
            cov_cols_harm = [c for c in harm_config.get("covariates", [])
                             if c in df_tr.columns and df_tr[c].notna().sum() > 0]

            # Build covars DataFrames (SITE + numeric covariates)
            def _make_covars(df_sub, cols, keep_cols=None):
                cov = df_sub[[site_col] + cols].copy().rename(columns={site_col: "SITE"})
                for c in cols:
                    if not pd.api.types.is_numeric_dtype(cov[c]):
                        cov[c] = pd.Categorical(cov[c]).codes.astype(float)
                    else:
                        cov[c] = cov[c].astype(float)
                if keep_cols is None:
                    # Train: drop constant covariates, record survivors
                    for c in list(cols):
                        if cov[c].nunique() <= 1:
                            cov = cov.drop(columns=c)
                else:
                    # Test: use exactly the same columns as train to avoid shape mismatch
                    cov = cov[[c for c in keep_cols if c in cov.columns]]
                return cov

            tr_covars = _make_covars(df_tr, cov_cols_harm)
            te_covars = _make_covars(df_te, cov_cols_harm, keep_cols=list(tr_covars.columns))

            # Remove zero-variance features (matching main pipeline)
            feat_vars = np.var(X_tr_raw, axis=0)
            valid_feat = feat_vars > 1e-10
            X_tr_v = X_tr_raw[:, valid_feat]
            X_te_v = X_te_raw[:, valid_feat]
            valid_cols_fold = [present_cols[i] for i, v in enumerate(valid_feat) if v]

            # Drop train rows with NaN site/covariates
            tr_nan = tr_covars.isna().any(axis=1).values
            te_nan = te_covars.isna().any(axis=1).values

            eb = harm_config.get("empirical_bayes", True)
            combat_model, X_tr_harm = harmonizationLearn(
                X_tr_v[~tr_nan], tr_covars[~tr_nan], eb=eb,
            )
            X_te_harm = harmonizationApply(X_te_v[~te_nan], te_covars[~te_nan], combat_model)

            X_tr_h = X_tr_harm
            X_te_h = X_te_harm
            y_tr_h = y_tr[~tr_nan]
            y_te_h = y_te[~te_nan]
            # Update bilateral pairs to match surviving columns
            bilateral_pairs_fold = [(n, l, r) for n, l, r in bilateral_pairs
                                    if l in valid_cols_fold and r in valid_cols_fold]
        except Exception as exc:
            logger.warning(f"ComBat fold {fold_idx} failed ({exc}), using raw features")
            X_tr_h, X_te_h = X_tr_raw, X_te_raw
            y_tr_h, y_te_h = y_tr, y_te
            valid_cols_fold = present_cols
            bilateral_pairs_fold = bilateral_pairs

        # Prepare features (use per-fold surviving columns after variance filter)
        X_tr_feat, _ = _prepare_features(X_tr_h, valid_cols_fold, bilateral_pairs_fold, feature_transform)
        X_te_feat, _ = _prepare_features(X_te_h, valid_cols_fold, bilateral_pairs_fold, feature_transform)

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_feat)
        X_te_s = scaler.transform(X_te_feat)

        # Scale target
        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(y_tr_h.reshape(-1, 1)).ravel()

        svr = MODEL_REGISTRY["svr"](reg_config, seed + fold_idx)
        svr.fit(X_tr_s, y_tr_s)
        pred = y_scaler.inverse_transform(svr.predict(X_te_s).reshape(-1, 1)).ravel()

        all_true.extend(y_te_h)
        all_pred.extend(pred)
        fold_data.append({
            "X_train": X_tr_s, "X_test": X_te_s,
            "y_train": y_tr_h, "y_test": y_te_h,
        })

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    r, p = pearsonr(all_true, all_pred)

    if label:
        print(f"  [{label}] r={r:.3f}, p={p:.4f}, n={len(y)}")

    return {
        "r": r, "p": p,
        "all_true": all_true, "all_pred": all_pred,
        "n": len(y), "fold_data": fold_data,
    }


def attrition_analysis(long_df, wide_df, env, target_col) -> dict:
    """Compare subjects lost to follow-up vs retained on key demographics.

    Returns dict with t-test and chi2 results for age, sex, baseline severity.
    """
    from scipy.stats import chi2_contingency

    config = env.configs.data
    id_col = config["columns"]["mapping"]["id"]
    bl_col = f"{target_col}_baseline"

    # Subjects with both baseline and Y2 data
    has_y2 = wide_df[id_col][wide_df.get(f"{target_col}_year2", pd.Series(dtype=float)).notna()].values \
              if f"{target_col}_year2" in wide_df.columns else np.array([])

    # Baseline rows only
    tp_col = config["columns"]["mapping"]["timepoint"]
    tps = config["timepoints"]
    bl_df = long_df[long_df[tp_col] == tps["baseline"]].copy()

    retained = bl_df[id_col].isin(has_y2)
    results = {}

    # Age comparison
    if "interview_age" in bl_df.columns:
        age_ret = bl_df.loc[retained, "interview_age"].dropna().values
        age_lost = bl_df.loc[~retained, "interview_age"].dropna().values
        if len(age_ret) > 5 and len(age_lost) > 5:
            t, p = ttest_ind(age_ret, age_lost)
            results["age"] = {"t": t, "p": p,
                              "mean_retained": age_ret.mean(),
                              "mean_lost": age_lost.mean()}

    # Sex comparison
    if "sex_mapped" in bl_df.columns:
        ct = pd.crosstab(retained, bl_df["sex_mapped"])
        if ct.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(ct)
            results["sex"] = {"chi2": chi2, "p": p}

    # Baseline severity
    if bl_col in wide_df.columns:
        wide_with_ret = wide_df.copy()
        wide_with_ret["retained"] = wide_with_ret[id_col].isin(has_y2)
        sev_ret = wide_with_ret.loc[wide_with_ret["retained"], bl_col].dropna().values
        sev_lost = wide_with_ret.loc[~wide_with_ret["retained"], bl_col].dropna().values
        if len(sev_ret) > 5 and len(sev_lost) > 5:
            t, p = ttest_ind(sev_ret, sev_lost)
            results["severity"] = {"t": t, "p": p,
                                   "mean_retained": sev_ret.mean(),
                                   "mean_lost": sev_lost.mean()}

    results["n_retained"] = int(retained.sum())
    results["n_lost"] = int((~retained).sum())
    return results


def paired_asymmetry_change(long_df, env, bilateral_pairs, feature_cols, target_col) -> pd.DataFrame:
    """Test whether AI changes from baseline to year 2 predict severity change.

    For each bilateral structure, computes ΔAI = AI_year2 - AI_baseline and
    correlates with Δseverity.

    Returns DataFrame: structure, r_delta_ai_delta_sev, p, n.
    """
    config = env.configs.data
    id_col = config["columns"]["mapping"]["id"]
    tp_col = config["columns"]["mapping"]["timepoint"]
    tps = config["timepoints"]
    harm_config = env.configs.harmonize
    feature_transform = _get_feature_transform(env)

    bl_tp = tps.get("baseline")
    y2_tp = tps.get("year2")

    if y2_tp is None:
        return pd.DataFrame()

    bl_df = long_df[long_df[tp_col] == bl_tp].copy()
    y2_df = long_df[long_df[tp_col] == y2_tp].copy()

    # Merge baseline and year2 on subject ID
    common_ids = np.intersect1d(bl_df[id_col].values, y2_df[id_col].values)
    bl_m = bl_df[bl_df[id_col].isin(common_ids)].set_index(id_col)
    y2_m = y2_df[y2_df[id_col].isin(common_ids)].set_index(id_col)
    bl_m = bl_m.loc[common_ids]
    y2_m = y2_m.loc[common_ids]

    present_cols = [c for c in feature_cols if c in bl_m.columns and c in y2_m.columns]
    X_bl = bl_m[present_cols].values.astype(float)
    X_y2 = y2_m[present_cols].values.astype(float)
    delta_sev = (y2_m[target_col].values.astype(float)
                 - bl_m[target_col].values.astype(float))

    # ComBat each timepoint separately
    X_bl_h, keep_bl, _ = _combat_harmonize(X_bl, bl_m.reset_index(), harm_config)
    X_y2_h, keep_y2, _ = _combat_harmonize(X_y2, y2_m.reset_index(), harm_config)

    keep = keep_bl & keep_y2
    # X_bl_h rows are already keep_bl-filtered; apply keep_y2 mask within those rows.
    # X_y2_h rows are already keep_y2-filtered; apply keep_bl mask within those rows.
    X_bl_h = X_bl_h[keep_y2[keep_bl]]
    X_y2_h = X_y2_h[keep_bl[keep_y2]]
    delta_sev_f = delta_sev[keep]

    if len(delta_sev_f) < 20:
        return pd.DataFrame()

    asym_bl = compute_asymmetry_features(X_bl_h, present_cols, bilateral_pairs)
    asym_y2 = compute_asymmetry_features(X_y2_h, present_cols, bilateral_pairs)

    rows = []
    for name, _, _ in bilateral_pairs:
        key = f"{name}_AI"
        if key not in asym_bl or key not in asym_y2:
            continue
        delta_ai = asym_y2[key] - asym_bl[key]
        valid = np.isfinite(delta_ai) & np.isfinite(delta_sev_f)
        if valid.sum() < 10:
            continue
        r, p = pearsonr(delta_ai[valid], delta_sev_f[valid])
        rows.append({"structure": name, "r_delta": r, "p": p, "n": int(valid.sum())})

    return pd.DataFrame(rows)


def reliability_analysis(long_df, env, bilateral_pairs, feature_cols,
                         target_col, icc_model="3,1") -> pd.DataFrame:
    """Compute ICC(3,1) test-retest reliability of AI features across timepoints.

    Args:
        long_df: Long-format dataframe with all timepoints.
        env: Environment with configs.
        bilateral_pairs: Bilateral ROI pairs.
        feature_cols: Raw imaging feature column names.
        target_col: Target column (unused here; kept for consistent API).
        icc_model: ICC model string (default "3,1" — two-way mixed, consistency).

    Returns DataFrame: structure, ICC, lower95, upper95, p.
    """
    try:
        import pingouin as pg
    except ImportError:
        logger.warning("pingouin not installed; skipping ICC. Install with: pip install pingouin")
        return pd.DataFrame(columns=["structure", "ICC", "lower95", "upper95", "p"])

    config = env.configs.data
    id_col = config["columns"]["mapping"]["id"]
    tp_col = config["columns"]["mapping"]["timepoint"]
    tps = config["timepoints"]
    harm_config = env.configs.harmonize

    rows = []
    for tp_name, tp_val in tps.items():
        tp_df = long_df[long_df[tp_col] == tp_val].copy()
        present_cols = [c for c in feature_cols if c in tp_df.columns]
        X_raw = tp_df[present_cols].values.astype(float)
        valid = np.all(np.isfinite(X_raw), axis=1)
        tp_df = tp_df[valid].reset_index(drop=True)
        X_raw = X_raw[valid]

        if len(tp_df) < 30:
            continue

        X_harm, keep, _ = _combat_harmonize(X_raw, tp_df, harm_config)
        tp_df = tp_df[keep].reset_index(drop=True)
        asym = compute_asymmetry_features(X_harm, present_cols, bilateral_pairs)

        for name, _, _ in bilateral_pairs:
            key = f"{name}_AI"
            if key not in asym:
                continue
            rows.append({id_col: tp_df[id_col].values,
                         "timepoint": tp_name,
                         "ai": asym[key],
                         "structure": name})

    if not rows:
        return pd.DataFrame()

    # Build long format for ICC
    long_icc = []
    for entry in rows:
        for subj, ai_val in zip(entry[id_col], entry["ai"]):
            long_icc.append({"subject": subj, "timepoint": entry["timepoint"],
                              "ai": ai_val, "structure": entry["structure"]})
    long_icc_df = pd.DataFrame(long_icc)

    icc_rows = []
    for structure, grp in long_icc_df.groupby("structure"):
        grp_wide = grp.pivot_table(index="subject", columns="timepoint", values="ai")
        if grp_wide.shape[1] < 2 or grp_wide.dropna().shape[0] < 10:
            continue
        grp_long = grp_wide.dropna().reset_index().melt(id_vars="subject",
                                                          value_name="ai",
                                                          var_name="timepoint")
        try:
            icc_res = pg.intraclass_corr(data=grp_long, targets="subject",
                                          raters="timepoint", ratings="ai")
            row_icc = icc_res[icc_res["Type"] == f"ICC{icc_model}"].iloc[0]
            icc_rows.append({
                "structure": structure,
                "ICC": row_icc["ICC"],
                "lower95": row_icc["CI95%"][0],
                "upper95": row_icc["CI95%"][1],
                "p": row_icc["pval"],
                "n": grp_wide.dropna().shape[0],
            })
        except Exception as e:
            logger.warning(f"ICC failed for {structure}: {e}")

    return pd.DataFrame(icc_rows)


def stability_analysis(long_df, env, bilateral_pairs, feature_cols, target_col,
                        min_severity=None, feature_transform=None):
    """Per-timepoint: ComBat -> features -> correlate with target.

    Filters to subjects above a severity threshold AT EACH TIMEPOINT independently.
    This is intentional: we want high-severity subjects at that specific scan,
    not just those who were high-severity at baseline. The sample therefore differs
    by timepoint (subjects who remitted may drop out of later timepoints).
    Compare with cross_timepoint_prediction() which filters on baseline severity only.

    Reports ALL per-feature correlations plus the best feature row.

    Returns DataFrame: timepoint, feature, r, p, n.
    """
    if feature_transform is None:
        feature_transform = _get_feature_transform(env)
    config = env.configs.data
    tp_col = config["columns"]["mapping"]["timepoint"]
    timepoints = config["timepoints"]
    harm_config = env.configs.harmonize

    # Get severity filter from regression config if not specified
    if min_severity is None:
        reg_config = env.configs.regression
        bins_config = reg_config.get("sample_weighting", {}).get("custom_bins", {})
        target_name = None
        for t in reg_config.get("targets", []):
            if t["column"] == target_col:
                target_name = t["name"]
                break
        if target_name and target_name in bins_config:
            min_severity = bins_config[target_name][0]

    rows = []
    for tp_name, tp_val in timepoints.items():
        tp_df = long_df[long_df[tp_col] == tp_val].copy()
        present_cols = [c for c in feature_cols if c in tp_df.columns]
        y = tp_df[target_col].values.astype(float)

        valid = np.all(tp_df[present_cols].notna().values, axis=1) & np.isfinite(y)
        tp_df = tp_df[valid].reset_index(drop=True)
        y = y[valid]

        # Apply severity filter
        if min_severity is not None:
            mask = y >= min_severity
            tp_df = tp_df[mask].reset_index(drop=True)
            y = y[mask]

        if len(y) < 30:
            rows.append({"timepoint": tp_name, "feature": "best", "r": np.nan,
                         "p": np.nan, "n": len(y)})
            continue

        # ComBat harmonization with complete covariate filtering
        X_raw = tp_df[present_cols].values.astype(float)
        X_harm, keep_mask, _ = _combat_harmonize(X_raw, tp_df, harm_config, min_site_n=5)
        y = y[keep_mask]

        if len(y) < 30:
            rows.append({"timepoint": tp_name, "feature": "best", "r": np.nan,
                         "p": np.nan, "n": len(y)})
            continue

        # Prepare features
        X_feat, feat_names = _prepare_features(X_harm, present_cols, bilateral_pairs, feature_transform)

        # Correlate each feature with target, report all + best
        best_r, best_p, best_name = 0, 1, ""
        for i, name in enumerate(feat_names):
            r, p = pearsonr(X_feat[:, i], y)
            rows.append({"timepoint": tp_name, "feature": name, "r": r,
                         "p": p, "n": len(y)})
            if abs(r) > abs(best_r):
                best_r, best_p, best_name = r, p, name

        rows.append({"timepoint": tp_name, "feature": f"best ({best_name})",
                     "r": best_r, "p": best_p, "n": len(y)})

    return pd.DataFrame(rows)


def worsening_group_analysis(wide_df, env, bilateral_pairs, feature_cols, target_col,
                             baseline_name="baseline", followup_name="year2",
                             feature_transform=None, threshold=None):
    """Compare baseline features: worseners vs stable, WITH ComBat harmonization.

    Args:
        threshold: Passed to compute_change_scores. None → delta > 0; numeric → delta >= threshold.

    Returns DataFrame with t-test results per feature.
    """
    if feature_transform is None:
        feature_transform = _get_feature_transform(env)
    harm_config = env.configs.harmonize

    df = compute_change_scores(wide_df, target_col, baseline_name, followup_name, threshold=threshold)
    present_cols = [c for c in feature_cols if c in df.columns]
    X_raw = df[present_cols].values.astype(float)

    valid = np.all(np.isfinite(X_raw), axis=1)
    df = df[valid].reset_index(drop=True)
    X_raw = X_raw[valid]

    # ComBat harmonization with complete covariate filtering
    X_harm, keep_mask, _ = _combat_harmonize(X_raw, df, harm_config, min_site_n=5)
    df = df[keep_mask].reset_index(drop=True)
    print(f"  ComBat harmonization applied (n={len(df)})")

    # Prepare features
    X_feat, feat_names = _prepare_features(X_harm, present_cols, bilateral_pairs, feature_transform)
    worsened = df["worsened"].values.astype(bool)

    rows = []
    for i, name in enumerate(feat_names):
        vals = X_feat[:, i]
        t, p = ttest_ind(vals[worsened], vals[~worsened])
        d_pool = np.sqrt(
            ((worsened.sum() - 1) * vals[worsened].std(ddof=1) ** 2
             + ((~worsened).sum() - 1) * vals[~worsened].std(ddof=1) ** 2)
            / (len(vals) - 2)
        )
        d = (vals[worsened].mean() - vals[~worsened].mean()) / d_pool if d_pool > 0 else 0
        rows.append({
            "feature": name,
            "mean_worsened": vals[worsened].mean(),
            "mean_stable": vals[~worsened].mean(),
            "t": t, "p": p, "cohens_d": d,
            "n_worsened": worsened.sum(),
            "n_stable": (~worsened).sum(),
        })
    result = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    _, p_fdr, _, _ = multipletests(result["p"].values, method="fdr_bh")
    result["p_fdr"] = p_fdr
    return result


# ── Small utilities used by longitudinal notebook cells ──────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size with pooled standard deviation."""
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0


def print_univariate_table(asym: dict, y: np.ndarray, label: str) -> dict:
    """Print AI feature correlations with target in a formatted table with FDR correction.

    Returns dict {feature_name: (r, p_raw, p_fdr)}.
    """
    from statsmodels.stats.multitest import multipletests

    ai_names = sorted(k for k in asym if k.endswith("_AI"))
    rs, ps = [], []
    for name in ai_names:
        r_ai, p_ai = pearsonr(asym[name], y)
        rs.append(r_ai)
        ps.append(p_ai)

    _, ps_fdr, _, _ = multipletests(ps, method="fdr_bh")

    print(f"\n  Univariate AI correlations ({label}, n={len(y)}):")
    print(f"  {'Feature':<20} {'r':>8} {'p_raw':>10} {'p_FDR':>10}")
    print(f"  {'-' * 52}")
    results = {}
    for name, r_ai, p_ai, p_fdr in zip(ai_names, rs, ps, ps_fdr):
        sig = "***" if p_fdr < 0.001 else ("**" if p_fdr < 0.01 else ("*" if p_fdr < 0.05 else ""))
        print(f"  {name:<20} {r_ai:>+8.3f} {p_ai:>10.4f} {p_fdr:>10.4f} {sig}")
        results[name] = (r_ai, p_ai, p_fdr)
    return results


def compute_group_ai(
    timepoint_df: pd.DataFrame,
    group_ids,
    env,
    feature_cols: list,
    bilateral_pairs: list,
    ai_key: str = "pallidum_AI",
) -> tuple:
    """Compute mean AI and standard error for a severity group at one timepoint.

    Args:
        timepoint_df: DataFrame for a single timepoint (e.g. year2 rows).
        group_ids: Subject IDs belonging to the group.
        env: Environment with configs (provides id column name).
        feature_cols: Raw feature column names.
        bilateral_pairs: List of (name, left_col, right_col) tuples.
        ai_key: Asymmetry feature key to return (default "pallidum_AI").

    Returns:
        (mean_ai, se_ai, n) — or (None, None, 0) if too few subjects.
    """
    id_col = env.configs.data["columns"]["mapping"]["id"]
    sub = timepoint_df[timepoint_df[id_col].isin(group_ids)].copy()
    present_cols = [c for c in feature_cols if c in sub.columns]
    valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs if l in present_cols and r in present_cols]
    X = sub[present_cols].values.astype(float)
    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    if len(X) < 5:
        return None, None, 0
    asym = compute_asymmetry_features(X, present_cols, valid_pairs)
    if ai_key not in asym:
        return None, None, 0
    ai = asym[ai_key]
    return ai.mean(), ai.std() / np.sqrt(len(ai)), len(ai)


def prep_longitudinal_df(
    df: "pd.DataFrame",
    roi_columns: list,
    target_col: str,
    id_col: str,
    cov_cols: list,
    sex_col: str,
    age_col_bl: str,
    age_col_long: str,
    label: str,
    bl_df: "pd.DataFrame | None" = None,
) -> tuple:
    """Prepare a longitudinal timepoint dataframe for SVR/analysis.

    Filters to rows with valid imaging + target, substitutes longitudinal age
    for baseline age column, fills sex from baseline df if missing, then drops
    rows with any missing covariate values.

    Returns:
        (d, y, present_cols) — cleaned DataFrame, float target array, valid ROI column names.
    """
    d = df.copy()
    present = [c for c in roi_columns if c in d.columns]
    mask = d[present].notna().all(axis=1) & d[target_col].notna()
    d = d[mask].reset_index(drop=True)
    if age_col_long in d.columns:
        d[age_col_bl] = d[age_col_long]
    if bl_df is not None and sex_col in d.columns and d[sex_col].isna().any():
        bl_sex = bl_df.drop_duplicates(id_col).set_index(id_col)[sex_col]
        d[sex_col] = d[sex_col].fillna(d[id_col].map(bl_sex))
    cov_present = [c for c in cov_cols if c in d.columns]
    cov_valid = d[cov_present].notna().all(axis=1)
    d = d[cov_valid].reset_index(drop=True)
    y = d[target_col].values.astype(float)
    y_valid = np.isfinite(y)
    d = d[y_valid].reset_index(drop=True)
    y = y[y_valid]
    print(f"  {label}: n = {len(d)}")
    return d, y, present


def run_univariate_ai_fdr(
    df: "pd.DataFrame",
    y: np.ndarray,
    present_cols: list,
    bilateral_pairs: list,
    label: str,
) -> tuple:
    """Compute univariate AI-severity correlations with Benjamini-Hochberg FDR correction.

    Args:
        df: Subject dataframe with ROI columns.
        y: Target array (same length as df).
        present_cols: ROI feature columns available in df.
        bilateral_pairs: List of (name, left_col, right_col) tuples.
        label: Description printed in the table header.

    Returns:
        (asym, results) — asym is dict {name: array}, results is dict {name: (r, p_raw, p_fdr)}.
    """
    from statsmodels.stats.multitest import multipletests

    valid_p = [(n, l, r) for n, l, r in bilateral_pairs if l in present_cols and r in present_cols]
    X = df[present_cols].values.astype(float)
    asym = compute_asymmetry_features(X, present_cols, valid_p)
    ai_names = sorted(k for k in asym if k.endswith("_AI"))
    print(f"\n  Univariate AI correlations ({label}, n={len(y)}):")
    raw_results: dict = {}
    p_values: list = []
    for name in ai_names:
        r, p = pearsonr(asym[name], y)
        raw_results[name] = (r, p)
        p_values.append(p)
    _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
    print(f"  {'Feature':<20} {'r':>8} {'p_raw':>10} {'p_FDR':>10}")
    print(f"  {'-' * 52}")
    results: dict = {}
    for i, name in enumerate(ai_names):
        r, p_raw = raw_results[name]
        p_corr = p_fdr[i]
        sig = "***" if p_corr < 0.001 else ("**" if p_corr < 0.01 else ("*" if p_corr < 0.05 else ""))
        print(f"  {name:<20} {r:>+8.3f} {p_raw:>10.4f} {p_corr:>10.4f} {sig}")
        results[name] = (r, p_raw, p_corr)
    return asym, results


def compute_timepoint_ai(
    df_tp: "pd.DataFrame",
    feat_cols: list,
    bil_pairs: list,
    id_col: str,
) -> tuple:
    """Compute pallidum AI for a single longitudinal timepoint dataframe.

    Args:
        df_tp: Timepoint dataframe (long format, one timepoint).
        feat_cols: Feature column names to consider.
        bil_pairs: List of (name, left_col, right_col) tuples.
        id_col: Subject ID column name.

    Returns:
        (ai_series, ids_series) — pallidum AI values and subject IDs,
        both indexed to the valid rows of df_tp.
    """
    valid_c = [c for c in feat_cols if c in df_tp.columns]
    valid_p = [(nm, l, r) for nm, l, r in bil_pairs if l in valid_c and r in valid_c]
    feat_ok = df_tp[valid_c].notna().all(axis=1)
    df_v = df_tp[feat_ok].copy()
    if len(df_v) == 0:
        return pd.Series(dtype=float, name="pal_ai"), df_v[id_col]
    X = df_v[valid_c].values.astype(float)
    asym = compute_asymmetry_features(X, valid_c, valid_p)
    pal_keys = [k for k in asym if "pallidum" in k.lower() and k.endswith("_AI")]
    if not pal_keys:
        return pd.Series(dtype=float, name="pal_ai"), df_v[id_col]
    return pd.Series(asym[pal_keys[0]], index=df_v.index, name="pal_ai"), df_v[id_col]


def run_sex_stratified_svr(
    X_sex: np.ndarray,
    y_sex: np.ndarray,
    df_sex: "pd.DataFrame",
    label: str,
    cov_cols: list,
    sex_col: str,
    scanner_col: str,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple:
    """Run fold-based SVR for one sex group with per-fold residualization.

    Sex is excluded from covariates (constant within group). Uses family-aware
    StratifiedGroupKFold on rel_family_id if available, else StratifiedKFold.

    Returns:
        (fold_data, all_true, all_pred) — list of fold dicts (for permutation reuse),
        concatenated true and predicted target arrays.
    """
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    cov_nosex = [c for c in cov_cols if c != sex_col and c in df_sex.columns]
    fam_col = "rel_family_id"

    if fam_col in df_sex.columns:
        fam_grps = pd.to_numeric(df_sex[fam_col], errors="coerce").values
        miss = np.isnan(fam_grps)
        if miss.any():
            mx = np.nanmax(fam_grps) if (~miss).any() else 0
            fam_grps[miss] = np.arange(mx + 1, mx + 1 + miss.sum())
        site_k = pd.Categorical(df_sex[scanner_col]).codes
        cv_sex = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (X_sex, site_k, fam_grps)
    else:
        site_k = pd.Categorical(df_sex[scanner_col]).codes
        cv_sex = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (X_sex, site_k)

    fold_data_sex: list = []
    all_t: list = []
    all_p: list = []

    for fi, (tr_i, te_i) in enumerate(cv_sex.split(*split_args)):
        X_tr, X_te = X_sex[tr_i], X_sex[te_i]
        y_tr, y_te = y_sex[tr_i], y_sex[te_i]
        cov_tr = df_sex.iloc[tr_i][cov_nosex].values.astype(float) if cov_nosex else None
        cov_te = df_sex.iloc[te_i][cov_nosex].values.astype(float) if cov_nosex else None

        if cov_nosex:
            rm = LinearRegression().fit(cov_tr, y_tr)
            y_tr_r = y_tr - rm.predict(cov_tr)
            y_te_r = y_te - rm.predict(cov_te)
        else:
            y_tr_r, y_te_r = y_tr.copy(), y_te.copy()

        sc_fold = StandardScaler()
        Xtr_s = sc_fold.fit_transform(X_tr)
        Xte_s = sc_fold.transform(X_te)

        svr = SVR(kernel="linear", C=1.0)
        svr.fit(Xtr_s, y_tr_r)
        y_pred = svr.predict(Xte_s)

        fold_data_sex.append({"Xtr": Xtr_s, "Xte": Xte_s, "ytr": y_tr_r, "yte": y_te_r})
        all_t.extend(y_te_r)
        all_p.extend(y_pred)
        r_fold = pearsonr(y_te_r, y_pred)[0]
        print(f"    Fold {fi+1}: r = {r_fold:+.3f}  (n_test={len(te_i)})")

    all_t_arr = np.array(all_t)
    all_p_arr = np.array(all_p)
    r_all, _ = pearsonr(all_t_arr, all_p_arr)
    print(f"  Overall r = {r_all:+.4f}")
    return fold_data_sex, all_t_arr, all_p_arr


def perm_and_boot_svr(
    fold_data: list,
    all_true: np.ndarray,
    all_pred: np.ndarray,
    n_perms: int = 300,
    n_boot: int = 5000,
    seed: int = 42,
    label: str = "",
) -> tuple:
    """Permutation test + bootstrap CIs for fold-based SVR results.

    Permutations shuffle training labels within each fold and refit SVR.
    Bootstrap resamples (with replacement) from the concatenated true/pred arrays.

    Returns:
        (p_emp, ci_lo, ci_hi) — empirical p-value and 95% bootstrap CI bounds.
    """
    from sklearn.svm import SVR
    from tqdm.auto import tqdm

    r_obs = pearsonr(all_true, all_pred)[0]
    rng_p = np.random.default_rng(seed)
    perm_rs: list = []
    for _ in tqdm(range(n_perms), desc=f"perm ({label})", leave=False):
        at: list = []
        ap: list = []
        for fd in fold_data:
            yp = rng_p.permutation(fd["ytr"])
            svr_p = SVR(kernel="linear", C=1.0).fit(fd["Xtr"], yp)
            ap.extend(svr_p.predict(fd["Xte"]))
            at.extend(fd["yte"])
        perm_rs.append(pearsonr(at, ap)[0])
    perm_rs_arr = np.array(perm_rs)
    n_exc = int(np.sum(perm_rs_arr >= r_obs))
    p_emp = (n_exc + 1) / (n_perms + 1)

    # Use a different seed offset so bootstrap is independent of permutation count.
    rng_b = np.random.default_rng(seed + 12345)
    boot_rs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng_b.integers(0, len(all_true), size=len(all_true))
        boot_rs[b] = pearsonr(all_true[idx], all_pred[idx])[0]
    return boot_rs, perm_rs_arr, p_emp


# ── Phase 2: notebook analysis helpers ────────────────────────────────────

def run_demographic_analysis(df_analysis, targets_to_show,
                              age_col="demo_brthdat_v2", sex_col="sex_mapped"):
    """Compute age correlation and sex difference for each target.

    Returns list of dicts with keys: target, age_r, age_p, male_mean,
    female_mean, sex_t, sex_p.
    """
    from scipy.stats import pearsonr, ttest_ind
    import numpy as np

    has_age = age_col in df_analysis.columns
    has_sex = sex_col in df_analysis.columns
    results = []

    if has_age:
        age = df_analysis[age_col].values
        print(f"\nAge: mean={np.nanmean(age):.1f} months, SD={np.nanstd(age):.1f}, "
              f"range=[{np.nanmin(age):.0f}, {np.nanmax(age):.0f}]")
    if has_sex:
        sc = df_analysis[sex_col].value_counts()
        print("\nSex distribution:")
        for s, c in sc.items():
            print(f"  {s}: n={c} ({100*c/len(df_analysis):.1f}%)")

    print("\n{:<25} {:>8} {:>10} {:>8} {:>8} {:>8} {:>10}".format(
        "Target", "Age r", "Age p", "M mean", "F mean", "Sex t", "Sex p"))
    print("-" * 90)

    for tgt in targets_to_show:
        col, name = tgt["column"], tgt["name"]
        if col not in df_analysis.columns:
            continue
        y = df_analysis[col].dropna()
        idx = y.index
        row = {"target": name}

        if has_age:
            av = df_analysis.loc[idx, age_col]
            msk = av.notna() & y.notna()
            if msk.sum() > 10:
                r, p = pearsonr(av[msk], y[msk])
                row["age_r"] = r; row["age_p"] = p

        if has_sex:
            sv = df_analysis.loc[idx, sex_col]
            ms = y[sv == "male"]; fs = y[sv == "female"]
            if len(ms) > 10 and len(fs) > 10:
                t, p = ttest_ind(ms, fs)
                row["male_mean"] = ms.mean(); row["female_mean"] = fs.mean()
                row["sex_t"] = t; row["sex_p"] = p

        results.append(row)
        as_ = "***" if row.get("age_p", 1) < 0.001 else ("**" if row.get("age_p", 1) < 0.01 else ("*" if row.get("age_p", 1) < 0.05 else ""))
        ss_ = "***" if row.get("sex_p", 1) < 0.001 else ("**" if row.get("sex_p", 1) < 0.01 else ("*" if row.get("sex_p", 1) < 0.05 else ""))
        print("{:<25} {:>+8.4f} {:>9.4f}{:<2}{:>8.2f} {:>8.2f} {:>+8.2f} {:>9.4f}{:<2}".format(
            name, row.get("age_r", 0), row.get("age_p", 1), as_,
            row.get("male_mean", 0), row.get("female_mean", 0),
            row.get("sex_t", 0), row.get("sex_p", 1), ss_))

    return results


def get_paired_bl_y2(long_df, tp_col, baseline_tp, year2_tp, roi_columns,
                     bilateral_pairs, id_col, target_col=None, harm_cfg=None):
    """Load paired BL+Y2 data with valid imaging.

    If harm_cfg is provided, applies joint ComBat harmonization across both
    timepoints (BL and Y2 pooled together) before computing asymmetry features.
    This removes site effects from both timepoints using a single model, making
    developmental change estimates free of scanner confounds.

    Returns:
        bl_v, y2_v  — filtered dataframes
        X_bl, X_y2  — feature arrays (ComBat-harmonized if harm_cfg given, else raw)
        asym_bl, asym_y2  — dicts from compute_asymmetry_features
        valid_pairs — filtered bilateral pair list
        present_cols — feature column names in X_bl/X_y2
    """
    import numpy as np
    import pandas as pd
    from .univariate import compute_asymmetry_features

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()
    shared = np.intersect1d(bl[id_col].values, y2[id_col].values)

    bl_s = bl[bl[id_col].isin(shared)].sort_values(id_col).reset_index(drop=True)
    y2_s = y2[y2[id_col].isin(shared)].sort_values(id_col).reset_index(drop=True)

    present_cols = [c for c in roi_columns if c in bl_s.columns and c in y2_s.columns]
    valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs if l in present_cols and r in present_cols]

    X_bl = bl_s[present_cols].values.astype(float)
    X_y2 = y2_s[present_cols].values.astype(float)

    extra = np.array([True] * len(bl_s))
    if target_col is not None:
        extra = bl_s[target_col].notna().values & y2_s[target_col].notna().values

    valid = np.all(np.isfinite(X_bl), axis=1) & np.all(np.isfinite(X_y2), axis=1) & extra
    X_bl = X_bl[valid]; X_y2 = X_y2[valid]
    bl_v = bl_s[valid].reset_index(drop=True)
    y2_v = y2_s[valid].reset_index(drop=True)

    if harm_cfg is not None:
        # Joint ComBat: pool BL + Y2 rows, harmonize together, split back.
        # Timepoint (0=BL, 1=Y2) is injected as an explicit covariate so ComBat
        # does not absorb real developmental signal into its site-effect estimate.
        # Subjects are sorted identically (by id_col) so index i in BL == index i in Y2.
        n_bl = len(bl_v)
        X_pool = np.vstack([X_bl, X_y2])
        df_pool = pd.concat([bl_v, y2_v], ignore_index=True)
        df_pool["_timepoint"] = [0] * n_bl + [1] * len(y2_v)

        # Extend harm_cfg covariates to include the timepoint indicator
        harm_cfg_tp = dict(harm_cfg)
        harm_cfg_tp["covariates"] = list(harm_cfg.get("covariates", [])) + ["_timepoint"]

        X_pool_h, keep_mask, _ = _combat_harmonize(X_pool, df_pool, harm_cfg_tp)

        bl_keep = keep_mask[:n_bl]        # which BL rows survived ComBat filters
        y2_keep = keep_mask[n_bl:]        # which Y2 rows survived ComBat filters
        pair_keep = bl_keep & y2_keep     # subjects valid in BOTH timepoints

        # Among each timepoint's ComBat survivors, keep only those also present in
        # the other timepoint (maintain 1:1 pairing).
        bl_selector = pair_keep[bl_keep]
        y2_selector = pair_keep[y2_keep]

        X_bl = X_pool_h[:bl_keep.sum()][bl_selector]
        X_y2 = X_pool_h[bl_keep.sum():][y2_selector]
        bl_v = bl_v[pair_keep].reset_index(drop=True)
        y2_v = y2_v[pair_keep].reset_index(drop=True)
        print(f"  ComBat: {n_bl} → {len(bl_v)} paired subjects after joint BL+Y2 harmonization")

    asym_bl = compute_asymmetry_features(X_bl, present_cols, valid_pairs)
    asym_y2 = compute_asymmetry_features(X_y2, present_cols, valid_pairs)

    return bl_v, y2_v, X_bl, X_y2, asym_bl, asym_y2, valid_pairs, present_cols


def run_family_history_analysis(full_df, env, network="dopamine_core"):
    """ComBat-harmonize the full sample, compute AI features, run FH+/FH- analysis.

    Returns dict with keys:
        merged, fhx_asym, ai_names, group_df, int_df,
        fh_pos_mask, fh_neg_mask, n_fhx_pos, n_fhx_neg,
        ai_pqbc, y_pqbc, fh_pqbc
    """
    import numpy as np, pandas as pd
    from scipy.stats import ttest_ind, pearsonr
    from statsmodels.formula.api import ols
    from neuroHarmonize import harmonizationLearn
    from ..preprocessing.transforms import load_family_history
    from ..tsne.embeddings import get_roi_columns_from_config

    reg_config = env.configs.regression
    harm_config = env.configs.harmonize
    roi_columns = get_roi_columns_from_config(env.configs.data, [network])

    fhx_path = env.repo_root / "data" / "raw" / "mh_p_fhx.csv"
    fhx_df = load_family_history(fhx_path, scope="psychosis")
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    fhx_baseline = fhx_df[fhx_df["eventname"] == baseline_tp].copy()

    merged = full_df.merge(
        fhx_baseline[["src_subject_id", "fhx_psychosis_any", "fhx_psychosis_parent", "fhx_psychosis_count"]],
        on="src_subject_id", how="left",
    )
    merged = merged[merged["fhx_psychosis_any"].notna()].reset_index(drop=True)

    # ComBat on full sample
    site_col = harm_config.get("site_column", "mri_info_manufacturer")
    n_splits = reg_config.get("cv", {}).get("n_outer_splits", 5)
    valid_cols = [c for c in roi_columns if c in merged.columns]
    feat_mask = merged[valid_cols].notna().all(axis=1)
    merged = merged[feat_mask].reset_index(drop=True)
    if site_col in merged.columns:
        sc = merged[site_col].value_counts()
        merged = merged[~merged[site_col].isin(sc[sc < n_splits].index)].reset_index(drop=True)

    X_raw = merged[valid_cols].values
    covars = merged[[site_col]].copy().rename(columns={site_col: "SITE"})
    for cc in harm_config.get("covariates", []):
        if cc in merged.columns:
            v = merged[cc]
            covars[cc] = pd.Categorical(v).codes if not pd.api.types.is_numeric_dtype(v) else v.values
    _, X_harm = harmonizationLearn(X_raw, covars, eb=harm_config.get("empirical_bayes", True),
                                   smooth_terms=harm_config.get("smooth_terms", []))

    bilateral_pairs, _ = extract_bilateral_pairs(env.configs.data, [network])
    valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs if l in valid_cols and r in valid_cols]
    fhx_asym = compute_asymmetry_features(X_harm, valid_cols, valid_pairs)
    ai_names = sorted(k for k in fhx_asym if k.endswith("_AI"))

    fh_pos_mask = merged["fhx_psychosis_any"].values == 1
    fh_neg_mask = merged["fhx_psychosis_any"].values == 0
    n_fhx_pos = fh_pos_mask.sum(); n_fhx_neg = fh_neg_mask.sum()

    # Analysis 1: FH+/FH- group differences
    group_rows = []
    for ai_name in ai_names:
        ai_vals = fhx_asym[ai_name]
        ai_pos = ai_vals[fh_pos_mask]; ai_neg = ai_vals[fh_neg_mask]
        t_stat, p_val = ttest_ind(ai_pos, ai_neg)
        pooled = np.sqrt(((len(ai_pos)-1)*ai_pos.std()**2 + (len(ai_neg)-1)*ai_neg.std()**2)
                         / (len(ai_pos) + len(ai_neg) - 2))
        d = (ai_pos.mean() - ai_neg.mean()) / pooled if pooled > 0 else 0.0
        group_rows.append({"feature": ai_name, "mean_FH+": ai_pos.mean(), "mean_FH-": ai_neg.mean(),
                            "cohen_d": d, "t": t_stat, "p": p_val})
    group_df = pd.DataFrame(group_rows).sort_values("p").reset_index(drop=True)
    m = len(group_df); group_df["rank"] = range(1, m+1)
    group_df["p_fdr"] = (group_df["p"] * m / group_df["rank"]).clip(upper=1.0)
    group_df["p_fdr"] = group_df["p_fdr"].iloc[::-1].cummin().iloc[::-1]

    # Analysis 2: FH x AI interaction on PQ-BC
    target_col = "pps_y_ss_severity_score"
    pqbc_mask = merged[target_col].notna()
    merged_pqbc = merged[pqbc_mask].reset_index(drop=True)
    y_pqbc = merged_pqbc[target_col].values
    fh_pqbc = merged_pqbc["fhx_psychosis_any"].values
    ai_pqbc = {k: fhx_asym[k][pqbc_mask.values] for k in ai_names}

    int_rows = []
    for ai_name in ai_names:
        ai_v = ai_pqbc[ai_name]; ai_z = (ai_v - ai_v.mean()) / ai_v.std()
        tmp = pd.DataFrame({"y": y_pqbc, "AI": ai_z, "FH": fh_pqbc.astype(float),
                            "AI_x_FH": ai_z * fh_pqbc})
        m_fit = ols("y ~ AI + FH + AI_x_FH", data=tmp).fit()
        int_rows.append({"feature": ai_name, "beta_AI": m_fit.params["AI"], "p_AI": m_fit.pvalues["AI"],
                         "beta_FH": m_fit.params["FH"], "p_FH": m_fit.pvalues["FH"],
                         "beta_interaction": m_fit.params["AI_x_FH"], "p_interaction": m_fit.pvalues["AI_x_FH"],
                         "R2": m_fit.rsquared})
    int_df = pd.DataFrame(int_rows).sort_values("p_interaction").reset_index(drop=True)
    mi = len(int_df); int_df["rank"] = range(1, mi+1)
    int_df["p_int_fdr"] = (int_df["p_interaction"] * mi / int_df["rank"]).clip(upper=1.0)
    int_df["p_int_fdr"] = int_df["p_int_fdr"].iloc[::-1].cummin().iloc[::-1]

    return {
        "merged": merged, "fhx_asym": fhx_asym, "ai_names": ai_names,
        "group_df": group_df, "int_df": int_df,
        "fh_pos_mask": fh_pos_mask, "fh_neg_mask": fh_neg_mask,
        "n_fhx_pos": n_fhx_pos, "n_fhx_neg": n_fhx_neg,
        "ai_pqbc": ai_pqbc, "y_pqbc": y_pqbc, "fh_pqbc": fh_pqbc,
    }


def run_sex_stratified_nested_cv(env, full_df, target_config, model_name, sex_col="sex_mapped"):
    """Run nested CV SVR separately per sex (high-severity + full-population).

    Returns dict:
        sex_results: {sex: result}
        fullpop_results: {sex: result}
        sex_fold_data: {sex: [fold dicts]}
        pooled_r, pooled_mae, pooled_r2, pooled_n
    """
    import numpy as np, copy, pickle
    from scipy.stats import pearsonr
    from core.regression.pipeline import run_target_with_nested_cv

    harm_config_orig = env.configs.harmonize
    reg_config_orig = env.configs.regression
    run_cfg = env.configs.run
    seed = run_cfg["seed"]
    target_name = target_config["name"]

    results_dir = (env.repo_root / "outputs" / run_cfg["run_name"] / run_cfg["run_id"]
                   / f"seed_{seed}" / "regression" / target_name / model_name)

    # Load pooled baseline results
    pooled_r = pooled_mae = pooled_r2 = pooled_n = None
    pooled_path = results_dir / "results.pkl"
    if pooled_path.exists():
        with open(pooled_path, "rb") as f: pooled_saved = pickle.load(f)
        pooled_folds = pooled_saved[f"{model_name}_folds"]
        pooled_true = np.concatenate([f["y_test"] for f in pooled_folds])
        pooled_pred = np.concatenate([f["y_pred"] for f in pooled_folds])
        pooled_r, _ = pearsonr(pooled_true, pooled_pred)
        pooled_mae = np.mean(np.abs(pooled_true - pooled_pred))
        pooled_r2 = 1 - np.sum((pooled_true - pooled_pred)**2) / np.sum((pooled_true - np.mean(pooled_true))**2)
        pooled_n = len(pooled_true)

    # Remove sex from ComBat + residualization (constant within single-sex group)
    harm_config_nosex = copy.deepcopy(dict(harm_config_orig))
    harm_config_nosex["covariates"] = [c for c in harm_config_nosex.get("covariates", [])
                                       if c not in ("demo_sex_v2", "sex_mapped")]
    reg_config_nosex = copy.deepcopy(dict(reg_config_orig))
    reg_config_nosex["covariates"]["columns"] = [
        c for c in reg_config_nosex["covariates"].get("columns", []) if c != "sex_mapped"]

    sex_fold_data = {}; sex_results = {}; fullpop_results = {}

    env.configs.harmonize = harm_config_nosex
    env.configs.regression = reg_config_nosex
    try:
        for sex_label in ["male", "female"]:
            sex_df = full_df[full_df[sex_col] == sex_label].copy().reset_index(drop=True)
            print(f"{'='*60}\n  {sex_label.upper()}-ONLY SVR (high-severity)\n{'='*60}")
            result = run_target_with_nested_cv(env, sex_df, target_config, model_name)
            sex_results[sex_label] = result
            with open(results_dir / "results.pkl", "rb") as f: saved = pickle.load(f)
            folds = saved[f"{model_name}_folds"]
            sex_fold_data[sex_label] = [{"X_train": f["X_train"], "X_test": f["X_test"],
                                         "y_train": f["y_train"], "y_test": f["y_test"],
                                         "y_pred": f["y_pred"]} for f in folds]
            r_val = result[model_name]["overall"]["pearson_r"]
            n_actual = sum(len(f["y_test"]) for f in folds)
            print(f"  {sex_label.upper()}: r = {r_val:.4f}, n = {n_actual}")

        reg_nofilt = copy.deepcopy(reg_config_nosex)
        reg_nofilt["sample_weighting"]["custom_bins"] = {}
        reg_nofilt["sample_weighting"]["enabled"] = False
        env.configs.regression = reg_nofilt
        for sex_label in ["male", "female"]:
            sex_full_df = full_df[full_df[sex_col] == sex_label].copy().reset_index(drop=True)
            print(f"{'='*60}\n  FULL POP {sex_label.upper()}-ONLY SVR (n={len(sex_full_df)})\n{'='*60}")
            fp_result = run_target_with_nested_cv(env, sex_full_df, target_config, model_name)
            fullpop_results[sex_label] = fp_result
            print(f"  {sex_label.upper()} FULL POP: r = {fp_result[model_name]['overall']['pearson_r']:.4f}")
    finally:
        env.configs.regression = reg_config_orig
        env.configs.harmonize = harm_config_orig

    return {"sex_results": sex_results, "fullpop_results": fullpop_results,
            "sex_fold_data": sex_fold_data,
            "pooled_r": pooled_r, "pooled_mae": pooled_mae,
            "pooled_r2": pooled_r2, "pooled_n": pooled_n}


def permutation_test_sex_stratified(
    env,
    full_df,
    target_config: dict,
    model_name: str,
    sex_col: str = "sex_mapped",
    n_permutations: int | None = None,
    seed: int | None = None,
) -> dict:
    """Pipeline-matched permutation test for sex-stratified SVR.

    Mirrors run_sex_stratified_nested_cv: removes sex from ComBat covariates
    and from target residualization (sex is constant within each group),
    then runs a fully pipeline-matched permutation_test per sex.

    Args:
        env: Environment with configs.
        full_df: Full dataset (same as passed to run_sex_stratified_nested_cv).
        target_config: Target configuration dict (name, column).
        model_name: Model name (e.g., "svr").
        sex_col: Column name for sex grouping (default "sex_mapped").
        n_permutations: Override regression.yaml permutation.n_permutations.
        seed: Override run.yaml seed.

    Returns:
        Dict keyed by sex_label ("male", "female"), each with keys:
            null_distribution, n_permutations, null_mean, null_std.
    """
    import copy
    from core.regression.evaluation import permutation_test

    harm_orig = env.configs.harmonize
    reg_orig = env.configs.regression

    harm_nosex = copy.deepcopy(dict(harm_orig))
    harm_nosex["covariates"] = [c for c in harm_nosex.get("covariates", [])
                                if c not in ("demo_sex_v2", "sex_mapped")]

    reg_nosex = copy.deepcopy(dict(reg_orig))
    reg_nosex["covariates"]["columns"] = [
        c for c in reg_nosex["covariates"].get("columns", []) if c != "sex_mapped"]

    results = {}
    env.configs.harmonize = harm_nosex
    env.configs.regression = reg_nosex
    try:
        for sex_label in ["male", "female"]:
            sex_df = full_df[full_df[sex_col] == sex_label].copy().reset_index(drop=True)
            print(f"\n{'='*55}")
            print(f"  Pipeline-matched permutation: {sex_label.upper()}")
            print(f"{'='*55}")
            results[sex_label] = permutation_test(
                env, sex_df, target_config, model_name,
                n_permutations=n_permutations, seed=seed,
            )
    finally:
        env.configs.harmonize = harm_orig
        env.configs.regression = reg_orig

    return results


def run_y2_deep_dive(long_df, bilateral_pairs, present_cols, valid_pairs,
                     target_col, n_bootstrap, seed, tp_col, env):
    """5 Year-2 follow-up analyses (NB09 cell 11): within-subject change, AI vs total at Y2,
    bootstrap CIs, sex decomposition at Y2, and AI stability regression.

    Returns dict with analysis results.
    """
    import numpy as np
    from scipy.stats import pearsonr, ttest_ind, ttest_rel

    id_col = env.configs.data["columns"]["mapping"]["id"]
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp = env.configs.data["timepoints"]["year2"]
    sex_col = env.configs.data["columns"]["mapping"].get("sex_mapped", "sex_mapped")
    cutoff = env.configs.regression.get("sample_weighting", {}).get("custom_bins", {}).get(
        target_col.replace("pps_y_ss_", "pps_severity_").replace("_score", "_raw"), [30])[0] if \
        env.configs.regression.get("sample_weighting") else 30

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()

    shared_ids = np.intersect1d(bl[id_col].values, y2[id_col].values)
    bl_s = bl[bl[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)
    y2_s = y2[y2[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)

    valid_bl = bl_s[present_cols].notna().all(axis=1) & bl_s[target_col].notna()
    valid_y2 = y2_s[present_cols].notna().all(axis=1) & y2_s[target_col].notna()
    both = valid_bl.values & valid_y2.values
    bl_v = bl_s[both].reset_index(drop=True)
    y2_v = y2_s[both].reset_index(drop=True)
    n_paired = len(bl_v)
    print(f"Paired BL+Y2: n = {n_paired}")

    X_bl = bl_v[present_cols].values.astype(float)
    X_y2 = y2_v[present_cols].values.astype(float)
    asym_bl = compute_asymmetry_features(X_bl, present_cols, valid_pairs)
    asym_y2 = compute_asymmetry_features(X_y2, present_cols, valid_pairs)
    pqbc_bl = bl_v[target_col].values.astype(float)
    pqbc_y2 = y2_v[target_col].values.astype(float)
    delta_pqbc = pqbc_y2 - pqbc_bl

    # 1. Within-subject change
    ai_names = sorted(k for k in asym_bl if k.endswith("_AI"))
    change_results = {}
    print("\n1. WITHIN-SUBJECT CHANGE: ΔAI vs ΔPQBC")
    for name in ai_names:
        delta_ai = asym_y2[name] - asym_bl[name]
        r_c, p_c = pearsonr(delta_ai, delta_pqbc)
        r_y2, p_y2 = pearsonr(delta_ai, pqbc_y2)
        sig1 = "*" if p_c < 0.05 else ""; sig2 = "*" if p_y2 < 0.05 else ""
        print(f"  {name:<20} r_change={r_c:+.3f}{sig1}  r_y2={r_y2:+.3f}{sig2}")
        change_results[name] = {"r_change": r_c, "p_change": p_c, "r_y2": r_y2, "p_y2": p_y2}

    # 2. AI vs Total at Y2
    y2_high = y2[y2[target_col] >= cutoff].copy()
    y2_mask = y2_high[present_cols].notna().all(axis=1) & y2_high[target_col].notna()
    y2_h = y2_high[y2_mask].reset_index(drop=True)
    X_y2h = y2_h[present_cols].values.astype(float)
    y_y2h = y2_h[target_col].values.astype(float)
    asym_y2h = compute_asymmetry_features(X_y2h, present_cols, valid_pairs)

    ai_vs_tot = {}
    print(f"\n2. AI vs TOTAL at Y2 (n={len(y_y2h)})")
    for name, lcol, rcol in valid_pairs:
        ak = f"{name}_AI"; tk = f"{name}_total"
        if ak in asym_y2h and tk in asym_y2h:
            r_ai, p_ai = pearsonr(asym_y2h[ak], y_y2h)
            r_tot, p_tot = pearsonr(asym_y2h[tk], y_y2h)
            winner = "AI" if abs(r_ai) > abs(r_tot) else "Total"
            sig_ai = "*" if p_ai < 0.05 else ""; sig_tot = "*" if p_tot < 0.05 else ""
            ai_vs_tot[name] = {"r_ai": r_ai, "p_ai": p_ai, "r_tot": r_tot, "p_tot": p_tot, "winner": winner}
            print(f"  {name:<15} AI r={r_ai:+.3f}{sig_ai}  Tot r={r_tot:+.3f}{sig_tot}  [{winner}]")

    # 3. Bootstrap CI for Y2 pallidum AI
    # IMPORTANT: use the SAME index for both X and y per resample to preserve pairing.
    # Independent indices break the correlation structure and center the CI on zero.
    pal_ai_y2h = asym_y2h.get("pallidum_AI", np.zeros(len(y_y2h)))
    r_obs, p_obs = pearsonr(pal_ai_y2h, y_y2h)
    rng = np.random.RandomState(seed)
    n_y2 = len(y_y2h)
    boot_r = np.array([
        pearsonr(pal_ai_y2h[idx := rng.randint(0, n_y2, n_y2)], y_y2h[idx])[0]
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    boot_result = {"r_obs": r_obs, "p_obs": p_obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
                   "boot_r": boot_r, "n": len(y_y2h)}
    print(f"\n3. Y2 PALLIDUM AI BOOTSTRAP: r={r_obs:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    return {
        "n_paired": n_paired, "change_results": change_results,
        "ai_vs_tot": ai_vs_tot, "boot_result": boot_result,
        "asym_y2h": asym_y2h, "y_y2h": y_y2h,
    }


def run_worsening_analysis(long_df, bilateral_pairs, target_col, env, n_splits, n_perms, seed, tp_col):
    """Worsening group analysis: baseline + Y2 SVR for clinically-worsened subjects.

    Returns dict with all SVR results and univariate tables.
    """
    import numpy as np
    from scipy.stats import pearsonr, linregress

    reg_config = env.configs.regression
    id_col = env.configs.data["columns"]["mapping"]["id"]
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp = env.configs.data["timepoints"]["year2"]
    cov_cols = reg_config["covariates"]["columns"]
    age_col_bl = cov_cols[0]
    age_col_long = "demo_brthdat_v2_l"
    sex_col = env.configs.data["columns"]["mapping"]["sex_mapped"]
    roi_columns = [c for c in long_df.columns if any(
        c in str(pair) for pair in bilateral_pairs)]

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()

    full_sd = bl[target_col].dropna().std()
    mcid = np.ceil(1.0 * full_sd)

    all_shared = np.intersect1d(bl[id_col].values, y2[id_col].values)
    bl_all = bl[bl[id_col].isin(all_shared)].sort_values(id_col).reset_index(drop=True)
    y2_all = y2[y2[id_col].isin(all_shared)].sort_values(id_col).reset_index(drop=True)
    bl_scores = bl_all[target_col].values.astype(float)
    y2_scores = y2_all[target_col].values.astype(float)
    valid = np.isfinite(bl_scores) & np.isfinite(y2_scores)
    bl_all = bl_all[valid].reset_index(drop=True)
    y2_all = y2_all[valid].reset_index(drop=True)
    delta = y2_scores[valid] - bl_scores[valid]
    worsened_mask = delta >= mcid
    worsen_ids = y2_all[id_col].values[worsened_mask]

    print(f"Full SD = {full_sd:.1f}, MCID = {mcid:.0f}")
    print(f"Worsened: n = {worsened_mask.sum()} ({worsened_mask.mean()*100:.1f}%)")

    present_cols = [c for c in long_df.columns if any(
        lc in c or rc in c for _, lc, rc in bilateral_pairs) and c in bl.columns]
    # Rebuild roi_columns from what's actually in the df
    from ..tsne.embeddings import get_roi_columns_from_config
    roi_nets = reg_config.get("roi_networks", ["dopamine_core"])
    try:
        roi_columns = get_roi_columns_from_config(env.configs.data, roi_nets)
    except Exception:
        roi_columns = [c for c in long_df.columns if "smri_vol_scs_" in c or "dmri_dtimd" in c]

    present_cols = [c for c in roi_columns if c in bl.columns]
    valid_pairs_w = [(n, l, r) for n, l, r in bilateral_pairs if l in present_cols and r in present_cols]

    def _prep(df, label):
        d = df.copy()
        if age_col_long in d.columns: d[age_col_bl] = d[age_col_long]
        if d[age_col_bl].isna().any():
            bl_age = bl.drop_duplicates(id_col).set_index(id_col)[age_col_bl]
            d[age_col_bl] = d[age_col_bl].fillna(d[id_col].map(bl_age))
        if sex_col in d.columns and d[sex_col].isna().any():
            bl_sex = bl.drop_duplicates(id_col).set_index(id_col)[sex_col]
            d[sex_col] = d[sex_col].fillna(d[id_col].map(bl_sex))
        cov_pres = [c for c in cov_cols if c in d.columns]
        mask = d[present_cols].notna().all(axis=1) & d[cov_pres].notna().all(axis=1)
        d = d[mask].reset_index(drop=True)
        y = d[target_col].values.astype(float)
        y_v = np.isfinite(y)
        d = d[y_v].reset_index(drop=True); y = y[y_v]
        print(f"  {label}: n = {len(d)}"); return d, y

    y2_worsen = y2[y2[id_col].isin(worsen_ids)].copy()
    y2_w_prep, y_y2w = _prep(y2_worsen, "Y2 worsening group")
    res_y2 = run_cross_sectional_svr(y2_w_prep, y_y2w, env, roi_columns, valid_pairs_w,
                                      "y2_worsening", n_splits)

    bl_worsen = bl[bl[id_col].isin(worsen_ids)].copy()
    bl_w_prep, y_blw = _prep(bl_worsen, "BL worsening group")
    res_bl = run_cross_sectional_svr(bl_w_prep, y_blw, env, roi_columns, valid_pairs_w,
                                      "bl_worsening", n_splits)

    return {"res_y2": res_y2, "res_bl": res_bl,
            "y2_prep": y2_w_prep, "y_y2w": y_y2w,
            "bl_prep": bl_w_prep, "y_blw": y_blw,
            "valid_pairs": valid_pairs_w, "present_cols": present_cols,
            "mcid": mcid, "n_worsened": worsened_mask.sum()}


def run_persistent_remitted(long_df, bilateral_pairs, target_col, env, cutoff, n_bootstrap, seed, tp_col):
    """Persistent vs remitted: ComBat on BL high-severity, compute AI, run t-tests + Cohen's d.

    Returns dict: ai_names, asym, mask_p, mask_r, n_persistent, n_remitted,
                  bl_matched, y2_matched, ai_names, pvals, dvals, results_df.
    """
    import numpy as np, pandas as pd
    from scipy.stats import ttest_ind, pearsonr
    from statsmodels.stats.multitest import multipletests
    from .univariate import prepare_harmonized_data, compute_asymmetry_features, extract_bilateral_pairs
    from ..tsne.embeddings import get_roi_columns_from_config

    id_col = env.configs.data["columns"]["mapping"]["id"]
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp = env.configs.data["timepoints"]["year2"]
    reg_config = env.configs.regression
    harm_config = env.configs.harmonize
    roi_nets = reg_config.get("roi_networks", ["dopamine_core"])
    feat_cols = get_roi_columns_from_config(env.configs.data, roi_nets)
    bil_pairs, _ = extract_bilateral_pairs(env.configs.data, roi_nets)

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()

    bl_high_ids = bl[bl[target_col] >= cutoff][id_col].values
    shared_ids = np.intersect1d(bl_high_ids, y2[id_col].values)
    bl_matched = bl[bl[id_col].isin(shared_ids)].copy()
    y2_matched = y2[y2[id_col].isin(shared_ids)].copy()
    bl_matched = bl_matched.sort_values(id_col).reset_index(drop=True)
    y2_matched = y2_matched.sort_values(id_col).reset_index(drop=True)

    valid_y2 = y2_matched[target_col].notna()
    bl_matched = bl_matched[valid_y2.values].reset_index(drop=True)
    y2_matched = y2_matched[valid_y2].reset_index(drop=True)
    print(f"Matched BL-Y2 with valid PQ-BC: n = {len(bl_matched)}")

    y2_scores_arr = y2_matched[target_col].values.astype(float)
    persistent_mask = y2_scores_arr >= cutoff
    remitted_mask = y2_scores_arr < cutoff
    n_persistent = persistent_mask.sum(); n_remitted = remitted_mask.sum()
    print(f"Persistent: {n_persistent}, Remitted: {n_remitted}")

    # ComBat on BL high-severity sample
    rc_nofilt = {**reg_config, "sample_weighting": {"enabled": False, "custom_bins": {}}}
    X_harm, y_bl, df_harm, valid_cols = prepare_harmonized_data(
        bl_matched, feat_cols, harm_config, rc_nofilt,
        target_col=target_col, target_name="pps_severity_raw", residualize_age_sex=False,
    )
    valid_p = [(nm, l, r) for nm, l, r in bil_pairs if l in valid_cols and r in valid_cols]
    asym = compute_asymmetry_features(X_harm, valid_cols, valid_p)
    ai_names = sorted(k for k in asym if k.endswith("_AI"))

    harm_ids = set(df_harm[id_col].values)
    bl_harm = bl_matched[bl_matched[id_col].isin(harm_ids)].reset_index(drop=True)
    y2_harm = y2_matched[y2_matched[id_col].isin(harm_ids)].reset_index(drop=True)
    y2_sc_harm = y2_harm[target_col].values.astype(float)
    mask_p = y2_sc_harm >= cutoff; mask_r = y2_sc_harm < cutoff

    rows = []
    for ai in ai_names:
        a_p = asym[ai][mask_p]; a_r = asym[ai][mask_r]
        if len(a_p) > 1 and len(a_r) > 1:
            t, p = ttest_ind(a_p, a_r, equal_var=False)
            d = cohens_d(a_p, a_r)
            rows.append({"feature": ai, "mean_p": a_p.mean(), "mean_r": a_r.mean(), "t": t, "p": p, "d": d})
        else:
            rows.append({"feature": ai, "mean_p": np.nan, "mean_r": np.nan, "t": np.nan, "p": 1.0, "d": 0.0})
    results_df = pd.DataFrame(rows)
    if len(results_df):
        _, p_fdr, _, _ = multipletests(results_df["p"].fillna(1.0), method="fdr_bh")
        results_df["p_fdr"] = p_fdr

    # Print summary
    print(f"\n{'Feature':<22} {'Persist':>8} {'Remit':>8} {'t':>7} {'p':>8} {'d':>7}")
    for _, row in results_df.iterrows():
        sig = "***" if row.get("p_fdr", 1) < 0.001 else ("**" if row.get("p_fdr", 1) < 0.01 else ("*" if row["p"] < 0.05 else ""))
        print(f"  {row['feature']:<20} {row['mean_p']:>+8.4f} {row['mean_r']:>+8.4f} {row['t']:>7.2f} {row['p']:>8.4f} {row['d']:>+7.3f} {sig}")

    return {"ai_names": ai_names, "asym": asym, "mask_p": mask_p, "mask_r": mask_r,
            "n_persistent": mask_p.sum(), "n_remitted": mask_r.sum(),
            "bl_matched": bl_harm, "y2_matched": y2_harm,
            "results_df": results_df}


def run_y2_replication(long_df, bilateral_pairs, roi_columns, target_col, env, cutoff, n_splits,
                       tp_col, include_worseners=False):
    """Run BL-defined, Y2-defined, and optionally worseners cross-sectional SVR at Year 2.

    Args:
        include_worseners: If True, adds a third analysis for worseners (BL<cutoff AND Y2>=cutoff).

    Returns dict: bldef, y2def, and optionally worseners (each with r, all_true, all_pred, uni, n).
    """
    import numpy as np
    from scipy.stats import pearsonr

    id_col = env.configs.data["columns"]["mapping"]["id"]
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp = env.configs.data["timepoints"]["year2"]
    cov_cols = env.configs.regression["covariates"]["columns"]
    age_col_bl = cov_cols[0]
    age_col_long = "demo_brthdat_v2_l"
    sex_col = env.configs.data["columns"]["mapping"]["sex_mapped"]

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()

    def _run_cohort(y2_subset, label, bl_df=None):
        df_prep, y_prep, present_cols = prep_longitudinal_df(
            y2_subset, roi_columns, target_col, id_col, cov_cols, sex_col,
            age_col_bl, age_col_long, label, bl_df=bl_df,
        )
        print(f"\n  {label}: n={len(df_prep)}, "
              f"mean={y_prep.mean():.1f}, still>={cutoff}: {(y_prep >= cutoff).sum()} ({(y_prep >= cutoff).mean()*100:.1f}%)")
        val_pairs = [(n, l, r) for n, l, r in bilateral_pairs if l in present_cols and r in present_cols]
        asym, uni = run_univariate_ai_fdr(df_prep, y_prep, present_cols, bilateral_pairs, label)
        print(f"  Running SVR...")
        res = run_cross_sectional_svr(df_prep, y_prep, env, roi_columns, val_pairs, label, n_splits)
        return {"r": res["r"], "all_true": res["all_true"], "all_pred": res["all_pred"],
                "uni": uni, "n": len(df_prep), "asym": asym}

    sep = "=" * 70
    print(f"\n{sep}\n  ANALYSIS 1: BASELINE-DEFINED (BL PQ-BC >= {cutoff} → Y2 data)\n{sep}")
    bl_high_ids = bl[bl[target_col] >= cutoff][id_col].values
    shared_ids = np.intersect1d(bl_high_ids, y2[id_col].values)
    y2_bldef = y2[y2[id_col].isin(shared_ids)].copy()
    bldef = _run_cohort(y2_bldef, "BL-defined at Y2", bl_df=bl)
    print(f"  SVR r = {bldef['r']:.4f}")

    print(f"\n{sep}\n  ANALYSIS 2: YEAR 2-DEFINED (Y2 PQ-BC >= {cutoff})\n{sep}")
    y2_y2def = y2[y2[target_col] >= cutoff].copy()
    y2def = _run_cohort(y2_y2def, "Y2-defined", bl_df=bl)
    print(f"  SVR r = {y2def['r']:.4f}")

    result = {"bldef": bldef, "y2def": y2def}

    if include_worseners:
        print(f"\n{sep}\n  ANALYSIS 3: WORSENERS (delta >= 0.5 SD)\n{sep}")
        # Consistent with worsening classifier: delta >= 0.5 * full BL SD
        shared_ids = np.intersect1d(bl[id_col].values, y2[id_col].values)
        bl_sc = bl.drop_duplicates(id_col).set_index(id_col)[target_col].reindex(shared_ids)
        y2_sc = y2.drop_duplicates(id_col).set_index(id_col)[target_col].reindex(shared_ids)
        full_sd = bl[target_col].dropna().std()
        threshold_w = 0.5 * full_sd
        delta_w = y2_sc - bl_sc
        worsen_mask = (delta_w >= threshold_w) & bl_sc.notna() & y2_sc.notna()
        worsener_ids = shared_ids[worsen_mask.values]
        y2_worseners = y2[y2[id_col].isin(worsener_ids)].copy()
        print(f"  Worseners (delta >= 0.5 SD = {threshold_w:.1f} pts): n={len(y2_worseners)}")
        worseners = _run_cohort(y2_worseners, "Worseners", bl_df=bl)
        print(f"  SVR r = {worseners['r']:.4f}")
        result["worseners"] = worseners

    print(f"\n{sep}\n  COMPARISON\n{sep}")
    print(f"  {'Approach':<25} {'r':>8} {'n':>6}")
    print(f"  {'-'*40}")
    print(f"  {'BL-defined':<25} {bldef['r']:>+8.4f} {bldef['n']:>6}")
    print(f"  {'Y2-defined':<25} {y2def['r']:>+8.4f} {y2def['n']:>6}")
    if include_worseners:
        print(f"  {'Worseners':<25} {result['worseners']['r']:>+8.4f} {result['worseners']['n']:>6}")

    return result


def build_y2_high_df(long_df, target_col, env, cutoff, tp_col, roi_columns,
                     cohort_mode="y2_defined"):
    """Build the Y2-defined high-severity dataframe ready for SVR.

    Args:
        cohort_mode: 'y2_defined' (Y2 >= cutoff) or 'worseners' (delta >= 0.5 SD from BL to Y2).
            Worseners are selected from the full paired sample regardless of absolute cutoff,
            consistent with the worsening classifier definition (mcid_factor=0.5).

    Returns (y2_high_df, y_vals, bilateral_pairs, valid_pairs, present_cols).
    """
    import numpy as np, copy

    id_col = env.configs.data["columns"]["mapping"]["id"]
    year2_tp = env.configs.data["timepoints"]["year2"]
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    cov_cols = env.configs.regression["covariates"]["columns"]
    age_long_col = "demo_brthdat_v2_l"
    age_bl_col = cov_cols[0]
    sex_col = env.configs.data["columns"]["mapping"].get("sex_mapped", "sex_mapped")
    scanner_col = env.configs.data["columns"]["mapping"]["scanner_model"]

    y2_all = long_df[long_df[tp_col] == year2_tp].copy()
    bl_all = long_df[long_df[tp_col] == baseline_tp].copy()

    present_y2 = [c for c in roi_columns if c in y2_all.columns]

    if cohort_mode == "worseners":
        # Select paired subjects with delta >= 0.5 SD (matches worsening classifier)
        shared_ids = np.intersect1d(bl_all[id_col].values, y2_all[id_col].values)
        bl_p = bl_all[bl_all[id_col].isin(shared_ids)].drop_duplicates(id_col).set_index(id_col)
        y2_p = y2_all[y2_all[id_col].isin(shared_ids)].drop_duplicates(id_col).set_index(id_col)
        full_sd = bl_all[target_col].dropna().std()
        threshold = 0.5 * full_sd
        bl_sc = bl_p[target_col].reindex(shared_ids)
        y2_sc = y2_p[target_col].reindex(shared_ids)
        delta = y2_sc - bl_sc
        worsen_mask = (delta >= threshold) & bl_sc.notna() & y2_sc.notna()
        worsen_ids = shared_ids[worsen_mask.values]
        y2_high = y2_all[y2_all[id_col].isin(worsen_ids)].copy()
        print(f"Worseners (delta >= 0.5 SD = {threshold:.1f} pts): n = {len(y2_high)}")
    else:
        y2_high = y2_all[y2_all[target_col] >= cutoff].copy()
        print(f"Y2-defined high-severity (PQ-BC >= {cutoff}): n = {len(y2_high)}")

    mask_valid = y2_high[present_y2].notna().all(axis=1) & y2_high[target_col].notna()
    y2_high = y2_high[mask_valid].reset_index(drop=True)

    if age_long_col in y2_high.columns:
        y2_high[age_bl_col] = y2_high[age_long_col]
    if y2_high[sex_col].isna().any():
        bl_sex = bl_all.drop_duplicates(id_col).set_index(id_col)[sex_col]
        y2_high[sex_col] = y2_high[sex_col].fillna(y2_high[id_col].map(bl_sex))

    cov_pres = [c for c in cov_cols if c in y2_high.columns]
    cov_valid = y2_high[cov_pres].notna().all(axis=1)
    y2_high = y2_high[cov_valid].reset_index(drop=True)

    y_vals = y2_high[target_col].values.astype(float)
    bilateral_pairs_local, _ = extract_bilateral_pairs(env.configs.data,
        env.configs.regression.get("roi_networks", ["dopamine_core"]))
    valid_pairs_local = [(n, l, r) for n, l, r in bilateral_pairs_local if l in present_y2 and r in present_y2]

    return y2_high, y_vals, bilateral_pairs_local, valid_pairs_local, present_y2


def run_prospective_svr(long_df, bilateral_pairs, target_col, env, cutoff, n_splits, n_perms,
                        n_bootstrap, seed, tp_col, cohort_mode="y2_defined"):
    """Baseline brain → Year 2 severity (prospective SVR) for Y2-defined high-severity subjects.

    Args:
        cohort_mode: 'y2_defined' (Y2 >= cutoff) or 'worseners' (delta >= 0.5 SD from BL to Y2).
            Worseners are selected from the full paired sample regardless of absolute cutoff,
            consistent with the worsening classifier definition (mcid_factor=0.5).

    Returns dict: r_prosp, all_true, all_pred, uni_results, boot_ci, perm_p, n.
    """
    import numpy as np, pandas as pd
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    from sklearn.linear_model import LinearRegression
    from core.regression.univariate import compute_asymmetry_features

    id_col = env.configs.data["columns"]["mapping"]["id"]
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp = env.configs.data["timepoints"]["year2"]
    cov_cols = env.configs.regression["covariates"]["columns"]
    age_bl_col = cov_cols[0]
    age_long_col = "demo_brthdat_v2_l"
    sex_col = env.configs.data["columns"]["mapping"].get("sex_mapped", "sex_mapped")

    bl_df = long_df[long_df[tp_col] == baseline_tp].copy()
    y2_df = long_df[long_df[tp_col] == year2_tp].copy()

    if cohort_mode == "worseners":
        # Select paired subjects with delta >= 0.5 SD (matches worsening classifier)
        shared_ids = np.intersect1d(bl_df[id_col].values, y2_df[id_col].values)
        bl_sc = bl_df.drop_duplicates(id_col).set_index(id_col)[target_col].reindex(shared_ids)
        y2_sc = y2_df.drop_duplicates(id_col).set_index(id_col)[target_col].reindex(shared_ids)
        full_sd = bl_df[target_col].dropna().std()
        threshold = 0.5 * full_sd
        delta = y2_sc - bl_sc
        worsen_mask = (delta >= threshold) & bl_sc.notna() & y2_sc.notna()
        y2_high_ids = shared_ids[worsen_mask.values]
        print(f"Worseners (delta >= 0.5 SD = {threshold:.1f} pts): n = {len(y2_high_ids)}")
    else:
        y2_high_ids = y2_df[y2_df[target_col] >= cutoff][id_col].values
        print(f"Y2 high-severity (PQ-BC >= {cutoff}): n = {len(y2_high_ids)}")

    # Get their baseline rows
    bl_prosp = bl_df[bl_df[id_col].isin(y2_high_ids)].copy()
    y2_sc = y2_df[y2_df[id_col].isin(y2_high_ids)][[id_col, target_col]].rename(columns={target_col: "y2_severity"})
    bl_prosp = bl_prosp.merge(y2_sc, on=id_col, how="inner")

    from core.tsne.embeddings import get_roi_columns_from_config
    roi_nets = env.configs.regression.get("roi_networks", ["dopamine_core"])
    roi_columns = get_roi_columns_from_config(env.configs.data, roi_nets)
    present_bl = [c for c in roi_columns if c in bl_prosp.columns]

    img_valid = bl_prosp[present_bl].notna().all(axis=1)
    cov_valid = bl_prosp[cov_cols].notna().all(axis=1) if all(c in bl_prosp.columns for c in cov_cols) else pd.Series([True]*len(bl_prosp))
    bl_prosp = bl_prosp[img_valid & cov_valid.values & bl_prosp["y2_severity"].notna()].reset_index(drop=True)
    print(f"Valid baseline imaging: n = {len(bl_prosp)}")

    if age_long_col in bl_prosp.columns:
        bl_prosp[age_bl_col] = bl_prosp[age_long_col].fillna(bl_prosp[age_bl_col])

    # Compute BL AI features
    X_bl = bl_prosp[present_bl].values.astype(float)
    valid_pairs_p = [(n, l, r) for n, l, r in bilateral_pairs if l in present_bl and r in present_bl]
    asym_bl = compute_asymmetry_features(X_bl, present_bl, valid_pairs_p)
    ai_names_p = sorted(k for k in asym_bl if k.endswith("_AI"))

    y_y2 = bl_prosp["y2_severity"].values.astype(float)
    # Residualize Y2 severity for BL age + sex
    # Exclude sex_col from numeric covariates — encode separately to avoid string→float error
    cov_present_p = [c for c in cov_cols if c in bl_prosp.columns and c != sex_col]
    cov_parts = []
    if cov_present_p:
        cov_parts.append(bl_prosp[cov_present_p].values.astype(float))
    if sex_col in bl_prosp.columns:
        sex_enc = pd.Categorical(bl_prosp[sex_col]).codes.astype(float).reshape(-1, 1)
        cov_parts.append(sex_enc)
    if cov_parts:
        cov_mat = np.column_stack(cov_parts)
        rm = LinearRegression().fit(cov_mat, y_y2)
        y_y2_r = y_y2 - rm.predict(cov_mat)
    else:
        y_y2_r = y_y2

    # Univariate: BL AI → Y2 PQ-BC
    uni_rs, uni_ps, uni_names = [], [], []
    for ai_name in ai_names_p:
        r_v, p_v = pearsonr(asym_bl[ai_name], y_y2_r)
        uni_rs.append(r_v); uni_ps.append(p_v); uni_names.append(ai_name)
    _, p_fdr, _, _ = multipletests(uni_ps, method="fdr_bh")
    uni_results = {n: {"r": r, "p": p, "p_fdr": f} for n, r, p, f in zip(uni_names, uni_rs, uni_ps, p_fdr)}
    print(f"\nUnivariate BL AI → Y2 PQ-BC (n={len(bl_prosp)}, residualized):")
    for n, r, p, f in zip(uni_names, uni_rs, uni_ps, p_fdr):
        sig = "***" if f < 0.001 else ("**" if f < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {n:<20} r={r:+.4f} p={p:.4f} p_fdr={f:.4f} {sig}")

    # SVR: BL brain → Y2 severity
    print(f"\nRunning BL → Y2 SVR (n={len(bl_prosp)})...")
    valid_p2 = [(n, l, r) for n, l, r in bilateral_pairs if l in present_bl and r in present_bl]
    res_prosp = run_cross_sectional_svr(bl_prosp, y_y2_r, env, roi_columns, valid_p2,
                                        "prospective_bl2y2", n_splits)

    # Permutation test: shuffle training labels per fold, refit SVR each permutation.
    # perm_and_boot_svr expects keys Xtr/Xte/ytr/yte; adapt from run_cross_sectional_svr's
    # X_train/X_test/y_train/y_test keys.
    fold_data_adapted = [
        {"Xtr": fd["X_train"], "Xte": fd["X_test"],
         "ytr": fd["y_train"], "yte": fd["y_test"]}
        for fd in res_prosp["fold_data"]
    ]
    boot_rs_arr, null_rs_arr, p_perm = perm_and_boot_svr(
        fold_data_adapted, res_prosp["all_true"], res_prosp["all_pred"],
        n_perms=n_perms, n_boot=n_bootstrap, seed=seed, label="prospective",
    )
    boot_ci = np.percentile(boot_rs_arr, [2.5, 97.5])

    print(f"\nProspective (BL brain → Y2 PQ-BC): r = {res_prosp['r']:.4f}, p_perm = {p_perm:.4f}")
    print(f"Bootstrap 95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")

    return {"r": res_prosp["r"], "p_perm": p_perm, "boot_ci": boot_ci,
            "all_true": res_prosp["all_true"], "all_pred": res_prosp["all_pred"],
            "uni_results": uni_results, "n": len(bl_prosp)}


def run_asymmetry_quantification(long_df, bilateral_pairs, target_col, env,
                                  n_perms, n_bootstrap, seed, cutoff, tp_col):
    """Developmental asymmetry quantification: PQ-BC trajectory, BL AI gap,
    developmental change, and cross-sectional SVR at multiple timepoints.

    Returns results dict.
    """
    import numpy as np
    from scipy.stats import ttest_ind, ttest_rel, pearsonr

    id_col = env.configs.data["columns"]["mapping"]["id"]
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp = env.configs.data["timepoints"]["year2"]
    year4_tp = env.configs.data["timepoints"]["year4"]
    cov_cols = env.configs.regression["covariates"]["columns"]

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()
    y4 = long_df[long_df[tp_col] == year4_tp].copy() if year4_tp in long_df[tp_col].values else None

    from core.tsne.embeddings import get_roi_columns_from_config
    roi_nets = env.configs.regression.get("roi_networks", ["dopamine_core"])
    roi_columns = get_roi_columns_from_config(env.configs.data, roi_nets)
    present_cols = [c for c in roi_columns if c in bl.columns]
    valid_pairs_all = [(n, l, r) for n, l, r in bilateral_pairs if l in present_cols and r in present_cols]

    # ── 0. PQ-BC symptom trajectory ──
    shared_bl_y2 = np.intersect1d(bl[id_col].values, y2[id_col].values)
    bl_2 = bl[bl[id_col].isin(shared_bl_y2)].sort_values(id_col).reset_index(drop=True)
    y2_2 = y2[y2[id_col].isin(shared_bl_y2)].sort_values(id_col).reset_index(drop=True)

    X_bl2 = bl_2[present_cols].values.astype(float)
    X_y22 = y2_2[present_cols].values.astype(float)
    valid2 = np.all(np.isfinite(X_bl2), axis=1) & np.all(np.isfinite(X_y22), axis=1)
    X_bl2 = X_bl2[valid2]; X_y22 = X_y22[valid2]
    bl_2v = bl_2[valid2].reset_index(drop=True); y2_2v = y2_2[valid2].reset_index(drop=True)

    asym_bl2 = compute_asymmetry_features(X_bl2, present_cols, valid_pairs_all)
    asym_y22 = compute_asymmetry_features(X_y22, present_cols, valid_pairs_all)
    sev_bl2 = bl_2v[target_col].values.astype(float)
    sev_y2_2 = y2_2v[target_col].values.astype(float)

    ctrl_m = sev_bl2 == 0; high_m = sev_bl2 >= cutoff
    t_sym, p_sym = ttest_rel(sev_bl2[high_m], sev_y2_2[high_m])
    print(f"High group BL→Y2: t={t_sym:.1f}, p={p_sym:.1e}")

    # ── 1. BL AI gap (permutation + bootstrap) ──
    ctrl_ai = asym_bl2["pallidum_AI"][ctrl_m]; high_ai = asym_bl2["pallidum_AI"][high_m]
    observed_gap = ctrl_ai.mean() - high_ai.mean()
    rng = np.random.RandomState(seed)
    n_ctrl = ctrl_m.sum(); n_high_g = high_m.sum()
    combined = np.concatenate([ctrl_ai, high_ai])
    null_gaps = np.array([
        rng.permutation(combined)[:n_ctrl].mean() - rng.permutation(combined)[n_ctrl:n_ctrl+n_high_g].mean()
        for _ in range(n_perms)])
    p_gap_perm = (np.sum(null_gaps >= observed_gap) + 1) / (n_perms + 1)
    boot_gaps = np.array([
        rng.choice(ctrl_ai, len(ctrl_ai), replace=True).mean() -
        rng.choice(high_ai, len(high_ai), replace=True).mean()
        for _ in range(n_bootstrap)])
    ci_gap = np.percentile(boot_gaps, [2.5, 97.5])
    print(f"\nBL AI gap: {observed_gap:.5f}, p_perm={p_gap_perm:.4f}, 95% CI=[{ci_gap[0]:.5f}, {ci_gap[1]:.5f}]")

    # ── 2. Developmental change BL→Y2 ──
    ai_names_all = sorted(k for k in asym_bl2 if k.endswith("_AI"))
    dev_results = {}
    for ai_name in ai_names_all:
        ai_bl_g = asym_bl2[ai_name]; ai_y2_g = asym_y22[ai_name]
        t_dev, p_dev = ttest_rel(ai_bl_g, ai_y2_g)
        dev_results[ai_name] = {"bl_mean": ai_bl_g.mean(), "y2_mean": ai_y2_g.mean(),
                                 "t": t_dev, "p": p_dev, "d": (ai_y2_g.mean() - ai_bl_g.mean()) / np.std(np.concatenate([ai_bl_g, ai_y2_g]))}
        sig = "*" if p_dev < 0.05 else ""
        print(f"  {ai_name}: BL={ai_bl_g.mean():+.4f} → Y2={ai_y2_g.mean():+.4f} (p={p_dev:.4f}{sig})")

    return {"dev_results": dev_results, "gap_result": {"observed": observed_gap, "p_perm": p_gap_perm, "ci": ci_gap},
            "sev_bl2": sev_bl2, "sev_y2_2": sev_y2_2, "ctrl_m": ctrl_m, "high_m": high_m,
            "asym_bl2": asym_bl2, "asym_y22": asym_y22, "ai_names_all": ai_names_all,
            "bl_2v": bl_2v, "y2_2v": y2_2v}


def run_delta_svr(long_df, bilateral_pairs, target_col, env, n_splits, seed, tp_col):
    """Continuous SVR: baseline brain → ΔPQ-BC (Year 2 − Baseline). Full paired sample, no bin filter.

    Returns dict with keys: r, p, all_true, all_pred, n, fold_data.
    """
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp    = env.configs.data["timepoints"]["year2"]
    id_col      = env.configs.data["columns"]["mapping"]["id"]

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()
    shared_ids = np.intersect1d(bl[id_col].values, y2[id_col].values)

    bl_p = bl[bl[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)
    y2_p = y2[y2[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)

    bl_scores = bl_p[target_col].values.astype(float)
    y2_scores = y2_p[target_col].values.astype(float)
    valid = np.isfinite(bl_scores) & np.isfinite(y2_scores)
    bl_p = bl_p[valid].reset_index(drop=True)
    delta_y = (y2_scores - bl_scores)[valid]

    # ── Fill longitudinal age + drop NaN covariates before SVR ────────────
    cov_cfg = env.configs.regression.get("covariates", {})
    cov_cols = cov_cfg.get("columns", [])
    age_long_col = "demo_brthdat_v2_l"
    if cov_cols and age_long_col in bl_p.columns:
        bl_p = bl_p.copy()
        bl_p[cov_cols[0]] = bl_p[age_long_col].fillna(bl_p[cov_cols[0]])
    if cov_cfg.get("residualize", False) and cov_cols:
        cov_present = [c for c in cov_cols if c in bl_p.columns]
        if cov_present:
            cov_valid = bl_p[cov_present].notna().all(axis=1).values
            n_drop = (~cov_valid).sum()
            if n_drop:
                print(f"  Dropping {n_drop} subjects with NaN covariates")
            bl_p = bl_p[cov_valid].reset_index(drop=True)
            delta_y = delta_y[cov_valid]
    # ──────────────────────────────────────────────────────────────────────

    present_cols = [c for c in long_df.columns if c in bl_p.columns
                    and any(lc in c or rc in c for _, lc, rc in bilateral_pairs)]
    valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs
                   if l in present_cols and r in present_cols]

    print(f"Delta SVR: n={len(bl_p)}, delta mean={delta_y.mean():.2f}, sd={delta_y.std():.2f}")
    return run_cross_sectional_svr(bl_p, delta_y, env, present_cols, valid_pairs,
                                   "delta_svr", n_splits)


def run_cross_sectional_svc(df, y_binary, env, feature_cols, bilateral_pairs,
                             label="", n_splits=None, n_downsample_iters=30):
    """Cross-sectional nested CV SVC with per-fold ComBat.

    Mirrors run_cross_sectional_svr but:
    - y_binary is 0/1 — no residualization, no y scaling
    - Uses SVC(kernel='linear', C=1.0, probability=True)
    - Stratifies CV on y_binary directly (not qcut)
    - When n_downsample_iters > 0: iterative majority-class downsampling to minority size,
      averaging predicted probabilities across iterations (mirrors main SVM pipeline).
      When n_downsample_iters == 0: falls back to class_weight='balanced' (old behavior).

    Returns dict with keys: auc, balanced_acc, all_true, all_proba, n.
    """
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    reg_config = env.configs.regression
    harm_config = env.configs.harmonize
    feature_transform = _get_feature_transform(env)
    seed = env.configs.run.get("seed", 42)

    if n_splits is None:
        n_splits = reg_config.get("cv", {}).get("n_outer_splits", 5)

    # Filter to rows with valid features
    present_cols = [c for c in feature_cols if c in df.columns]
    X_raw = df[present_cols].values.astype(float)
    y_binary = np.asarray(y_binary)
    valid = np.all(np.isfinite(X_raw), axis=1) & np.isfinite(y_binary.astype(float))
    df = df[valid].reset_index(drop=True)
    y_binary = y_binary[valid]
    X_raw = X_raw[valid]

    if len(y_binary) < 20:
        print(f"  [{label}] Too few subjects after filtering: n={len(y_binary)}")
        return {"auc": np.nan, "balanced_acc": np.nan, "all_true": np.array([]),
                "all_proba": np.array([]), "n": len(y_binary)}

    site_col = harm_config.get("site_column", "mri_info_manufacturer")

    family_groups = None
    if "rel_family_id" in df.columns:
        family_groups = pd.to_numeric(df["rel_family_id"], errors="coerce").values
        missing = np.isnan(family_groups)
        if missing.any():
            max_id = np.nanmax(family_groups) if (~missing).any() else 0
            family_groups[missing] = np.arange(max_id + 1, max_id + 1 + missing.sum())

    if family_groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (df, y_binary, family_groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (df, y_binary)

    all_true, all_proba = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(*split_args)):
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_te = df.iloc[test_idx].reset_index(drop=True)
        X_tr_raw = X_raw[train_idx]
        X_te_raw = X_raw[test_idx]
        y_tr = y_binary[train_idx].copy()
        y_te = y_binary[test_idx].copy()

        # Per-fold ComBat: fit on train only, apply to test (no leakage).
        try:
            from neuroHarmonize import harmonizationLearn, harmonizationApply

            cov_cols_harm = [c for c in harm_config.get("covariates", [])
                             if c in df_tr.columns and df_tr[c].notna().sum() > 0]

            def _make_covars(df_sub, cols, keep_cols=None):
                cov = df_sub[[site_col] + cols].copy().rename(columns={site_col: "SITE"})
                for c in cols:
                    if not pd.api.types.is_numeric_dtype(cov[c]):
                        cov[c] = pd.Categorical(cov[c]).codes.astype(float)
                    else:
                        cov[c] = cov[c].astype(float)
                if keep_cols is None:
                    for c in list(cols):
                        if cov[c].nunique() <= 1:
                            cov = cov.drop(columns=c)
                else:
                    cov = cov[[c for c in keep_cols if c in cov.columns]]
                return cov

            tr_covars = _make_covars(df_tr, cov_cols_harm)
            te_covars = _make_covars(df_te, cov_cols_harm, keep_cols=list(tr_covars.columns))

            feat_vars = np.var(X_tr_raw, axis=0)
            valid_feat = feat_vars > 1e-10
            X_tr_v = X_tr_raw[:, valid_feat]
            X_te_v = X_te_raw[:, valid_feat]
            valid_cols_fold = [present_cols[i] for i, v in enumerate(valid_feat) if v]

            tr_nan = tr_covars.isna().any(axis=1).values
            te_nan = te_covars.isna().any(axis=1).values

            eb = harm_config.get("empirical_bayes", True)
            combat_model, X_tr_harm = harmonizationLearn(
                X_tr_v[~tr_nan], tr_covars[~tr_nan], eb=eb,
            )
            X_te_harm = harmonizationApply(X_te_v[~te_nan], te_covars[~te_nan], combat_model)

            X_tr_h = X_tr_harm
            X_te_h = X_te_harm
            y_tr_h = y_tr[~tr_nan]
            y_te_h = y_te[~te_nan]
            bilateral_pairs_fold = [(n, l, r) for n, l, r in bilateral_pairs
                                    if l in valid_cols_fold and r in valid_cols_fold]
        except Exception as exc:
            logger.warning(f"ComBat fold {fold_idx} failed ({exc}), using raw features")
            X_tr_h, X_te_h = X_tr_raw, X_te_raw
            y_tr_h, y_te_h = y_tr, y_te
            valid_cols_fold = present_cols
            bilateral_pairs_fold = bilateral_pairs

        # Prepare features (asymmetry or raw)
        X_tr_feat, _ = _prepare_features(X_tr_h, valid_cols_fold, bilateral_pairs_fold, feature_transform)
        X_te_feat, _ = _prepare_features(X_te_h, valid_cols_fold, bilateral_pairs_fold, feature_transform)

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_feat)
        X_te_s = scaler.transform(X_te_feat)

        # ── Classifier: downsampled iterations or class-weighted fallback ─
        if n_downsample_iters > 0:
            rng = np.random.default_rng(seed + fold_idx)
            pos_idx = np.where(y_tr_h == 1)[0]
            neg_idx = np.where(y_tr_h == 0)[0]
            min_n = min(len(pos_idx), len(neg_idx))
            proba_runs = []
            for _ in range(n_downsample_iters):
                chosen_pos = rng.choice(pos_idx, min_n, replace=False)
                chosen_neg = rng.choice(neg_idx, min_n, replace=False)
                idx_bal = np.concatenate([chosen_pos, chosen_neg])
                svc = SVC(kernel="linear", C=1.0, probability=True,
                          random_state=int(rng.integers(1 << 31)))
                svc.fit(X_tr_s[idx_bal], y_tr_h[idx_bal])
                proba_runs.append(svc.predict_proba(X_te_s)[:, 1])
            proba = np.mean(proba_runs, axis=0)
        else:
            svc = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True,
                      random_state=seed + fold_idx)
            svc.fit(X_tr_s, y_tr_h)
            proba = svc.predict_proba(X_te_s)[:, 1]
        # ──────────────────────────────────────────────────────────────────

        all_true.extend(y_te_h)
        all_proba.extend(proba)

    all_true = np.array(all_true)
    all_proba = np.array(all_proba)
    all_pred_bin = (all_proba >= 0.5).astype(int)

    auc = roc_auc_score(all_true, all_proba)
    bal_acc = balanced_accuracy_score(all_true, all_pred_bin)

    if label:
        print(f"  [{label}] AUC={auc:.3f}, balanced_acc={bal_acc:.3f}, n={len(y_binary)}")

    return {"auc": auc, "balanced_acc": bal_acc,
            "all_true": all_true, "all_proba": all_proba,
            "n": len(y_binary)}


def run_worsening_classifier(long_df, bilateral_pairs, target_col, env, n_splits, seed,
                              tp_col, mcid_factor=0.5):
    """Binary SVC: baseline brain → worsened/stable (MCID = mcid_factor × SD).

    Returns dict: auc, balanced_acc, all_true, all_proba, n, threshold, n_worsened, n_stable.
    """
    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp    = env.configs.data["timepoints"]["year2"]
    id_col      = env.configs.data["columns"]["mapping"]["id"]

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()
    shared_ids = np.intersect1d(bl[id_col].values, y2[id_col].values)
    bl_p = bl[bl[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)
    y2_p = y2[y2[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)

    bl_scores = bl_p[target_col].values.astype(float)
    y2_scores = y2_p[target_col].values.astype(float)
    valid = np.isfinite(bl_scores) & np.isfinite(y2_scores)
    bl_p = bl_p[valid].reset_index(drop=True)
    delta = (y2_scores - bl_scores)[valid]

    full_sd   = bl[target_col].dropna().std()
    threshold = mcid_factor * full_sd
    worsened  = (delta >= threshold).astype(int)
    n_worse, n_stable = int(worsened.sum()), int((worsened == 0).sum())
    print(f"Classifier threshold: {mcid_factor}×SD = {threshold:.1f} pts")
    print(f"Worsened: n={n_worse} ({100*n_worse/len(worsened):.1f}%)  Stable: n={n_stable}")

    present_cols = [c for c in long_df.columns if c in bl_p.columns
                    and any(lc in c or rc in c for _, lc, rc in bilateral_pairs)]
    valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs
                   if l in present_cols and r in present_cols]

    result = run_cross_sectional_svc(bl_p, worsened, env, present_cols, valid_pairs,
                                     "worsening_svc", n_splits)
    result["threshold"] = threshold
    result["n_worsened"] = n_worse
    result["n_stable"] = n_stable
    return result


def run_multi_algo_worsening_classifier(long_df, bilateral_pairs, target_col, env,
                                        n_splits, seed, tp_col, mcid_factor=0.5):
    """Compare SVC, RF, and MLP classifiers for worsening prediction.

    Uses the same paired BL/Y2 setup as run_worsening_classifier. Per-fold ComBat
    and asymmetry preprocessing are handled inside run_cross_sectional_svc for SVC;
    RF and MLP use a shared ComBat-on-train approach via _run_fold_clf.

    Returns dict:
        svc, rf, mlp  — each: {auc, balanced_acc, n_worsened, n_stable}
        comparison_df — pd.DataFrame sortable by AUC
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler as _SS
    from neuroHarmonize import harmonizationLearn, harmonizationApply

    baseline_tp = env.configs.data["timepoints"]["baseline"]
    year2_tp    = env.configs.data["timepoints"]["year2"]
    id_col      = env.configs.data["columns"]["mapping"]["id"]
    harm_config = env.configs.harmonize
    site_col    = harm_config.get("site_column", "mri_info_manufacturer")
    feature_transform = _get_feature_transform(env)

    bl = long_df[long_df[tp_col] == baseline_tp].copy()
    y2 = long_df[long_df[tp_col] == year2_tp].copy()
    shared_ids = np.intersect1d(bl[id_col].values, y2[id_col].values)
    bl_p = bl[bl[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)
    y2_p = y2[y2[id_col].isin(shared_ids)].sort_values(id_col).reset_index(drop=True)

    bl_scores = bl_p[target_col].values.astype(float)
    y2_scores = y2_p[target_col].values.astype(float)
    valid = np.isfinite(bl_scores) & np.isfinite(y2_scores)
    bl_p = bl_p[valid].reset_index(drop=True)
    delta = (y2_scores - bl_scores)[valid]

    full_sd   = bl[target_col].dropna().std()
    threshold = mcid_factor * full_sd
    worsened  = (delta >= threshold).astype(int)
    n_worse, n_stable = int(worsened.sum()), int((worsened == 0).sum())
    print(f"Multi-algo classifier threshold: {mcid_factor}×SD = {threshold:.1f} pts")
    print(f"Worsened: n={n_worse} ({100*n_worse/len(worsened):.1f}%)  Stable: n={n_stable}")

    present_cols = [c for c in long_df.columns if c in bl_p.columns
                    and any(lc in c or rc in c for _, lc, rc in bilateral_pairs)]
    valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs
                   if l in present_cols and r in present_cols]

    # ── SVC via existing function (handles ComBat + asymmetry + downsampling) ──
    print("\n  [SVC]")
    svc_res = run_cross_sectional_svc(bl_p, worsened, env, present_cols, valid_pairs,
                                      "worsening_svc", n_splits, n_downsample_iters=30)

    # ── Drop rows with NaN in any feature (RF/MLP don't handle NaN natively) ─
    X_raw_full = bl_p[present_cols].values.astype(float)
    feat_valid = np.all(np.isfinite(X_raw_full), axis=1)
    if (~feat_valid).any():
        print(f"  Dropping {(~feat_valid).sum()} subjects with NaN features for RF/MLP")
        bl_p = bl_p[feat_valid].reset_index(drop=True)
        worsened = worsened[feat_valid]
        X_raw_full = X_raw_full[feat_valid]

    # ── Shared CV setup for RF + MLP (same fold structure as SVC) ──────────
    family_groups = None
    if "rel_family_id" in bl_p.columns:
        family_groups = pd.to_numeric(bl_p["rel_family_id"], errors="coerce").values
        missing = np.isnan(family_groups)
        if missing.any():
            max_id = np.nanmax(family_groups) if (~missing).any() else 0
            family_groups[missing] = np.arange(max_id + 1, max_id + 1 + missing.sum())

    if family_groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (bl_p, worsened, family_groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_args = (bl_p, worsened)

    def _run_tree_mlp_folds(algo_name, make_clf_fn):
        all_true, all_proba = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(*split_args)):
            df_tr = bl_p.iloc[train_idx].reset_index(drop=True)
            df_te = bl_p.iloc[test_idx].reset_index(drop=True)
            X_tr_raw = X_raw_full[train_idx]
            X_te_raw = X_raw_full[test_idx]
            y_tr = worsened[train_idx].copy()
            y_te = worsened[test_idx].copy()

            # Per-fold ComBat
            try:
                cov_cols_harm = [c for c in harm_config.get("covariates", [])
                                 if c in df_tr.columns and df_tr[c].notna().sum() > 0]

                def _make_covars(df_sub, cols, keep_cols=None):
                    cov = df_sub[[site_col] + cols].copy().rename(columns={site_col: "SITE"})
                    for c in cols:
                        if not pd.api.types.is_numeric_dtype(cov[c]):
                            cov[c] = pd.Categorical(cov[c]).codes.astype(float)
                        else:
                            cov[c] = cov[c].astype(float)
                    if keep_cols is None:
                        # Train side: drop constant columns and record survivors
                        for c in list(cols):
                            if cov[c].nunique() <= 1:
                                cov = cov.drop(columns=c)
                    else:
                        # Test side: use exactly the same columns as train
                        cov = cov[[c for c in keep_cols if c in cov.columns]]
                    return cov

                tr_covars = _make_covars(df_tr, cov_cols_harm)
                te_covars = _make_covars(df_te, cov_cols_harm, keep_cols=list(tr_covars.columns))
                feat_vars = np.var(X_tr_raw, axis=0)
                valid_feat = feat_vars > 1e-10
                X_tr_v = X_tr_raw[:, valid_feat]
                X_te_v = X_te_raw[:, valid_feat]
                valid_cols_fold = [present_cols[i] for i, v in enumerate(valid_feat) if v]
                tr_nan = tr_covars.isna().any(axis=1).values
                te_nan = te_covars.isna().any(axis=1).values
                eb = harm_config.get("empirical_bayes", True)
                combat_model, X_tr_harm = harmonizationLearn(
                    X_tr_v[~tr_nan], tr_covars[~tr_nan], eb=eb)
                X_te_harm = harmonizationApply(X_te_v[~te_nan], te_covars[~te_nan], combat_model)
                X_tr_h = X_tr_harm; X_te_h = X_te_harm
                y_tr_h = y_tr[~tr_nan]; y_te_h = y_te[~te_nan]
                bilateral_pairs_fold = [(n, l, r) for n, l, r in valid_pairs
                                        if l in valid_cols_fold and r in valid_cols_fold]
            except Exception as exc:
                logger.warning(f"ComBat fold {fold_idx} failed ({exc}), using raw features")
                X_tr_h, X_te_h = X_tr_raw, X_te_raw
                y_tr_h, y_te_h = y_tr, y_te
                valid_cols_fold = present_cols
                bilateral_pairs_fold = valid_pairs

            X_tr_feat, _ = _prepare_features(X_tr_h, valid_cols_fold, bilateral_pairs_fold, feature_transform)
            X_te_feat, _ = _prepare_features(X_te_h, valid_cols_fold, bilateral_pairs_fold, feature_transform)
            scaler = _SS()
            X_tr_s = scaler.fit_transform(X_tr_feat)
            X_te_s = scaler.transform(X_te_feat)

            clf = make_clf_fn(y_tr_h, fold_idx)
            clf.fit(X_tr_s, y_tr_h)
            all_true.extend(y_te_h)
            all_proba.extend(clf.predict_proba(X_te_s)[:, 1])

        all_true = np.array(all_true)
        all_proba = np.array(all_proba)
        all_pred_bin = (all_proba >= 0.5).astype(int)
        auc = roc_auc_score(all_true, all_proba)
        bal_acc = balanced_accuracy_score(all_true, all_pred_bin)
        print(f"  [{algo_name}] AUC={auc:.3f}, balanced_acc={bal_acc:.3f}")
        return {"auc": auc, "balanced_acc": bal_acc, "n_worsened": n_worse, "n_stable": n_stable}

    print("\n  [RF]")
    rf_res = _run_tree_mlp_folds(
        "RF",
        lambda y_tr_h, fold_idx: RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            max_features="sqrt", random_state=seed + fold_idx, n_jobs=-1),
    )

    print("\n  [MLP]")
    mlp_res = _run_tree_mlp_folds(
        "MLP",
        lambda y_tr_h, fold_idx: MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True,
            random_state=seed + fold_idx, max_iter=300),
    )

    comparison_df = pd.DataFrame([
        {"algorithm": "SVC (downsampled)", **{k: svc_res[k] for k in ["auc", "balanced_acc"]}},
        {"algorithm": "RandomForest",      **{k: rf_res[k]  for k in ["auc", "balanced_acc"]}},
        {"algorithm": "MLP",               **{k: mlp_res[k] for k in ["auc", "balanced_acc"]}},
    ]).sort_values("auc", ascending=False).reset_index(drop=True)

    print(f"\n  {'Algorithm':<22} {'AUC':>7} {'Bal.Acc':>9}")
    print(f"  {'-'*40}")
    for _, row in comparison_df.iterrows():
        print(f"  {row['algorithm']:<22} {row['auc']:>7.4f} {row['balanced_acc']:>9.4f}")

    return {
        "svc": svc_res, "rf": rf_res, "mlp": mlp_res,
        "comparison_df": comparison_df,
        "n_worsened": n_worse, "n_stable": n_stable, "threshold": threshold,
    }
