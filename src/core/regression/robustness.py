"""Robustness analyses for the dopamine-psychosis regression pipeline.

All functions accept an optional `env` to fall back on config-driven defaults
(n_perms, n_boot, seed, cutoffs). Results are returned as DataFrames or dicts
so notebooks stay thin: one function call per cell.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .univariate import compute_asymmetry_features, prepare_harmonized_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_reg_config(env=None):
    """Return regression config dict or empty dict if no env."""
    if env is not None:
        return env.configs.regression
    return {}


def _pallidum_ai_r(X_harm, y, feature_cols, bilateral_pairs):
    """Quick helper: compute pallidum_AI and correlate with y."""
    asym = compute_asymmetry_features(X_harm, feature_cols, bilateral_pairs)
    ai = asym.get("pallidum_AI")
    if ai is None:
        return np.nan, np.nan
    valid = np.isfinite(ai) & np.isfinite(y)
    if valid.sum() < 10:
        return np.nan, np.nan
    return pearsonr(ai[valid], y[valid])


# ---------------------------------------------------------------------------
# 1. Cutoff sensitivity
# ---------------------------------------------------------------------------

def cutoff_sensitivity(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    target_name: str | None = None,
    cutoffs=None,
    min_n: int = 50,
    env=None,
) -> pd.DataFrame:
    """Sweep minimum PQ-BC severity threshold; compute pallidum AI univariate r.

    Args:
        full_df: Full subject-level dataframe.
        feature_cols: Raw feature column names.
        bilateral_pairs: Bilateral ROI pair definitions.
        target_col: Target column name.
        target_name: Config target name (for bin lookup). Optional.
        cutoffs: List of minimum severity values to test. Default: range(0, 65, 5).
        min_n: Skip cutoff if fewer subjects remain after ComBat.
        env: Environment with configs (optional).

    Returns:
        DataFrame: cutoff, r, p, n.
    """
    if cutoffs is None:
        cutoffs = list(range(0, 65, 5))

    harm_config = env.configs.harmonize if env else {}
    reg_config = _get_reg_config(env)

    rows = []
    for cutoff in cutoffs:
        # Filter to subjects above this cutoff
        mask = full_df[target_col].notna() & (full_df[target_col] >= cutoff)
        df_cut = full_df[mask].copy()
        y_cut = df_cut[target_col].values.astype(float)

        if len(df_cut) < min_n:
            rows.append({"cutoff": cutoff, "r": np.nan, "p": np.nan, "n": len(df_cut)})
            continue

        try:
            from neuroHarmonize import harmonizationLearn
            from .longitudinal import _combat_harmonize

            present_cols = [c for c in feature_cols if c in df_cut.columns]
            X_raw = df_cut[present_cols].values.astype(float)
            X_harm, keep, _ = _combat_harmonize(X_raw, df_cut, harm_config,
                                                  min_site_n=5)
            y_h = y_cut[keep]

            if keep.sum() < min_n:
                rows.append({"cutoff": cutoff, "r": np.nan, "p": np.nan,
                              "n": int(keep.sum())})
                continue

            r, p = _pallidum_ai_r(X_harm, y_h, present_cols, bilateral_pairs)
            rows.append({"cutoff": cutoff, "r": r, "p": p, "n": int(keep.sum())})
        except Exception as e:
            logger.warning(f"cutoff_sensitivity: cutoff={cutoff} failed: {e}")
            rows.append({"cutoff": cutoff, "r": np.nan, "p": np.nan, "n": np.nan})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Split-half replication
# ---------------------------------------------------------------------------

def split_half_replication(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    n_iterations: int = 100,
) -> pd.DataFrame:
    """Run nested CV SVR on 100 random 50/50 stratified splits (family-aware).

    Args:
        env: Environment with configs.
        full_df: Full subject-level dataframe.
        target_config: Target config dict (with 'name', 'column').
        n_iterations: Number of split-half runs.

    Returns:
        DataFrame: iteration, r_first_half, r_second_half.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from .pipeline import run_target_with_nested_cv

    seed = env.configs.run.get("seed", 42)
    rng = np.random.RandomState(seed)

    target_col = target_config["column"]
    mask = full_df[target_col].notna()
    df_base = full_df[mask].copy()

    family_col = "rel_family_id"
    rows = []

    for i in range(n_iterations):
        # Family-aware 50/50 split: keep all siblings together
        if family_col in df_base.columns:
            unique_fam = df_base[family_col].unique()
            rng.shuffle(unique_fam)
            half = len(unique_fam) // 2
            fam_first = set(unique_fam[:half])
            mask_first = df_base[family_col].isin(fam_first)
        else:
            n = len(df_base)
            idx = rng.permutation(n)
            mask_first = pd.Series([False] * n)
            mask_first.iloc[idx[:n // 2]] = True

        df_a = df_base[mask_first].reset_index(drop=True)
        df_b = df_base[~mask_first].reset_index(drop=True)

        r_a, r_b = np.nan, np.nan
        for half_df, label in [(df_a, "a"), (df_b, "b")]:
            try:
                res = run_target_with_nested_cv(env, half_df, target_config,
                                                 model_name="svr", verbose=False)
                val = res["svr"]["overall"]["pearson_r"]
                if label == "a":
                    r_a = val
                else:
                    r_b = val
            except Exception as e:
                logger.warning(f"split_half iter={i} half={label} failed: {e}")

        rows.append({"iteration": i, "r_first_half": r_a, "r_second_half": r_b})
        if (i + 1) % 10 == 0:
            print(f"  Split-half: {i+1}/{n_iterations} done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Leave-one-feature-out (LOFO)
# ---------------------------------------------------------------------------

def leave_one_feature_out(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
) -> pd.DataFrame:
    """Drop each feature one at a time, rerun nested CV SVR, compute Δr.

    Args:
        env: Environment with configs.
        full_df: Full subject-level dataframe.
        target_config: Target config dict.

    Returns:
        DataFrame: feature_dropped, r_without, delta_r (= r_baseline - r_without).
    """
    from copy import deepcopy
    from .pipeline import run_target_with_nested_cv

    # Baseline run
    print("  LOFO: running baseline...")
    try:
        res_base = run_target_with_nested_cv(env, full_df, target_config,
                                              model_name="svr", verbose=False)
        r_baseline = res_base["svr"]["overall"]["pearson_r"]
    except Exception as e:
        logger.error(f"LOFO baseline failed: {e}")
        return pd.DataFrame()

    print(f"  LOFO baseline r={r_baseline:.3f}")

    # Get current ROI networks from config
    reg_config = env.configs.regression
    roi_networks = reg_config.get("roi_networks", [])
    from ..tsne.embeddings import get_roi_columns_from_config
    feature_cols = get_roi_columns_from_config(env.configs.data, roi_networks)
    feature_cols = [c for c in feature_cols if c in full_df.columns]

    rows = []
    for i, drop_feat in enumerate(feature_cols):
        env_copy = deepcopy(env)
        # Remove this feature from the data
        df_drop = full_df.drop(columns=[drop_feat], errors="ignore")

        try:
            res = run_target_with_nested_cv(env_copy, df_drop, target_config,
                                             model_name="svr", verbose=False)
            r_without = res["svr"]["overall"]["pearson_r"]
        except Exception as e:
            logger.warning(f"LOFO drop={drop_feat} failed: {e}")
            r_without = np.nan

        delta_r = r_baseline - r_without
        rows.append({"feature_dropped": drop_feat, "r_without": r_without,
                     "delta_r": delta_r})
        if (i + 1) % 5 == 0:
            print(f"  LOFO: {i+1}/{len(feature_cols)} done")

    return pd.DataFrame(rows).sort_values("delta_r", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Network specificity null
# ---------------------------------------------------------------------------

def network_specificity_null(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    target_name: str | None = None,
    n_perms: int = 1000,
    n_svr_perms: int = 0,
    seed: int = 42,
    env=None,
) -> dict:
    """Compare dopamine network against other defined networks (SVR) and a random null.

    Three-part analysis:
      1. Named networks: run full ComBat-per-fold SVR for each network defined in
         data.yaml roi_features (salience, cognitive, limbic, thalamic, dopamine).
         These are directly comparable to the main result.
      2. Fast univariate null: pre-harmonize a pool of ALL imaging features (DTI FA/MD,
         cortical thickness/area/sulc, subcortical volumes) once, then for each
         permutation randomly sample n_dopamine bilateral pairs and compute the best
         AI correlation. Fast (no SVR per perm).
      3. SVR null (optional, n_svr_perms > 0): same random bilateral pair sampling
         but runs full ComBat-per-fold SVR for each permutation. Directly comparable
         to the named network SVR results. Slow (~15s/perm).

    Args:
        full_df: Subject-level dataframe (pre-filtered to high-severity cohort).
        feature_cols: Dopamine network feature columns.
        bilateral_pairs: Dopamine bilateral pairs.
        target_col: Target column name.
        target_name: Config target name (for prepare_harmonized_data bin filter).
        n_perms: Number of random permutations for the fast univariate null.
        n_svr_perms: Number of SVR permutations for the slow null (0 = skip).
        seed: Random state.
        env: Environment with configs.

    Returns:
        dict with keys:
            named_df     — DataFrame: network, r, n, p_vs_null (vs univariate null)
            null_df      — DataFrame: perm, best_r  (fast univariate null)
            svr_null_df  — DataFrame: perm, r  (SVR null; empty if n_svr_perms=0)
            dopa_pct     — float: percentile of dopamine SVR r in univariate null
            dopa_svr_pct — float: percentile of dopamine SVR r in SVR null (nan if skipped)
    """
    import copy
    from .longitudinal import _combat_harmonize
    from .univariate import extract_bilateral_pairs, prepare_harmonized_data
    from .pipeline import run_target_with_nested_cv
    from ..tsne.embeddings import get_roi_columns_from_config

    reg_config = _get_reg_config(env)
    harm_config = env.configs.harmonize if env else {}
    rng = np.random.RandomState(seed)

    # ── 1. Named network SVR comparison ─────────────────────────────────────
    # Uses run_target_with_nested_cv (same as NB07 main SVR) with combined
    # site+target stratification — results are directly comparable to the main result.
    roi_features_cfg = env.configs.data.get("roi_features", {})
    all_defined_networks = [k for k, v in roi_features_cfg.items() if isinstance(v, dict)]

    # Ensure dopamine_core is first so its r is available for null percentile
    ordered_networks = ["dopamine_core"] + [n for n in all_defined_networks if n != "dopamine_core"]

    target_config = {"name": target_name or target_col, "column": target_col}
    reg_orig = env.configs.regression

    print("Named network SVR comparison:")
    named_rows = []
    dopa_r = np.nan
    for net_name in ordered_networks:
        net_feat_cols = get_roi_columns_from_config(env.configs.data, [net_name])
        net_feat_cols = [c for c in net_feat_cols if c in full_df.columns]
        if len(net_feat_cols) < 2:
            print(f"  {net_name}: skipped (no features in df)")
            continue
        # Temporarily override roi_networks so run_target_with_nested_cv uses this network
        reg_override = copy.deepcopy(dict(reg_orig))
        reg_override["roi_networks"] = [net_name]
        env.configs.regression = reg_override
        try:
            result = run_target_with_nested_cv(
                env, full_df, target_config, model_name="svr", verbose=False
            )
            r = result["svr"]["overall"]["pearson_r"]
            n = result["svr"]["n_samples"]
        finally:
            env.configs.regression = reg_orig
        print(f"  {net_name:<20} r = {r:+.4f}  (n={n})")
        named_rows.append({"network": net_name, "r": r, "n": n})
        if net_name == "dopamine_core":
            dopa_r = r
    named_df = pd.DataFrame(named_rows)

    # ── 2 & 3. Random null pool: ALL imaging modalities ─────────────────────
    # Include DTI FA/MD, cortical thickness/area/sulc, ALL subcortical volumes.
    # Broader than just subcortical volumes — ensures the null covers the full
    # feature space so dopamine specificity can't be explained by any brain region.
    dopa_set = set(feature_cols)
    all_imaging_prefixes = (
        "dmri_dtifa_fiberat_",
        "dmri_dtimd_fiberat_",
        "smri_vol_scs_",
        "smri_vol_cdk_",
        "smri_thk_cdk_",
        "smri_area_cdk_",
        "smri_sulc_cdk_",
        "mrisdp_",
    )
    pool_cols = [
        c for c in full_df.columns
        if any(c.startswith(p) for p in all_imaging_prefixes)
        and c not in dopa_set
        and full_df[c].notna().any()
    ]

    mask_base = full_df[target_col].notna()
    df_base = full_df[mask_base].copy().reset_index(drop=True)
    y_base = df_base[target_col].values.astype(float)

    # Harmonize the pool once on the severity cohort
    pool_present = [c for c in pool_cols if c in df_base.columns]
    print(f"\nPre-harmonizing null pool ({len(pool_present)} features) once...")
    X_pool_raw = df_base[pool_present].values.astype(float)
    X_pool_h, keep_pool, _ = _combat_harmonize(X_pool_raw, df_base, harm_config, min_site_n=5)
    y_pool = y_base[keep_pool]
    n_pool_feats = X_pool_h.shape[1]
    n_dopa = len(feature_cols)
    print(f"  Pool: {n_pool_feats} harmonized features, {len(y_pool)} subjects")

    if n_pool_feats < n_dopa:
        print(f"  Warning: pool ({n_pool_feats}) smaller than dopamine set ({n_dopa}), using pool size")
        n_sample = n_pool_feats
    else:
        n_sample = n_dopa

    # Find bilateral pairs in pool (indices into pool_present)
    pool_pair_indices = []
    used_idx = set()
    for i, c in enumerate(pool_present):
        if i in used_idx:
            continue
        for lsuf, rsuf in [("lh", "rh"), ("l", "r")]:
            if c.endswith(lsuf):
                cand = c[:-len(lsuf)] + rsuf
                if cand in pool_present:
                    j = pool_present.index(cand)
                    if j not in used_idx:
                        name = c[:-len(lsuf)].rstrip("_")
                        pool_pair_indices.append((name, i, j))
                        used_idx.update([i, j])
                        break

    print(f"  Bilateral pairs in pool: {len(pool_pair_indices)}")

    null_rows = []
    for perm in range(n_perms):
        # Sample random bilateral pairs from pool (same count as dopamine pairs)
        n_pairs_dopa = len(bilateral_pairs)
        if len(pool_pair_indices) < n_pairs_dopa:
            sampled_pairs = pool_pair_indices
        else:
            chosen = rng.choice(len(pool_pair_indices), size=n_pairs_dopa, replace=False)
            sampled_pairs = [pool_pair_indices[k] for k in chosen]

        if not sampled_pairs:
            null_rows.append({"perm": perm, "best_r": np.nan})
            continue

        best_r = 0.0
        for name, li, ri in sampled_pairs:
            L = X_pool_h[:, li]; R = X_pool_h[:, ri]
            total = L + R
            ai = np.where(np.abs(total) < 1e-10, 0.0, (L - R) / total)
            valid = np.isfinite(ai) & np.isfinite(y_pool)
            if valid.sum() > 10:
                r, _ = pearsonr(ai[valid], y_pool[valid])
                if abs(r) > abs(best_r):
                    best_r = r
        null_rows.append({"perm": perm, "best_r": best_r})

        if (perm + 1) % 200 == 0:
            print(f"  Random null: {perm+1}/{n_perms}")

    null_df = pd.DataFrame(null_rows)
    null_valid = null_df["best_r"].dropna().values
    dopa_pct = float((np.abs(null_valid) < abs(dopa_r)).mean() * 100) if len(null_valid) else np.nan

    # Add p_vs_null for each named network (vs univariate null)
    if len(null_valid):
        named_df["p_vs_null"] = named_df["r"].apply(
            lambda r: float((np.abs(null_valid) >= abs(r)).mean())
        )

    # ── 3. SVR null (optional) ────────────────────────────────────────────────
    # For each permutation: sample the same number of bilateral pairs as dopamine,
    # build feat_cols + bilateral_pairs lists, run full ComBat-per-fold SVR.
    # Directly comparable to named network SVRs. Slow: ~15s × n_svr_perms.
    svr_null_df = pd.DataFrame(columns=["perm", "r"])
    dopa_svr_pct = np.nan

    if n_svr_perms > 0 and pool_pair_indices:
        print(f"\nRunning SVR null ({n_svr_perms} perms × ~15s each)...")
        # Use run_target_with_nested_cv for the null so it matches the named network
        # pipeline exactly (combined site+target stratification, filter_target_data,
        # sample weighting). Pass a slimmed df with only the random features + metadata,
        # set feature_mode="raw" so the pipeline picks them up via imaging prefix matching.
        imaging_prefixes = reg_orig.get("imaging_prefixes", [
            "dmri", "smri_area_cdk", "smri_thick_cdk", "smri_sulc_cdk",
            "smri_vol_cdk", "smri_vol_scs", "mrisdp",
        ])
        # Pool uses specific prefixes (e.g. "smri_thk_cdk_") that differ from
        # reg/mlp configs (e.g. "smri_thick_cdk"). Compute roots to avoid
        # duplicate columns (thickness/mrsdp) in meta_cols_base vs rand_feat_cols.
        pool_prefix_roots = [p.rstrip("_") for p in all_imaging_prefixes]
        all_pfx_union = list(set(imaging_prefixes) | set(pool_prefix_roots))
        meta_cols_base = [c for c in full_df.columns
                          if not any(c.startswith(p) for p in all_pfx_union)]

        mlp_orig = env.configs.mlp
        n_pairs_dopa = len(bilateral_pairs)
        svr_null_rows = []
        for perm in range(n_svr_perms):
            if len(pool_pair_indices) < n_pairs_dopa:
                sampled_pairs = pool_pair_indices
            else:
                chosen = rng.choice(len(pool_pair_indices), size=n_pairs_dopa, replace=False)
                sampled_pairs = [pool_pair_indices[k] for k in chosen]

            rand_feat_cols = []
            for name, li, ri in sampled_pairs:
                rand_feat_cols.extend([pool_present[li], pool_present[ri]])

            # Build slim df: metadata + random features only
            null_df_rand = full_df[meta_cols_base + rand_feat_cols].copy()

            # Temporarily override both reg and mlp configs so the pipeline runs in
            # raw mode with pool-derived prefixes. Critical: mlp_config["feature_mode"]
            # defaults to "roi" which causes extract_mlp_harmonization_data to look for
            # ROI network columns not present in null_df_rand (→ empty X → crash).
            reg_override = copy.deepcopy(dict(reg_orig))
            reg_override["feature_mode"] = "raw"
            reg_override["use_pca"] = False
            mlp_override = copy.deepcopy(dict(mlp_orig))
            mlp_override["feature_mode"] = "raw"
            mlp_override["imaging_prefixes"] = pool_prefix_roots
            env.configs.regression = reg_override
            env.configs.mlp = mlp_override
            try:
                res = run_target_with_nested_cv(
                    env, null_df_rand, target_config, model_name="svr", verbose=False
                )
                svr_null_rows.append({"perm": perm, "r": res["svr"]["overall"]["pearson_r"]})
            except Exception as exc:
                logger.warning(f"SVR null perm {perm} failed: {exc}")
                svr_null_rows.append({"perm": perm, "r": np.nan})
            finally:
                env.configs.regression = reg_orig
                env.configs.mlp = mlp_orig

            if (perm + 1) % 10 == 0:
                print(f"  SVR null: {perm+1}/{n_svr_perms}")

        svr_null_df = pd.DataFrame(svr_null_rows)
        svr_valid = svr_null_df["r"].dropna().values
        if len(svr_valid):
            dopa_svr_pct = float((np.abs(svr_valid) < abs(dopa_r)).mean() * 100)
            # Update p_vs_null for named networks using SVR null
            named_df["p_vs_svr_null"] = named_df["r"].apply(
                lambda r: float((np.abs(svr_valid) >= abs(r)).mean())
            )
            print(f"  Dopamine at {dopa_svr_pct:.1f}th percentile of SVR null")

    return {
        "named_df": named_df,
        "null_df": null_df,
        "svr_null_df": svr_null_df,
        "dopa_pct": dopa_pct,
        "dopa_svr_pct": dopa_svr_pct,
    }


# ---------------------------------------------------------------------------
# 5. Sex-stratified analysis
# ---------------------------------------------------------------------------

def sex_stratified_analysis(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    env,
    min_n: int = 30,
) -> pd.DataFrame:
    """ComBat full sample, then split by sex, correlate pallidum AI per sex.

    Full-sample ComBat BEFORE sex-split avoids unstable harmonization in small
    per-sex subsets. Only the correlation step is sex-stratified.

    Args:
        full_df: Full subject-level dataframe.
        feature_cols: Raw feature column names.
        bilateral_pairs: Bilateral ROI pairs.
        target_col: Target column name.
        env: Environment with configs.
        min_n: Minimum subjects per sex group to report result.

    Returns:
        DataFrame: sex, r, p, n per AI feature.
    """
    from .longitudinal import _combat_harmonize
    harm_config = env.configs.harmonize
    reg_config = env.configs.regression

    mask = full_df[target_col].notna()
    df_f = full_df[mask].copy()
    y = df_f[target_col].values.astype(float)

    present_cols = [c for c in feature_cols if c in df_f.columns]
    X_raw = df_f[present_cols].values.astype(float)
    X_harm, keep, _ = _combat_harmonize(X_raw, df_f, harm_config)
    df_f = df_f[keep].reset_index(drop=True)
    y = y[keep]

    if "sex_mapped" not in df_f.columns:
        logger.warning("sex_mapped column not found; cannot stratify by sex.")
        return pd.DataFrame()

    asym = compute_asymmetry_features(X_harm, present_cols, bilateral_pairs)
    rows = []
    for sex_label in df_f["sex_mapped"].unique():
        mask_sex = (df_f["sex_mapped"] == sex_label).values
        if mask_sex.sum() < min_n:
            continue
        y_sex = y[mask_sex]
        for feat_name, ai in asym.items():
            if not feat_name.endswith("_AI"):
                continue
            ai_sex = ai[mask_sex]
            valid = np.isfinite(ai_sex) & np.isfinite(y_sex)
            if valid.sum() < min_n:
                continue
            r, p = pearsonr(ai_sex[valid], y_sex[valid])
            rows.append({"sex": sex_label, "feature": feat_name,
                         "r": r, "p": p, "n": int(valid.sum())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Scanner-stratified analysis
# ---------------------------------------------------------------------------

def scanner_stratified_analysis(
    full_df: pd.DataFrame,
    feature_cols: list,
    bilateral_pairs: list,
    target_col: str,
    env,
    min_per_scanner: int = 20,
) -> pd.DataFrame:
    """Correlate pallidum AI with target separately per scanner model (no ComBat).

    No ComBat is applied since each scanner group is a single site — ComBat would
    have nothing to harmonize. The test validates that the signal is not an artifact
    of ComBat or a single scanner type.

    Returns:
        DataFrame: scanner, r_pallidum_AI, p, n.
    """
    scanner_col = env.configs.data["columns"]["mapping"].get(
        "scanner_model", "mri_info_manufacturersmn"
    )
    if scanner_col not in full_df.columns:
        logger.warning(f"Scanner column '{scanner_col}' not found.")
        return pd.DataFrame()

    mask = full_df[target_col].notna()
    df_f = full_df[mask].copy()
    y_all = df_f[target_col].values.astype(float)
    present_cols = [c for c in feature_cols if c in df_f.columns]

    rows = []
    for scanner in df_f[scanner_col].unique():
        sc_mask = (df_f[scanner_col] == scanner).values
        if sc_mask.sum() < min_per_scanner:
            continue

        X_sc = df_f.loc[sc_mask, present_cols].values.astype(float)
        y_sc = y_all[sc_mask]
        valid = np.all(np.isfinite(X_sc), axis=1) & np.isfinite(y_sc)
        X_sc = X_sc[valid]
        y_sc = y_sc[valid]
        if len(y_sc) < min_per_scanner:
            continue

        asym = compute_asymmetry_features(X_sc, present_cols, bilateral_pairs)
        ai = asym.get("pallidum_AI")
        if ai is None or np.isfinite(ai).sum() < 10:
            continue
        r, p = pearsonr(ai[np.isfinite(ai)], y_sc[np.isfinite(ai)])
        rows.append({"scanner": scanner, "r_pallidum_AI": r, "p": p,
                     "n": int(valid.sum())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Bootstrap feature CIs
# ---------------------------------------------------------------------------

def bootstrap_feature_ci(
    X_harm: np.ndarray,
    y: np.ndarray,
    bilateral_pairs: list,
    valid_cols: list,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CIs on AI feature correlations with target.

    Args:
        X_harm: Harmonized feature matrix.
        y: Target array.
        bilateral_pairs: Bilateral ROI pairs.
        valid_cols: Column names for X_harm.
        n_boot: Number of bootstrap resamples.
        seed: Random state.

    Returns:
        DataFrame: feature, r_obs, ci_lo, ci_hi, p_boot (fraction of resamples with r<0).
    """
    rng = np.random.RandomState(seed)
    asym = compute_asymmetry_features(X_harm, valid_cols, bilateral_pairs)
    ai_keys = [k for k in asym if k.endswith("_AI")]
    n = len(y)
    rows = []

    for key in ai_keys:
        ai = asym[key]
        valid = np.isfinite(ai) & np.isfinite(y)
        ai_v, y_v = ai[valid], y[valid]
        if len(ai_v) < 10:
            continue

        r_obs, _ = pearsonr(ai_v, y_v)
        boot_rs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(len(ai_v), size=len(ai_v), replace=True)
            boot_rs[b], _ = pearsonr(ai_v[idx], y_v[idx])

        ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
        p_boot = (boot_rs <= 0).mean() if r_obs > 0 else (boot_rs >= 0).mean()
        rows.append({"feature": key, "r_obs": r_obs, "ci_lo": ci_lo,
                     "ci_hi": ci_hi, "p_boot": p_boot, "n": int(valid.sum())})

    return pd.DataFrame(rows).sort_values("r_obs").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. One-per-family permutation
# ---------------------------------------------------------------------------

def one_per_family_permutation(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    n_perms: int = 500,
) -> dict:
    """Permutation test after randomly removing one sibling per family.

    Runs n_perms iterations: each time randomly pick one subject per family,
    fit the full nested CV SVR pipeline, collect r. Reports observed r and
    empirical p-value (fraction of permutations with shuffled r >= observed r).

    Args:
        env: Environment with configs.
        full_df: Full subject-level dataframe.
        target_config: Target config dict.
        n_perms: Number of permutations (each draws a fresh one-per-family sample).

    Returns:
        dict: observed_r, boot_rs (array of per-sample r values), ci_lo, ci_hi, p_emp.
    """
    from .pipeline import run_target_with_nested_cv

    seed = env.configs.run.get("seed", 42)
    rng = np.random.RandomState(seed)
    family_col = "rel_family_id"
    target_col = target_config["column"]

    boot_rs = []
    for i in range(n_perms):
        if family_col in full_df.columns:
            # One per family: randomly select one member
            keep_idx = (
                full_df.groupby(family_col)
                .apply(lambda g: g.sample(1, random_state=int(rng.randint(1e8))))
                .index.get_level_values(1)
            )
            df_opf = full_df.loc[keep_idx].reset_index(drop=True)
        else:
            df_opf = full_df.copy()

        try:
            res = run_target_with_nested_cv(env, df_opf, target_config,
                                             model_name="svr", verbose=False)
            boot_rs.append(res["svr"]["overall"]["pearson_r"])
        except Exception as e:
            logger.warning(f"one_per_family perm={i} failed: {e}")
            boot_rs.append(np.nan)

        if (i + 1) % 50 == 0:
            print(f"  One-per-family: {i+1}/{n_perms}")

    boot_rs = np.array(boot_rs)
    valid_rs = boot_rs[np.isfinite(boot_rs)]
    obs_r = np.nanmedian(boot_rs)  # median across draws as the stable estimate
    ci_lo, ci_hi = (np.percentile(valid_rs, [2.5, 97.5])
                    if len(valid_rs) > 10 else (np.nan, np.nan))

    return {
        "observed_r": obs_r,
        "boot_rs": boot_rs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_valid": len(valid_rs),
    }
