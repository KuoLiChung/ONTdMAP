#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
methylQC_PCAnHM.py

Description:
    Group-level CpG methylation PCA and hierarchical clustering heatmap analysis.
    Parses multi-sample methylation matrices (TSV format), converts beta values to
    M-values, and generates PCA pair plots and clustermap heatmaps for global and
    promoter-restricted CpG site sets. Supports depth filtering, variance-only mode
    for threshold exploration, and TSV cache I/O for fast re-analysis.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
    - adjustText (optional; recommended for non-overlapping PCA labels)
"""

import argparse
import os
import sys
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging
import tempfile
from collections import defaultdict
import re
from scipy.stats import zscore
from itertools import combinations

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==========================================
# Sample Column Matching
# ==========================================
def is_in_group(col_name: str, group_input: str) -> bool:
    """
    Determine whether a sample column belongs to a group using dual-mode matching.

    Mode 1 (comma-separated input): exact name match or name-prefix match
    (e.g., 'C1,C2,C3' matches columns named 'C1', 'C1_rep1', 'C1.bam').

    Mode 2 (single string input): extracts the alphabetic/symbolic prefix before
    the first digit and matches columns sharing that prefix (e.g., 'C' matches
    'C1', 'C2', 'C10').

    Parameters
    ----------
    col_name : str
        Sample column name from the input matrix header.
    group_input : str
        User-supplied group identifier string.

    Returns
    -------
    bool
        True if the column belongs to the specified group.
    """
    if not group_input:
        return False

    group_input = group_input.strip()

    if ',' in group_input:
        # Mode 1: match against a comma-separated list of exact names or name prefixes
        allowed_samples = [s.strip() for s in group_input.split(',')]
        for sample in allowed_samples:
            if (col_name == sample
                    or col_name.startswith(f"{sample}_")
                    or col_name.startswith(f"{sample}.")):
                return True
        return False

    # Mode 2: match columns whose leading non-numeric prefix equals the target
    match = re.match(r'^([^0-9]+)', col_name)
    if match:
        extracted_prefix = match.group(1)
        if extracted_prefix == group_input:
            return True

    return False


# ==========================================
# BED File Processing
# ==========================================
def parse_bed_to_arrays(bed_path):
    """
    Parse a BED file into chromosome-keyed sorted start/end arrays.

    Intervals are sorted by start position per chromosome to enable
    binary-search-based overlap queries downstream.

    Parameters
    ----------
    bed_path : str
        Path to the BED file.

    Returns
    -------
    dict or None
        Dictionary mapping chromosome names to (starts, ends) NumPy arrays,
        or None on parse failure.
    """
    logging.info(f"Parsing BED file: {bed_path}")
    intervals = defaultdict(list)
    try:
        with open(bed_path) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    intervals[parts[0]].append((int(parts[1]), int(parts[2])))
    except Exception as e:
        logging.error(f"Error parsing BED file: {e}")
        return None

    merged = {}
    for chrom, ivs in intervals.items():
        ivs.sort(key=lambda x: x[0])
        starts = np.array([x[0] for x in ivs], dtype=np.int32)
        ends   = np.array([x[1] for x in ivs], dtype=np.int32)
        merged[chrom] = (starts, ends)
    return merged


def get_promoter_mask(chroms, positions, bed_arrays):
    """
    Return a boolean mask indicating which positions fall within BED-defined intervals.

    Uses binary search (np.searchsorted) on pre-sorted end arrays for efficient
    overlap detection without iterating over all intervals.

    Parameters
    ----------
    chroms : array-like
        Chromosome labels for each position.
    positions : array-like
        Genomic start positions.
    bed_arrays : dict
        Output of parse_bed_to_arrays().

    Returns
    -------
    np.ndarray
        Boolean mask of length len(positions).
    """
    mask   = np.zeros(len(positions), dtype=bool)
    df_tmp = pd.DataFrame({'chr': chroms, 'pos': positions})

    for chrom, group in df_tmp.groupby('chr'):
        if chrom in bed_arrays:
            starts, ends = bed_arrays[chrom]
            pos_arr = group['pos'].values
            idx     = np.searchsorted(ends, pos_arr, side='right')
            valid   = idx < len(starts)
            if np.any(valid):
                valid_idx  = np.where(valid)[0]
                in_interval = starts[idx[valid_idx]] <= pos_arr[valid_idx]
                mask[group.index.values[valid_idx[in_interval]]] = True
    return mask


# ==========================================
# Multiprocessing Worker
# ==========================================
def process_chunk(args):
    """
    Parse a chunk of the input matrix, compute beta values, apply depth filtering,
    and return valid sites with an optional promoter overlap mask.

    Sites where any sample has a missing or sub-threshold value are dropped entirely
    (strict complete-case filtering).

    Parameters
    ----------
    args : tuple
        (chunk_df, sample_cols, min_cov, bed_arrays)

    Returns
    -------
    tuple
        (beta_matrix, site_ids, promoter_mask) for valid sites, or (None, None, None)
        if no sites pass filtering.
    """
    chunk_df, sample_cols, min_cov, bed_arrays = args

    beta_list = []
    for col in sample_cols:
        series   = chunk_df[col].astype(str)
        split_df = series.str.split(',', expand=True)

        if split_df.shape[1] < 2:
            beta = np.full(len(series), np.nan, dtype=np.float32)
        else:
            total    = pd.to_numeric(split_df[0], errors='coerce').values
            modified = pd.to_numeric(split_df[1], errors='coerce').values

            with np.errstate(divide='ignore', invalid='ignore'):
                beta = np.where(total >= min_cov, modified / total, np.nan)

        beta_list.append(beta)

    beta_mat   = np.column_stack(beta_list).astype(np.float32)
    valid_mask = ~np.any(np.isnan(beta_mat), axis=1)

    if not np.any(valid_mask):
        return None, None, None

    valid_beta = beta_mat[valid_mask]
    valid_ids  = (chunk_df['#chr'].astype(str) + ":" + chunk_df['start'].astype(str))[valid_mask].values

    promoter_mask = None
    if bed_arrays:
        promoter_mask = get_promoter_mask(
            chunk_df['#chr'].values[valid_mask],
            chunk_df['start'].values[valid_mask],
            bed_arrays
        )

    return valid_beta, valid_ids, promoter_mask


# ==========================================
# Analysis and Plotting
# ==========================================
def run_analysis(data_mat, feature_ids, sample_names, groups, outdir, prefix,
                 top_n, top_heatmap, control_name, case_name, variance_only=False):
    """
    Run PCA and hierarchical clustering heatmap analysis on a CpG beta-value matrix.

    Workflow:
      1. Clip beta values and convert to M-values.
      2. Compute per-site variance across samples.
      3. If variance_only: output variance distribution plot and exit.
      4. Otherwise: select top_n most variable sites, run PCA, generate PC pair plots
         and save PC loadings; then generate beta-value and M-value z-score heatmaps
         using hierarchical clustering for the top_heatmap most variable sites.

    Parameters
    ----------
    data_mat : np.ndarray
        Beta-value matrix of shape (n_sites, n_samples).
    feature_ids : np.ndarray
        CpG site identifiers of length n_sites.
    sample_names : list of str
        Ordered sample names matching matrix columns.
    groups : list of str
        Group label for each sample.
    outdir : str
        Output directory for all generated files.
    prefix : str
        Filename prefix identifying the analysis context (e.g., 'Global', 'Promoter').
    top_n : int
        Number of most variable sites to use for PCA.
    top_heatmap : int
        Number of most variable sites to visualize in heatmaps.
    control_name : str
        Display label for the control group.
    case_name : str
        Display label for the case group.
    variance_only : bool
        If True, only generate the variance distribution plot and skip PCA/heatmaps.
    """
    num_sites   = len(data_mat)
    num_samples = len(sample_names)

    if num_sites < 10:
        logging.warning(f"[{prefix}] Too few sites ({num_sites}), skipping analysis.")
        return

    logging.info(f"[{prefix}] Analyzing {num_sites} sites...")

    epsilon      = 1e-6
    beta_clipped = np.clip(data_mat, epsilon, 1 - epsilon)
    m_all        = np.log2(beta_clipped / (1 - beta_clipped))
    row_vars     = np.var(m_all, axis=1)

    # Variance-only mode: output distribution plot and return
    if variance_only:
        logging.info(f"[{prefix}] Generating M-value variance distribution plot...")
        fig, ax1 = plt.subplots(figsize=(8, 6))
        sns.histplot(row_vars, bins=100, color='tab:blue', edgecolor='white', ax=ax1, label='Binned Count')
        sorted_vars       = np.sort(row_vars)[::-1]
        cumulative_counts = np.arange(1, len(sorted_vars) + 1)
        ax1.plot(sorted_vars, cumulative_counts, color='tab:orange', linewidth=2, label='Cumulative Count')
        ax1.set_yscale('log')
        ax1.set_xlabel("Variance")
        ax1.set_ylabel("Log10(Count)")
        ax1.legend(frameon=False, loc='best')
        plt.title(f"{prefix} M-value Variance Distribution (n={num_sites})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_Mvalue_VarDist.png"), dpi=300)
        plt.close(fig)
        logging.info(f"[{prefix}] Variance-only mode: skipping PCA and heatmap generation.")
        return

    # Select top_n most variable sites for PCA
    actual_top_n = min(top_n, num_sites)
    top_indices  = np.argpartition(row_vars, -actual_top_n)[-actual_top_n:]
    m_values     = m_all[top_indices]

    n_pcs       = min(5, num_samples, actual_top_n)
    pca         = PCA(n_components=n_pcs)
    pca_results = pca.fit_transform(m_values.T)

    pc_cols = [f'PC{i + 1}' for i in range(n_pcs)]
    pca_df  = pd.DataFrame(pca_results, columns=pc_cols)
    pca_df['Sample'] = sample_names
    pca_df['Group']  = groups

    # PCA pair plots across all PC combinations
    if n_pcs >= 2:
        logging.info(f"[{prefix}] Generating PC pair plots...")
        pairs     = list(combinations(range(n_pcs), 2))
        num_pairs = len(pairs)
        ncols     = 2
        nrows     = (num_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
        axes      = axes.flatten()
        var_ratio = pca.explained_variance_ratio_ * 100

        for i, (pc_a_idx, pc_b_idx) in enumerate(pairs):
            ax   = axes[i]
            pc_a = pc_cols[pc_a_idx]
            pc_b = pc_cols[pc_b_idx]

            sns.scatterplot(x=pc_a, y=pc_b, hue='Group', style='Group',
                            data=pca_df, s=100, palette='Set1', edgecolor='k', ax=ax)

            texts = []
            for j in range(len(pca_df)):
                texts.append(ax.text(pca_df.iloc[j][pc_a], pca_df.iloc[j][pc_b],
                                     pca_df.iloc[j]['Sample'], fontsize=9))

            if HAS_ADJUST_TEXT:
                adjust_text(texts, ax=ax,
                            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.6))

            ax.set_xlabel(f"{pc_a} ({var_ratio[pc_a_idx]:.2f}%)")
            ax.set_ylabel(f"{pc_b} ({var_ratio[pc_b_idx]:.2f}%)")
            ax.set_title(f"{prefix} PCA: {pc_a} vs {pc_b}")
            ax.legend(frameon=False, loc='best', fontsize='small')

        for j in range(len(pairs), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_PCA_Pairs.png"), dpi=300)
        plt.close()
    else:
        logging.warning(f"[{prefix}] Insufficient samples or sites for PC pair plots.")

    # Save PC1 loadings sorted by absolute weight
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}_Loading' for i in range(n_pcs)],
        index=feature_ids[top_indices]
    )
    loadings['PC1_Abs'] = loadings['PC1_Loading'].abs()
    loadings.sort_values('PC1_Abs', ascending=False).to_csv(
        os.path.join(outdir, f"{prefix}_PCA_loadings.csv")
    )

    # Select top_heatmap most variable sites for heatmaps
    logging.info(f"[{prefix}] Generating heatmaps (beta-value and M-value z-score)...")
    actual_top_h = min(top_heatmap, num_sites)
    top_h_indices = np.argpartition(row_vars, -actual_top_h)[-actual_top_h:]

    df_beta = pd.DataFrame(data_mat[top_h_indices],
                           index=feature_ids[top_h_indices], columns=sample_names)
    m_top = m_all[top_h_indices]
    m_z   = zscore(m_top, axis=1)
    df_mz = pd.DataFrame(m_z, index=feature_ids[top_h_indices], columns=sample_names)

    lut        = {control_name: "#4DBBD5", case_name: "#E64B35"}
    col_colors = pd.Series(groups, index=sample_names).map(lut)
    sns.set_theme(style="white", color_codes=True, font_scale=1.1)
    show_row_tree = (actual_top_h <= 200)

    def draw_heatmap(data_matrix, is_zscore, title_suffix, filename_suffix):
        """
        Render a hierarchical clustering heatmap with group color annotations.

        Parameters
        ----------
        data_matrix : pd.DataFrame
            Site-by-sample matrix to cluster and display.
        is_zscore : bool
            If True, use a diverging color scale centered at zero (Z-score).
            If False, use a 0–1 scale (beta-value).
        title_suffix : str
            Data type label for the plot title.
        filename_suffix : str
            Tag appended to the output filename.
        """
        cmap       = "RdBu_r" if is_zscore else "RdYlBu_r"
        vmin, vmax = (-3, 3) if is_zscore else (0, 1)
        cbar_label = "Z-score" if is_zscore else "Methylation Level"

        g = sns.clustermap(
            data_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
            method='average', metric='euclidean',
            col_cluster=True, row_cluster=True,
            col_colors=col_colors,
            figsize=(14, 10), yticklabels=False, xticklabels=True,
            tree_kws={'linewidths': 1.2, 'colors': '#333333'},
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            cbar_kws={'label': cbar_label}
        )
        g.ax_heatmap.set_ylabel('')
        g.ax_col_colors.tick_params(right=False, left=False, top=False, bottom=False)
        for spine in g.ax_col_colors.spines.values():
            spine.set_visible(False)
        if not show_row_tree:
            g.ax_row_dendrogram.set_visible(False)

        if actual_top_h >= num_sites:
            g.fig.suptitle(
                f"All Filtered Sites (n={actual_top_h}) Heatmap\n"
                f"Sorted by Hierarchical Clustering, Data = {title_suffix}",
                fontweight='bold', fontsize=18, y=1.05)
            g.savefig(os.path.join(outdir, f"{prefix}_AllSites_{filename_suffix}_byHC.png"),
                      dpi=300, bbox_inches='tight')
        else:
            g.fig.suptitle(
                f"Top {actual_top_h} Sites Heatmap\n"
                f"Sorted by Hierarchical Clustering, Data = {title_suffix}",
                fontweight='bold', fontsize=18, y=1.05)
            g.savefig(os.path.join(outdir, f"{prefix}_Top{actual_top_h}_{filename_suffix}_byHC.png"),
                      dpi=300, bbox_inches='tight')
        plt.close(g.fig)

    draw_heatmap(df_beta, is_zscore=False, title_suffix="Beta-value",       filename_suffix="Beta")
    draw_heatmap(df_mz,   is_zscore=True,  title_suffix="Z-score (M-value)", filename_suffix="ZM")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Group-level CpG methylation PCA and hierarchical clustering heatmap analysis."
    )

    # I/O Arguments
    parser.add_argument('-i', '--input', required=True,
                        help="Input multi-sample methylation matrix (TSV format). Example: Union.bed")
    parser.add_argument('-b', '--bed',
                        help="BED file defining specific genomic regions (e.g., TSS/Promoters) or other target regions for subset analysis.")
    parser.add_argument('-o', '--outdir', required=True, type=str,
                        help="Output directory. Created automatically if it does not exist.")
    parser.add_argument('-t', '--threads', type=int, default=max(1, multiprocessing.cpu_count() - 2),
                        help="Number of CPU threads for parallel processing. (Default: available cores - 2)")

    # Group Configuration
    parser.add_argument('--control', type=str, default="",
                        help="Control group identifiers. Mode 1: comma-separated exact names (e.g., 'C1,C2,C3'). "
                             "Mode 2: shared prefix before the first digit (e.g., 'C'). "
                             "Leave empty for single-group analysis.")
    parser.add_argument('--case', type=str, default="",
                        help="Case group identifiers. Supports the same matching modes as --control. "
                             "Leave empty for single-group analysis.")
    parser.add_argument('--control_name', type=str, default="Control",
                        help="Display label for the Control group in plots and reports. (Default: 'Control')")
    parser.add_argument('--case_name', type=str, default="Case",
                        help="Display label for the Case group in plots and reports. (Default: 'Case')")

    # Feature Selection and Dimensionality Reduction
    parser.add_argument('--top_global', type=int, default=2000,
                        help="Number of most variable CpG sites (by M-value variance) for global PCA. (Default: 2000)")
    parser.add_argument('--top_promoter', type=int, default=2000,
                        help="Number of most variable CpG sites within promoter regions for targeted PCA. (Default: 2000)")
    parser.add_argument('--top_heatmap', type=int, default=500,
                        help="Number of most variable CpG sites to include in hierarchical clustering heatmaps. (Default: 500)")
    parser.add_argument('--variance_only', action='store_true',
                        help="Output only the M-value variance distribution plot. Skips PCA and heatmap generation. "
                             "Useful for selecting variance thresholds before full analysis.")

    # QC Filtering
    parser.add_argument('--min-depth', type=int, default=0,
                        help="Minimum read depth. Sites with depth below this threshold are masked as NaN. (Default: 0)")

    # Cache I/O
    parser.add_argument('--save_cache', type=str, default="cache",
                        help="File path for saving the aggregated DMR beta matrix as a TSV cache.")
    parser.add_argument('--load_cache', type=str, default="cache",
                        help="File prefix of a previously saved TSV cache to load directly, bypassing parsing.")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not HAS_ADJUST_TEXT:
        logging.warning("Package 'adjustText' not found. PCA plot labels may overlap. "
                        "Install with: pip install adjustText")

    # ==========================================
    # Sample Column Resolution
    # ==========================================
    cache_global   = f"{args.load_cache}_Global.tsv"   if args.load_cache else None
    cache_promoter = f"{args.load_cache}_Promoter.tsv" if args.load_cache else None
    use_cache      = args.load_cache and (
        (cache_global and os.path.exists(cache_global)) or
        (cache_promoter and os.path.exists(cache_promoter))
    )

    if use_cache:
        # Read sample columns from the cache header
        cache_file = cache_global if os.path.exists(cache_global) else cache_promoter
        try:
            with open(cache_file, 'r') as f:
                header = f.readline().strip().split('\t')
            all_sample_cols = header[0:]
        except Exception as e:
            logging.error(f"Could not read cache file header: {e}")
            sys.exit(1)
    else:
        try:
            with open(args.input, 'r') as f:
                header = f.readline().strip().split('\t')
            all_sample_cols = header[3:]
        except Exception as e:
            logging.error(f"Could not read input file header: {e}")
            sys.exit(1)

    def match_cols(targets):
        if not targets:
            return []
        return [c for c in all_sample_cols if is_in_group(c, targets)]

    c_cols = match_cols(args.control)
    k_cols = match_cols(args.case)

    if not c_cols and not k_cols:
        logging.error("No samples matched the provided group identifiers.")
        sys.exit(1)

    final_cols = c_cols + k_cols
    groups     = [args.control_name] * len(c_cols) + [args.case_name] * len(k_cols)

    logging.info(f"Matched {len(c_cols)} control(s), {len(k_cols)} case(s). Total: {len(final_cols)} samples.")

    # ==========================================
    # Route A: Load from Cache
    # ==========================================
    if use_cache:
        logging.info(f"Cache prefix '{args.load_cache}' found. Loading cached matrices...")

        if os.path.exists(cache_global):
            logging.info("Loading global cache...")
            df_global = pd.read_csv(cache_global, sep='\t', index_col=0)
            run_analysis(df_global.values, df_global.index.values, final_cols, groups,
                         args.outdir, "Global", args.top_global, args.top_heatmap,
                         args.control_name, args.case_name, args.variance_only)

        if os.path.exists(cache_promoter):
            logging.info("Loading promoter cache...")
            df_promoter = pd.read_csv(cache_promoter, sep='\t', index_col=0)
            run_analysis(df_promoter.values, df_promoter.index.values, final_cols, groups,
                         args.outdir, "Promoter", args.top_promoter, args.top_heatmap,
                         args.control_name, args.case_name, args.variance_only)

        logging.info(f"All analyses completed from cache. Results saved to: {args.outdir}")
        return

    # ==========================================
    # Route B: Standard Extraction (Multiprocessing)
    # ==========================================
    bed_arrays = parse_bed_to_arrays(args.bed) if args.bed else None

    dtype_dict = {col: str for col in final_cols}
    dtype_dict['#chr']  = str
    dtype_dict['start'] = int

    temp_global   = tempfile.NamedTemporaryFile(delete=False)
    temp_promoter = tempfile.NamedTemporaryFile(delete=False) if bed_arrays else None
    global_ids    = []
    promoter_ids  = []

    pool       = multiprocessing.Pool(args.threads)
    chunk_iter = pd.read_csv(args.input, sep='\t', chunksize=50000,
                             usecols=['#chr', 'start'] + final_cols,
                             dtype=dtype_dict, low_memory=False)

    logging.info("Starting multi-process data extraction...")
    try:
        for res in pool.imap(process_chunk,
                             [(c, final_cols, args.min_depth, bed_arrays) for c in chunk_iter]):
            if res is None or res[0] is None:
                continue
            mat, ids, p_mask = res

            mat.tofile(temp_global)
            global_ids.extend(ids)

            if temp_promoter and p_mask is not None and np.any(p_mask):
                mat[p_mask].tofile(temp_promoter)
                promoter_ids.extend(ids[p_mask])

    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")
        pool.terminate()
        sys.exit(1)
    finally:
        pool.close()
        pool.join()

    temp_global.close()
    if temp_promoter:
        temp_promoter.close()

    num_samples = len(final_cols)

    # Global analysis
    if len(global_ids) > 0:
        logging.info(f"Running global analysis on {len(global_ids)} sites...")
        global_mat = np.fromfile(temp_global.name, dtype=np.float32).reshape(-1, num_samples)

        if args.save_cache:
            pd.DataFrame(global_mat, index=global_ids, columns=final_cols).to_csv(
                f"{args.save_cache}_Global.tsv", sep='\t')

        run_analysis(global_mat, np.array(global_ids), final_cols, groups,
                     args.outdir, "Global", args.top_global, args.top_heatmap,
                     args.control_name, args.case_name, args.variance_only)
        del global_mat
    else:
        logging.warning("No sites passed the global filter.")

    if os.path.exists(temp_global.name):
        os.unlink(temp_global.name)

    # Promoter-restricted analysis
    if temp_promoter:
        if len(promoter_ids) > 0:
            logging.info(f"Running promoter analysis on {len(promoter_ids)} sites...")
            promoter_mat = np.fromfile(temp_promoter.name, dtype=np.float32).reshape(-1, num_samples)

            if args.save_cache:
                pd.DataFrame(promoter_mat, index=promoter_ids, columns=final_cols).to_csv(
                    f"{args.save_cache}_Promoter.tsv", sep='\t')

            run_analysis(promoter_mat, np.array(promoter_ids), final_cols, groups,
                         args.outdir, "Promoter", args.top_promoter, args.top_heatmap,
                         args.control_name, args.case_name, args.variance_only)
            del promoter_mat
        else:
            logging.warning("No sites passed the promoter filter.")

        if os.path.exists(temp_promoter.name):
            os.unlink(temp_promoter.name)

    logging.info(f"All analyses completed. Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()