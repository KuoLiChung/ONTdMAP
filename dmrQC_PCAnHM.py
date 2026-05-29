#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dmrQC_PCAnHM.py

Description:
    PCA and hierarchical clustering heatmap analysis for modkit DMR outputs.
    Overlaps CpG sites from a multi-sample methylation matrix (Union BED) with
    DMR intervals, aggregates read depth per DMR per sample, converts beta values
    to M-values, and generates PCA pair plots and dual clustermaps (group-constrained
    and unconstrained hierarchical clustering) for both beta-value and M-value
    z-score matrices. Supports TSV cache I/O for fast re-analysis.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, intervaltree
    - adjustText (optional; recommended for non-overlapping PCA labels)
"""

import argparse
import os
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from intervaltree import IntervalTree
from collections import defaultdict
from scipy.stats import zscore
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from itertools import combinations

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


# ==========================================
# Sample Column Matching
# ==========================================
def is_in_group(col_name: str, group_input: str) -> bool:
    """
    Determine whether a sample column belongs to a group using dual-mode matching.

    Mode 1 (comma-separated input): exact name match or name-prefix match
    (e.g., 'C1,C2,C3' matches 'C1', 'C1_rep1', 'C1.bam').

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
# Read Depth Parsing
# ==========================================
def extract_depths(val):
    """
    Parse a comma-delimited depth field and return total depth and methylated read count.

    Accepts two formats:
      - Three-field: 'total,unmethylated,methylated' -> returns (total, unmeth + meth)
      - Two-field:   'total,methylated'              -> returns (total, methylated)

    Parameters
    ----------
    val : str or float
        Raw field value from the Union BED matrix.

    Returns
    -------
    tuple of (float, float)
        (total_depth, methylated_depth). Returns (0, 0) for missing or invalid values.
    """
    if pd.isna(val) or val in ("NA", "."):
        return (0, 0)
    try:
        parts = str(val).split(',')
        if len(parts) >= 3:
            a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
            return (a, b + c)
        elif len(parts) == 2:
            a, b = float(parts[0]), float(parts[1])
            return (a, b)
        return (0, 0)
    except (ValueError, AttributeError):
        return (0, 0)


# ==========================================
# Group-Constrained Hierarchical Clustering
# ==========================================
def get_group_constrained_linkage(df_matrix, valid_samples, sample_to_group):
    """
    Compute a linkage matrix that forces separation between group groups.

    Adds a large inter-group distance penalty to the pairwise distance matrix
    so that hierarchical clustering separates Control from Case at the top-level
    merge. The penalty-inflated merge heights are then rescaled to just above the
    tallest within-group merge height, preserving readable dendrogram proportions.

    Parameters
    ----------
    df_matrix : pd.DataFrame
        Site-by-sample beta-value matrix.
    valid_samples : list of str
        Ordered list of sample column names.
    sample_to_group : dict
        Mapping from sample name to group label.

    Returns
    -------
    np.ndarray or None
        Linkage matrix for use with seaborn.clustermap, or None if fewer than
        two samples are provided.
    """
    if len(valid_samples) <= 1:
        return None

    col_data    = df_matrix[valid_samples].T
    dist_square = distance.squareform(distance.pdist(col_data, metric='euclidean'))

    max_dist = np.max(dist_square) if np.max(dist_square) > 0 else 1.0
    penalty  = max_dist * 1000

    for i in range(len(valid_samples)):
        for j in range(i + 1, len(valid_samples)):
            if sample_to_group[valid_samples[i]] != sample_to_group[valid_samples[j]]:
                dist_square[i, j] += penalty
                dist_square[j, i] += penalty

    custom_dist  = distance.squareform(dist_square)
    linkage_mat  = hierarchy.linkage(custom_dist, method='average')

    # Rescale inter-group merge heights to just above the tallest within-group merge
    normal_heights   = [row[2] for row in linkage_mat if row[2] < penalty]
    top_normal_height = max(normal_heights) if normal_heights else max_dist

    for i in range(len(linkage_mat)):
        if linkage_mat[i, 2] >= penalty:
            linkage_mat[i, 2] = top_normal_height * 1.1

    return linkage_mat


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="DMR-level PCA and hierarchical clustering heatmap analysis from modkit Union BED output."
    )

    # I/O Arguments
    parser.add_argument('-d', '--input_dmr', required=True,
                        help="QC-filtered DMR BED file (output of DMRQC.py). Example: QC_pass.bed")
    parser.add_argument('-u', '--input_bed', required=True,
                        help="Multi-sample methylation matrix in TSV/BED format. Example: Union.bed")
    parser.add_argument('-o', '--outdir', required=True, type=str,
                        help="Output directory. Created automatically if it does not exist.")
    parser.add_argument('--top_heatmap', type=int, default=150,
                        help="Number of top DMRs (by absolute effect size) to include in heatmaps. "
                             "Set to 0 to use all DMRs. (Default: 150)")

    # Group Configuration
    parser.add_argument('--control', type=str, default="",
                        help="Control group identifiers. Mode 1: comma-separated exact names (e.g., 'C1,C2,C3'). "
                             "Mode 2: shared prefix before the first digit (e.g., 'C'). "
                             "Leave empty for single-group analysis.")
    parser.add_argument('--case', type=str, default="",
                        help="Case group identifiers. Supports the same matching modes as --control. "
                             "Leave empty for single-group analysis.")
    parser.add_argument('--control_name', type=str, default="Control",
                        help="Display label for the Control group. (Default: 'Control')")
    parser.add_argument('--case_name', type=str, default="Case",
                        help="Display label for the Case group. (Default: 'Case')")

    # Cache I/O
    parser.add_argument('--save_cache', type=str, default="cache",
                        help="File path for saving the aggregated DMR beta matrix as a TSV cache.")
    parser.add_argument('--load_cache', type=str, default="cache",
                        help="File path of a previously saved TSV cache to load directly, bypassing parsing.")

    args = parser.parse_args()

    if not HAS_ADJUST_TEXT:
        print("[WARNING] Package 'adjustText' not found. PCA plot labels may overlap. "
              "Install with: pip install adjustText")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # ==========================================
    # Route A: Load from Cache
    # ==========================================
    if args.load_cache and os.path.exists(args.load_cache):
        print(f"[INFO] Loading cached matrix: {args.load_cache}")
        merged_df = pd.read_csv(args.load_cache, sep='\t', index_col='feature_id')

        sample_to_group = {}
        valid_samples   = []
        for col in merged_df.columns:
            if is_in_group(col, args.control):
                sample_to_group[col] = args.control_name
                valid_samples.append(col)
            elif is_in_group(col, args.case):
                sample_to_group[col] = args.case_name
                valid_samples.append(col)

        print(f"      -> Loaded {len(merged_df)} DMRs, identified {len(valid_samples)} samples.")

    else:
        # ==========================================
        # Route B: Parse from Input Files
        # ==========================================

        # Step 1: Load DMR BED and build interval trees
        print(f"[INFO] Step 1. Loading DMR file and building interval trees: {args.input_dmr}")
        dmr_cols = ['chrom', 'chrom_start', 'chrom_end', 'dmr_id', 'cohen_h',
                    'strand', 'effect_size', 'DM_status']
        dmr_df = pd.read_csv(args.input_dmr, sep='\t', header=None,
                             usecols=range(8), names=dmr_cols)

        chrom_trees = defaultdict(IntervalTree)
        for _, row in dmr_df.iterrows():
            chrom_trees[row['chrom']].addi(row['chrom_start'], row['chrom_end'], row['dmr_id'])
        print(f"      -> Built interval trees for {len(dmr_df)} DMRs.")

        # Step 2: Parse Union BED header and assign samples to groups
        print(f"[INFO] Step 2. Scanning Union BED header and assigning groups: {args.input_bed}")

        with open(args.input_bed, 'r') as f:
            header = f.readline().strip().split('\t')

        sample_to_group = {}
        valid_samples   = []
        valid_indices   = []

        for idx, col in enumerate(header):
            if idx < 3:
                continue  # Skip chrom, start, end coordinate columns

            if is_in_group(col, args.control):
                sample_to_group[col] = args.control_name
                valid_samples.append(col)
                valid_indices.append(idx)
                print(f"  [Match] Column: {col:15} -> Group: {args.control_name}")

            elif is_in_group(col, args.case):
                sample_to_group[col] = args.case_name
                valid_samples.append(col)
                valid_indices.append(idx)
                print(f"  [Match] Column: {col:15} -> Group: {args.case_name}")

        if not valid_samples:
            print("[ERROR] No matching samples found in Union BED. Check --control and --case.")
            sys.exit(1)

        print(f"      -> Identified {len(valid_samples)} target samples.")

        # Step 3: Stream-parse CpG sites and aggregate depth per DMR (pooled read depth)
        print("[INFO] Step 3. Streaming CpG sites and aggregating read depth per DMR...")
        dmr_cpg_values = defaultdict(lambda: defaultdict(list))

        with open(args.input_bed, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                chrom = parts[0]
                try:
                    start, end = int(parts[1]), int(parts[2])
                except ValueError:
                    continue

                if chrom in chrom_trees:
                    overlaps = chrom_trees[chrom].overlap(start, end)
                    if overlaps:
                        row_vals = {}
                        for s_name, idx in zip(valid_samples, valid_indices):
                            raw_val    = parts[idx] if idx < len(parts) else "NA"
                            a, meth    = extract_depths(raw_val)
                            if a > 0:
                                row_vals[s_name] = (a, meth)

                        for overlap in overlaps:
                            dmr_id = overlap.data
                            for s_name, depth_tuple in row_vals.items():
                                dmr_cpg_values[dmr_id][s_name].append(depth_tuple)

        # Compute pooled methylation fraction per DMR per sample
        aggregated_rows = []
        for dmr_id, sample_dict in dmr_cpg_values.items():
            row_dict = {'dmr_id': dmr_id}
            for sample in valid_samples:
                vals = sample_dict.get(sample, [])
                if not vals:
                    row_dict[sample] = np.nan
                else:
                    sum_a    = sum(v[0] for v in vals)
                    sum_meth = sum(v[1] for v in vals)
                    row_dict[sample] = (sum_meth / sum_a) if sum_a > 0 else np.nan
            aggregated_rows.append(row_dict)

        union_df = pd.DataFrame(aggregated_rows)
        if union_df.empty:
            print("[ERROR] No CpG sites overlapped with target DMRs.")
            return

        union_df_clean = union_df.dropna(subset=valid_samples).copy()
        print(f"      -> Aggregation complete: {len(union_df_clean)} DMRs with valid values.")

        # Step 4: Merge DMR metadata with aggregated beta values
        print("[INFO] Step 4. Merging DMR metadata with aggregated values...")
        merged_df = pd.merge(dmr_df, union_df_clean, on='dmr_id')
        merged_df['abs_effect_size'] = merged_df['effect_size'].abs()
        merged_df = merged_df.sort_values(by='abs_effect_size', ascending=False)

        merged_df['feature_id'] = (merged_df['chrom'].astype(str) + ":" +
                                   merged_df['chrom_start'].astype(str) + "-" +
                                   merged_df['chrom_end'].astype(str))
        merged_df.set_index('feature_id', inplace=True)

        if args.save_cache:
            print(f"[INFO] Saving aggregated matrix cache to: {args.save_cache}")
            merged_df.to_csv(args.save_cache, sep='\t')

    # ==========================================
    # Step 5: PCA Analysis
    # ==========================================
    print("[INFO] Step 5. Running PCA...")
    beta_matrix_all = merged_df[valid_samples].values
    feature_ids_all = merged_df.index.values

    epsilon      = 1e-6
    beta_clipped = np.clip(beta_matrix_all, epsilon, 1 - epsilon)
    m_all        = np.log2(beta_clipped / (1 - beta_clipped))

    num_samples = len(valid_samples)
    num_sites   = len(merged_df)
    n_pcs       = min(5, num_samples, num_sites)

    pca         = PCA(n_components=n_pcs)
    pca_results = pca.fit_transform(m_all.T)

    pc_cols = [f'PC{i + 1}' for i in range(n_pcs)]
    pca_df  = pd.DataFrame(pca_results, columns=pc_cols)
    pca_df['Sample'] = valid_samples
    pca_df['Group']  = [sample_to_group[s] for s in valid_samples]

    if n_pcs >= 2:
        pairs     = list(combinations(range(n_pcs), 2))
        ncols     = 2
        nrows     = (len(pairs) + ncols - 1) // ncols
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
            ax.set_title(f"PCA: {pc_a} vs {pc_b}")
            ax.legend(frameon=False, loc='best', fontsize='small')

        for j in range(len(pairs), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "PCA_Pairs_AllSites.png"), dpi=300)
        plt.close()

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}_Loading' for i in range(n_pcs)],
        index=feature_ids_all
    )
    loadings['PC1_Abs'] = loadings['PC1_Loading'].abs()
    loadings.sort_values('PC1_Abs', ascending=False).to_csv(
        os.path.join(outdir, "PCA_Loadings.csv")
    )

    # ==========================================
    # Steps 6–8: Heatmap Generation
    # ==========================================
    use_all = (args.top_heatmap == 0 or args.top_heatmap >= len(merged_df))
    if use_all:
        print(f"[INFO] Step 6. Preparing heatmaps for all {len(merged_df)} DMRs...")
        top_dmrs   = merged_df
        actual_top = len(merged_df)
    else:
        print(f"[INFO] Step 6. Preparing heatmaps for top {args.top_heatmap} DMRs by effect size...")
        top_dmrs   = merged_df.head(args.top_heatmap)
        actual_top = args.top_heatmap

    heatmap_beta   = top_dmrs[valid_samples]
    m_top          = m_all[:actual_top, :]
    m_z_matrix     = zscore(m_top, axis=1)
    heatmap_zscore = pd.DataFrame(m_z_matrix, index=top_dmrs.index, columns=valid_samples)

    groups     = [sample_to_group[s] for s in valid_samples]
    lut        = {args.control_name: "#4DBBD5", args.case_name: "#E64B35"}
    col_colors = pd.Series(groups, index=valid_samples).map(lut)
    sns.set_theme(style="white", color_codes=True, font_scale=1.1)
    show_row_tree = (actual_top <= 200)

    def draw_heatmap(data_matrix, is_zscore, title_suffix, filename_suffix):
        """
        Render two clustermaps for a given data matrix:
          1. Group-constrained clustering: inter-group distance penalty forces
             Control/Case separation at the top-level dendrogram merge.
          2. Unconstrained hierarchical clustering: standard average-linkage Euclidean.

        Parameters
        ----------
        data_matrix : pd.DataFrame
            DMR-by-sample matrix to cluster and display.
        is_zscore : bool
            If True, use a diverging RdBu_r color scale centered at zero.
            If False, use a 0–1 RdYlBu_r scale for beta values.
        title_suffix : str
            Data type label for plot titles (e.g., 'Beta-value').
        filename_suffix : str
            Tag appended to output filenames (e.g., 'Beta').
        """
        cmap       = "RdBu_r" if is_zscore else "RdYlBu_r"
        vmin, vmax = (-3, 3) if is_zscore else (0, 1)
        cbar_label = "Z-score" if is_zscore else "Methylation Level"

        # Heatmap 1: Group-constrained column ordering
        col_linkage = get_group_constrained_linkage(data_matrix, valid_samples, sample_to_group)

        g1 = sns.clustermap(
            data_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
            col_linkage=col_linkage, row_cluster=True, col_colors=col_colors,
            figsize=(14, 10), yticklabels=False, xticklabels=True,
            tree_kws={'linewidths': 1.2, 'colors': '#333333'},
            cbar_pos=(0.02, 0.8, 0.03, 0.15), cbar_kws={'label': cbar_label}
        )
        g1.ax_heatmap.set_ylabel('')
        g1.ax_col_colors.tick_params(right=False, left=False, top=False, bottom=False)
        for spine in g1.ax_col_colors.spines.values():
            spine.set_visible(False)
        if not show_row_tree:
            g1.ax_row_dendrogram.set_visible(False)

        if use_all:
            g1.fig.suptitle(f"All Filtered DMRs (n={actual_top}) Heatmap\n"
                            f"Sorted by Group, Data = {title_suffix}",
                            fontweight='bold', fontsize=18, y=1.05)
            g1.savefig(os.path.join(outdir, f"AllSites_{filename_suffix}_byGroup.png"),
                       dpi=300, bbox_inches='tight')
        else:
            g1.fig.suptitle(f"Top {actual_top} DMRs Heatmap\n"
                            f"Sorted by Group, Data = {title_suffix}",
                            fontweight='bold', fontsize=18, y=1.05)
            g1.savefig(os.path.join(outdir, f"Top{actual_top}_{filename_suffix}_byGroup.png"),
                       dpi=300, bbox_inches='tight')
        plt.close(g1.fig)

        # Heatmap 2: Unconstrained hierarchical clustering
        g2 = sns.clustermap(
            data_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
            method='average', metric='euclidean',
            col_cluster=True, row_cluster=True, col_colors=col_colors,
            figsize=(14, 10), yticklabels=False, xticklabels=True,
            tree_kws={'linewidths': 1.2, 'colors': '#333333'},
            cbar_pos=(0.02, 0.8, 0.03, 0.15), cbar_kws={'label': cbar_label}
        )
        g2.ax_heatmap.set_ylabel('')
        g2.ax_col_colors.tick_params(right=False, left=False, top=False, bottom=False)
        for spine in g2.ax_col_colors.spines.values():
            spine.set_visible(False)
        if not show_row_tree:
            g2.ax_row_dendrogram.set_visible(False)

        if use_all:
            g2.fig.suptitle(f"All Filtered DMRs (n={actual_top}) Heatmap\n"
                            f"Sorted by Hierarchical Clustering, Data = {title_suffix}",
                            fontweight='bold', fontsize=18, y=1.05)
            g2.savefig(os.path.join(outdir, f"AllSites_{filename_suffix}_byHC.png"),
                       dpi=300, bbox_inches='tight')
        else:
            g2.fig.suptitle(f"Top {actual_top} DMRs Heatmap\n"
                            f"Sorted by Hierarchical Clustering, Data = {title_suffix}",
                            fontweight='bold', fontsize=18, y=1.05)
            g2.savefig(os.path.join(outdir, f"Top{actual_top}_{filename_suffix}_byHC.png"),
                       dpi=300, bbox_inches='tight')
        plt.close(g2.fig)

    print("[INFO] Step 7. Generating beta-value heatmaps...")
    draw_heatmap(heatmap_beta,   is_zscore=False, title_suffix="Beta-value",        filename_suffix="Beta")

    print("[INFO] Step 8. Generating M-value z-score heatmaps...")
    draw_heatmap(heatmap_zscore, is_zscore=True,  title_suffix="Z-score (M-value)", filename_suffix="ZM")

    print("[SUCCESS] All figures generated successfully.")


if __name__ == "__main__":
    main()
