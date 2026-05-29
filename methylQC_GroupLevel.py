#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ONT cfDNA CpG 5mC Methylation QC (Group-Level)

Description:
    A group-level methylation QC tool for ONT multi-sample TSV matrices.
    Optimized for cfDNA, this multiprocessing-enabled utility evaluates CpG
    methylation patterns across experimental groups. It systematically compares
    distributions of read depth, methylation level, and detection rates
    between control and case groups.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, matplotlib, seaborn
"""

import argparse
import os
import sys
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import time
import warnings
from collections import defaultdict
import logging
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# Global Constants & Configurations
# ==========================================
MAX_DEPTH_BIN = 1000
BETA_BINS = 100
CHUNK_SIZE = 100000

sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.2

COLORS = {'control': '#0072B2', 'case': '#D55E00', 'single': '#7F8C8D'}


# ==========================================
# Helper Function: Dual-Mode Group Matching
# ==========================================
def is_in_group(col_name: str, group_input: str) -> bool:
    """
    Dual-mode matching engine:
    1. Exact match (triggered by comma separation).
    2. Prefix auto-extraction (captures all characters before the first digit).
    """
    if not group_input:
        return False

    group_input = group_input.strip()

    # Mode 1: Exact match (comma detected)
    if ',' in group_input:
        allowed_samples = [s.strip() for s in group_input.split(',')]
        for sample in allowed_samples:
            if col_name == sample or col_name.startswith(f"{sample}_") or col_name.startswith(f"{sample}."):
                return True
        return False

    # Mode 2: Prefix auto-extraction (no comma)
    import re
    match = re.match(r'^([^0-9]+)', col_name)
    if match:
        extracted_prefix = match.group(1)
        if extracted_prefix == group_input:
            return True

    return False


# ==========================================
# BED Processing (Optimized O(log N) with ID Mapping)
# ==========================================
def parse_bed_to_arrays(bed_path):
    """Parses a BED file into dictionary of numpy arrays for vectorized interval matching."""
    logging.info(f"Parsing BED file for interval mapping: {bed_path}")
    intervals = defaultdict(list)
    with open(bed_path) as f:
        for i, line in enumerate(f):
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 3:
                intervals[parts[0]].append((int(parts[1]), int(parts[2]), i))

    merged = {}
    for chrom, ivs in intervals.items():
        ivs.sort(key=lambda x: x[0])
        starts = np.array([x[0] for x in ivs], dtype=np.int32)
        ends = np.array([x[1] for x in ivs], dtype=np.int32)
        ids = np.array([x[2] for x in ivs], dtype=np.int32)
        merged[chrom] = (starts, ends, ids)
    return merged


def check_promoter_and_get_ids(chroms, positions, bed_arrays):
    """Identifies CpG sites located within defined promoter intervals using binary search."""
    mask = np.zeros(len(positions), dtype=bool)
    p_ids = np.full(len(positions), -1, dtype=np.int32)
    df_temp = pd.DataFrame({'chr': chroms, 'pos': positions})

    for chrom, group in df_temp.groupby('chr'):
        if chrom in bed_arrays:
            starts, ends, ids = bed_arrays[chrom]
            pos_arr = group['pos'].values
            idx = np.searchsorted(ends, pos_arr, side='right')
            valid = idx < len(starts)

            if np.any(valid):
                valid_idx = np.where(valid)[0]
                in_interval = starts[idx[valid_idx]] <= pos_arr[valid_idx]
                final_valid = valid_idx[in_interval]
                global_indices = group.index.values[final_valid]

                mask[global_indices] = True
                p_ids[global_indices] = ids[idx[final_valid]]

    return mask, p_ids


# ==========================================
# MapReduce Worker Logic
# ==========================================
def parse_column_fast(series):
    """Extracts depth and 5mC beta values efficiently from comma-separated strings."""
    valid = series.replace('NA', np.nan).dropna()
    d_out = np.full(len(series), np.nan, dtype=np.float32)
    b_out = np.full(len(series), np.nan, dtype=np.float32)
    if valid.empty:
        return d_out, b_out

    triplets = valid.str.split(',', expand=True).astype(np.float32)

    depth = triplets[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        beta = triplets[1] / depth.replace(0, np.nan)

    d_out[valid.index] = depth.values
    b_out[valid.index] = beta.values
    return d_out, b_out


def process_chunk(args):
    """Processes a single DataFrame chunk to compute core methylation and depth statistics."""
    warnings.simplefilter("ignore", category=RuntimeWarning)

    chunk_df, groups_info, sample_cols, bed_arrays = args
    N = len(chunk_df)
    depth_mat = np.zeros((N, len(sample_cols)), dtype=np.float32)
    beta_mat = np.zeros((N, len(sample_cols)), dtype=np.float32)

    for i, col in enumerate(sample_cols):
        d_arr, b_arr = parse_column_fast(chunk_df[col].reset_index(drop=True))
        depth_mat[:, i] = d_arr
        beta_mat[:, i] = b_arr

    scopes = {'Global': np.ones(N, dtype=bool)}
    if bed_arrays:
        mask_promoter, p_ids = check_promoter_and_get_ids(chunk_df['#chr'].values, chunk_df['start'].values, bed_arrays)
        scopes['Promoter'] = mask_promoter

    res = {'Global': {}, 'Promoter': {}}

    for scope_name, mask in scopes.items():
        if mask.sum() == 0: continue
        d_mask = depth_mat[mask]
        b_mask = beta_mat[mask]

        res[scope_name]['sample_depth_hist'] = np.array([np.bincount(
            np.clip(np.nan_to_num(d_mask[:, i]), 0, MAX_DEPTH_BIN).astype(int), minlength=MAX_DEPTH_BIN + 1) for i in
            range(len(sample_cols))])
        res[scope_name]['sample_beta_hist'] = np.array(
            [np.histogram(b_mask[:, i][~np.isnan(b_mask[:, i])], bins=BETA_BINS, range=(0, 1))[0] for i in
             range(len(sample_cols))])
        res[scope_name]['sample_beta_sum'] = np.nansum(b_mask, axis=0)
        res[scope_name]['sample_beta_count'] = np.sum(~np.isnan(b_mask), axis=0)
        res[scope_name]['sample_depth_sum'] = np.nansum(d_mask, axis=0)

        res[scope_name]['groups'] = {}
        for g_name, g_idx, _ in groups_info:
            d_g = d_mask[:, g_idx]

            non_na_count = np.sum(~np.isnan(d_g), axis=1)
            det_count_hist = np.bincount(non_na_count, minlength=len(g_idx) + 1)

            res[scope_name]['groups'][g_name] = {
                'det_counts': det_count_hist,
            }
    return res


# ==========================================
# Visualization Implementations
# ==========================================
def plot_distribution(x_vals, hists, groups_info, title, xlabel, ylabel, filename, xlim=None):
    """Plots group-level distribution curves with Interquartile Range (IQR) shading."""
    warnings.simplefilter("ignore", category=RuntimeWarning)
    fig, ax = plt.subplots(figsize=(8, 6))
    for g_name, g_idx, color in groups_info:
        g_hists = hists[g_idx, :]

        with np.errstate(invalid='ignore'):
            median = np.nanmedian(g_hists, axis=0)
            p25 = np.nanpercentile(g_hists, 25, axis=0)
            p75 = np.nanpercentile(g_hists, 75, axis=0)

        ax.plot(x_vals, median, color=color, label=f'{g_name} Median', linewidth=2.5)
        ax.fill_between(x_vals, p25, p75, color=color, alpha=0.2, label=f'{g_name} IQR')

    if xlim is not None:
        ax.set_xlim(xlim)

    if "depth" in xlabel.lower():
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax.set_xlim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    sns.despine(ax=ax, trim=False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_detection_rate_lines(data_dict, groups_info, title, xlabel, ylabel, filename):
    """Plots detection rate distributions utilizing a smooth line graph geometry."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for g_name, g_idx, color in groups_info:
        counts = data_dict[g_name]
        N = len(g_idx)
        if N == 0: continue

        x_vals = np.arange(N + 1) / N
        ax.plot(x_vals, counts, color=color, label=g_name, linewidth=2.5, markersize=4)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

    ax.grid(axis='y', linestyle=':', alpha=0.6)
    sns.despine(ax=ax, trim=False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_sample_avg_hist(sample_means, groups_info, title, filename):
    """Generates stacked histograms representing the distribution of average methylation levels across samples."""
    n_groups = len(groups_info)
    fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 3 * n_groups), sharex=True)
    if n_groups == 1: axes = [axes]

    all_vals = []
    for _, g_idx, _ in groups_info:
        vals = sample_means[g_idx]
        vals = vals[~np.isnan(vals)]
        vals = np.round(vals * 100, 1)
        all_vals.extend(vals)

    all_vals = np.array(all_vals)

    if len(all_vals) > 0:
        min_v = np.nanmin(all_vals)
        max_v = np.nanmax(all_vals)

        if min_v == max_v:
            min_v = max(0.0, min_v - 2.0)
            max_v = min(100.0, max_v + 2.0)

        shared_bins = np.arange(np.floor(min_v) - 0.5, np.ceil(max_v) + 1.5, 1)
    else:
        shared_bins = 'auto'

    for ax, (g_name, g_idx, color) in zip(axes, groups_info):
        vals = sample_means[g_idx]
        vals = vals[~np.isnan(vals)]
        vals_pct = vals * 100

        if len(vals_pct) > 0:
            ax.hist(vals_pct, bins=shared_bins, alpha=0.6, color=color, label=g_name, edgecolor='white')

            median_val = np.nanmedian(vals_pct)
            mean_val = np.nanmean(vals_pct)

            ax.axvline(median_val, color=color, linestyle='dashed', linewidth=1.5,
                       label=f'Median: {median_val:.1f}%')
            ax.axvline(mean_val, color=color, linestyle='dotted', linewidth=1.5,
                       label=f'Mean: {mean_val:.1f}%')

        ax.set_ylabel("Sample Count")
        ax.set_title(g_name, color=color, fontweight='bold', loc='left')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        sns.despine(ax=ax, trim=False)
        ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

    axes[-1].set_xlabel("Average Methylation Level (%)")
    axes[-1].set_xlim(left=0, right=100)

    fig.suptitle(title, y=1.02, fontsize=15)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_depth_cdf_range(sample_depth_hist, groups_info, title, xlabel, filename, xlim=None):
    """
    Plots the Cumulative Distribution Function (CDF) of sequencing depth with IQR shading.
    Maintains a continuous smooth geometry, injecting explicit markers strictly at defined
    fractional thresholds (e.g., 0.2, 0.4) for quantitative interpretation.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x_vals = np.arange(sample_depth_hist.shape[1])
    target_ys = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    group_results = {}
    for g_idx_num, (g_name, g_idx, color) in enumerate(groups_info):
        if len(g_idx) == 0: continue

        cdfs = []
        for idx in g_idx:
            hist = sample_depth_hist[idx]
            total = np.nansum(hist)
            if total > 0:
                cdfs.append(np.cumsum(hist) / total)

        if cdfs:
            cdfs = np.array(cdfs)

            p25_cdf = np.percentile(cdfs, 25, axis=0)
            p75_cdf = np.percentile(cdfs, 75, axis=0)
            median_cdf = np.median(cdfs, axis=0)

            ax.fill_between(x_vals, p25_cdf, p75_cdf, color=color, alpha=0.2, linewidth=0)
            ax.plot(x_vals, median_cdf, color=color, linewidth=2)

            group_results[g_name] = {
                'median_cdf': median_cdf,
                'color': color,
                'idx_num': g_idx_num
            }

    for ty in target_ys:
        pts_to_draw = []
        for g_name, data in group_results.items():
            median_cdf = data['median_cdf']
            idx = np.argmax(median_cdf >= ty)
            if idx == 0 or median_cdf[idx] < ty:
                continue

            x0, y0 = x_vals[idx - 1], median_cdf[idx - 1]
            x1, y1 = x_vals[idx], median_cdf[idx]
            exact_x = x0 + (ty - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1

            pts_to_draw.append((exact_x, data['idx_num'], g_name, data['color']))

        if not pts_to_draw: continue

        pts_to_draw.sort(key=lambda item: (round(item[0], 1), item[1]))

        for rank, (exact_x, g_idx_num, g_name, color) in enumerate(pts_to_draw):
            ax.scatter(exact_x, ty, color=color, edgecolor='white', zorder=5, s=50)

            if rank < len(pts_to_draw) / 2.0:
                offset = (-12, 0)
                ha_align = 'right'
            else:
                offset = (12, 0)
                ha_align = 'left'

            ax.annotate(f"{exact_x:.1f}", (exact_x, ty),
                        textcoords="offset points", xytext=offset,
                        ha=ha_align, va='center', fontsize=10, color=color, fontweight='bold')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Fraction of CpGs")
    if xlim: ax.set_xlim(xlim)
    ax.set_ylim(0, 1.05)

    for ty in target_ys:
        ax.axhline(ty, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.grid(axis='x', linestyle=':', alpha=0.6)
    sns.despine(ax=ax, trim=False)

    legend_elements = []
    for g_name, _, color in groups_info:
        legend_elements.append(Line2D([0], [0], color=color, lw=2, marker='o', markersize=5, label=f"{g_name} Median"))
        legend_elements.append(Patch(facecolor=color, alpha=0.2, edgecolor='none', label=f"{g_name} IQR"))

    ax.legend(handles=legend_elements, frameon=False, loc='lower right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_det_rate_cdf(det_data, groups_info, title, xlabel, filename):
    """
    Plots the Cumulative Distribution Function (CDF) of CpG detection rates.
    Implements dynamic numerical interpolation for targeted percentile annotation.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    target_ys = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    group_results = {}
    for g_idx_num, (g_name, g_idx, color) in enumerate(groups_info):
        hist = det_data[g_name]
        N = len(g_idx)
        if N == 0: continue

        x_vals = np.arange(N + 1) / N
        total = np.sum(hist)

        if total > 0:
            cdf = np.cumsum(hist) / total
            ax.plot(x_vals, cdf, color=color, linewidth=2.5)

            group_results[g_name] = {
                'cdf': cdf,
                'x_vals': x_vals,
                'color': color,
                'idx_num': g_idx_num
            }

    for ty in target_ys:
        pts_to_draw = []

        for g_name, data in group_results.items():
            cdf = data['cdf']
            x_vals = data['x_vals']
            idx = np.argmax(cdf >= ty)
            if idx == 0 or cdf[idx] < ty:
                continue

            x0, y0 = x_vals[idx - 1], cdf[idx - 1]
            x1, y1 = x_vals[idx], cdf[idx]
            exact_x = x0 + (ty - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1

            pts_to_draw.append((exact_x, data['idx_num'], g_name, data['color']))

        if not pts_to_draw: continue

        pts_to_draw.sort(key=lambda item: (round(item[0], 3), item[1]))

        for rank, (exact_x, g_idx_num, g_name, color) in enumerate(pts_to_draw):
            ax.scatter(exact_x, ty, color=color, edgecolor='white', zorder=5, s=50)

            if rank < len(pts_to_draw) / 2.0:
                offset = (-12, 0)
                ha_align = 'right'
            else:
                offset = (12, 0)
                ha_align = 'left'

            display_val = exact_x * 100
            ax.annotate(f"{display_val:.1f}", (exact_x, ty),
                        textcoords="offset points", xytext=offset,
                        ha=ha_align, va='center', fontsize=10, color=color, fontweight='bold')

    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=''))
    if "(%)" not in xlabel:
        xlabel = f"{xlabel} (%)"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Fraction of CpGs")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)

    for ty in target_ys:
        ax.axhline(ty, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.grid(axis='x', linestyle=':', alpha=0.6)
    sns.despine(ax=ax, trim=False)

    legend_elements = []
    for g_name, _, color in groups_info:
        legend_elements.append(Line2D([0], [0], color=color, lw=2.5, marker='o', markersize=5, label=g_name))
    ax.legend(handles=legend_elements, frameon=False, loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ==========================================
# Main Orchestration & Subprocess Invocation
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="ONT cfDNA CpG 5mC Methylation QC Analysis and Visualization (Group-Level)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Core Input/Output Configurations
    parser.add_argument('-i', '--input', required=True,
                        help="Path to the input multi-sample methylation matrix (TSV format). Example: Union.bed")
    parser.add_argument('-b', '--bed',
                        help="BED file defining specific genomic regions (e.g., TSS/Promoters) for targeted spatial profiling.")
    parser.add_argument('-o', '--outdir', required=True, type=str,
                        help="Target directory for output files. It will be created automatically if it does not exist.")
    parser.add_argument('-t', '--threads', type=int, default=max(1, multiprocessing.cpu_count() - 2),
                        help="Number of dedicated CPU threads for parallel execution. (Default: Available Cores - 2)")

    # Group Settings & Dual-Mode Engine
    parser.add_argument('--control', type=str, default="",
                        help="Identifiers for the Control group. Modes: 1) Exact match (e.g., 'C1,C2,C3') 2) Prefix match auto-extracting letters before the first number (e.g., 'C'). Leave empty if single-group analysis.")
    parser.add_argument('--case', type=str, default="",
                        help="Identifiers for the Case group. Supports the same matching modes as --control. Leave empty if single-group analysis.")
    parser.add_argument('--control_name', type=str, default="Control",
                        help="Display nomenclature for the Control group in generated plots and reports. (Default: 'Control')")
    parser.add_argument('--case_name', type=str, default="Case",
                        help="Display nomenclature for the Case group in generated plots and reports. (Default: 'Case')")

    # Output & Plotting Controls
    parser.add_argument('--plots', type=str, default="1,2,3,4,5,6,7,8,9,10,11,12",
                        help="Comma-separated numeric string dictating which figures to render (1-12).\n"
                             "(Default: All plots 1-12)")
    parser.add_argument('--report_only', action='store_true',
                        help="Flag to bypass visual rendering and exclusively output the statistical\n"
                             "Markdown report. Useful for fast CI/CD pipelines or headless servers.")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_to_run = [int(p) for p in args.plots.split(',')]

    with open(args.input, 'r') as f:
        header = f.readline().strip().split('\t')
    all_sample_cols = header[3:]

    def match_cols(targets):
        if not targets:
            return []

        matched = []
        for c in all_sample_cols:
            if is_in_group(c, targets):
                matched.append(c)
        return matched

    c_cols = match_cols(args.control)
    k_cols = match_cols(args.case)

    if not c_cols and not k_cols:
        logging.error("Critical: No samples matched the provided group criteria.")
        sys.exit(1)

    final_cols = c_cols + k_cols
    col_to_idx = {col: i for i, col in enumerate(final_cols)}

    groups_info = []
    if c_cols and k_cols:
        groups_info.append((args.control_name, [col_to_idx[c] for c in c_cols], COLORS['control']))
        groups_info.append((args.case_name, [col_to_idx[c] for c in k_cols], COLORS['case']))
    elif c_cols:
        groups_info.append((args.control_name, [col_to_idx[c] for c in c_cols], COLORS['single']))
    elif k_cols:
        groups_info.append((args.case_name, [col_to_idx[c] for c in k_cols], COLORS['single']))

    logging.info(f"Target Groups: {len(c_cols)} Controls, {len(k_cols)} Cases. Allocated Threads: {args.threads}")

    bed_arrays = parse_bed_to_arrays(args.bed) if args.bed else None

    agg = {
        scope: {
            'sample_depth_hist': np.zeros((len(final_cols), MAX_DEPTH_BIN + 1), dtype=np.int64),
            'sample_beta_hist': np.zeros((len(final_cols), BETA_BINS), dtype=np.int64),
            'sample_beta_sum': np.zeros(len(final_cols), dtype=np.float64),
            'sample_beta_count': np.zeros(len(final_cols), dtype=np.int64),
            'sample_depth_sum': np.zeros(len(final_cols), dtype=np.float64),
            'total_rows': 0,
            'groups': {g[0]: {
                'det_counts': np.zeros(len(g[1]) + 1, dtype=np.int64),
            } for g in groups_info}
        } for scope in (['Global', 'Promoter'] if bed_arrays else ['Global'])
    }

    start_time = time.time()
    chunk_iterator = pd.read_csv(args.input, sep='\t', chunksize=CHUNK_SIZE,
                                 usecols=['#chr', 'start', 'end'] + final_cols)
    pool = multiprocessing.Pool(processes=args.threads)

    def arg_generator():
        for chunk in chunk_iterator: yield (chunk, groups_info, final_cols, bed_arrays)

    chunks_processed = 0
    for res in pool.imap_unordered(process_chunk, arg_generator()):
        chunks_processed += 1
        if chunks_processed % 10 == 0: logging.info(
            f"Processed {chunks_processed * CHUNK_SIZE:,} genomic coordinates...")

        for scope in agg.keys():
            if scope not in res: continue
            r_scope = res[scope]
            if 'sample_depth_hist' not in r_scope: continue

            agg[scope]['sample_depth_hist'] += r_scope['sample_depth_hist']
            agg[scope]['sample_beta_hist'] += r_scope['sample_beta_hist']
            agg[scope]['sample_beta_sum'] += r_scope['sample_beta_sum']
            agg[scope]['sample_beta_count'] += r_scope['sample_beta_count']
            agg[scope]['sample_depth_sum'] += r_scope['sample_depth_sum']
            agg[scope]['total_rows'] += r_scope['sample_depth_hist'][0].sum()

            for g_name, _, _ in groups_info:
                agg[scope]['groups'][g_name]['det_counts'] += r_scope['groups'][g_name]['det_counts']

    pool.close()
    pool.join()
    logging.info(f"Data aggregation completed natively in {(time.time() - start_time) / 60:.1f} minutes.")

    # Generate Statistical Markdown Report
    report_path = os.path.join(args.outdir, "Group_Methylation_QC_Report.md")
    with open(report_path, "w") as f:
        f.write("# Group-Level Methylation QC Report\n\n")
        for scope in agg.keys():
            f.write(f"## {scope} Epigenomic Analysis\n")
            f.write(f"- Processed CpG Sites: **{agg[scope]['total_rows']:,}**\n")
            for g_name, g_idx, _ in groups_info:
                g_avg_depth = np.nansum(agg[scope]['sample_depth_sum'][g_idx]) / max(1, np.sum(
                    agg[scope]['sample_beta_count'][g_idx]))
                g_avg_beta = np.nansum(agg[scope]['sample_beta_sum'][g_idx]) / max(1, np.sum(
                    agg[scope]['sample_beta_count'][g_idx]))
                f.write(f"\n### {g_name} Group\n")
                f.write(f"- Average Read Depth: {g_avg_depth:.2f}x\n")
                f.write(f"- Global Average Methylation Fraction (\u03B2): {g_avg_beta:.4f}\n")

    if args.report_only: return

    logging.info("Initializing vector graphics rendering...")
    x_depth = np.arange(MAX_DEPTH_BIN + 1)
    x_beta_edges = np.linspace(0, 1, BETA_BINS + 1)
    x_beta_centers = 0.5 * (x_beta_edges[1:] + x_beta_edges[:-1])

    for scope, p_offsets in zip(['Global', 'Promoter'], [0, 1]):
        if scope not in agg: continue
        a = agg[scope]

        total_depth_counts = np.sum(a['sample_depth_hist'], axis=0)
        cum_counts = np.cumsum(total_depth_counts)
        if cum_counts[-1] > 0:
            p99_depth = int(np.searchsorted(cum_counts, cum_counts[-1] * 0.99))
        else:
            p99_depth = MAX_DEPTH_BIN
        p99_depth = max(10, p99_depth)

        depth_xlim = (0, p99_depth)

        # Rendering Engine
        if (1 + p_offsets) in plots_to_run:
            plot_depth_hist = a['sample_depth_hist'].astype(float).copy()
            plot_depth_hist[:, 0] = np.nan

            plot_distribution(x_depth, plot_depth_hist, groups_info,
                              f"{scope}: Read Depth Distribution", "Read Depth", "CpG Count",
                              os.path.join(args.outdir, f"Fig{1 + p_offsets:02d}_{scope}_Depth_Dist.png"),
                              xlim=depth_xlim)

        if (3 + p_offsets) in plots_to_run:
            plot_distribution(x_beta_centers, a['sample_beta_hist'], groups_info,
                              f"{scope}: Methylation Distribution", "Methylation Level (%)",
                              "CpG Count",
                              os.path.join(args.outdir, f"Fig{3 + p_offsets:02d}_{scope}_Methyl_Dist.png"))

        if (5 + p_offsets) in plots_to_run:
            det_data = {g: a['groups'][g]['det_counts'] for g in a['groups']}
            plot_detection_rate_lines(det_data, groups_info, f"{scope}: Detection Rate Distribution",
                                      "Detection Rate (%)", "CpG Count",
                                      os.path.join(args.outdir, f"Fig{5 + p_offsets:02d}_{scope}_DetRate_Dist.png"))

        if (7 + p_offsets) in plots_to_run:
            plot_depth_cdf_range(a['sample_depth_hist'], groups_info,
                                 f"{scope}: Cumulative Distribution of Read Depth",
                                 "Read Depth",
                                 os.path.join(args.outdir, f"Fig{7 + p_offsets:02d}_{scope}_Depth_CDF.png"),
                                 xlim=depth_xlim)

        if (9 + p_offsets) in plots_to_run:
            det_data = {g: a['groups'][g]['det_counts'] for g in a['groups']}
            plot_det_rate_cdf(det_data, groups_info,
                              f"{scope}: Cumulative Distribution of Detection Rate",
                              "Detection Rate (%)",
                              os.path.join(args.outdir, f"Fig{9 + p_offsets:02d}_{scope}_DetRate_CDF.png"))

        if (11 + p_offsets) in plots_to_run:
            with np.errstate(invalid='ignore'):
                s_means = a['sample_beta_sum'] / a['sample_beta_count']
            plot_sample_avg_hist(s_means, groups_info, f"{scope}: Distribution of Average Methylation across Samples",
                                 os.path.join(args.outdir, f"Fig{11 + p_offsets:02d}_{scope}_Sample_Methyl_Dist.png"))

    logging.info(f"[Done] QC Analysis successfully completed! Artifacts securely written to {args.outdir}")


if __name__ == "__main__":
    main()