#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
methylQC_GroupLevel_TSS.py

Description:
    Group-level CpG methylation QC tool for spatial profiling across transcription start site
    (TSS) regions. Designed for multi-sample TSV matrices produced by CreatUnionBed.py.
    Profiles CpG count, read depth, methylation level, and detection rate in 50bp bins
    within a +/- 2000bp window around each TSS. Supports optional depth and detection-rate
    filtering, and outputs per-group spatial plots and a Markdown QC report.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, matplotlib, seaborn
"""

import argparse
import os
import sys
import logging
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# Global Visualization Configuration
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.2

# Fixed color assignments for visual consistency across the pipeline
COLOR_CONTROL = '#1f77b4'  # Blue
COLOR_CASE    = '#ff7f0e'  # Orange

# Spatial bin configuration: -2000 to +2000bp in 50bp steps (80 bins total)
BIN_SIZE    = 50
WINDOW_SIZE = 2000
BINS        = np.arange(-WINDOW_SIZE, WINDOW_SIZE + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BINS[:-1] + (BIN_SIZE / 2)
NUM_BINS    = len(BIN_CENTERS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group-level TSS QC: spatial profiling of CpG methylation and read depth across TSS regions."
    )

    # I/O Arguments
    parser.add_argument('-i', '--input', required=True,
                        help="Input multi-sample methylation matrix (TSV). Example: Union.bed")
    parser.add_argument('-b', '--bed',
                        help="BED file defining specific genomic regions (e.g., TSS/Promoters) for targeted spatial profiling.")
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

    # QC Filtering Parameters
    parser.add_argument('--min-depth', type=int, default=0,
                        help="Minimum read depth threshold. Sites below this depth are masked as NaN. (Default: 0)")
    parser.add_argument('--min-det-rate', type=float, default=0.0,
                        help="Minimum group-wise detection rate (0.0–1.0). Sites below this rate are masked as NaN. (Default: 0.0)")

    return parser.parse_args()


def load_tss_bed(bed_path):
    """
    Load a TSS BED file and build a chromosome-keyed lookup dictionary.

    Computes the midpoint of each region and sorts by chromosome and midpoint
    to enable O(log N) spatial matching via pandas.merge_asof downstream.
    """
    logging.info(f"Loading TSS BED from {bed_path}...")
    df = pd.read_csv(bed_path, sep='\t', header=None, comment='#',
                     usecols=[0, 1, 2, 4], names=['chr', 'start', 'end', 'strand'])
    df['midpoint'] = (df['start'] + df['end']) // 2
    df.sort_values(['chr', 'midpoint'], inplace=True)

    tss_dict = {chrom: sub_df[['midpoint', 'strand']].reset_index(drop=True)
                for chrom, sub_df in df.groupby('chr')}
    return tss_dict


def process_chunk(chunk_df, tss_dict, groups, final_cols, min_depth, min_det_rate):
    """
    Multiprocessing worker: maps CpG sites to the nearest TSS, parses depth and
    methylation counts, applies sequential QC filters (depth then detection rate),
    and accumulates per-sample metrics into 50bp spatial bins.

    Parameters
    ----------
    chunk_df : pd.DataFrame
        A chunk of the input methylation matrix.
    tss_dict : dict
        Chromosome-keyed DataFrames of TSS midpoints and strand from load_tss_bed().
    groups : list of dict
        Group definitions including sample indices and display metadata.
    final_cols : list of str
        Ordered list of sample column names.
    min_depth : int
        Minimum read depth for a site to be retained.
    min_det_rate : float
        Minimum detection rate (proportion of samples with valid depth) per group.

    Returns
    -------
    dict or None
        Dictionary with keys 'U' (unfiltered) and 'F' (filtered), each containing
        accumulated bin-level statistics, or None if no sites map to a TSS.
    """
    # Step 1: O(N log M) spatial TSS mapping via nearest-neighbor merge
    mapped_chunks = []
    for chrom, grp in chunk_df.groupby('chr'):
        if chrom not in tss_dict:
            continue
        bed_df = tss_dict[chrom]
        grp = grp.sort_values('start')
        merged = pd.merge_asof(grp, bed_df, left_on='start', right_on='midpoint',
                               direction='nearest', tolerance=WINDOW_SIZE)
        merged = merged.dropna(subset=['midpoint'])
        mapped_chunks.append(merged)

    if not mapped_chunks:
        return None

    valid = pd.concat(mapped_chunks)

    # Compute strand-aware distance relative to the TSS midpoint
    valid['distance'] = valid['start'] - valid['midpoint']
    valid.loc[valid['strand'] == '-', 'distance'] *= -1

    # Assign each site to a 50bp bin index (0 to NUM_BINS - 1)
    bin_idx = ((valid['distance'] + WINDOW_SIZE) // BIN_SIZE).astype(int).values
    bin_idx = np.clip(bin_idx, 0, NUM_BINS - 1)

    num_samples = len(final_cols)
    num_sites   = len(valid)

    # Step 2: Parse depth and methylation count from "depth,mc" formatted columns
    depth_arr = np.zeros((num_sites, num_samples), dtype=np.float32)
    mc_arr    = np.zeros((num_sites, num_samples), dtype=np.float32)

    for i, col in enumerate(final_cols):
        s_data = valid[col].astype(str).str.split(',', expand=True)
        if s_data.shape[1] >= 2:
            depth_arr[:, i] = pd.to_numeric(s_data[0], errors='coerce').values
            mc_arr[:, i]    = pd.to_numeric(s_data[1], errors='coerce').values
        else:
            depth_arr[:, i] = np.nan
            mc_arr[:, i]    = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        beta_arr = (mc_arr / depth_arr) * 100.0

    # Step 3: Sequential QC filtering — depth filter followed by detection rate filter
    apply_filters = (min_depth > 0) or (min_det_rate > 0.0)

    if apply_filters:
        # Phase A: Mask sites below the minimum depth threshold
        d_mask     = depth_arr >= min_depth if min_depth > 0 else (~np.isnan(depth_arr))
        filt_depth = np.where(d_mask, depth_arr, np.nan)
        filt_beta  = np.where(d_mask, beta_arr,  np.nan)

        # Phase B: Mask sites below the minimum group-wise detection rate
        if min_det_rate > 0.0:
            for g in groups:
                idx = g['idx']
                if len(idx) == 0:
                    continue

                g_valid    = ~np.isnan(filt_depth[:, idx])
                g_det_rate = np.sum(g_valid, axis=1) / len(idx)
                bad_sites  = g_det_rate < min_det_rate

                bad_row_indices = np.where(bad_sites)[0]
                if len(bad_row_indices) > 0:
                    for c_idx in idx:
                        filt_depth[bad_row_indices, c_idx] = np.nan
                        filt_beta[bad_row_indices, c_idx]  = np.nan
    else:
        filt_depth = depth_arr
        filt_beta  = beta_arr

    # Step 4: Initialize spatial bin accumulators (unfiltered and filtered)
    res_U = {'d_sum': np.zeros((NUM_BINS, num_samples)), 'd_cnt': np.zeros((NUM_BINS, num_samples)),
             'b_sum': np.zeros((NUM_BINS, num_samples)), 'b_cnt': np.zeros((NUM_BINS, num_samples)),
             'cpg_cnt': np.zeros((NUM_BINS, num_samples)), 'site_cnt': np.zeros(NUM_BINS)}

    res_F = {'d_sum': np.zeros((NUM_BINS, num_samples)), 'd_cnt': np.zeros((NUM_BINS, num_samples)),
             'b_sum': np.zeros((NUM_BINS, num_samples)), 'b_cnt': np.zeros((NUM_BINS, num_samples)),
             'cpg_cnt': np.zeros((NUM_BINS, num_samples)), 'site_cnt': np.zeros(NUM_BINS)}

    # Step 5: Accumulate per-bin statistics
    for b in range(NUM_BINS):
        mask_b       = (bin_idx == b)
        sites_in_bin = np.sum(mask_b)
        if sites_in_bin == 0:
            continue

        res_U['site_cnt'][b] = sites_in_bin
        res_F['site_cnt'][b] = sites_in_bin

        # Unfiltered accumulation
        u_d     = depth_arr[mask_b, :]
        u_b     = beta_arr[mask_b, :]
        u_val_d = ~np.isnan(u_d)
        u_val_b = ~np.isnan(u_b)
        res_U['d_sum'][b, :]   = np.nansum(u_d, axis=0)
        res_U['d_cnt'][b, :]   = np.sum(u_val_d, axis=0)
        res_U['b_sum'][b, :]   = np.nansum(u_b, axis=0)
        res_U['b_cnt'][b, :]   = np.sum(u_val_b, axis=0)
        res_U['cpg_cnt'][b, :] = np.sum(u_val_d, axis=0)

        # Filtered accumulation
        f_d     = filt_depth[mask_b, :]
        f_b     = filt_beta[mask_b, :]
        f_val_d = ~np.isnan(f_d)
        f_val_b = ~np.isnan(f_b)
        res_F['d_sum'][b, :]   = np.nansum(f_d, axis=0)
        res_F['d_cnt'][b, :]   = np.sum(f_val_d, axis=0)
        res_F['b_sum'][b, :]   = np.nansum(f_b, axis=0)
        res_F['b_cnt'][b, :]   = np.sum(f_val_b, axis=0)
        res_F['cpg_cnt'][b, :] = np.sum(f_val_d, axis=0)

    return {'U': res_U, 'F': res_F}


def compute_metrics(mast, groups):
    """
    Compute per-bin mean and interquartile range (IQR) for each group.

    Uses np.nanpercentile across the sample axis to capture inter-sample
    variation per spatial bin while ignoring missing values.

    Parameters
    ----------
    mast : dict
        Accumulated bin-level statistics from the master accumulator.
    groups : list of dict
        Group definitions including sample indices and display metadata.

    Returns
    -------
    dict
        Nested dictionary of per-metric, per-group statistics (mean, q25, q75, color).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        s_mean_depth = mast['d_sum'] / mast['d_cnt']
        s_mean_beta  = mast['b_sum'] / mast['b_cnt']

    stats = {'depth': {}, 'beta': {}, 'cpg': {}, 'det_rate': {}}

    for g in groups:
        name  = g['name']
        idx   = g['idx']
        color = g['color']

        if len(idx) == 0:
            continue

        # Read depth statistics
        g_d = s_mean_depth[:, idx]
        stats['depth'][name] = {
            'mean':  np.nanmean(g_d, axis=1),
            'q25':   np.nanpercentile(g_d, 25, axis=1),
            'q75':   np.nanpercentile(g_d, 75, axis=1),
            'color': color
        }

        # Methylation level (beta) statistics
        g_b = s_mean_beta[:, idx]
        stats['beta'][name] = {
            'mean':  np.nanmean(g_b, axis=1),
            'q25':   np.nanpercentile(g_b, 25, axis=1),
            'q75':   np.nanpercentile(g_b, 75, axis=1),
            'color': color
        }

        # CpG count statistics
        g_cpg = mast['cpg_cnt'][:, idx]
        stats['cpg'][name] = {
            'mean':  np.nanmean(g_cpg, axis=1),
            'q25':   np.nanpercentile(g_cpg, 25, axis=1),
            'q75':   np.nanpercentile(g_cpg, 75, axis=1),
            'color': color
        }

        # Bin-level detection rate: valid CpG observations relative to total capacity
        valid_cpgs = np.nansum(mast['cpg_cnt'][:, idx], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            g_det = (valid_cpgs / (mast['site_cnt'] * len(idx))) * 100.0

        # Population-level aggregate — IQR is flat by definition
        stats['det_rate'][name] = {
            'mean':  g_det,
            'q25':   g_det,
            'q75':   g_det,
            'color': color
        }

    return stats


def plot_metric(data_dict, title, ylabel, ylim, out_path):
    """
    Generate a spatial profile plot showing group mean and IQR band across TSS bins.

    Parameters
    ----------
    data_dict : dict
        Per-group statistics with keys 'mean', 'q25', 'q75', 'color'.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    ylim : tuple or None
        Y-axis limits, or None for auto-scaling.
    out_path : str
        Output file path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, stats in data_dict.items():
        color = stats['color']
        ax.plot(BIN_CENTERS, stats['mean'], color=color, linewidth=2.5, label=f"{name} Mean")
        ax.fill_between(BIN_CENTERS, stats['q25'], stats['q75'],
                        color=color, alpha=0.25, edgecolor="none", label=f"{name} IQR")

    ax.set_title(title)
    ax.set_xlabel("Distance to TSS (bp)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-WINDOW_SIZE, WINDOW_SIZE)
    if ylim:
        ax.set_ylim(ylim)

    ax.legend(frameon=False, loc='best')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    sns.despine(ax=ax, trim=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_text_report(args, mast_U, mast_F, groups):
    """
    Write a Markdown QC report summarizing pre- and post-filter statistics per group.

    Reports include data retention rate, average read depth, global methylation fraction,
    and global detection rate for each group across all TSS-proximal CpG sites.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    mast_U : dict
        Unfiltered master accumulator.
    mast_F : dict
        Filtered master accumulator.
    groups : list of dict
        Group definitions including sample indices and display metadata.
    """
    report_path     = os.path.join(args.outdir, "Group_Methylation_TSS_QC_Report.md")
    total_cpg_sites = int(np.sum(mast_U['site_cnt']))

    with open(report_path, 'w') as f:
        f.write("# Group-Level Methylation TSS QC Report\n\n")

        f.write("## TSS Epigenomic Analysis\n")
        f.write(f"- Processed CpG Sites: **{total_cpg_sites:,}**\n")
        f.write(f"- Analyzed Region: +/- {WINDOW_SIZE} bp ({BIN_SIZE}bp bins)\n")

        filters_applied = args.min_depth > 0 or args.min_det_rate > 0.0
        if filters_applied:
            f.write(f"- Applied Filters: Min Depth >= {args.min_depth}, Min Detection Rate >= {args.min_det_rate}\n")
        else:
            f.write("- Applied Filters: None\n")
        f.write("\n")

        for g in groups:
            name        = g['name']
            idx         = g['idx']
            num_samples = len(idx)
            if num_samples == 0:
                continue

            total_possible = total_cpg_sites * num_samples

            # Unfiltered metrics
            u_valid_pts = np.sum(mast_U['d_cnt'][:, idx])
            u_mean_depth = (np.nansum(mast_U['d_sum'][:, idx]) / u_valid_pts
                            if u_valid_pts > 0 else 0)
            u_b_cnt     = np.sum(mast_U['b_cnt'][:, idx])
            u_mean_beta = (np.nansum(mast_U['b_sum'][:, idx]) / u_b_cnt
                           if u_b_cnt > 0 else 0)
            u_det_rate  = (u_valid_pts / total_possible) * 100 if total_possible > 0 else 0

            f.write(f"### {name} Group\n")
            f.write(f"- Analyzed Samples: {num_samples}\n")

            if filters_applied:
                # Filtered metrics
                f_valid_pts = np.sum(mast_F['d_cnt'][:, idx])
                f_mean_depth = (np.nansum(mast_F['d_sum'][:, idx]) / f_valid_pts
                                if f_valid_pts > 0 else 0)
                f_b_cnt     = np.sum(mast_F['b_cnt'][:, idx])
                f_mean_beta = (np.nansum(mast_F['b_sum'][:, idx]) / f_b_cnt
                               if f_b_cnt > 0 else 0)
                f_det_rate  = (f_valid_pts / total_possible) * 100 if total_possible > 0 else 0

                filtered_out_pts = u_valid_pts - f_valid_pts
                retention_rate   = (f_valid_pts / u_valid_pts) * 100 if u_valid_pts > 0 else 0

                f.write(f"- Data Retention Rate: **{retention_rate:.2f}%**\n")
                f.write(f"- Retained Data Points: {int(f_valid_pts):,} (Filtered out: {int(filtered_out_pts):,})\n")
                f.write(f"- Average Read Depth: {u_mean_depth:.2f}x -> **{f_mean_depth:.2f}x**\n")
                f.write(f"- Global Average Methylation Fraction (β): {u_mean_beta / 100:.4f} -> **{f_mean_beta / 100:.4f}**\n")
                f.write(f"- Global Detection Rate: {u_det_rate:.2f}% -> **{f_det_rate:.2f}%**\n")
            else:
                f.write(f"- Total Data Points: {int(u_valid_pts):,}\n")
                f.write(f"- Average Read Depth: **{u_mean_depth:.2f}x**\n")
                f.write(f"- Global Average Methylation Fraction (β): **{u_mean_beta / 100:.4f}**\n")
                f.write(f"- Global Detection Rate: **{u_det_rate:.2f}%**\n")

            f.write("\n")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tss_dict = load_tss_bed(args.bed)

    logging.info("Parsing UnionBed header...")
    with open(args.input, 'r') as f:
        header_line = f.readline().strip()

    col_names = header_line.split('\t')
    if col_names[0].startswith('#'):
        col_names[0] = col_names[0].lstrip('#')

    all_sample_cols = col_names[3:]

    # ==========================================
    # Sample Column Matching (Two Modes)
    # ==========================================
    def match_cols(targets):
        """
        Match sample column names against group identifiers.

        Mode 1 (comma-separated): exact name match or name-prefix match.
        Mode 2 (single prefix): extracts the alphabetic prefix before the first digit
        and matches columns sharing that prefix.
        """
        if not targets:
            return []

        import re
        targets = targets.strip()
        matched = []

        for col_name in all_sample_cols:
            is_matched = False

            if ',' in targets:
                # Mode 1: exact or prefix-delimited match against a comma-separated list
                allowed_samples = [s.strip() for s in targets.split(',')]
                for sample in allowed_samples:
                    if (col_name == sample
                            or col_name.startswith(f"{sample}_")
                            or col_name.startswith(f"{sample}.")):
                        is_matched = True
                        break
            else:
                # Mode 2: match columns whose leading alphabetic prefix equals the target
                match = re.match(r'^([^0-9]+)', col_name)
                if match:
                    extracted_prefix = match.group(1)
                    if extracted_prefix == targets:
                        is_matched = True

            if is_matched:
                matched.append(col_name)

        return matched

    c_cols = match_cols(args.control)
    k_cols = match_cols(args.case)

    if not c_cols and not k_cols:
        logging.error("No samples matched the provided group identifiers.")
        sys.exit(1)

    final_cols  = c_cols + k_cols
    col_to_idx  = {col: i for i, col in enumerate(final_cols)}

    groups = []
    if c_cols:
        groups.append({'name': args.control_name, 'cols': c_cols,
                        'idx': np.array([col_to_idx[c] for c in c_cols], dtype=int),
                        'color': COLOR_CONTROL})
    if k_cols:
        groups.append({'name': args.case_name, 'cols': k_cols,
                        'idx': np.array([col_to_idx[c] for c in k_cols], dtype=int),
                        'color': COLOR_CASE})

    logging.info(f"Groups: {len(c_cols)} control(s), {len(k_cols)} case(s). Threads: {args.threads}")

    # ==========================================
    # Master Accumulator Initialization
    # ==========================================
    num_samples = len(final_cols)

    def make_mast():
        return {
            'd_sum':    np.zeros((NUM_BINS, num_samples)),
            'd_cnt':    np.zeros((NUM_BINS, num_samples)),
            'b_sum':    np.zeros((NUM_BINS, num_samples)),
            'b_cnt':    np.zeros((NUM_BINS, num_samples)),
            'cpg_cnt':  np.zeros((NUM_BINS, num_samples)),
            'site_cnt': np.zeros(NUM_BINS)
        }

    mast_U = make_mast()
    mast_F = make_mast()

    logging.info("Starting multi-process TSS mapping...")
    pool = multiprocessing.Pool(processes=args.threads)

    chunk_iter = pd.read_csv(args.input, sep='\t', chunksize=100000,
                             names=col_names, header=0,
                             dtype={'chr': str, 'start': int, 'end': int})

    worker_func = partial(process_chunk, tss_dict=tss_dict, groups=groups,
                          final_cols=final_cols, min_depth=args.min_depth,
                          min_det_rate=args.min_det_rate)

    for i, res in enumerate(pool.imap_unordered(worker_func, chunk_iter)):
        if res is not None:
            for k in mast_U.keys():
                mast_U[k] += res['U'][k]
                mast_F[k] += res['F'][k]
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {(i + 1) * 100000} lines...")

    pool.close()
    pool.join()
    logging.info("Aggregation complete. Generating spatial profile plots...")

    # ==========================================
    # Output: Figures and QC Report
    # ==========================================
    def generate_all_figures(mast, suffix):
        stats = compute_metrics(mast, groups)

        plot_metric(stats['cpg'],
                    "CpG Distribution across TSS", "Mean CpG Count", None,
                    os.path.join(args.outdir, f"Fig13_TSS_CpG_Dist_{suffix}.png"))

        plot_metric(stats['depth'],
                    "Read Depth Profile across TSS", "Mean Read Depth", None,
                    os.path.join(args.outdir, f"Fig14_TSS_Read_Depth_Profile_{suffix}.png"))

        plot_metric(stats['beta'],
                    "Methylation Profile across TSS", "Mean Methylation Level (%)", (-5, 105),
                    os.path.join(args.outdir, f"Fig15_TSS_Methyl_Profile_{suffix}.png"))

        plot_metric(stats['det_rate'],
                    "Detection Rate across TSS", "Mean Detection Rate (%)", (-5, 105),
                    os.path.join(args.outdir, f"Fig16_TSS_DetRate_Profile_{suffix}.png"))

    generate_all_figures(mast_U, "unfiltered")

    if args.min_depth > 0 or args.min_det_rate > 0.0:
        logging.info("Generating filtered plot variants...")
        generate_all_figures(mast_F, "filtered")

        logging.info("Writing QC text report...")
        generate_text_report(args, mast_U, mast_F, groups)

    logging.info("Done. All spatial profile plots saved successfully.")


if __name__ == '__main__':
    main()