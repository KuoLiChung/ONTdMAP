#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
methylBox.py

Description:
    CpG-based methylation boxplot and statistical comparison tool.
    Supports targeting single CpG sites or multiple CpG sites (via BED) grouped by name.
    Automatically adjusts the aggregation mode ('cpg', 'combine-depth', 'combine-beta')
    based on the number of valid 1-bp CpG sites provided per target group.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, matplotlib, seaborn, scipy
    - htslib (bgzip, tabix)
"""

import argparse
import logging
import sys
import subprocess
import os
import re
import io
from pathlib import Path
from typing import Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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
# Logging Setup
# ==========================================
def setup_logging(log_file: Path) -> logging.Logger:
    """
    Configure a dual-handler logger writing DEBUG to file and INFO to stdout.

    Parameters
    ----------
    log_file : Path
        Path for the file log handler output.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger    = logging.getLogger("MethyPlotter")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ==========================================
# Data Loading and QC Filtering
# ==========================================
class FastDataProcessor:
    """Load methylation data via tabix and apply depth/detection-rate QC filters."""

    def __init__(self, logger: logging.Logger, outdir: Path):
        self.logger = logger
        self.outdir = outdir

    def ensure_tabix_ready(self, input_path: Path) -> Path:
        """
        Verify that the input file is bgzip-compressed and tabix-indexed.

        If validation fails or a plain BED is supplied, automatically runs
        coordinate sort -> bgzip -> tabix index and returns the resulting .gz path.

        Parameters
        ----------
        input_path : Path
            Path to the input methylation matrix (.bed or .bed.gz).

        Returns
        -------
        Path
            Path to a valid bgzip + tabix-indexed file.
        """
        is_valid = False
        if str(input_path).endswith('.gz'):
            tbi_path = Path(str(input_path) + '.tbi')
            if tbi_path.exists():
                try:
                    subprocess.run(["tabix", "-H", str(input_path)],
                                   check=True, capture_output=True)
                    is_valid = True
                    return input_path
                except subprocess.CalledProcessError:
                    pass

        if not is_valid:
            self.logger.info("Building tabix-compatible format: sort -> bgzip -> index...")
            file_name = os.path.basename(input_path)
            base_name = file_name.replace('.gz', '').replace('.bed', '')
            sorted_bed = self.outdir / f"{base_name}_sorted.bed"
            final_gz   = self.outdir / f"{base_name}_sorted.bed.gz"

            try:
                cat_cmd  = "zcat" if str(input_path).endswith('.gz') else "cat"
                sort_cmd = f"{cat_cmd} {input_path} | sort -k1,1 -k2,2n > {sorted_bed}"
                subprocess.run(sort_cmd, shell=True, check=True)
                subprocess.run(["bgzip", "-f", str(sorted_bed)], check=True)
                subprocess.run(["tabix", "-p", "bed", str(final_gz)], check=True)
                return final_gz
            except Exception as e:
                self.logger.error(f"Failed to build tabix index: {e}")
                sys.exit(1)

    def fast_intersect_and_load(self, union_bed_path: Path, target_input: str) -> pd.DataFrame:
        """
        Query a tabix-indexed file for a genomic region and parse depth and beta values.

        Reads the file header to recover column names, queries the target region
        with tabix, and expands each comma-delimited sample column into
        '<sample>_depth' and '<sample>_beta' float columns in-place.

        Parameters
        ----------
        union_bed_path : Path
            Path to the methylation matrix file.
        target_input : str
            Genomic region string in 'chr:start-end' format.

        Returns
        -------
        pd.DataFrame
            Parsed CpG-level DataFrame with depth and beta columns, or an empty
            DataFrame if no sites overlap the target region.
        """
        ready_gz_path = self.ensure_tabix_ready(union_bed_path)
        target_input  = target_input.strip()

        try:
            header_cmd    = ["tabix", "-H", str(ready_gz_path)]
            header_result = subprocess.run(header_cmd, capture_output=True, text=True, check=True)
            header_line   = header_result.stdout.strip().split('\n')[-1].lstrip('#')
            columns       = header_line.split('\t')

            cmd    = ["tabix", str(ready_gz_path), target_input]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if not result.stdout.strip():
                return pd.DataFrame(columns=columns)

            df = pd.read_csv(io.StringIO(result.stdout), sep='\t', names=columns)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Tabix query failed: {e.stderr}")
            sys.exit(1)

        if not df.empty:
            df['region_name'] = (df.iloc[:, 0] + ":" +
                                 df.iloc[:, 1].astype(str) + "-" +
                                 df.iloc[:, 2].astype(str))
            sample_cols = [c for c in columns
                           if c not in [columns[0], columns[1], columns[2]]]

            for col in sample_cols:
                split_df = df[col].astype(str).str.split(',', expand=True)
                if split_df.shape[1] >= 2:
                    depth = pd.to_numeric(split_df[0], errors='coerce')
                    meth  = pd.to_numeric(split_df[1], errors='coerce')
                    df[f"{col}_depth"] = depth
                    with np.errstate(divide='ignore', invalid='ignore'):
                        df[f"{col}_beta"] = np.where(depth > 0, meth / depth, np.nan)
                else:
                    df[f"{col}_depth"] = np.nan
                    df[f"{col}_beta"]  = np.nan
                df.drop(columns=[col], inplace=True)

        return df

    def apply_qc_filters(self, df: pd.DataFrame, control_prefix: str,
                         case_prefix: Optional[str], min_depth: int,
                         min_det_rate: float) -> pd.DataFrame:
        """
        Apply depth and detection-rate filters to the CpG beta matrix.

        Sites below min_depth are masked as NaN per sample. Sites where either
        group falls below min_det_rate (fraction of non-NaN samples) are dropped.

        Parameters
        ----------
        df : pd.DataFrame
            Parsed CpG DataFrame with '_beta' and '_depth' columns.
        control_prefix : str
            Group identifier for the control group.
        case_prefix : str or None
            Group identifier for the case group, or None for single-group mode.
        min_depth : int
            Minimum read depth; sites below this value are masked as NaN.
        min_det_rate : float
            Minimum detection rate (0.0-1.0) required per group.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame retaining only sites passing both criteria.
        """
        beta_cols  = [c for c in df.columns if c.endswith('_beta')]
        depth_cols = [c for c in df.columns if c.endswith('_depth')]

        masked_df = df.copy()
        for b_col, d_col in zip(beta_cols, depth_cols):
            mask = masked_df[d_col] < min_depth
            masked_df.loc[mask, b_col] = np.nan

        ctrl_betas = [c for c in beta_cols if is_in_group(c, control_prefix)]
        case_betas = ([c for c in beta_cols if is_in_group(c, case_prefix)]
                      if case_prefix else [])

        if not ctrl_betas:
            self.logger.error(
                f"No columns matched the control identifier '{control_prefix}'. "
                "Check --control and the column names in the input file."
            )
            sys.exit(1)

        def check_det_rate(row):
            ctrl_rate = row[ctrl_betas].notna().mean()
            if case_betas:
                case_rate = row[case_betas].notna().mean()
                return (ctrl_rate >= min_det_rate) and (case_rate >= min_det_rate)
            return ctrl_rate >= min_det_rate

        valid_mask = masked_df.apply(check_det_rate, axis=1)
        return masked_df[valid_mask].copy()


# ==========================================
# Region Aggregation
# ==========================================
class Aggregator:
    """
    Aggregate CpG-level beta values within a named target group.

    Three modes are supported:
      - 'cpg'           : Return each CpG site as a separate row.
      - 'combine-beta'  : Simple mean of per-site beta values across all CpGs.
      - 'combine-depth' : Depth-weighted pooled methylation fraction.
    """

    def __init__(self, mode: str):
        self.mode = mode

    def aggregate(self, df: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """
        Aggregate the CpG matrix according to the selected mode.

        Parameters
        ----------
        df : pd.DataFrame
            QC-filtered CpG DataFrame.
        target_name : str
            Display name for the region group (used in plot titles and filenames).

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with one row per group (or per CpG in 'cpg' mode).
        """
        df = df.copy()

        if self.mode == 'cpg':
            base_names        = (df.iloc[:, 0] + ":" +
                                 df.iloc[:, 1].astype(str) + "-" +
                                 df.iloc[:, 2].astype(str))
            df['region_name'] = target_name + "-" + base_names
            return df

        df['region_name'] = target_name
        region_col        = 'region_name'
        beta_cols         = [c for c in df.columns if c.endswith('_beta')]
        depth_cols        = [c for c in df.columns if c.endswith('_depth')]

        if self.mode == 'combine-beta':
            agg_df     = df.groupby(region_col)[beta_cols].mean().reset_index()
            cpg_counts = df.groupby(region_col).size().reset_index(name='combined_cpg_count')
            return pd.merge(agg_df, cpg_counts, on=region_col)

        elif self.mode == 'combine-depth':
            agg_data = []
            for name, group in df.groupby(region_col):
                row = {region_col: name, 'combined_cpg_count': len(group)}
                for b_col, d_col in zip(beta_cols, depth_cols):
                    mod_reads   = (group[b_col] * group[d_col]).sum(skipna=True)
                    total_reads = group[d_col].sum(skipna=True)
                    row[b_col]  = mod_reads / total_reads if total_reads > 0 else np.nan
                agg_data.append(row)
            return pd.DataFrame(agg_data)


# ==========================================
# Visualization and Reporting
# ==========================================
class ReporterAndVisualizer:
    """
    Generate per-region methylation boxplots, export raw data CSVs, and
    compile a statistical summary table.
    """

    def __init__(self, out_dir: Path, control_name: str, case_name: str):
        self.out_dir       = out_dir
        self.csv_dir       = out_dir / "csv_data"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.control_name  = control_name
        self.case_name     = case_name
        self.palette       = {self.control_name: "tab:blue", self.case_name: "tab:orange"}
        self.summary_stats = []

    def plot_and_export(self, title: str, ctrl_data: pd.Series,
                        case_data: Optional[pd.Series], p_val: float,
                        filename: str, combined_cpgs: int, mode: str,
                        stat_test: str = 'N/A'):
        """
        Save a boxplot with overlaid strip plot and export raw data as CSV.

        Parameters
        ----------
        title : str
            Plot title and region label.
        ctrl_data : pd.Series
            Beta values for the control group.
        case_data : pd.Series or None
            Beta values for the case group, or None for single-group mode.
        p_val : float
            p-value from the statistical test (NaN if not computed).
        filename : str
            Base filename for output files (special characters sanitized).
        combined_cpgs : int
            Number of CpG sites combined into this aggregated value.
        mode : str
            Aggregation mode label recorded in the summary CSV.
        stat_test : str
            Name of the statistical test used (recorded in the summary CSV).
        """
        data_dict = {self.control_name: pd.Series(ctrl_data.values)}
        if case_data is not None:
            data_dict[self.case_name] = pd.Series(case_data.values)

        raw_df    = pd.DataFrame(data_dict)
        safe_name = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in filename])
        safe_name = re.sub(r'_+', '_', safe_name)
        csv_path  = self.csv_dir / f"{safe_name}_raw.csv"
        raw_df.to_csv(csv_path, index=False)

        stars = "ns"
        if not np.isnan(p_val):
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        self.summary_stats.append({
            "Plot_Title":       title,
            "Aggregation_Mode": mode,
            "Combined_CpGs":    combined_cpgs,
            "Control_N":        ctrl_data.notna().sum(),
            "Case_N":           case_data.notna().sum() if case_data is not None else 0,
            "Control_Mean":     ctrl_data.mean(),
            "Case_Mean":        case_data.mean() if case_data is not None else np.nan,
            "Stat_Test":        stat_test if case_data is not None else "None",
            "P_Value":          p_val,
            "Significance":     stars
        })

        plt.figure(figsize=(6, 8))
        plot_df = pd.DataFrame({'Value': ctrl_data.tolist(), 'Group': self.control_name})
        if case_data is not None:
            plot_df = pd.concat(
                [plot_df, pd.DataFrame({'Value': case_data.tolist(), 'Group': self.case_name})],
                ignore_index=True
            )
        plot_df = plot_df.dropna()

        sns.boxplot(x='Group', y='Value', hue='Group', data=plot_df,
                    palette=self.palette, showfliers=False, width=0.5, legend=False)
        sns.stripplot(x='Group', y='Value', data=plot_df,
                      color='black', alpha=0.6, jitter=True)
        plt.title(title, fontsize=12)
        plt.ylabel("Methylation Level")
        plt.ylim(-0.05, 1.05)

        if case_data is not None and not np.isnan(p_val):
            p_text = f"{p_val:.1e}" if p_val < 1e-5 else f"{p_val:.4f}"
            plt.text(0.5, 0.95, f"p = {p_text} ({stars})",
                     ha='center', va='center', transform=plt.gca().transAxes)

        plot_path = self.out_dir / f"{safe_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def export_summary(self):
        """Write the accumulated per-region statistics to a CSV file."""
        if self.summary_stats:
            summary_df   = pd.DataFrame(self.summary_stats)
            summary_path = self.out_dir / "statistical_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            with open(summary_path, 'a') as f:
                f.write("\n# ns: p>=0.05, *: p<0.05, **: p<0.01, ***: p<0.001\n")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="CpG-based methylation boxplot and statistical comparison."
    )

    # I/O Arguments
    parser.add_argument('-i', '--input', required=True,
                        help="Multi-sample methylation matrix (TSV/BED or bgzip-compressed). "
                             "Example: Union.bed or Union.bed.gz")
    parser.add_argument('-o', '--outdir', required=True, type=str,
                        help="Output directory. Created automatically if it does not exist.")
    parser.add_argument('-t', '--target', type=str, required=True,
                        help="Target genomic region(s). Two modes:\n"
                             "  1. Single coordinate string: 'chr1:10400-10500'\n"
                             "  2. BED file path for batch processing. "
                             "     (--title is disabled in this mode; use column 4 for region names.)")

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

    # Analysis Mode
    parser.add_argument('--mode', choices=['cpg', 'combine-beta', 'combine-depth'],
                        default='combine-depth',
                        help="CpG aggregation mode:\n"
                             "  'combine-depth' : Depth-weighted average across all CpG sites (Default)\n"
                             "  'combine-beta'  : Simple mean of per-site beta values\n"
                             "  'cpg'           : Analyze each CpG site independently")
    parser.add_argument('--title', type=str, default="",
                        help="Custom name prefix for output files and plot titles. "
                             "Output format: 'PREFIX-chr:start-end'.")

    # QC Filtering
    parser.add_argument('--min-depth', type=int, default=0,
                        help="Minimum read depth. Sites below this threshold are masked as NaN. (Default: 0)")
    parser.add_argument('--min-det-rate', type=float, default=0.0,
                        help="Minimum group-wise detection rate (0.0-1.0). "
                             "Sites below this rate in either group are dropped. (Default: 0.0)")

    # Statistical Test
    parser.add_argument('--stat_test', type=str, default='ttest',
                        choices=['ttest', 'student_ttest', 'mannwhitney',
                                 'paired_ttest', 'wilcoxon', 'ks'],
                        help="Statistical test for Control vs Case comparison:\n"
                             "  'ttest'         : Welch's t-test (independent, unequal variance)\n"
                             "  'student_ttest' : Student's t-test (independent, equal variance)\n"
                             "  'mannwhitney'   : Mann-Whitney U test (non-parametric, independent)\n"
                             "  'paired_ttest'  : Paired t-test (requires equal N per group)\n"
                             "  'wilcoxon'      : Wilcoxon signed-rank test (requires equal N per group)\n"
                             "  'ks'            : Kolmogorov-Smirnov test (distribution shape)\n"
                             "(Default: ttest)")

    args        = parser.parse_args()
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.outdir / "analysis.log")

    # ==========================================
    # Target Resolution: Parse into groups keyed by name
    # ==========================================
    targets_to_process = defaultdict(list)
    target_str         = str(args.target)

    if os.path.isfile(target_str):
        if args.title:
            logger.error("--title is not allowed when --target is a BED file. "
                         "Place custom region names in column 4 of the BED file.")
            sys.exit(1)

        logger.info(f"BED file detected. Parsing region groups: {target_str}")
        with open(target_str, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    chrom  = parts[0]
                    start  = int(parts[1])
                    end    = int(parts[2])
                    coord  = f"{chrom}:{start}-{end}"
                    name   = parts[3] if len(parts) >= 4 else coord
                    length = end - start
                    targets_to_process[name].append(
                        {'coord': coord, 'length': length, 'start': start, 'end': end}
                    )
    else:
        parts = re.split(r'[:-]', target_str.strip())
        if len(parts) >= 3:
            chrom  = parts[0]
            start  = int(parts[1])
            end    = int(parts[2])
            length = end - start
            name   = args.title.strip() if args.title.strip() else target_str.strip()
            targets_to_process[name].append(
                {'coord': target_str.strip(), 'length': length, 'start': start, 'end': end}
            )
        else:
            logger.error("Target string must be in 'chr:start-end' format.")
            sys.exit(1)

    if not targets_to_process:
        logger.error("No valid target regions found.")
        sys.exit(1)

    # ==========================================
    # Batch Analysis Loop
    # ==========================================
    processor = FastDataProcessor(logger, args.outdir)

    for t_name, lines in targets_to_process.items():
        logger.info("-" * 50)
        logger.info(f"Analyzing group: {t_name}")

        current_mode = args.mode
        valid_lines  = []

        # Determine effective aggregation mode based on interval widths
        if len(lines) == 1:
            line = lines[0]
            if line['length'] > 1:
                # Multi-bp interval: retain user-specified mode
                logger.warning(
                    f"[{t_name}] Interval {line['coord']} spans more than 1 bp. "
                    f"Proceeding with user-specified mode '{current_mode}'."
                )
                valid_lines.append(line)
            else:
                # Single CpG site: force 'cpg' mode if a combine mode was requested
                if current_mode != 'cpg':
                    logger.warning(
                        f"[{t_name}] Single CpG site ({line['coord']}) is incompatible "
                        f"with '{current_mode}' mode. Switching to 'cpg' mode."
                    )
                    current_mode = 'cpg'
                valid_lines.append(line)
        else:
            for line in lines:
                if line['length'] > 1:
                    # Multi-bp entries within a named group are skipped
                    logger.warning(
                        f"[{t_name}] Skipping {line['coord']}: interval spans more than 1 bp "
                        "and cannot be treated as a single CpG site."
                    )
                else:
                    valid_lines.append(line)

            if len(valid_lines) == 0:
                logger.warning(f"[{t_name}] All entries exceeded 1 bp. Skipping group.")
                continue
            elif len(valid_lines) == 1:
                line = valid_lines[0]
                if current_mode != 'cpg':
                    logger.warning(
                        f"[{t_name}] Only one valid CpG site remains ({line['coord']}) after "
                        f"filtering. '{current_mode}' mode requires multiple sites. "
                        "Switching to 'cpg' mode."
                    )
                    current_mode = 'cpg'

        # Fetch and concatenate methylation data for all valid intervals
        df_list = []
        for line in valid_lines:
            df_part = processor.fast_intersect_and_load(args.input, line['coord'])
            if not df_part.empty:
                df_list.append(df_part)

        if not df_list:
            logger.warning(f"[{t_name}] No methylation data found for any interval. Skipping.")
            continue

        df_combined = pd.concat(df_list, ignore_index=True)

        df_filtered = processor.apply_qc_filters(
            df_combined, args.control, args.case, args.min_depth, args.min_det_rate
        )
        if df_filtered.empty:
            logger.warning(f"[{t_name}] No sites remained after QC filtering. Skipping.")
            continue

        df_agg = Aggregator(current_mode).aggregate(df_filtered, t_name)

        safe_dir_name  = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in t_name])
        safe_dir_name = re.sub(r'_+', '_', safe_dir_name)
        current_outdir = args.outdir / safe_dir_name / current_mode
        current_outdir.mkdir(parents=True, exist_ok=True)

        visualizer = ReporterAndVisualizer(current_outdir, args.control_name, args.case_name)

        ctrl_cols = [c for c in df_agg.columns
                     if is_in_group(c, args.control) and c.endswith('_beta')]
        case_cols = ([c for c in df_agg.columns
                      if is_in_group(c, args.case) and c.endswith('_beta')]
                     if args.case else [])

        for idx, row in df_agg.iterrows():
            title     = str(row['region_name'])
            filename  = title.replace(":", "_").replace("-", "_")
            cpg_count = row.get('combined_cpg_count', 1)

            ctrl_vals = pd.to_numeric(row[ctrl_cols], errors='coerce')
            case_vals = (pd.to_numeric(row[case_cols], errors='coerce')
                         if case_cols else None)

            p_val = np.nan
            if args.case and len(ctrl_vals.dropna()) > 0 and len(case_vals.dropna()) > 0:
                v_ctrl = ctrl_vals.dropna()
                v_case = case_vals.dropna()
                try:
                    if args.stat_test == 'ttest':
                        _, p_val = stats.ttest_ind(v_ctrl, v_case, equal_var=False)
                    elif args.stat_test == 'student_ttest':
                        _, p_val = stats.ttest_ind(v_ctrl, v_case, equal_var=True)
                    elif args.stat_test == 'mannwhitney':
                        _, p_val = stats.mannwhitneyu(v_ctrl, v_case, alternative='two-sided')
                    elif args.stat_test == 'ks':
                        _, p_val = stats.ks_2samp(v_ctrl, v_case)
                    elif args.stat_test == 'paired_ttest':
                        if len(v_ctrl) == len(v_case):
                            _, p_val = stats.ttest_rel(v_ctrl, v_case)
                        else:
                            logger.warning(
                                f"Skipping {title}: paired_ttest requires equal N "
                                f"(Control: {len(v_ctrl)}, Case: {len(v_case)})"
                            )
                            continue
                    elif args.stat_test == 'wilcoxon':
                        if len(v_ctrl) == len(v_case):
                            _, p_val = stats.wilcoxon(v_ctrl, v_case)
                        else:
                            logger.warning(
                                f"Skipping {title}: wilcoxon requires equal N "
                                f"(Control: {len(v_ctrl)}, Case: {len(v_case)})"
                            )
                            continue
                except Exception as e:
                    # Zero-variance data (all identical values) may raise errors in scipy
                    logger.warning(f"Statistical test failed for {title}: {e}")

            visualizer.plot_and_export(
                title=title,
                ctrl_data=ctrl_vals,
                case_data=case_vals,
                p_val=p_val,
                filename=filename,
                combined_cpgs=cpg_count,
                mode=current_mode,
                stat_test=args.stat_test
            )

        visualizer.export_summary()
        logger.info(f"Group '{t_name}' complete. Outputs saved to: {current_outdir}")

    logger.info("All batch tasks complete.")


if __name__ == "__main__":
    main()