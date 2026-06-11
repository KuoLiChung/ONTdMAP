#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dmrQC.py

Description:
    QC filtering, genomic annotation, and visualization pipeline for DMRs identified
    by modkit. Applies multi-criterion QC filters (effect size, CpG site count,
    Cohen's h, read depth, depth ratio), optionally masks ENCODE blacklist regions
    via bedtools, runs annotatr-based genomic annotation in R, and generates a
    standardized set of summary figures (pie charts, stacked bar plots, raincloud
    effect size distribution, co-occurrence heatmap, volcano plot, and QC histograms).

Dependencies:
    - Python >= 3.8
    - pandas, numpy, matplotlib, seaborn
    - R with packages: annotatr, GenomicRanges, dplyr
    - bedtools (optional; required for blacklist filtering)
"""

import os
import sys
import shutil
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib

# Force the Agg backend for rendering in headless server environments (e.g., SSH sessions)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import PolyCollection, PathCollection
from matplotlib.colors import LogNorm


def check_dependency(tool_name):
    """Check whether an external tool is available on the system PATH."""
    return shutil.which(tool_name) is not None


def parse_depth(count_str):
    """
    Parse a modkit counts field into a total read depth integer.

    Accepts both key:value formats (e.g., 'valid:10,modified:2') and
    plain integer strings. Returns 0 for missing or unparseable values.

    Parameters
    ----------
    count_str : str or float
        Raw counts field value from the modkit DMR output.

    Returns
    -------
    int
        Total read depth summed across all categories in the field.
    """
    if pd.isna(count_str) or count_str == "":
        return 0
    try:
        total = 0
        for part in str(count_str).split(','):
            if ':' in part:
                try:
                    total += int(part.split(':')[1])
                except (ValueError, IndexError):
                    pass
            else:
                try:
                    total += int(part)
                except ValueError:
                    pass
        return total
    except Exception:
        return 0


def run_cmd(cmd, desc, out_file=None):
    """
    Execute an external command, optionally capturing stdout to a file.

    For Rscript calls (out_file=None), stdout and stderr are passed through
    directly to the terminal. For bedtools calls, stdout is captured and
    written to out_file.

    Parameters
    ----------
    cmd : list of str
        Command and arguments to execute.
    desc : str
        Human-readable description of the command (used in log messages).
    out_file : str or None
        If provided, captured stdout is written to this path.

    Returns
    -------
    bool
        True on success. Exits the process on non-zero return code or exception.
    """
    print(f"[INFO] Running external tool ({desc}): {' '.join(cmd)}")
    try:
        if not out_file:
            result = subprocess.run(cmd)
        else:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with open(out_file, "w") as f:
                f.write(result.stdout)

        if result.returncode != 0:
            print(f"[ERROR] {desc} failed (exit code: {result.returncode})")
            sys.exit(1)
        return True
    except Exception as e:
        print(f"[ERROR] Exception while running {desc}: {e}")
        sys.exit(1)


def generate_r_script(bed_file, genome, outdir):
    """
    Write an R script that runs annotatr genomic annotation and saves results.

    The script annotates DMR regions against CpG context, basic gene structure,
    and intergenic regions, then exports a detailed per-annotation TSV.

    Parameters
    ----------
    bed_file : str
        Path to the QC-filtered DMR BED file.
    genome : str
        Reference genome build (e.g., 'hg38', 'mm10').
    outdir : str
        Output directory for the annotation TSV.

    Returns
    -------
    str
        Path to the written R script file.
    """
    r_script_path = os.path.join(outdir, "annotatr_compute.R")

    r_code = f"""
suppressPackageStartupMessages({{
    library(annotatr)
    library(GenomicRanges)
    library(dplyr)
}})

bed_file <- "{bed_file}"
genome <- "{genome}"
outdir <- "{outdir}"

cat("[R INFO] Running annotatr annotation...\\n")

# Build annotation database: CpG context, basic gene structure, and intergenic regions
annots <- c(paste0(genome, '_cpgs'), paste0(genome, '_basicgenes'), paste0(genome, '_genes_intergenic'))
annotations <- build_annotations(genome, annotations = annots)

# Read QC-filtered DMR regions in BED format
extraCols <- c(effect_size='numeric', DM_status='character', depth_A='numeric', depth_B='numeric')
regions <- read_regions(con = bed_file, genome = genome, format = 'bed', rename_score = 'cohen_h', extraCols = extraCols)

# Restrict to chromosomes present in both the regions and annotations, and trim out-of-bounds coordinates
regions <- keepSeqlevels(regions, intersect(seqlevels(regions), seqlevels(annotations)), pruning.mode="coarse")
regions <- trim(regions)

# Run annotation overlap
dm_annotated <- annotate_regions(regions = regions, annotations = annotations, ignore.strand = TRUE, quiet = TRUE)

# Export full annotation table as TSV
df_annotated <- data.frame(dm_annotated)
write.table(df_annotated, file=file.path(outdir, "annotatr_details.tsv"), sep='\\t', quote=FALSE, row.names=FALSE)

cat("[R SUCCESS] Annotatr annotation complete.\\n")
"""
    with open(r_script_path, "w") as f:
        f.write(r_code)
    return r_script_path


def plot_all_results(outdir, genome, df_pass):
    """
    Generate all DMR summary figures from annotation and QC data.

    Produces nine figures covering CpG and gene region distributions (pie charts
    and stacked bar plots), effect size distribution (raincloud plot), CpG-by-gene
    region co-occurrence heatmap, Cohen's h vs effect size scatter plot, CpG site
    count distribution, and read depth ratio histogram.

    Parameters
    ----------
    outdir : str
        Output directory containing 'annotatr_details.tsv' and for saving figures.
    genome : str
        Reference genome build (informational; not used directly in plots).
    df_pass : pd.DataFrame
        QC-filtered DMR DataFrame including effect_size, cohen_h, DM_status, etc.

    Returns
    -------
    pd.DataFrame or None
        Cleaned annotation DataFrame, or None if the annotation file is missing.
    """
    print("[5/5] Generating DMR QC figures...")
    sns.set_theme(style="whitegrid", context="paper")

    # Fixed category orders and color palettes for visual consistency
    cpg_order = ["Island", "Shore", "Shelf", "InterCGI"]
    cpg_colors = {
        "Island":   "#ff7f0e",
        "Shore":    "#2ca02c",
        "Shelf":    "#17becf",
        "InterCGI": "#bcbd22"
    }

    genic_order = [
        "Promoter (<1kb)", "Promoter (1-5kb)", "5'UTR",
        "Exon", "Intron", "3'UTR", "Intergenic"
    ]
    genic_colors = {
        "Promoter (<1kb)":  "#d62728",
        "Promoter (1-5kb)": "#ff9896",
        "5'UTR":            "#9467bd",
        "Exon":             "#8c564b",
        "Intron":           "#1f77b4",
        "3'UTR":            "#c5b0d5",
        "Intergenic":       "#7f7f7f"
    }

    full_palette = {**cpg_colors, **genic_colors}
    full_order   = cpg_order + genic_order

    try:
        df_annot = pd.read_csv(os.path.join(outdir, "annotatr_details.tsv"), sep='\t')
    except Exception:
        return

    def clean_label(text):
        """Map raw annotatr annotation type strings to standardized display labels."""
        if pd.isna(text):
            return "Unknown"
        t = text.lower()
        if "cpg" in t:
            if "islands" in t:  return "Island"
            if "shores"  in t:  return "Shore"
            if "shelves" in t:  return "Shelf"
            return "InterCGI"
        else:
            if "promoters"  in t: return "Promoter (<1kb)"
            if "1to5kb"     in t: return "Promoter (1-5kb)"
            if "5utr"       in t: return "5'UTR"
            if "3utr"       in t: return "3'UTR"
            if "exons"      in t: return "Exon"
            if "introns"    in t: return "Intron"
            return "Intergenic"

    df_annot['clean_type'] = df_annot['annot.type'].apply(clean_label)
    df_annot['clean_type'] = pd.Categorical(df_annot['clean_type'], categories=full_order, ordered=True)

    # --- Figure 1: CpG Region Pie Chart ---
    plt.figure(figsize=(10, 8))
    cpg_df = df_annot[df_annot['clean_type'].isin(cpg_order)]
    if not cpg_df.empty:
        counts = cpg_df.groupby('clean_type')['name'].nunique().reindex(cpg_order).dropna()
        colors = [cpg_colors[idx] for idx in counts.index]
        wedges, texts, autotexts = plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
                                           startangle=140, colors=colors,
                                           textprops={'fontweight': 'bold'})
        plt.gca().add_artist(plt.Circle((0, 0), 0.60, fc='white'))
        leg = plt.legend(wedges, [f'{l}: {v:,}' for l, v in zip(counts.index, counts.values)],
                         title=f"Total: {counts.sum()}", loc="center left",
                         bbox_to_anchor=(1.1, 0.5), fontsize=12)
        plt.setp(leg.get_title(), fontsize=14, fontweight='bold')
        plt.title("DMR Distribution in CpG Regions", fontweight='bold', fontsize=16)
        plt.savefig(os.path.join(outdir, "Fig01_DMR_Dist_CpG_Pie.png"), dpi=300, bbox_inches='tight')

    # --- Figure 2: Gene Region Pie Chart ---
    plt.figure(figsize=(10, 8))
    genic_df = df_annot[df_annot['clean_type'].isin(genic_order)]
    if not genic_df.empty:
        counts_g = genic_df.groupby('clean_type')['name'].nunique().reindex(genic_order).dropna()
        colors_g = [genic_colors[idx] for idx in counts_g.index]
        wedges, texts, autotexts = plt.pie(counts_g, labels=counts_g.index, autopct='%1.1f%%',
                                           startangle=140, colors=colors_g,
                                           textprops={'fontweight': 'bold'})
        plt.gca().add_artist(plt.Circle((0, 0), 0.60, fc='white'))
        leg = plt.legend(wedges, [f'{l}: {v:,}' for l, v in zip(counts_g.index, counts_g.values)],
                         title=f"Total: {counts_g.sum()}", loc="center left",
                         bbox_to_anchor=(1.1, 0.5), fontsize=12)
        plt.setp(leg.get_title(), fontsize=14, fontweight='bold')
        plt.title("DMR Distribution in Gene Regions", fontweight='bold', fontsize=16)
        plt.savefig(os.path.join(outdir, "Fig02_DMR_Dist_Gene_Pie.png"), dpi=300, bbox_inches='tight')

    # --- Figure 3: CpG Region Stacked Proportion Bar ---
    if not cpg_df.empty:
        plt.figure(figsize=(10, 6))
        cpg_u  = cpg_df.drop_duplicates(subset=['name', 'clean_type'])
        ct2    = pd.crosstab(cpg_u['clean_type'], cpg_u['DM_status'])
        target_order = [c for c in cpg_order if c in ct2.index]
        ct2    = ct2.reindex(target_order[::-1])
        ct2_p  = ct2.div(ct2.sum(axis=1), axis=0)
        h_sum  = int(ct2['Hyper'].sum()) if 'Hyper' in ct2.columns else 0
        m_sum  = int(ct2['Hypo'].sum())  if 'Hypo'  in ct2.columns else 0

        ax2 = ct2_p.plot(kind='barh', stacked=True, color=['tab:blue', 'tab:orange'],
                         ax=plt.gca(), width=0.7, alpha=0.75)
        ax2.xaxis.set_major_formatter(PercentFormatter(1.0))

        for i, (idx, row) in enumerate(ct2_p.iterrows()):
            curr_x = 0
            for col in ct2_p.columns:
                w     = row[col]
                count = int(ct2.loc[idx, col])
                if w > 0.03:
                    ax2.text(curr_x + w / 2, i, f'{count}', va='center', ha='center',
                             color='white', fontweight='bold', fontsize=12)
                curr_x += w

        legend_elements = [
            Patch(facecolor='tab:blue',   label=f'Hyper: {h_sum}'),
            Patch(facecolor='tab:orange', label=f'Hypo: {m_sum}')
        ]
        plt.legend(handles=legend_elements, title="Methylation Status",
                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
        plt.ylabel("")
        plt.xlabel("Proportion")
        plt.xlim(0, 1)
        plt.yticks(fontsize=14)
        plt.title("DMR Status Proportions by CpG Regions", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "Fig03_CpG_Prop.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # --- Figure 4: Gene Region Stacked Proportion Bar ---
    if not genic_df.empty:
        plt.figure(figsize=(11, 7))
        genic_u = genic_df.drop_duplicates(subset=['name', 'clean_type'])
        ct3     = pd.crosstab(genic_u['clean_type'], genic_u['DM_status'])
        target_g_order = [c for c in genic_order if c in ct3.index]
        ct3     = ct3.reindex(target_g_order[::-1])
        ct3_p   = ct3.div(ct3.sum(axis=1), axis=0)
        h_sum_g = int(ct3['Hyper'].sum()) if 'Hyper' in ct3.columns else 0
        m_sum_g = int(ct3['Hypo'].sum())  if 'Hypo'  in ct3.columns else 0

        ax3 = ct3_p.plot(kind='barh', stacked=True, color=['tab:blue', 'tab:orange'],
                         ax=plt.gca(), width=0.7, alpha=0.75)
        ax3.xaxis.set_major_formatter(PercentFormatter(1.0))

        for i, (idx, row) in enumerate(ct3_p.iterrows()):
            curr_x = 0
            for col in ct3_p.columns:
                w     = row[col]
                count = int(ct3.loc[idx, col])
                if w > 0.03:
                    ax3.text(curr_x + w / 2, i, f'{count}', va='center', ha='center',
                             color='white', fontweight='bold', fontsize=12)
                curr_x += w

        legend_elements = [
            Patch(facecolor='tab:blue',   label=f'Hyper: {h_sum_g}'),
            Patch(facecolor='tab:orange', label=f'Hypo: {m_sum_g}')
        ]
        plt.legend(handles=legend_elements, title="Methylation Status",
                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
        plt.ylabel("")
        plt.xlabel("Proportion")
        plt.xlim(0, 1)
        plt.yticks(fontsize=14)
        plt.title("DMR Status Proportions by Gene Regions", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "Fig04_Gene_Prop.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # --- Figure 5: Effect Size Raincloud Plot ---
    plt.figure(figsize=(12, 10))
    plot_order = [c for c in full_order if c in df_annot['clean_type'].unique()]

    ax = sns.violinplot(
        data=df_annot, x='effect_size', y='clean_type', order=plot_order,
        palette=full_palette, hue='clean_type', legend=False,
        inner=None, linewidth=0, alpha=0.5, density_norm="width"
    )

    # Clip violin polygons to the upper half (center line and above) to form a half-violin
    for art in ax.findobj(PolyCollection):
        path   = art.get_paths()[0]
        v      = path.vertices
        center = np.round(np.mean(v[:, 1]))
        v[:, 1] = np.clip(v[:, 1], a_min=None, a_max=center)

    # Overlay jittered strip plot offset below the half-violin (raincloud effect)
    sns.stripplot(
        data=df_annot, x='effect_size', y='clean_type', order=plot_order,
        palette=full_palette, hue='clean_type', legend=False,
        size=3, alpha=0.3, jitter=0.45, ax=ax, zorder=2
    )

    # Shift strip plot points downward to separate them from the violin body
    offset = -0.25
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            offsets = collection.get_offsets()
            offsets[:, 1] += offset
            collection.set_offsets(offsets)

    plt.axvline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1, zorder=0)
    plt.ylabel("")
    plt.xlabel("Effect Size", fontsize=13, fontweight='bold', labelpad=10)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=12)
    sns.despine(left=True, bottom=False)
    plt.title("Distribution of Effect Sizes", fontweight='bold', fontsize=16, pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Fig05_EffectSize_Dist.png"), dpi=300, bbox_inches='tight')

    # --- Figure 6: CpG x Gene Region Co-occurrence Heatmap ---
    # De-duplicate annotations so each DMR contributes at most once per category
    df_annot_unique  = df_annot[['name', 'clean_type']].drop_duplicates()
    co_mat_raw       = pd.crosstab(df_annot_unique['name'], df_annot_unique['clean_type'])
    co_occurrence_full = co_mat_raw.T.dot(co_mat_raw)

    rows       = [c for c in cpg_order   if c in co_occurrence_full.index]
    cols       = [g for g in genic_order if g in co_occurrence_full.columns]
    sub_matrix = co_occurrence_full.loc[rows, cols]

    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap("OrRd").copy()
    cmap.set_under("white")
    vmax = sub_matrix.values.max()

    ax = sns.heatmap(
        sub_matrix, annot=True, fmt="d", cmap=cmap, vmin=1,
        norm=LogNorm(vmin=1, vmax=vmax if vmax > 1 else 10),
        cbar_kws={'label': 'Unique DMR Count (Log Scale)'},
        linewidths=0.5, linecolor='#dddddd'
    )
    plt.xlabel("Gene Regions",  fontweight='bold', fontsize=12)
    plt.ylabel("CpG Regions",   fontweight='bold', fontsize=12)
    plt.title("DMR Intersection: CpG Regions x Gene Regions\n(Unique DMR Counts Only)",
              fontweight='bold', fontsize=15, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.savefig(os.path.join(outdir, "Fig06_CpGxGene_HM.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 7: Cohen's h vs Effect Size Scatter Plot ---
    plt.figure(figsize=(8, 7))
    custom_palette = {'Hyper': 'tab:blue', 'Hypo': 'tab:orange'}
    ax = sns.scatterplot(
        data=df_pass, x='effect_size', y='cohen_h',
        hue='DM_status', hue_order=['Hyper', 'Hypo'],
        palette=custom_palette, alpha=0.4, s=15, edgecolor=None
    )
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(r"DMR Cohen's $\bf{\mathit{h}}$ vs Effect Size", fontweight='bold', fontsize=15, pad=15)
    plt.xlabel("Effect Size", fontsize=12)
    plt.ylabel("Cohen's $h$", fontsize=12)
    leg = plt.legend(title="Methylation Status", frameon=True, facecolor='white', loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Fig07_CohensH_vs_EffectSize.png"), dpi=300)
    plt.close()

    # --- Figure 8: CpG Sites per DMR Distribution (Log Scale) ---
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df_pass, x='num_sites', legend=False, color="#2ca02c")
    plt.yscale('log')

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 9), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='black')

    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    plt.title("Distribution of CpG Sites per DMR", fontweight='bold', fontsize=14)
    plt.xlabel("Number of CpG Sites", fontsize=12)
    plt.ylabel("DMR Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Fig08_CpGsites_perDMR_Dist.png"), dpi=300)
    plt.close()

    # --- Figure 9: Read Depth Ratio Distribution ---
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_pass, x='log2_depth_ratio', bins=40, color='purple', kde=True)
    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    plt.title("DMR Read Depth Ratio by Groups", fontweight='bold')
    plt.xlabel("$\log_2$(A / B)")
    plt.ylabel("DMR Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Fig09_ReadDepthRatio.png"), dpi=300)

    print("All figures generated successfully.")
    return df_annot


def merge_annotation_and_qc(df_pass, annot_file, outdir):
    """
    Merge the QC-filtered DMR table with the annotatr annotation output.

    Performs a coordinate-system conversion from R's 1-based closed intervals
    to BED-style 0-based half-open intervals before joining on genomic coordinates.
    One DMR may produce multiple rows if it overlaps multiple annotation features.

    Parameters
    ----------
    df_pass : pd.DataFrame
        QC-filtered DMR DataFrame with 'chrom', 'chrom_start', 'chrom_end' columns.
    annot_file : str
        Path to the annotatr_details.tsv file produced by the R script.
    outdir : str
        Output directory for the integrated report TSV.

    Returns
    -------
    pd.DataFrame or None
        Merged DataFrame, or None if the annotation file cannot be read.
    """
    print("[INFO] Merging QC statistics with genomic annotation by coordinate...")

    try:
        df_annot = pd.read_csv(annot_file, sep='\t')
    except Exception as e:
        print(f"[ERROR] Failed to read annotation file: {e}")
        return None

    # Convert R 1-based closed coordinates to 0-based half-open (BED convention)
    df_annot['chrom']       = df_annot['seqnames']
    df_annot['chrom_start'] = df_annot['start'] - 1
    df_annot['chrom_end']   = df_annot['end']

    # Drop columns that conflict with df_pass fields to prevent _x/_y suffixes
    cols_to_drop    = ['strand', 'width', 'name', 'score']
    df_annot_clean  = df_annot.drop(columns=[c for c in cols_to_drop if c in df_annot.columns])

    # Left join on genomic coordinates; one DMR generates one row per annotation feature
    df_merged = pd.merge(
        df_annot_clean, df_pass,
        on=['chrom', 'chrom_start', 'chrom_end'],
        how='left'
    )

    out_file = os.path.join(outdir, "Integrated_DMRs.tsv")
    df_merged.to_csv(out_file, sep='\t', index=False)
    print(f"      -> Integrated annotation table written: {out_file} ({len(df_merged)} records)")

    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description="DMR QC, genomic annotation, and visualization pipeline for modkit output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # I/O Arguments
    parser.add_argument('-i', '--input', required=True,
                        help="Input raw DMR BED file from modkit. Example: dmr.seg.bed")
    parser.add_argument('-o', '--outdir', required=True, type=str,
                        help="Output directory. Created automatically if it does not exist.")

    # QC Filtering Parameters
    parser.add_argument("--min_effect_size", type=float, default=0.2,
                        help="Minimum absolute delta-beta (methylation fraction difference).")
    parser.add_argument("--min_sites", type=int, default=3,
                        help="Minimum number of CpG sites per DMR.")
    parser.add_argument("--min_cohen_h", type=float, default=0.5,
                        help="Minimum absolute Cohen's h statistic.")
    parser.add_argument("--min_depth", type=int, default=8,
                        help="Minimum average read depth per CpG site.")
    parser.add_argument("--max_depth_ratio", type=float, default=2.0,
                        help="Maximum absolute log2(depth_A / depth_B). Filters CNV-driven pseudo-DMRs.")

    # Genomic Annotation and Artifact Masking
    parser.add_argument("--genome", default="hg38",
                        help="Reference genome build for annotatr annotation (e.g., 'hg38').")
    parser.add_argument("--blacklist",
                        help="Optional ENCODE blacklist BED file for masking artifact-prone regions.")

    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("=" * 65)
    print(f"ONT DMR Analysis Pipeline | Output: {outdir}")
    print("=" * 65)

    # Step 1: Load data and compute read depth
    print("[1/5] Loading input data and computing read depth...")
    df = pd.read_csv(args.input, sep='\t', comment=None)
    df.columns    = [col.replace('#', '').strip() for col in df.columns]
    initial_count = len(df)

    mod_counts_A = df['a_counts'].apply(parse_depth)
    mod_counts_B = df['b_counts'].apply(parse_depth)
    df['depth_A'] = np.round(mod_counts_A / df['a_frac_modified'].replace(0, 1e-6)).fillna(0).astype(int)
    df['depth_B'] = np.round(mod_counts_B / df['b_frac_modified'].replace(0, 1e-6)).fillna(0).astype(int)

    df['log2_depth_ratio'] = np.log2(df['depth_A'] / df['depth_B'])
    df['dmr_length']       = df['chrom_end'] - df['chrom_start']
    df['dmr_id']           = "DMR_" + df.index.astype(str)

    # Step 2: Apply QC filters
    print(f"[2/5] Applying QC filters (min sites: {args.min_sites}, min effect size: {args.min_effect_size})...")
    standard_chroms = [f"chr{i}" for i in range(1, 23)] + ['chrX', 'chrY']
    mask = (
        (df['name'] == 'different') &
        (df['chrom'].isin(standard_chroms)) &
        (df['effect_size'].abs() >= args.min_effect_size) &
        (df['num_sites'] >= args.min_sites) &
        (df['cohen_h'].abs() >= args.min_cohen_h) &
        ((df['cohen_h_low'] * df['cohen_h_high']) > 0) &  # CI does not cross zero
        (df['depth_A'] / df['num_sites'] >= args.min_depth) &
        (df['depth_B'] / df['num_sites'] >= args.min_depth) &
        (df['log2_depth_ratio'].abs() <= args.max_depth_ratio)
    )
    df_pass = df[mask].copy()
    print(f"      -> DMRs passing QC: {len(df_pass)} / {initial_count}")

    if len(df_pass) == 0:
        print("[ERROR] No DMRs passed the filter criteria. Exiting.")
        sys.exit(0)

    # Step 3: Blacklist filtering (optional)
    if args.blacklist and check_dependency("bedtools"):
        print("[3/5] Applying blacklist filter with bedtools...")
        tmp_bed   = os.path.join(outdir, "before_bl.bed")
        clean_bed = os.path.join(outdir, "after_bl.bed")
        df_pass.to_csv(tmp_bed, sep='\t', index=False, header=False)
        run_cmd(["bedtools", "intersect", "-v", "-a", tmp_bed, "-b", args.blacklist],
                "bedtools intersect", out_file=clean_bed)
        df_pass = pd.read_csv(clean_bed, sep='\t', header=None, names=df_pass.columns)
        print(f"      -> DMRs after blacklist exclusion: {len(df_pass)}")
        for f in [tmp_bed, clean_bed]:
            if os.path.exists(f):
                os.remove(f)
    else:
        print("[3/5] Skipping blacklist filter.")

    # Step 4: Genomic annotation via R/annotatr
    df_pass['strand']    = '*'
    df_pass['DM_status'] = np.where(0 > df_pass['effect_size'], 'Hypo', 'Hyper')
    bed_out = os.path.join(outdir, "DMR_QC_pass.bed")
    cols = ['chrom', 'chrom_start', 'chrom_end', 'dmr_id', 'cohen_h', 'strand',
            'effect_size', 'DM_status', 'depth_A', 'depth_B']
    df_pass[cols].to_csv(bed_out, sep='\t', index=False, header=False)

    if check_dependency("Rscript"):
        print("[4/5] Running genomic annotation via Rscript...")
        r_script  = generate_r_script(bed_out, args.genome, outdir)
        run_cmd(["Rscript", r_script], "R annotation engine")

        annot_file = os.path.join(outdir, "annotatr_details.tsv")
        if os.path.exists(annot_file):
            df_integrated      = merge_annotation_and_qc(df_pass, annot_file, outdir)
            df_annot_cleaned   = plot_all_results(outdir, args.genome, df_pass)
        else:
            print(f"[ERROR] Annotation output not found: {annot_file}. Skipping figures.")

        if os.path.exists(r_script):
            os.remove(r_script)

    print(f"\n[SUCCESS] Pipeline complete. All results saved to: {outdir}/")


if __name__ == "__main__":
    main()
