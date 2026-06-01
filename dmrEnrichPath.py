#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dmrEnrichPath.py

Description:
    Functional annotation, filtering, and pathway enrichment analysis for annotated DMRs.
    Reads the integrated DMR annotation table produced by DMRQC.py, applies region-based
    and depth-based filters, resolves bidirectional gene conflicts, exports gene ID lists,
    and generates effect size summary plots. Automatically writes and executes an R script
    that runs GO, Disease Ontology, Reactome, and g:Profiler enrichment analyses via
    clusterProfiler, DOSE, ReactomePA, and gprofiler2, producing dot plots, bar plots,
    Manhattan plots, and result CSVs for both hyper- and hypomethylated gene sets.

Dependencies:
    - Python >= 3.8
    - pandas, matplotlib, seaborn
    - R with packages: clusterProfiler, org.Hs.eg.db, DOSE, ReactomePA, gprofiler2,
                       ggplot2, enrichplot
    - conda environment 'r-enrich' (used for Rscript invocation)
"""

import argparse
import datetime
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_comma_separated(valid_choices):
    """
    A helper function (closure) for custom type parsing and validation in argparse.
    It splits a comma-separated string into a list and verifies that each element
    is within the allowed choices.
    """

    def parse(value):
        # Split the string by commas and remove leading/trailing whitespaces from each item
        items = [item.strip() for item in value.split(',')]

        for item in items:
            if item not in valid_choices:
                # Raise a standard argparse error if an invalid input is detected
                raise argparse.ArgumentTypeError(
                    f"Invalid choice: '{item}'. Please use comma-separated values. "
                    f"Available options are: {', '.join(valid_choices)}"
                )
        return items

    return parse


def parse_args():
    parser = argparse.ArgumentParser(
        description="DMR functional annotation filtering and pathway enrichment analysis."
    )
    parser.add_argument('-i', '--input', required=True,
                        help="Integrated DMR annotation TSV file (output of DMRQC.py). Example: Integrated_DMRs.tsv")
    parser.add_argument('-o', '--outdir', required=True, type=str,
                        help="Output directory. Created automatically if it does not exist.")

    ALLOWED_GENE_REGIONS = {'all', 'promoter', '5utr', 'exon', 'intron', '3utr', 'intergenic'}
    ALLOWED_CPG_REGIONS  = {'all', 'island', 'shore', 'shelf', 'intercgi'}

    # Region Subset Selection
    parser.add_argument('--gene_regions', default='all', type=check_comma_separated(ALLOWED_GENE_REGIONS),
                            help="Available Regions: all, promoter, 5utr, exon, intron, 3utr, and intergenic \n"
                                 "Gene region filter (Default: all). Comma-separated for multiple values.\n")
    parser.add_argument('--cpg_regions', default='all', type=check_comma_separated(ALLOWED_CPG_REGIONS),
                            help="Available Regions: all, island, shore, shelf, and intercgi \n"
                                 "CpG context region filter (Default: all). Comma-separated for multiple values.")

    # QC Filtering
    parser.add_argument("--min_sites", type=int, default=3,
                        help="Minimum number of CpG sites per DMR. (Default: 3)")
    parser.add_argument("--min_depth", type=int, default=8,
                        help="Minimum average read depth per CpG site. (Default: 8)")

    # Gene-Level Filtering and Visualization
    parser.add_argument("--remove_overlap_genes", action='store_true',
                        help="Remove genes with both hyper- and hypomethylated DMRs. "
                             "Resolves ambiguous bidirectional regulatory signals. (Default: False)")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top genes (by absolute effect size) to include in barplots. (Default: 10)")

    # Enrichment Analysis Statistical Parameters
    parser.add_argument("--p_cutoff", type=float, default=0.05,
                        help="Nominal p-value threshold for enrichment analysis. (Default: 0.05)")
    parser.add_argument("--q_cutoff", type=float, default=0.05,
                        help="FDR-adjusted p-value (q-value) threshold for multiple testing correction. (Default: 0.05)")
    parser.add_argument("--enrich_correction", default="BH",
                        choices=["holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"],
                        help="Multiple testing correction method for clusterProfiler. "
                             "See https://yulab-smu.top/biomedical-knowledge-mining-book/02-Enrichment.html. (Default: 'BH')")
    parser.add_argument("--gprof_correction", default="fdr",
                        choices=["gSCS", "fdr", "bonferroni"],
                        help="Correction algorithm for gprofiler2. "
                             "See https://biit.cs.ut.ee/gprofiler/page/r. (Default: 'fdr')")
    return parser.parse_args()


def map_regions_to_annot_types(gene_regions, cpg_regions):
    """
    Map user-supplied region labels to annotatr annotation type strings.

    Parameters
    ----------
    : list of str
        Selected gene region labels (e.g., ['promoter', 'exon']).
    cpg_regions : list of str
        Selected CpG region labels (e.g., ['island', 'shore']).

    Returns
    -------
    tuple of (list, list)
        (target_gene_types, target_cpg_types) as annotatr-compatible type strings.
        Returns empty lists when 'all' is selected.
    """
    gene_map = {
        'promoter': ['hg38_genes_promoters', 'hg38_genes_1to5kb'],
        '5UTR':     ['hg38_genes_5UTRs'],
        'exon':     ['hg38_genes_exons'],
        'intron':   ['hg38_genes_introns'],
        '3UTR':     ['hg38_genes_3UTRs']
    }
    cpg_map = {
        'island':   ['hg38_cpg_islands'],
        'shore':    ['hg38_cpg_shores'],
        'shelf':    ['hg38_cpg_shelves'],
        'interCGI': ['hg38_cpg_inter']
    }

    target_genes = []
    if 'all' not in gene_regions:
        for r in gene_regions:
            target_genes.extend(gene_map[r])

    target_cpgs = []
    if 'all' not in cpg_regions:
        for r in cpg_regions:
            target_cpgs.extend(cpg_map[r])

    return target_genes, target_cpgs


def main():
    args      = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"[Info] Loading data from {args.input}...")
    df = pd.read_csv(args.input, sep='\t', na_values=['NA', 'NaN'])

    initial_dmr_count = df['dmr_id'].nunique()
    initial_row_count = len(df)

    # ==========================================
    # Initialize Markdown Report
    # ==========================================
    md_lines = [
        "# ONT DMR Functional Annotation & Filtering Report",
        f"**Execution Time** : {timestamp}  ",
        f"**Input File** : `{args.input}`  ",
        f"**Output Path** : `{outdir}`  ",
        "\n## 1. Filtering & Analysis Parameters",
        f"- **min_sites** : {args.min_sites}",
        f"- **min_depth** : {args.min_depth}",
        f"- **gene_regions** : {', '.join(args.gene_regions)}",
        f"- **CpG_regions** : {', '.join(args.cpg_regions)}",
        f"- **remove_overlap_genes** : {args.remove_overlap_genes}",
        f"- **P-value cutoff** (Rscript) : {args.p_cutoff}",
        f"- **Q-value cutoff** (Rscript) : {args.q_cutoff}",
        f"- **enrich_correction** (Rscript) : {args.enrich_correction}",
        f"- **gprof_correction** (Rscript) : {args.gprof_correction}",
        "\n## 2. Input Data Summary",
        f"- **Total Input Annotation Rows** : {initial_row_count}",
        f"- **Total Unique Input DMRs** : {initial_dmr_count}",
        "\n## 3. Step-by-Step Filtering Tracking"
    ]

    df_target = df.copy()

    # ==========================================
    # Filter 1: Minimum Depth and CpG Site Count
    # ==========================================
    if args.min_sites > 0 or args.min_depth > 0:
        df_target = df_target[
            (df_target['num_sites']  >= args.min_sites) &
            (df_target['depth_A_x'] >= args.min_depth) &
            (df_target['depth_B_x'] >= args.min_depth)
        ]
        current_dmr_count = df_target['dmr_id'].nunique()
        retention_rate    = (current_dmr_count / initial_dmr_count) * 100 if initial_dmr_count > 0 else 0
        md_lines.append("### Step 1: Depth & Sites Filtering")
        md_lines.append(f"- **Remaining DMRs**: {current_dmr_count} (Retention Rate: {retention_rate:.2f}%)")
    else:
        md_lines.append("### Step 1: Depth & Sites Filtering bypassed.")

    # ==========================================
    # Filter 2: Region Intersection (Gene and CpG)
    # ==========================================
    target_gene_types, target_cpg_types = map_regions_to_annot_types(args.gene_regions, args.cpg_regions)

    dmr_set_genes = set(df_target['dmr_id'])
    if target_gene_types:
        dmr_set_genes = set(df_target[df_target['annot.type'].isin(target_gene_types)]['dmr_id'])

    dmr_set_cpg = set(df_target['dmr_id'])
    if target_cpg_types:
        dmr_set_cpg = set(df_target[df_target['annot.type'].isin(target_cpg_types)]['dmr_id'])

    valid_dmrs    = dmr_set_genes.intersection(dmr_set_cpg)
    df_target     = df_target[df_target['dmr_id'].isin(valid_dmrs)]
    df_target     = df_target[df_target['annot.symbol'].notna()]

    current_dmr_count  = df_target['dmr_id'].nunique()
    current_gene_count = df_target['annot.symbol'].nunique()
    retention_rate     = (current_dmr_count / initial_dmr_count) * 100 if initial_dmr_count > 0 else 0

    md_lines.append("### Step 2: Region Intersection Filtering")
    if target_gene_types or target_cpg_types:
        md_lines.append("- Applied Logic: Intersection of valid Gene regions and CpG regions.")
    else:
        md_lines.append("- Applied Logic: All regions retained (default).")
    md_lines.append(
        f"- **Remaining**: {current_dmr_count} DMRs (Retention Rate: {retention_rate:.2f}%) "
        f"mapping to {current_gene_count} genes."
    )

    # ==========================================
    # Filter 3: Hyper/Hypo Direction Conflict Check
    # ==========================================
    hyper_genes   = set(df_target[df_target['DM_status_x'] == 'Hyper']['annot.symbol'])
    hypo_genes    = set(df_target[df_target['DM_status_x'] == 'Hypo']['annot.symbol'])
    overlap_genes = hyper_genes.intersection(hypo_genes)

    md_lines.append("### Step 3: Direction Conflict (Hyper/Hypo) Check")
    if overlap_genes:
        print(f"[Warning] {len(overlap_genes)} genes have both Hyper and Hypo DMRs.")
        md_lines.append(f"- **Warning**: Detected {len(overlap_genes)} overlapping genes.")
        if args.remove_overlap_genes:
            df_target          = df_target[~df_target['annot.symbol'].isin(overlap_genes)]
            current_gene_count = df_target['annot.symbol'].nunique()
            md_lines.append(f"- **Action**: Removed overlapping genes. Remaining: {current_gene_count} genes.")
        else:
            md_lines.append("- **Action**: Retained overlapping genes.")
    else:
        md_lines.append("- No overlapping genes detected.")

    # ==========================================
    # Final Report Summary
    # ==========================================
    hyper_final_genes = df_target[df_target['DM_status_x'] == 'Hyper']['annot.symbol'].nunique()
    hypo_final_genes  = df_target[df_target['DM_status_x'] == 'Hypo']['annot.symbol'].nunique()

    md_lines.append("\n## 4. Final Output Summary")
    md_lines.append(f"- **Total Unique Genes Analyzed**: {df_target['annot.symbol'].nunique()}")
    md_lines.append(f"- **Final Hypermethylated Genes**: {hyper_final_genes}")
    md_lines.append(f"- **Final Hypomethylated Genes**: {hypo_final_genes}")

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[Info] Report saved to {report_path}")

    # ==========================================
    # Export Filtered Annotation and Gene ID Lists
    # ==========================================
    filtered_tsv_path = os.path.join(outdir, "annotatr_details.filtered.tsv")
    df_target.to_csv(filtered_tsv_path, sep='\t', index=False)

    hyper_df      = df_target[df_target['DM_status_x'] == 'Hyper']
    hyper_ids     = hyper_df['annot.gene_id'].dropna().unique().astype(int).astype(str).tolist()
    hyper_symbols = hyper_df['annot.symbol'].dropna().unique().tolist()

    hypo_df      = df_target[df_target['DM_status_x'] == 'Hypo']
    hypo_ids     = hypo_df['annot.gene_id'].dropna().unique().astype(int).astype(str).tolist()
    hypo_symbols = hypo_df['annot.symbol'].dropna().unique().tolist()

    pd.Series(hyper_ids).to_csv(    os.path.join(outdir, "hyper_ids.txt"),     index=False, header=False)
    pd.Series(hypo_ids).to_csv(     os.path.join(outdir, "hypo_ids.txt"),      index=False, header=False)
    pd.Series(hyper_symbols).to_csv(os.path.join(outdir, "hyper_symbols.txt"), index=False, header=False)
    pd.Series(hypo_symbols).to_csv( os.path.join(outdir, "hypo_symbols.txt"),  index=False, header=False)

    print("[Info] Entrez ID and gene symbol lists exported.")

    # ==========================================
    # Figures
    # ==========================================

    # Effect size distribution histogram
    plt.figure(figsize=(10, 6))
    plot_df = df_target.copy()
    plot_df['abs_effect_size'] = plot_df['effect_size_x'].abs()
    plot_df.rename(columns={'DM_status_x': 'Methylation Status'}, inplace=True)

    sns.histplot(data=plot_df, x='abs_effect_size', hue='Methylation Status',
                 hue_order=['Hyper', 'Hypo'], multiple='stack', bins=30,
                 palette={'Hyper': 'tab:blue', 'Hypo': 'tab:orange'},
                 edgecolor='white', linewidth=0.5)
    plt.xlim(0, 1)
    plt.title('Distribution of Absolute Effect Sizes')
    plt.xlabel('Absolute Effect Size')
    plt.ylabel('DMR Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_size_dist.png"), dpi=300)
    plt.close()

    # Top N gene barplots (hyper and hypo)
    top_hyper = (hyper_df.groupby('annot.symbol')['effect_size_x']
                 .max().sort_values(ascending=False).head(args.top_n))
    top_hypo  = (hypo_df.groupby('annot.symbol')['effect_size_x']
                 .min().sort_values(ascending=True).head(args.top_n))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    if not top_hyper.empty:
        sns.barplot(x=top_hyper.values, y=top_hyper.index, ax=axes[0], color='tab:blue', alpha=0.8)
        axes[0].set_title(f'Top {args.top_n} Hypermethylated Genes')
        axes[0].set_xlabel('Effect Size')
        axes[0].set_ylabel('Gene Symbol')

    if not top_hypo.empty:
        sns.barplot(x=top_hypo.values, y=top_hypo.index, ax=axes[1], color='tab:orange', alpha=0.8)
        axes[1].set_title(f'Top {args.top_n} Hypomethylated Genes')
        axes[1].set_xlabel('Effect Size')
        axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"top{args.top_n}_effect_size.png"), dpi=300)
    plt.close()

    # ==========================================
    # Generate and Execute R Enrichment Script
    # ==========================================
    r_script_content = f"""
# Suppress Rplots.pdf generation in non-interactive sessions
pdf(NULL)

suppressPackageStartupMessages({{
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(DOSE)
  library(ReactomePA)
  library(gprofiler2)
  library(ggplot2)
  library(enrichplot)
}})

# Helper: format p-values using fixed notation above 0.0001, scientific below
format_pval <- function(breaks) {{
  sapply(breaks, function(val) {{
    if (is.na(val)) return(NA)
    if (val >= 0.0001) {{
      return(formatC(val, format = "f", digits = 4, drop0trailing = TRUE))
    }} else {{
      return(sprintf("%.1e", val))
    }}
  }})
}}

# Parse optional command-line overrides; fall back to Python-injected defaults
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(idx, default) {{
  if (length(args) >= idx && !is.na(args[idx]) && args[idx] != "") args[idx] else default
}}

P_CUTOFF          <- as.numeric(get_arg(1, {args.p_cutoff}))
Q_CUTOFF          <- as.numeric(get_arg(2, {args.q_cutoff}))
enrich_CORRECTION <- get_arg(3, "{args.enrich_correction}")
GPROF_CORRECTION  <- get_arg(4, "{args.gprof_correction}")
OUTDIR            <- get_arg(5, "{outdir}")
HYPER_ID          <- get_arg(6, file.path(OUTDIR, "hyper_ids.txt"))
HYPO_ID           <- get_arg(7, file.path(OUTDIR, "hypo_ids.txt"))

run_comprehensive_enrichment <- function(id_file, title_prefix) {{
  if (!file.exists(id_file)) return(NULL)
  entrez_ids <- as.character(read.table(id_file, header=FALSE, stringsAsFactors=FALSE)$V1)

  if (length(entrez_ids) < 10) {{
    cat(paste("\\n[Skip]", title_prefix, "- fewer than 10 genes; skipping enrichment.\\n"))
    return(NULL)
  }}

  cat(paste("\\n=======================================================\\n"))
  cat(paste(">>>", title_prefix, "enrichment analysis (Entrez IDs)...\\n"))
  cat(paste("=======================================================\\n"))

  # Helper: save result table and generate dot/bar plots
  save_results <- function(obj, name_suffix) {{
    if (!is.null(obj) && nrow(as.data.frame(obj)) > 0) {{
      write.csv(as.data.frame(obj),
                file.path(OUTDIR, paste0(title_prefix, "_", name_suffix, "_results.csv")),
                row.names=FALSE)

      suppressWarnings(suppressMessages({{
        p_dot <- dotplot(obj, showCategory=15) +
          ggtitle(paste(title_prefix, name_suffix, "Enrichment")) +
          scale_color_continuous(low="red", high="blue", name="p.adjust", labels=format_pval)
        ggsave(file.path(OUTDIR, paste0(title_prefix, "_", name_suffix, "_dotplot.png")),
               plot=p_dot, width=8, height=6, dpi=300)

        p_bar <- barplot(obj, showCategory=15) +
          ggtitle(paste(title_prefix, name_suffix, "Enrichment")) +
          scale_fill_continuous(low="red", high="blue", name="p.adjust", labels=format_pval)
        ggsave(file.path(OUTDIR, paste0(title_prefix, "_", name_suffix, "_barplot.png")),
               plot=p_bar, width=8, height=6, dpi=300)
      }}))
      cat(paste("     [Done]", name_suffix, "results and figures saved.\\n"))
    }}
  }}

  # GO Biological Process enrichment
  cat("  -> Running GO enrichment...\\n")
  tryCatch({{
    ego <- enrichGO(gene=entrez_ids, OrgDb=org.Hs.eg.db, ont="BP",
                    pAdjustMethod=enrich_CORRECTION, pvalueCutoff=P_CUTOFF,
                    qvalueCutoff=Q_CUTOFF, readable=TRUE)
    save_results(ego, "GO")
    if (!is.null(ego) && nrow(as.data.frame(ego)) > 0) {{
      suppressWarnings({{
        p_net <- cnetplot(ego, showCategory=5)
        ggsave(file.path(OUTDIR, paste0(title_prefix, "_GO_cnetplot.png")),
               plot=p_net, width=12, height=9, dpi=300)
      }})
    }}
  }}, error=function(e) {{ cat("\\n[GO Error]:\\n"); print(e) }})

  # Disease Ontology enrichment
  cat("  -> Running DO enrichment...\\n")
  tryCatch({{
    edo <- enrichDO(gene=entrez_ids, pvalueCutoff=P_CUTOFF, qvalueCutoff=Q_CUTOFF,
                    readable=TRUE, pAdjustMethod=enrich_CORRECTION)
    save_results(edo, "DO")
  }}, error=function(e) {{ cat("\\n[DO Error]:\\n"); print(e) }})

  # Reactome pathway enrichment
  cat("  -> Running Reactome enrichment...\\n")
  tryCatch({{
    ereact <- enrichPathway(gene=entrez_ids, pvalueCutoff=P_CUTOFF, qvalueCutoff=Q_CUTOFF,
                            readable=TRUE, pAdjustMethod=enrich_CORRECTION)
    save_results(ereact, "Reactome")
  }}, error=function(e) {{ cat("\\n[Reactome Error]:\\n"); print(e) }})

  # g:Profiler multi-database enrichment
  cat("  -> Running gprofiler2 enrichment...\\n")
  tryCatch({{
    set_base_url("https://biit.cs.ut.ee/gprofiler")
    gostres <- gost(query=entrez_ids, organism="hsapiens",
                    sources=c("KEGG", "REAC", "WP", "TF", "MIRNA", "HP", "GO:BP", "GO:CC", "GO:MF"),
                    user_threshold=P_CUTOFF, correction_method=GPROF_CORRECTION)

    if (!is.null(gostres$result)) {{
      res_df <- gostres$result
      res_df[] <- lapply(res_df, function(x) if (is.list(x)) sapply(x, paste, collapse=";") else x)
      colnames(res_df)[colnames(res_df) == "p_value"] <- "p.adjust"

      write.csv(res_df,
                file.path(OUTDIR, paste0(title_prefix, "_gProfiler_results.csv")),
                row.names=FALSE)

      suppressWarnings(suppressMessages({{
        # Annotated Manhattan plot highlighting top 3 terms per source
        p_gost   <- gostplot(gostres, capped=FALSE, interactive=FALSE)
        top_terms <- c()
        for (s in unique(res_df$source)) {{
          s_df      <- res_df[res_df$source == s, ]
          s_df      <- s_df[order(s_df$p.adjust), ]
          top_terms <- c(top_terms, s_df$term_id[1:min(3, nrow(s_df))])
        }}
        p_gost_pub <- publish_gostplot(p_gost, highlight_terms=top_terms)
        ggsave(file.path(OUTDIR, paste0(title_prefix, "_gProfiler_Manhattan_Annotated.png")),
               plot=p_gost_pub, width=14, height=10, dpi=300)

        # Per-source dot plots and bar plots
        for (s in unique(res_df$source)) {{
          s_df <- res_df[res_df$source == s, ]
          s_df <- s_df[order(s_df$p.adjust), ]
          if (nrow(s_df) > 15) s_df <- s_df[1:15, ]

          s_df$term_name_short <- substr(s_df$term_name, 1, 60)
          s_df$term_name_short <- ifelse(nchar(s_df$term_name) > 60,
                                          paste0(s_df$term_name_short, "..."),
                                          s_df$term_name_short)
          s_df$term_name_short <- factor(s_df$term_name_short, levels=rev(s_df$term_name_short))
          safe_s <- gsub(":", "_", s)

          p_dot_src <- ggplot(s_df, aes(x=intersection_size/query_size, y=term_name_short,
                                         size=intersection_size, color=p.adjust)) +
            geom_point() +
            scale_color_continuous(low="red", high="blue", name="p.adjust", labels=format_pval) +
            theme_minimal() +
            labs(title=paste(title_prefix, s, "Dotplot"), x="Gene Ratio", y="", size="Count")
          ggsave(file.path(OUTDIR, paste0(title_prefix, "_gProfiler_", safe_s, "_dotplot.png")),
                 plot=p_dot_src, width=10, height=6, dpi=300)

          p_bar_src <- ggplot(s_df, aes(x=-log10(p.adjust), y=term_name_short, fill=p.adjust)) +
            geom_bar(stat="identity") +
            scale_fill_continuous(low="red", high="blue", name="p.adjust", labels=format_pval) +
            theme_minimal() +
            labs(title=paste(title_prefix, s, "Barplot"), x="-log10(p.adjust)", y="")
          ggsave(file.path(OUTDIR, paste0(title_prefix, "_gProfiler_", safe_s, "_barplot.png")),
                 plot=p_bar_src, width=10, height=6, dpi=300)
        }}
      }}))
      cat("     [Done] gprofiler2 figures saved.\\n")
    }}
  }}, error=function(e) {{ cat("\\n[gprofiler2 Error]:\\n"); print(e) }})
}}

run_comprehensive_enrichment(HYPER_ID, "Hyper")
run_comprehensive_enrichment(HYPO_ID,  "Hypo")

cat("\\n>>> R enrichment analysis complete.\\n")
"""

    r_script_path = os.path.join(outdir, "run_pathway.R")
    with open(r_script_path, "w", encoding="utf-8") as f:
        f.write(r_script_content)

    print("\n[Info] Executing R enrichment script...")
    try:
        subprocess.run(
            ["Rscript", r_script_path],
            check=True
        )
        print("[Info] Enrichment analysis complete.")
    except subprocess.CalledProcessError:
        print("[Error] R script execution failed. Ensure all required Bioconductor packages "
              "and gprofiler2 are installed in the 'r-enrich' conda environment.")
    except FileNotFoundError:
        print("[Error] 'Rscript' or 'conda' command not found.")


if __name__ == "__main__":
    main()
