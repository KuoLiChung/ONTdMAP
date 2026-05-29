#!/usr/bin/env bash


########## Reference Datasets ##########
gtf_promoter_bed='/path/to/gencode.v49.annotation.protein_coding.promoter.TSSflanking2000.bed'
hg38_cpg_bed='/path/to/hg38.CpG_2bp.bed'
hg38_blacklist='/path/to/GRCh38_unified_blacklist.bed'
hg38_annotcpg_bed='/path/to/hg38.CpG_2bp.annotated.bed.gz'
########## Reference Datasets ##########


# 0. methylQC_SampleLevel.py
ontdmap sample-qc \
  -i sampleID.filtered.aligned.sorted.mapQ20.primary.CpG.combine_strands.consistent.bedmethyl.bgz \
  -b ${gtf_promoter_bed} \
  -o methylQC \
  -p ${sample_id} \
  --min-depth 0 \
  --plots all



# 1. CreateUnionBed.py
ontdmap uni \
  -b ${hg38_cpg_bed} \
  -o union.bed \
  sample1.filtered.aligned.sorted.mapQ20.primary.CpG.combine_strands.consistent.bedmethyl.bgz \
  sample2.filtered.aligned.sorted.mapQ20.primary.CpG.combine_strands.consistent.bedmethyl.bgz \
  sample3.filtered.aligned.sorted.mapQ20.primary.CpG.combine_strands.consistent.bedmethyl.bgz \
  ......



# 2. methylQC_GroupLevel.py
ontdmap group-qc \
  -i union.bed \
  -b ${gtf_promoter_bed} \
  -o methylQC \
  --control A \
  --case B \
  --control_name ABC \
  --case_name DEF \
  -t 32



# 3.1. methylQC_GroupLevel_TSS.py: Explore data distribution (before filtering)
ontdmap group-tss \
  -i union.bed \
  -b ${gtf_promoter_bed} \
  -o methylQC \
  --control A \
  --case B \
  --control_name ABC \
  --case_name DEF \
  --min-depth 0 \
  --min-det-rate 0 \
  -t 32

# 3.2. methylQC_GroupLevel_TSS.py: Implement data filtering
ontdmap group-tss \
  -i union.bed \
  -b ${gtf_promoter_bed} \
  -o methylQC \
  --control A \
  --case B \
  --control_name ABC \
  --case_name DEF \
  --min-depth N \
  --min-det-rate N \
  -t 32



# 4.1. methylQC_PCAnHM.py: Explore variance distribution and save cache file
ontdmap group-pca \
  -i union.bed \
  -b ${gtf_promoter_bed} \
  -o methylQC \
  --control A \
  --case B \
  --control_name ABC \
  --case_name DEF \
  --min-depth N \
  --variance_only \
  --save_cache cache \
  -t 32

# 4.2. methylQC_PCAnHM.py: Load cache file and implement PCA and hierarchical clustering (HC)
ontdmap group-pca \
  -i union.bed \
  -b ${gtf_promoter_bed} \
  -o methylQC \
  --control A \
  --case B \
  --control_name ABC \
  --case_name DEF \
  --min-depth N \
  --top_global N \
  --top_promoter N \
  --top_heatmap N \
  --load_cache cache \
  -t 32



# 5. dmrQC.py: after DMR analysis using modkit dmr pair command with --segment option
ontdmap dmr-qc \
  -i dmr.mapQ20.primary.CpG.combine_strands.seg.bed \
  -o dmrQC \
  --min_sites 0 \
  --min_depth 8 \
  --genome hg38 \
  --blacklist ${hg38_blacklist}



# 6. dmrQC_PCAnHM.py
ontdmap dmr-pca \
  -d DMR_QC_pass.bed \
  -u union.bed \
  -o dmrQC \
  --top_heatmap N \
  --control A \
  --case B \
  --control_name ABC \
  --case_name DEF \
  --save_cache cache # save cache file for rerun
  # --load_cache



# 7. dmrEnrichPath.py
ontdmap dmr-pathway \
  -i Integrated_DMRs.tsv \
  -o PEA \
  --gene_regions promoter \
  --CpG_regions island,shore,shelf \
  --min_sites 0 \
  --min_depth 8 \
  --remove_overlap_genes \
  --enrich_correction BH \
  --gprof_correction fdr



# 8.1 CpGsearch.py: Single mode for Gene ID, Gene Symbol, or genomic region
ontdmap cpg single\
  -q GeneSymbol \
  --query_type symbol \
  --cpg_bed ${hg38_annotcpg_bed} \
  --gene_region promoter \
  --CpG_regions island,shore,shelf \
  --name 'GeneSymbol; promoter; island, shore, shelf' \
  --output GeneSymbol_promoter_island_shore_shelf.bed \
  --overwrite

# 8.1.1 CpGsearch.py: Batch mode for Gene List
ontdmap cpg batch\
  -q genelist.txt \
  --query_type genelist \
  --cpg_bed ${hg38_annotcpg_bed} \
  --gene_region promoter \
  --CpG_regions island,shore,shelf \
  ----name_format 'A; B; C' \
  --output genelist_promoter_island_shore_shelf.bed \
  --header True \
  --overwrite
# 8.1.2 CpGsearch.py: Batch mode for BED File
ontdmap cpg batch\
  -q query.bed \
  --query_type bed \
  --cpg_bed ${hg38_annotcpg_bed} \
  --output cpgsearch_output.bed \
  --header True \
  --overwrite
# 8.1.3 CpGsearch.py: Batch mode for TSV Format
ontdmap cpg batch\
  -q query.tsv \
  --query_type tsv \
  --cpg_bed ${hg38_annotcpg_bed} \
  --output cpgsearch_output.bed \
  --header True \
  --overwrite



# 9. methylBox.py
ontdmap box \
  -i union.bed \
  -o boxplot \
  -t genelist_promoter_island_shore_shelf.bed \
  --control A \
  --case B \
  --control_name ABC \
  --case_name DEF \
  --mode combine-depth \
  --min-depth 8 \
  --min-det-rate 0.5 \
  --stat_test ttest
