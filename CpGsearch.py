#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CpGsearch.py

Description:
    CpG site annotation search tool for querying a bgzip-compressed, tabix-indexed
    annotated CpG BED database. Supports gene ID, gene symbol, and genomic region
    queries in both single and batch modes. Filters results by CpG context region
    (island, shore, shelf, interCGI) and gene structural region (promoter, exon,
    intron, UTR, intergenic). Reports per-query failure diagnostics indicating which
    filter criterion was not satisfied.

Dependencies:
    - Python >= 3.8
    - pysam
    - htslib (tabix index required alongside the input .bgz file)

Usage:
    CpGsearch.py single --cpg_bed <db.bgz> -q <GENE|region> --query_type <id|symbol|region>
    CpGsearch.py batch  --cpg_bed <db.bgz> -q <input_file>  --query_type <genelist|bed|tsv>
"""

import argparse
import sys
import os
import re
import pysam
from collections import defaultdict


# ==========================================
# Annotation Label Utilities
# ==========================================
def clean_label(text):
    """
    Map a raw annotatr annotation type string to a standardized display label.

    Parameters
    ----------
    text : str
        Raw annotation type string from the database.

    Returns
    -------
    str
        Standardized label (e.g., 'Island', 'Promoter (<1kb)', 'Intron').
    """
    if not text or text == "NA":
        return "Unknown"
    t = text.lower()
    if "cpg" in t:
        if "islands" in t:  return "Island"
        if "shores"  in t:  return "Shore"
        if "shelves" in t:  return "Shelf"
        return "InterCGI"
    else:
        if "promoters" in t: return "Promoter (<1kb)"
        if "1to5kb"    in t: return "Promoter (1-5kb)"
        if "5utr"      in t: return "5'UTR"
        if "3utr"      in t: return "3'UTR"
        if "exons"     in t: return "Exon"
        if "introns"   in t: return "Intron"
        return "Intergenic"


def parse_gene_column(gene_col_str):
    """
    Parse the fifth column of the annotated CpG database.

    Expected format: 'Gene1:Reg1|Reg2,Gene2:Reg1' or the literal 'intergenic'.
    Multiple gene identifiers before the colon are pipe-delimited; multiple
    region labels after the colon are also pipe-delimited.

    Parameters
    ----------
    gene_col_str : str
        Raw string from column 5 of the database BED record.

    Returns
    -------
    dict
        Mapping of uppercase gene identifier -> list of lowercase region labels.
        Returns {'INTERGENIC': ['intergenic']} for intergenic entries.
    """
    if gene_col_str.lower() == "intergenic":
        return {"INTERGENIC": ["intergenic"]}

    gene_dict = {}
    for block in gene_col_str.split(','):
        if ':' not in block:
            continue
        genes_part, regs_part = block.split(':', 1)
        gene_identifiers = genes_part.split('|')
        regions          = regs_part.split('|')
        for identifier in gene_identifiers:
            gene_dict[identifier.upper()] = [r.lower() for r in regions]
    return gene_dict


def check_criteria(cpg_val, parsed_genes, target_genes_set,
                   req_cpgs, req_gene_regs, is_region_query):
    """
    Evaluate whether a CpG site satisfies the target, CpG region, and gene region criteria.

    Parameters
    ----------
    cpg_val : str
        CpG context annotation string from column 4 of the database record.
    parsed_genes : dict
        Output of parse_gene_column() for the current record.
    target_genes_set : set
        Uppercase gene identifiers to match (empty for region queries).
    req_cpgs : list of str
        Required CpG region labels (lowercase); ['all'] to skip filtering.
    req_gene_regs : list of str
        Required gene region labels (lowercase); ['all'] to skip filtering.
    is_region_query : bool
        If True, skip gene-name matching and accept any gene at the locus.

    Returns
    -------
    tuple of (bool, bool, bool)
        (found_target, passed_cpg_filter, passed_gene_region_filter)
    """
    found_target    = False
    passed_cpg      = False
    passed_gene_reg = False

    # Step 1: Determine whether the target gene or region is present at this site
    if is_region_query:
        genes_to_check = list(parsed_genes.keys())
        found_target   = True
    else:
        matched_targets = target_genes_set.intersection(set(parsed_genes.keys()))
        if not matched_targets:
            return False, False, False
        genes_to_check = list(matched_targets)
        found_target   = True

    # Step 2: Evaluate CpG context filter
    if 'all' in req_cpgs:
        passed_cpg = True
    else:
        if any(req in cpg_val.lower() for req in req_cpgs):
            passed_cpg = True

    # Step 3: Evaluate gene region filter
    if 'all' in req_gene_regs:
        passed_gene_reg = True
    else:
        for g in genes_to_check:
            site_regs_for_gene = parsed_genes[g]
            for req in req_gene_regs:
                search_term = req
                if req == "5utr": search_term = "5'utr"
                if req == "3utr": search_term = "3'utr"
                if any(search_term in r for r in site_regs_for_gene):
                    passed_gene_reg = True
                    break
            if passed_gene_reg:
                break

    return found_target, passed_cpg, passed_gene_reg


# ==========================================
# Argument Parsing
# ==========================================
def parse_args():
    """
    Build the argument parser with 'single' and 'batch' subcommands.

    Returns
    -------
    argparse.Namespace
        Validated parsed arguments.
    """
    # Shared options inherited by both subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_group  = parent_parser.add_argument_group('Common Options')

    parent_group.add_argument('--cpg_bed', required=True,
                              help="[Required] Path to the annotated CpG BED database (.bgz). "
                                   "A matching .tbi tabix index must exist alongside it.")
    parent_group.add_argument('--output',
                              help="[Optional] Output BED file path. "
                                   "Defaults to stdout if not specified.")
    parent_group.add_argument('--overwrite', action='store_true',
                              help="[Optional] Overwrite the output file if it already exists.\n"
                                   "If NOT provided, new results will be APPENDED to the existing file.\n"
                                   "*** Note on grouping when using --cpg_region and --gene_region options: \n"
                                   "Querying multiple regions in a single run groups them into a single output name.\n"
                                   "To separate them, run this tool multiple times for each region without --overwrite.")

    main_parser = argparse.ArgumentParser(
        description="CpG Site Annotation Search Tool.\n"
                    "Provides 'single' and 'batch' modes for precise CpG annotation retrieval.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = main_parser.add_subparsers(
        dest='mode', required=True,
        help="Execution mode: 'single' or 'batch'."
    )

    # Single mode
    parser_single = subparsers.add_parser(
        'single',
        parents=[parent_parser],
        help="Single query: search for a gene ID, gene symbol, or genomic region.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    single_req = parser_single.add_argument_group('Single Mode - Required Arguments')
    single_req.add_argument('-q', '--query', required=True,
                            help="Query target: gene ID, gene symbol, or genomic region "
                                 "(e.g., 'chr1:10000-20000').")
    single_req.add_argument('--query_type', required=True, type=str.lower,
                            choices=['id', 'symbol', 'region'],
                            help="Data type of the -q input.")

    single_opt = parser_single.add_argument_group('Single Mode - Filtering & Formatting Options')
    single_opt.add_argument('--gene_region', default='all',
                            help="Available Regions: all, promoter, 5utr, exon, intron, 3utr, and intergenic \n"
                                 "Gene region filter (Default: all). Comma-separated for multiple values.\n"
                                 "Note: 'intergenic' is not valid when --query_type is 'id' or 'symbol'.")
    single_opt.add_argument('--cpg_region', default='all',
                            help="Available Regions: all, island, shore, shelf, and intercgi \n"
                                 "CpG context region filter (Default: all). Comma-separated for multiple values.")
    single_opt.add_argument('--name', default=None,
                            help="Custom string for column 4 of the output BED.\n"
                                 "Default: '<query>; <gene_region>; <cpg_region>'")

    # Batch mode
    parser_batch = subparsers.add_parser(
        'batch',
        parents=[parent_parser],
        help="Batch query: process multiple targets from a file.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    batch_req = parser_batch.add_argument_group('Batch Mode - Required Arguments')
    batch_req.add_argument('-q', '--query', required=True,
                           help="Path to the input query file.")
    batch_req.add_argument('--query_type', required=True, type=str.lower,
                           choices=['genelist', 'bed', 'tsv'],
                           help="Format of the input file:\n"
                                "  'genelist' : One gene symbol or ID per line.\n"
                                "  'bed'      : BED file with columns: chr, start, end, name, gene_region, cpg_region.\n"
                                "  'tsv'      : TSV with columns: query*, type, name, gene_region, cpg_region. \n"
                                "  * Query in TSV format can be either of id, symbol, or region (e.g., 'chr1:10000-20000').")

    batch_opt = parser_batch.add_argument_group('Batch Mode - Options')
    batch_opt.add_argument('--header', type=lambda x: (str(x).lower() == 'true'), default=True,
                           help="Whether the input file has a header row. (Default: True)")
    batch_opt.add_argument('--gene_region', default='all',
                           help="[genelist only] Available Regions: all, promoter, 5utr, exon, intron, 3utr, and intergenic \n"
                                "[genelist only] Gene region filter (Default: all). Comma-separated for multiple values.")
    batch_opt.add_argument('--cpg_region', default='all',
                           help="[genelist only] Available Regions: all, island, shore, shelf, and intercgi \n"
                                "[genelist only] CpG context region filter (Default: all). Comma-separated for multiple values.")
    batch_opt.add_argument('--name_format', default='A; B; C',
                           help="[genelist only] Column 4 format template. (Default: 'A; B; C')\n"
                                "[genelist only] A: query | B: gene_region | C: cpg_region")

    # Validate parsed arguments
    args = main_parser.parse_args()

    ALLOWED_GENE_REGIONS = {'all', 'promoter', '5utr', 'exon', 'intron', '3utr', 'intergenic'}
    ALLOWED_CPG_REGIONS  = {'all', 'island', 'shore', 'shelf', 'intercgi'}

    req_genes = [x.strip().lower() for x in args.gene_region.split(',')]
    invalid_genes = [x for x in req_genes if x not in ALLOWED_GENE_REGIONS]
    if invalid_genes:
        main_parser.error(
            f"Invalid '--gene_region' value(s): {', '.join(invalid_genes)}\n"
            f"Allowed values: {', '.join(ALLOWED_GENE_REGIONS)}"
        )

    req_cpgs = [x.strip().lower() for x in args.cpg_region.split(',')]
    invalid_cpgs = [x for x in req_cpgs if x not in ALLOWED_CPG_REGIONS]
    if invalid_cpgs:
        main_parser.error(
            f"Invalid '--cpg_region' value(s): {', '.join(invalid_cpgs)}\n"
            f"Allowed values: {', '.join(ALLOWED_CPG_REGIONS)}"
        )

    # Gene queries cannot target intergenic regions
    if args.mode == 'single' and args.query_type in ['id', 'symbol']:
        if 'intergenic' in req_genes:
            main_parser.error(
                "When '--query_type' is 'id' or 'symbol', '--gene_region' cannot include 'intergenic'."
            )

    return args


# ==========================================
# Search Engine
# ==========================================
def run_search_tasks(tasks, tbx, out_fh):
    """
    Execute a list of standardized search tasks against a tabix-indexed CpG database.

    Processes region queries with targeted tabix fetches and gene queries with a
    single full-file scan. Reports per-task diagnostics for any query that yields
    no output, indicating which filter criterion (target presence, CpG region,
    or gene region) was not satisfied.

    Parameters
    ----------
    tasks : list of dict
        Each task dict must contain: 'query', 'qtype', 'name', 'req_cpg', 'req_gene'.
    tbx : pysam.TabixFile
        Open tabix file handle for the annotated CpG database.
    out_fh : file-like object
        Output file handle (stdout or an open file).
    """
    for task in tasks:
        task['status'] = {
            'found_target':    False,
            'passed_cpg':      False,
            'passed_gene_reg': False,
            'success':         False
        }

    region_tasks = [t for t in tasks if t['qtype'] == 'region']
    gene_tasks   = [t for t in tasks if t['qtype'] in ['id', 'symbol']]
    printed_sites = set()

    # Process region queries with targeted tabix fetches
    for task in region_tasks:
        chrom, pos   = task['query'].split(':')
        start, end   = map(int, pos.split('-'))
        try:
            for row in tbx.fetch(chrom, start, end):
                fields        = row.split('\t')
                coord_id      = f"{fields[0]}:{fields[1]}-{fields[2]}"
                unique_out_id = f"{coord_id}_{task['name']}"
                if unique_out_id in printed_sites:
                    continue

                cpg_val      = fields[3]
                parsed_genes = parse_gene_column(fields[4])

                ft, pc, pgr = check_criteria(
                    cpg_val, parsed_genes, set(),
                    task['req_cpg'], task['req_gene'],
                    is_region_query=True
                )

                if ft:  task['status']['found_target']    = True
                if pc:  task['status']['passed_cpg']      = True
                if pgr: task['status']['passed_gene_reg'] = True

                if ft and pc and pgr:
                    task['status']['success'] = True
                    out_fh.write(f"{fields[0]}\t{fields[1]}\t{fields[2]}\t{task['name']}\n")
                    printed_sites.add(unique_out_id)
        except ValueError:
            pass  # Coordinate range not present in the index

    # Process gene queries with a single full-file scan
    if gene_tasks:
        print(
            f"[Info] Batch contains {len(gene_tasks)} gene query/queries. "
            "Running a single full-file scan...",
            file=sys.stderr
        )
        global_target_genes = set(t['query'].upper() for t in gene_tasks)

        for row in tbx.fetch():
            fields       = row.split('\t')
            coord_id     = f"{fields[0]}:{fields[1]}-{fields[2]}"
            cpg_val      = fields[3]
            parsed_genes = parse_gene_column(fields[4])

            site_genes      = set(parsed_genes.keys())
            matched_targets = global_target_genes.intersection(site_genes)
            if not matched_targets:
                continue

            for task in gene_tasks:
                if task['query'].upper() not in matched_targets:
                    continue

                unique_out_id = f"{coord_id}_{task['name']}"
                if unique_out_id in printed_sites:
                    continue

                ft, pc, pgr = check_criteria(
                    cpg_val, parsed_genes, {task['query'].upper()},
                    task['req_cpg'], task['req_gene'],
                    is_region_query=False
                )

                if ft:  task['status']['found_target']    = True
                if pc:  task['status']['passed_cpg']      = True
                if pgr: task['status']['passed_gene_reg'] = True

                if ft and pc and pgr:
                    task['status']['success'] = True
                    out_fh.write(f"{fields[0]}\t{fields[1]}\t{fields[2]}\t{task['name']}\n")
                    printed_sites.add(unique_out_id)

    # Report per-task failure diagnostics
    for task in tasks:
        if not task['status']['success']:
            q = task['query']
            s = task['status']

            if not s['found_target']:
                print(
                    f"[Warning] Query '{q}': gene not found in database, or no CpG sites "
                    "in the specified coordinate range.",
                    file=sys.stderr
                )
            elif not s['passed_cpg'] and not s['passed_gene_reg']:
                print(
                    f"[Warning] Query '{q}': target found, but no sites satisfy both "
                    f"cpg_region ({task['req_cpg']}) and gene_region ({task['req_gene']}).",
                    file=sys.stderr
                )
            elif not s['passed_cpg']:
                print(
                    f"[Warning] Query '{q}': target found, but no sites satisfy "
                    f"cpg_region ({task['req_cpg']}).",
                    file=sys.stderr
                )
            elif not s['passed_gene_reg']:
                print(
                    f"[Warning] Query '{q}': target found, but no sites satisfy "
                    f"gene_region ({task['req_gene']}).",
                    file=sys.stderr
                )


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()

    # File output logic: Support overwrite ('w') and append ('a') modes
    if args.output:
        if os.path.exists(args.output):
            if args.overwrite:
                out_fh = open(args.output, 'w')   # Force overwrite
            else:
                out_fh = open(args.output, 'a')   # File exists and overwrite is not set; append new lines
        else:
            out_fh = open(args.output, 'w')       # File does not exist; create a new file
    else:
        out_fh = sys.stdout                       # Output to standard output (terminal)

    try:
        tbx = pysam.TabixFile(args.cpg_bed)
    except Exception as e:
        sys.exit(f"Error opening --cpg_bed: {e}")

    tasks = []

    # ==========================================
    # Single Mode
    # ==========================================
    if args.mode == 'single':
        req_gene = [x.strip().lower() for x in args.gene_region.split(',')]
        req_cpg  = [x.strip().lower() for x in args.cpg_region.split(',')]

        if args.query_type in ['id', 'symbol'] and 'intergenic' in req_gene:
            sys.exit("Error: '--gene_region' cannot include 'intergenic' when '--query_type' is 'id' or 'symbol'.")

        name_str = (args.name if args.name is not None
                    else f"{args.query}; {args.gene_region.replace(',', ', ')}; "
                         f"{args.cpg_region.replace(',', ', ')}")

        tasks.append({
            'query':    args.query,
            'qtype':    args.query_type,
            'name':     name_str,
            'req_gene': req_gene,
            'req_cpg':  req_cpg
        })

    # ==========================================
    # Batch Mode
    # ==========================================
    elif args.mode == 'batch':
        if not os.path.isfile(args.query):
            sys.exit(f"Error: batch query file '{args.query}' does not exist.")

        with open(args.query, 'r') as f:
            lines = f.readlines()

        if args.header and lines:
            lines = lines[1:]  # Skip header row

        if args.query_type == 'genelist':
            req_gene = [x.strip().lower() for x in args.gene_region.split(',')]
            req_cpg  = [x.strip().lower() for x in args.cpg_region.split(',')]
            fmt_gene = args.gene_region.replace(',', ', ')
            fmt_cpg  = args.cpg_region.replace(',', ', ')

            for line in lines:
                q = line.strip()
                if not q:
                    continue
                fmt_name = (args.name_format.replace('A', '{0}').replace('B', '{1}').replace('C', '{2}')
                            .format(q, fmt_gene, fmt_cpg))
                tasks.append({
                    'query':    q,
                    'qtype':    'symbol',
                    'name':     fmt_name,
                    'req_gene': req_gene,
                    'req_cpg':  req_cpg
                })

        elif args.query_type == 'bed':
            for line in lines:
                fields = line.strip().split('\t')
                if len(fields) < 6:
                    continue
                # Columns: chr, start, end, name, gene_region, cpg_region
                q_region = f"{fields[0]}:{fields[1]}-{fields[2]}"
                tasks.append({
                    'query':    q_region,
                    'qtype':    'region',
                    'name':     fields[3],
                    'req_gene': [x.strip().lower() for x in fields[4].split(',')],
                    'req_cpg':  [x.strip().lower() for x in fields[5].split(',')]
                })

        elif args.query_type == 'tsv':
            for line in lines:
                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue
                # Columns: query, type, name, gene_region, cpg_region
                tasks.append({
                    'query':    fields[0],
                    'qtype':    fields[1].lower(),
                    'name':     fields[2],
                    'req_gene': [x.strip().lower() for x in fields[3].split(',')],
                    'req_cpg':  [x.strip().lower() for x in fields[4].split(',')]
                })

    # ==========================================
    # Execute Tasks
    # ==========================================
    run_search_tasks(tasks, tbx, out_fh)

    if args.output:
        out_fh.close()


if __name__ == '__main__':
    main()