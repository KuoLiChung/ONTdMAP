#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CreateUnionBed.py

Description:
    Synchronizes multi-sample bedMethyl datasets from modkit pileup into a
    unified TSV matrix for group-level methylation analysis. Consolidates
    genomic coordinates and sample-specific depth and modification counts.

Dependencies:
    - Python >= 3.6
    - gzip
"""

import os
import gzip
import sys
import argparse
from contextlib import ExitStack


def get_coords(cols):
    """
    Converts chromosome and genomic start positions into a comparable tuple
    for synchronized linear scanning across multiple file streams.
    """
    if cols is None: return (float('inf'), float('inf'))
    try:
        return (cols[0], int(cols[1]))
    except (ValueError, IndexError):
        return (float('inf'), float('inf'))


def get_next_m_record(handler):
    """
    Retrieves the subsequent 'm' (5mC) modification record from the bedMethyl
    file stream, filtering out headers and non-target modification types ('h').
    """
    while True:
        try:
            line = next(handler).strip().split('\t')
            if line[0].startswith('#') or "chrom" in line[0].lower():
                continue
            if len(line) > 3 and line[3] == 'm':
                return line
        except StopIteration:
            return None


def main():
    # 1. Command-line interface for multi-sample matrix construction
    parser = argparse.ArgumentParser(
        description="Synchronized scanning for multi-sample methylation matrix construction.")
    parser.add_argument('-b', '--bed', required=True,
                        help="Reference BED file defining target genomic loci (e.g., CpG sites).")
    parser.add_argument('-o', '--output', required=True,
                        help="Path for the generated union bedMethyl TSV output.")
    parser.add_argument('samples', nargs='+',
                        help="Input sample files in bgzipped bedMethyl format (e.g., S1.bedmethyl.bgz S2.bedmethyl.bgz).")
    args = parser.parse_args()

    # Extracting sample identifiers from filenames
    sample_names = []
    for f in args.samples:
        base = os.path.basename(f)
        name = base.split('.')[0]
        sample_names.append(name)

    sys.stderr.write(f"[*] Loading genomic template: {args.bed}\n")
    sys.stderr.write(f"[*] Total samples for alignment: {len(args.samples)}\n")

    # 2. Initialize multi-file stream handling via ExitStack context management
    with ExitStack() as stack:
        template_h = stack.enter_context(open(args.bed, 'r'))
        out_h = stack.enter_context(open(args.output, 'w'))

        # Open multiple file handles simultaneously (supports both plain text and gzip)
        handlers = []
        for f in args.samples:
            if f.endswith('.bgz') or f.endswith('.gz'):
                handlers.append(stack.enter_context(gzip.open(f, 'rt')))
            else:
                handlers.append(stack.enter_context(open(f, 'r')))

        # Initialization: Seed the buffer with the first record from each sample
        current_lines = [get_next_m_record(h) for h in handlers]

        # Write output matrix header
        out_h.write("\t".join(["#chr", "start", "end"] + sample_names) + "\n")

        # Start synchronized scanning across the template BED
        for t_count, t_line in enumerate(template_h):
            if t_line.startswith('#') or "chrom" in t_line.lower():
                continue

            t_cols = t_line.strip().split('\t')
            t_coords = (t_cols[0], int(t_cols[1]))

            row_output = [t_cols[0], t_cols[1], t_cols[2]]

            for i in range(len(handlers)):
                # Pointer Advancement: Skip sample records that precede the current template coordinate
                while current_lines[i] and get_coords(current_lines[i]) < t_coords:
                    current_lines[i] = get_next_m_record(handlers[i])

                # Alignment Check: Verify if sample coordinate matches the template
                s_coords = get_coords(current_lines[i])
                if s_coords == t_coords:
                    # Capture metrics: index 4 (coverage), 11 (5mC count), 13 (5hmC count)
                    vals = f"{current_lines[i][4]},{current_lines[i][11]},{current_lines[i][13]}"
                    row_output.append(vals)
                    # Advance to next record for this sample
                    current_lines[i] = get_next_m_record(handlers[i])
                else:
                    # Assign NA for missing loci or insufficient coverage in sample
                    row_output.append("NA")

            # Commit synchronized row to output
            out_h.write("\t".join(row_output) + "\n")

            # Status update for large-scale processing monitoring
            if (t_count + 1) % 500000 == 0:
                sys.stderr.write(f"[*] Processed loci: {t_count + 1}...\n")

    sys.stderr.write(f"[*] Processing completed successfully. Output saved to: {args.output}\n")


if __name__ == "__main__":
    main()