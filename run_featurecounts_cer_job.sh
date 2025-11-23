#!/bin/bash

# Define input/output
OUT_FILE="/rnaseq_project/new_annotation/mapping_independent/featurecounts_T_cer_stringent/featurecounts_results_T_cer_stringent/gene_counts_T_cer_stringent.txt"
GTF="/rnaseq_project/new_annotation/genome/Saccharomyces_cerevisiae.R64-1-1.114.gtf"

# Setup working directories
cd /rnaseq_project/new_annotation/mapping_independent/featurecounts_T_cer_stringent

mkdir -p featurecounts_results_T_cer_stringent
mkdir -p logs

echo "Running featureCounts..."

featureCounts \
    -T 8 \
    -s 2 \
    -p -B -C \
    -a "$GTF" \
    -o "$OUT_FILE" \
    *.bam

echo "featureCounts complete."
