#!/bin/bash

cd $SLURM_SUBMIT_DIR

# Define input/output for schoenii
OUT_FILE="/rnaseq_project/new_annotation/mapping_independent/featurecounts_schoenii/featurecounts_results/gene_counts_schoenii.txt"
GTF="/rnaseq_project/new_annotation/genome/schoenii_annotation.gtf"

# Setup working directories
cd /rnaseq_project/new_annotation/mapping_independent/featurecounts_schoenii

mkdir -p featurecounts_results
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
