#!/bin/bash

cd $SLURM_SUBMIT_DIR

# Define input/output
OUT_FILE="/rnaseq_project/new_annotation/mapping_independent/featurecounts_cerevisiae/featurecounts_results/gene_counts_cerevisiae.txt"
GTF="/rnaseq_project/new_annotation/genome/Saccharomyces_cerevisiae.R64-1-1.114.gtf"

# Setup working directories
cd /rnaseq_project/new_annotation/mapping_independent/featurecounts_cerevisiae

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
