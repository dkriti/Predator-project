#!/bin/bash
#SBATCH --job-name="featurecounts_T_cer_stringent_rnaseq"
#SBATCH --account=st-cnislow-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x-%j.log
#SBATCH --error=logs/%x_%j.err

cd $SLURM_SUBMIT_DIR

module load zlib-ng/2.0.7
module load gcc/9.4.0
module load r/4.4.0
module load miniconda3/4.9.2
module load samtools
source activate variant_env

# Define input/output
OUT_FILE="/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/mapping_independent/featurecounts_T_cer_stringent/featurecounts_results_T_cer_stringent/gene_counts_T_cer_stringent.txt"
GTF="/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/Saccharomyces_cerevisiae.R64-1-1.114.gtf"

# Setup working directories
cd /scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/mapping_independent/featurecounts_T_cer_stringent

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
