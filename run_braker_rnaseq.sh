#!/bin/bash
#SBATCH --job-name="bakerprotrnaseq-rnaseq"
#SBATCH --account=st-cnislow-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x-%j.log
#SBATCH --error=logs/%x_%j.err

cd $SLURM_SUBMIT_DIR

module load zlib-ng/2.0.7
module load gcc/9.4.0
module load miniconda3/4.9.2
source activate braker_env

# Setup working directories
cd /scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/braker_run_rnaseq2

AUGUSTUS_CONFIG_PATH=/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/augustus_config

braker.pl --genome=/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/Schoenii-final-20250514-annotated.fa.masked --bam=/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/rnaseq_evidence/T0R1p_clean_Aligned.out.sorted.bam,/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/rnaseq_evidence/T0R2p_clean_Aligned.out.sorted.bam,/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/rnaseq_evidence/T0R3p_clean_Aligned.out.sorted.bam --softmasking --fungus --species=s_schoenii --GENEMARK_PATH=/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/gmes_linux_64 --AUGUSTUS_CONFIG_PATH=/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/augustus_config --workingdir=/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/braker_run_rnaseq2

