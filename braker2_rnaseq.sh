#!/bin/bash

cd $SLURM_SUBMIT_DIR

# Setup working directories
cd /rnaseq_project/new_annotation/genome/braker2/braker_rnaseq2

AUGUSTUS_CONFIG_PATH=/scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/genome/braker2/augustus_config

braker.pl --genome=/rnaseq_project/new_annotation/genome/braker2/Schoenii-final-20250514-annotated.fa.masked --bam=/rnaseq_project/new_annotation/genome/braker2/rnaseq_evidence/sample1_Aligned.out.sorted.bam,/rnaseq_project/new_annotation/genome/braker2/rnaseq_evidence/sample2_Aligned.out.sorted.bam,/rnaseq_project/new_annotation/genome/braker2/rnaseq_evidence/sample3_Aligned.out.sorted.bam --softmasking --fungus --species=s_schoenii --GENEMARK_PATH=/rnaseq_project/new_annotation/genome/braker2/gmes_linux_64 --AUGUSTUS_CONFIG_PATH=/rnaseq_project/new_annotation/genome/braker2/augustus_config --workingdir=/rnaseq_project/new_annotation/genome/braker2/braker_rnaseq

