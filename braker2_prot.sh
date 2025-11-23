#!/bin/bash

cd $SLURM_SUBMIT_DIR

# Setup working directories
cd /rnaseq_project/new_annotation/genome/braker2/braker_prot

AUGUSTUS_CONFIG_PATH=/rnaseq_project/new_annotation/genome/braker2/augustus_config

braker.pl --genome=/rnaseq_project/new_annotation/genome/braker2/Schoenii-final-20250514-annotated.fa.masked --prot_seq=/rnaseq_project/new_annotation/genome/braker2/combined_proteins.fa --softmasking --species=s_schoenii --GENEMARK_PATH=/rnaseq_project/new_annotation/genome/braker2/gmes_linux_64 --AUGUSTUS_CONFIG_PATH=/rnaseq_project/new_annotation/genome/braker2/augustus_config --fungus --workingdir=/rnaseq_project/new_annotation/genome/braker2/braker_prot

