#!/bin/bash

cd $SLURM_SUBMIT_DIR

# Setup working directories
cd /rnaseq_project/new_annotation/mapping_independent/data/

STAR --runThreadN 12 --genomeDir /rnaseq_project/new_annotation/mapping_independent/STARgenome/STARgenome_sjdb134 --readFilesIn filename_1.trimmed.fq.gz filename_2.trimmed.fq.gz --readFilesCommand zcat --outSAMtype BAM Unsorted --outFileNamePrefix filename_  --outFilterMatchNminOverLread 0.9 --outFilterMismatchNoverReadLmax 0.03 --outFilterMultimapNmax 1

