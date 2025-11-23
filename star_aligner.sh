#!/bin/bash
#SBATCH --job-name="st-T0R1L-star-sch-rnaseq"
#SBATCH --account=st-cnislow-1
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

# Setup working directories
cd /scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/mapping_independent/data/T0R1L

STAR --runThreadN 12 --genomeDir /scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/mapping_independent/STARgenome/STARgenome_schoneii_final_sjdb134 --readFilesIn clean_T0R1L_1.trimmed.fq.gz clean_T0R1L_2.trimmed.fq.gz --readFilesCommand zcat --outSAMtype BAM Unsorted --outFileNamePrefix T0R1L_clean_stringent_sch_  --outFilterMatchNminOverLread 0.9 --outFilterMismatchNoverReadLmax 0.03 --outFilterMultimapNmax 1

