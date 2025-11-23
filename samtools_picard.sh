#!/bin/bash
#SBATCH --job-name="TL-samtoolsandpicard-rnaseq"
#SBATCH --account=st-cnislow-1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.log
#SBATCH --error=logs/%x_%j.err

cd $SLURM_SUBMIT_DIR

module load zlib-ng/2.0.7
module load gcc/9.4.0
module load samtools
module load r/4.4.0
module load miniconda3/4.9.2
source activate variant_env

# Setup working directories
cd /scratch/st-cnislow-1/divya_scratch/rnaseq_project/new_annotation/mapping_independent/data

for i in T*L
do
    cd $i;
    samtools sort -o ${i}_clean_default_cer_Aligned.out.sorted.bam ${i}_clean_default_cer_Aligned.out.bam;
    samtools sort -o ${i}_clean_default_sch_Aligned.out.sorted.bam ${i}_clean_default_sch_Aligned.out.bam;
    samtools sort -o ${i}_clean_stringent_cer_Aligned.out.sorted.bam ${i}_clean_stringent_cer_Aligned.out.bam;
    samtools sort -o ${i}_clean_stringent_sch_Aligned.out.sorted.bam ${i}_clean_stringent_sch_Aligned.out.bam;
    samtools index ${i}_clean_default_sch_Aligned.out.sorted.bam;
    samtools index ${i}_clean_default_cer_Aligned.out.sorted.bam;
    samtools index ${i}_clean_stringent_sch_Aligned.out.sorted.bam;
    samtools index ${i}_clean_stringent_cer_Aligned.out.sorted.bam;
    picard MarkDuplicates I=${i}_clean_default_sch_Aligned.out.sorted.bam O=${i}_clean_default_sch_Aligned.out.sorted.dedup.bam M=${i}_default_sch_dup_metrics.txt VALIDATION_STRINGENCY=LENIENT CREATE_INDEX=true;
    picard MarkDuplicates I=${i}_clean_default_cer_Aligned.out.sorted.bam O=${i}_clean_default_cer_Aligned.out.sorted.dedup.bam M=${i}_default_cer_dup_metrics.txt VALIDATION_STRINGENCY=LENIENT CREATE_INDEX=true;
    picard MarkDuplicates I=${i}_clean_stringent_sch_Aligned.out.sorted.bam O=${i}_clean_stringent_sch_Aligned.out.sorted.dedup.bam M=${i}_stringent_sch_dup_metrics.txt VALIDATION_STRINGENCY=LENIENT CREATE_INDEX=true;
    picard MarkDuplicates I=${i}_clean_stringent_cer_Aligned.out.sorted.bam O=${i}_clean_stringent_cer_Aligned.out.sorted.dedup.bam M=${i}_stringent_cer_dup_metrics.txt VALIDATION_STRINGENCY=LENIENT CREATE_INDEX=true;
    picard CollectInsertSizeMetrics I=${i}_clean_default_sch_Aligned.out.sorted.dedup.bam O=${i}_default_sch_insert_size_metrics.txt H=${i}_default_sch_insert_size_histogram.pdf M=0.5;
    picard CollectInsertSizeMetrics I=${i}_clean_default_cer_Aligned.out.sorted.dedup.bam O=${i}_default_cer_insert_size_metrics.txt H=${i}_default_cer_insert_size_histogram.pdf M=0.5;
    picard CollectInsertSizeMetrics I=${i}_clean_stringent_sch_Aligned.out.sorted.dedup.bam O=${i}_stringent_sch_insert_size_metrics.txt H=${i}_stringent_sch_insert_size_histogram.pdf M=0.5;
    picard CollectInsertSizeMetrics I=${i}_clean_stringent_cer_Aligned.out.sorted.dedup.bam O=${i}_stringent_cer_insert_size_metrics.txt H=${i}_stringent_cer_insert_size_histogram.pdf M=0.5;
    cd ../;
done

