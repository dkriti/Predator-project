#!/bin/bash

cd $SLURM_SUBMIT_DIR

# Setup working directories
cd /rnaseq_project/new_annotation/mapping_independent/data

for i in *
do
    cd $i;
    samtools sort -o ${i}_Aligned.out.sorted.bam ${i}_Aligned.out.bam;
    samtools index ${i}_Aligned.out.sorted.bam;
    picard MarkDuplicates I=${i}_Aligned.out.sorted.bam O=${i}_Aligned.out.sorted.dedup.bam M=${i}_dup_metrics.txt VALIDATION_STRINGENCY=LENIENT CREATE_INDEX=true;
    picard CollectInsertSizeMetrics I=${i}_Aligned.out.sorted.dedup.bam O=${i}_insert_size_metrics.txt H=${i}_insert_size_histogram.pdf M=0.5;
    cd ../;
done

