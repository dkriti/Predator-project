import re
from collections import defaultdict

def parse_gtf_attributes(attr_str):
    attr_dict = {}
    for attr in re.findall(r'(\S+) "([^"]+)"', attr_str):
        attr_dict[attr[0]] = attr[1]
    return attr_dict

def gtf_stats(gtf_path):
    genes = {}
    transcripts = defaultdict(set)
    exons_per_gene = defaultdict(set)
    exons_per_transcript = defaultdict(int)
    alt_spliced_genes = set()
    genes_with_introns = set()
    single_exon_genes = set()
    annotated_genes = set()
    unannotated_genes = set()

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue
            feature = parts[2]
            attrs = parse_gtf_attributes(parts[8])
            gene_id = attrs.get('gene_id', None)
            transcript_id = attrs.get('transcript_id', None)

            if feature == 'gene' and gene_id:
                genes[gene_id] = attrs
                # functional annotation: must have at least one of gene_name, go_terms, or cog_category non-empty/non "-"
                gene_name = attrs.get('gene_name', None)
                go_terms = attrs.get('go_terms', "-")
                cog_category = attrs.get('cog_category', "-")
                if ((gene_name and gene_name != "-") or
                    (go_terms and go_terms != "-") or
                    (cog_category and cog_category != "-")):
                    annotated_genes.add(gene_id)
                else:
                    unannotated_genes.add(gene_id)

            if feature == 'transcript' and transcript_id and gene_id:
                transcripts[gene_id].add(transcript_id)
            if feature == 'exon' and gene_id and transcript_id:
                exons_per_gene[gene_id].add(parts[3] + "-" + parts[4])  # unique exon positions
                exons_per_transcript[transcript_id] += 1

    # Single/multi-exon/intronic genes
    for gene, exons in exons_per_gene.items():
        if len(exons) == 1:
            single_exon_genes.add(gene)
        else:
            genes_with_introns.add(gene)

    for gene, tids in transcripts.items():
        if len(tids) > 1:
            alt_spliced_genes.add(gene)

    print(f"Total genes: {len(genes)}")
    print(f"Total transcripts: {sum(len(tset) for tset in transcripts.values())}")
    print(f"Total exons: {sum(len(exonset) for exonset in exons_per_gene.values())}")
    print(f"Single-exon genes: {len(single_exon_genes)}")
    print(f"Genes with introns (multi-exon): {len(genes_with_introns)}")
    print(f"Genes with >1 transcript (alternative splicing): {len(alt_spliced_genes)}")
    print(f"Functionally annotated genes: {len(annotated_genes)} ({len(annotated_genes)/float(len(genes))*100:.1f}%)")
    print(f"Unannotated genes: {len(unannotated_genes)} ({len(unannotated_genes)/float(len(genes))*100:.1f}%)")

if __name__ == "__main__":
    gtf_path = "final_merged.gtf"  # change the file as required
    gtf_stats(gtf_path)
