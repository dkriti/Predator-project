# ============================
# End-to-end DE + GO + GSEA pipeline (S. cerevisiae)
# + VolcanoCompare (strict) + 6 category heatmaps
# + GSEA diverging NES bars for comparison pairs
# L/D semantics: L = prey + live predator; D = prey + dead predator
# ============================

# ---- File and directory setup ----
species <- "S. cerevisiae"
counts_file <- "gene_counts_cerevisiae.txt"
gtf_file    <- "Saccharomyces_cerevisiae.R64-1-1.114.gtf"
output_dir  <- "csv_results_cerevisiae"
plots_dir   <- "pdf_plots_cerevisiae"
dir.create(output_dir, showWarnings = FALSE)
dir.create(plots_dir,  showWarnings = FALSE)

# ---- Thresholds (shown in plot subtitles) ----
core_padj_thr <- 0.05   # DE padj threshold for volcano/MA and ORA seed
core_lfc_thr  <- 1.0
gsea_fdr_thr  <- 0.05   # GSEA FDR threshold
net_padj_thr  <- 0.05   # Network filter
net_lfc_thr   <- 0.5
padj_thr_vc   <- 0.01   # VolcanoCompare stricter thresholds
lfc_thr_vc    <- 1.0
condition_note <- "L = prey + live predator; D = prey + dead predator"

# ---- Libraries ----
suppressPackageStartupMessages({
  library(DESeq2); library(EnhancedVolcano); library(pheatmap)
  library(AnnotationDbi); library(org.Sc.sgd.db)
  library(clusterProfiler); library(enrichplot)
  library(readr); library(dplyr); library(tibble); library(tidyr); library(stringr)
  library(ggplot2); library(ggrepel); library(gridExtra); library(ggplotify)
  library(igraph); library(ggraph)
})
if (!requireNamespace("ashr", quietly=TRUE)) install.packages("ashr")

# ---- Small helpers ----
ratio_num <- function(x){parts<-strsplit(x,"/",fixed=TRUE);sapply(parts,function(v) as.numeric(v[1])/as.numeric(v[2]))}
head_n <- function(df,n) df[seq_len(min(n,nrow(df))), , drop=FALSE]
is_standard_symbol <- function(x){
  x <- as.character(x)
  orf <- grepl("^Y[A-Z]{2}[0-9]{3}[CW](?:-[A-Z0-9]+)?$", x)
  sym <- grepl("^[A-Z0-9][A-Z0-9-]{1,8}$", x)
  orf | sym
}
choose_plot_label <- function(alias_symbol, orf_id, final_label){
  cand1 <- ifelse(!is.na(alias_symbol)&nzchar(alias_symbol), alias_symbol, NA_character_)
  cand1[!is_standard_symbol(cand1)] <- NA_character_
  cand2 <- ifelse(!is.na(orf_id)&nzchar(orf_id), orf_id, NA_character_)
  cand2[!is_standard_symbol(cand2)] <- NA_character_
  lab_no_cer <- gsub("^cer_","", ifelse(!is.na(final_label), final_label, ""))
  lab_no_cer[!is_standard_symbol(lab_no_cer)] <- NA_character_
  ifelse(!is.na(cand1), cand1, ifelse(!is.na(cand2), cand2, ifelse(!is.na(lab_no_cer), lab_no_cer, NA_character_)))
}
wrap_comp_label <- function(x){
  sapply(x, function(s){
    parts <- strsplit(s,"__minus__",fixed=TRUE)[[1]]
    if(length(parts)!=2) return(s)
    fmt <- function(p) stringr::str_replace_all(p, "_vs_", "\nvs\n")
    paste0(fmt(parts[1]), "\n–\n", fmt(parts[2]))
  })
}
wrap_text <- function(x, width=42){
  vapply(x, function(s) paste(strwrap(s, width=width), collapse="\n"), character(1))
}

# ---- Load & preprocess ----
counts <- read.table(counts_file, header=TRUE, row.names=1, comment.char="#")
counts <- counts[, 6:ncol(counts)]
colnames(counts) <- gsub("_clean_stringent_cer_Aligned.out.sorted.dedup.bam", "", colnames(counts))

sample_names <- colnames(counts)
timepoint <- sub("^(T[0-9]+).*$", "\\1", sample_names)
treatment <- sub("^T[0-9]+R[0-9]+([A-Za-z])$", "\\1", sample_names)  # L/D as per biology
group <- factor(paste(timepoint, treatment, sep="_"),
                levels=c("T0_L","T6_L","T6_D","T24_L","T24_D"))
replicate <- sub("^T[0-9]+(R[0-9]+).*$", "\\1", sample_names)
samples <- data.frame(
  row.names = sample_names,
  timepoint = factor(timepoint, levels=c("T0","T6","T24")),
  replicate = factor(replicate),
  treatment = factor(treatment, levels=c("L","D")),
  group     = group
)

keep <- rowSums(counts >= 10) >= 2
counts <- counts[keep, ]
zero_samples <- colSums(counts)==0
if(any(zero_samples)){ counts <- counts[,!zero_samples,drop=FALSE]; samples <- samples[!zero_samples,,drop=FALSE] }

dds <- DESeqDataSetFromMatrix(countData=counts, colData=samples, design=~group)
dds <- DESeq(dds, fitType="local", sfType="poscounts")
vsd <- vst(dds, blind=FALSE)

norm_counts <- counts(dds, normalized=TRUE)
write.csv(as.data.frame(norm_counts), file.path(output_dir,"normalized_counts.csv"), row.names=TRUE)

# ---- Annotation (ORF ↔ COMMON; no BLAST needed) ----
gtf_raw <- readr::read_tsv(gtf_file, comment="#", col_names=FALSE, show_col_types=FALSE)
gtf_df <- gtf_raw %>% filter(X3=="gene") %>%
  mutate(raw_id=str_extract(X9,'(?<=gene_id \")[^\"]+'),
         gtf_name=str_extract(X9,'(?<=gene_name \")[^\"]+')) %>% select(raw_id, gtf_name) %>% distinct()

gene_df <- tibble(gene_id=rownames(dds))
annot0 <- gene_df %>%
  mutate(orf_id=sub("^cer_","", gene_id)) %>%  # gene_id are your rownames; strip cer_ if present
  left_join(gtf_df, by=c("gene_id"="raw_id"))

# COMMON = standard yeast gene symbol; ENTREZID available for GO/GSEA
sgd_map <- AnnotationDbi::select(org.Sc.sgd.db,
                                 keys=unique(annot0$orf_id),
                                 keytype="ORF",
                                 columns=c("COMMON","ALIAS","ENTREZID")) %>%
  as_tibble() %>% distinct(ORF, .keep_all = TRUE)

annot_df <- annot0 %>%
  mutate(alias_symbol = sgd_map$COMMON[match(orf_id, sgd_map$ORF)],
         base_label   = coalesce(gtf_name, alias_symbol, gene_id),
         final_label  = if_else(str_starts(base_label,"cer_"), base_label, paste0("cer_", base_label)))
annot_df <- as.data.frame(annot_df); rownames(annot_df) <- annot_df$gene_id

# ---- Global ORF → ENTREZ map (once) + safe join helper ----
entrez_map_all <- AnnotationDbi::select(org.Sc.sgd.db,
                                        keys   = unique(na.omit(annot_df$orf_id)),
                                        keytype= "ORF",
                                        columns= "ENTREZID") %>%
  as_tibble() %>%
  distinct(ORF, ENTREZID)

join_entrez <- function(df) {
  df %>%
    left_join(entrez_map_all, by = c("orf_id" = "ORF")) %>%
    mutate(ENTREZID = as.character(ENTREZID))
}

# ---- Contrasts ----
contrast_list <- list(
  T6_L_vs_T0_L   = c("group","T6_L","T0_L"),
  T6_D_vs_T0_L   = c("group","T6_D","T0_L"),
  T24_L_vs_T0_L  = c("group","T24_L","T0_L"),
  T24_D_vs_T0_L  = c("group","T24_D","T0_L"),
  T6_L_vs_T6_D   = c("group","T6_L","T6_D"),
  T24_L_vs_T24_D = c("group","T24_L","T24_D"),
  T24_D_vs_T6_D  = c("group","T24_D","T6_D"),
  T24_L_vs_T6_L  = c("group","T24_L","T6_L")
)

get_res <- function(contrast_vec, lfc_limit=10){
  res_shrink <- lfcShrink(dds, contrast=contrast_vec, type="ashr")
  as.data.frame(res_shrink) %>%
    rownames_to_column("gene_id") %>%
    left_join(annot_df, by="gene_id") %>%
    mutate(plot_label=choose_plot_label(alias_symbol, orf_id, final_label)) %>%
    filter(abs(log2FoldChange) <= lfc_limit)
}

# ---- ORA (effect-size only) ----
plot_go_ora_views <- function(go_table, contrast_name, topn=20){
  if (is.null(go_table) || !nrow(go_table)) return(invisible(NULL))
  df <- go_table %>%
    mutate(GeneRatio_num=ratio_num(GeneRatio), BgRatio_num=ratio_num(BgRatio),
           FoldEnrichment=GeneRatio_num/BgRatio_num, negLog10Padj=-log10(p.adjust+1e-300))
  top_eff <- df[order(-df$FoldEnrichment, df$p.adjust), , drop=FALSE] %>% head_n(topn)
  p_eff <- ggplot(top_eff, aes(x=FoldEnrichment, y=reorder(Description, FoldEnrichment),
                               size=Count, color=negLog10Padj)) +
    geom_point() +
    scale_size_continuous(name="Gene count") +
    scale_color_gradient(name=expression(-log[10]~"FDR"), low="#bdd7e7", high="#08519c") +
    labs(title=paste0("GO ORA (BP) — ", species, ": ", contrast_name),
         subtitle=sprintf("Effect size view. ORA seeded with DE genes (padj<%.2g). %s",
                          core_padj_thr, condition_note),
         x="Fold enrichment (GeneRatio/BgRatio)", y=NULL) +
    theme_bw(base_size=11)
  pdf(file.path(plots_dir, sprintf("GO_ORA_effectsize_%s.pdf", contrast_name)), width=8, height=6)
  print(p_eff); dev.off()
}

run_go <- function(res_df){
  sig_orfs <- res_df %>% filter(padj < core_padj_thr, !is.na(orf_id)) %>% pull(orf_id) %>% unique()
  if (!length(sig_orfs)) return(NULL)

  em <- entrez_map_all %>% filter(ORF %in% sig_orfs)
  entrez <- na.omit(em$ENTREZID)
  if (!length(entrez)) return(NULL)

  go_result <- enrichGO(
    gene          = entrez,
    OrgDb         = org.Sc.sgd.db,
    keyType       = "ENTREZID",
    ont           = "BP",
    pvalueCutoff  = core_padj_thr
  )
  if (is.null(go_result) || nrow(as.data.frame(go_result))==0) return(NULL)

  # map back ENTREZID → ORF, then ORF → COMMON symbol
  orf_for_entrez <- setNames(em$ORF, em$ENTREZID)
  symbol_for_orf <- setNames(sgd_map$COMMON, sgd_map$ORF)

  go_df <- as.data.frame(go_result)
  go_df$geneNames <- sapply(strsplit(go_df$geneID,"/"), function(ids){
    orfs <- orf_for_entrez[ids]; syms <- symbol_for_orf[orfs]
    paste(ifelse(!is.na(syms)&nzchar(syms), syms, orfs), collapse="/")
  })
  list(result=go_result, table=go_df, entrez_map=em)
}

run_go_clusters <- function(go_result, contrast_name, entrez_map){
  if (is.null(go_result)) return(NULL)
  go_df <- as.data.frame(go_result); if (nrow(go_df)<2) return(NULL)
  go_simplified <- tryCatch({ clusterProfiler::simplify(go_result, cutoff=0.7, by="p.adjust", select_fun=min) },
                            error=function(e) go_result)
  go_df2 <- as.data.frame(go_simplified)
  orf_for_entrez <- setNames(entrez_map$ORF, entrez_map$ENTREZID)
  symbol_for_orf <- setNames(sgd_map$COMMON, sgd_map$ORF)
  go_df2$geneNames <- sapply(strsplit(go_df2$geneID,"/"), function(ids){
    orfs <- orf_for_entrez[ids]; syms <- symbol_for_orf[orfs]
    paste(ifelse(!is.na(syms)&nzchar(syms), syms, orfs), collapse="/")
  })
  write.csv(go_df2, file.path(output_dir, sprintf("GO_clusters_%s.csv", contrast_name)), row.names=FALSE)

  go_sim <- tryCatch({ enrichplot::pairwise_termsim(go_simplified) }, error=function(e) NULL)
  pdf(file.path(plots_dir, sprintf("GO_clusters_%s.pdf", contrast_name)), width=8, height=6)
  if (!is.null(go_sim)) print(emapplot(go_sim, showCategory=min(20, nrow(go_df2))) +
                                ggtitle(sprintf("GO Clusters (BP) — %s: %s", species, contrast_name)) +
                                labs(subtitle=sprintf("Seed: DE genes (padj<%.2g). %s",
                                                      core_padj_thr, condition_note)))
  else print(ggplot() + ggtitle('emapplot error (no similarity)'))
  dev.off()

  df <- go_df2 %>% mutate(GeneRatio_num=ratio_num(GeneRatio), BgRatio_num=ratio_num(BgRatio),
                          FoldEnrichment=GeneRatio_num/BgRatio_num, negLog10Padj=-log10(p.adjust+1e-300))
  df_top <- df[order(-df$FoldEnrichment, df$p.adjust), , drop=FALSE] %>% head_n(20)
  p_eff <- ggplot(df_top, aes(x=FoldEnrichment, y=reorder(Description, FoldEnrichment),
                              size=Count, color=negLog10Padj)) +
    geom_point() + scale_size_continuous(name="Gene count") +
    scale_color_gradient(name=expression(-log[10]~"FDR"), low="#bdd7e7", high="#08519c") +
    labs(title=sprintf("GO Clusters (BP) — %s: %s", species, contrast_name),
         subtitle=sprintf("Effect size view. Seed: DE genes (padj<%.2g). %s",
                          core_padj_thr, condition_note),
         x="Fold enrichment (GeneRatio/BgRatio)", y=NULL) +
    theme_bw(base_size=11)
  pdf(file.path(plots_dir, sprintf("GO_clusters_dot_effectsize_%s.pdf", contrast_name)), width=8, height=5)
  print(p_eff); dev.off()
}

# ---- Bipartite GO network (significant genes only; Up=firebrick2, Down=royalblue) ----
plot_go_network <- function(cluster_go_gene_csv, volcano_file, contrast_name,
                            padj_thresh=net_padj_thr, lfc_thresh=net_lfc_thr, top_n_clusters=20){
  if (!file.exists(cluster_go_gene_csv)) return(invisible(NULL))
  dat <- read.csv(cluster_go_gene_csv, stringsAsFactors=FALSE)
  if (!nrow(dat)) return(invisible(NULL))
  top_clusters <- unique(dat$GO_Cluster)[1:min(top_n_clusters, length(unique(dat$GO_Cluster)))]
  dat <- dat[dat$GO_Cluster %in% top_clusters, , drop=FALSE]

  edges_raw <- dat %>% tidyr::separate_rows(Top_DE_Genes, sep="; ") %>%
    mutate(gene_symbol=gsub(" \\(log2FC.*","", Top_DE_Genes)) %>%
    select(GO_Cluster, gene_symbol) %>% distinct()

  vol <- read.csv(volcano_file, stringsAsFactors=FALSE) %>%
    mutate(reg = ifelse(padj<padj_thresh & abs(log2FoldChange)>=lfc_thresh,
                        ifelse(log2FoldChange>=0,"Up","Down"), NA_character_))
  vol_sig <- vol %>% filter(!is.na(reg)) %>% select(alias_symbol, reg) %>% rename(gene_symbol=alias_symbol)
  edges <- edges_raw %>% inner_join(vol_sig, by="gene_symbol")
  if (!nrow(edges)) { message("No significant edges for network."); return(invisible(NULL)) }

  terms <- unique(edges$GO_Cluster); genes <- unique(edges$gene_symbol)
  verts <- bind_rows(
    data.frame(name=terms, is_term=TRUE,  reg=NA_character_, stringsAsFactors=FALSE),
    data.frame(name=genes, is_term=FALSE, reg=vol_sig$reg[match(genes, vol_sig$gene_symbol)], stringsAsFactors=FALSE)
  )
  g <- igraph::graph_from_data_frame(edges[,c("GO_Cluster","gene_symbol")], vertices=verts, directed=FALSE)

  pdf(file.path(plots_dir, sprintf("GO_Network_%s.pdf", contrast_name)), width=10, height=7)
  print(
    ggraph(g, layout="fr") +
      geom_edge_link(alpha=0.25, colour="grey60") +
      geom_node_point(aes(shape=ifelse(is_term,"Term","Gene"), color=ifelse(is_term, NA, reg)),
                      size=ifelse(igraph::V(g)$is_term, 3, 2.5), show.legend=TRUE) +
      geom_node_text(aes(label=name), repel=TRUE, size=2.6, max.overlaps=100) +
      scale_shape_manual(name="Node type", values=c("Term"=15,"Gene"=16)) +
      scale_color_manual(name="Regulation", values=c("Down"="royalblue","Up"="firebrick2"),
                         breaks=c("Down","Up"), na.value="black", drop=TRUE) +  # no NA in legend
      guides(color=guide_legend(order=1),
             shape=guide_legend(order=2, override.aes=list(color="black", size=3))) +
      ggtitle(paste("GO Cluster–Gene Network (BP) —", species, ":", contrast_name)) +
      labs(subtitle=sprintf("Nodes: Terms & significant genes. Filter: padj<%.2g & |log2FC|≥%.2f. %s",
                            padj_thresh, lfc_thresh, condition_note)) +
      theme_void() + theme(legend.position="right")
  )
  dev.off()
}

# ---- Term–term projection (by shared sig DE genes) ----
plot_go_term_projection <- function(cluster_go_gene_csv, contrast_name,
                                    top_n_clusters=20, min_shared=2){
  if (!file.exists(cluster_go_gene_csv)) return(invisible(NULL))
  dat <- read.csv(cluster_go_gene_csv, stringsAsFactors=FALSE)
  if (!nrow(dat)) return(invisible(NULL))
  top_clusters <- unique(dat$GO_Cluster)[1:min(top_n_clusters, length(unique(dat$GO_Cluster)))]
  dat <- dat[dat$GO_Cluster %in% top_clusters, , drop=FALSE]

  edges <- dat %>% tidyr::separate_rows(Top_DE_Genes, sep="; ") %>%
    mutate(gene_symbol=gsub(" \\(log2FC.*","", Top_DE_Genes)) %>%
    select(GO_Cluster, gene_symbol) %>% distinct()
  if (!nrow(edges)) return(invisible(NULL))

  terms <- unique(edges$GO_Cluster); genes <- unique(edges$gene_symbol)
  verts <- bind_rows(data.frame(name=terms,type=TRUE), data.frame(name=genes,type=FALSE))
  g <- igraph::graph_from_data_frame(edges, vertices=verts, directed=FALSE)
  proj <- igraph::bipartite_projection(g)
  term_g <- if (igraph::vcount(proj$proj1)==length(terms)) proj$proj1 else proj$proj2
  term_g <- igraph::delete_edges(term_g, igraph::E(term_g)[weight < min_shared])
  if (igraph::ecount(term_g)==0) { message("No term–term edges after filtering."); return(invisible(NULL)) }

  pdf(file.path(plots_dir, sprintf("GO_TermProjection_%s.pdf", contrast_name)), width=9, height=6)
  print(
    ggraph(term_g, layout="fr") +
      geom_edge_link(aes(width=weight), alpha=0.4, colour="grey50") +
      scale_edge_width(range=c(0.3,2.5), name="Shared DE genes") +
      geom_node_point(size=4, colour="#2b8cbe") +
      geom_node_text(aes(label=name), repel=TRUE, size=3) +
      ggtitle(sprintf("Term–term projection (BP) — %s: %s", species, contrast_name)) +
      labs(subtitle=sprintf("Edges = shared significant DE genes in cluster CSV; min shared = %d. %s",
                            min_shared, condition_note)) +
      theme_void()
  )
  dev.off()
}

# ---- Cluster → GO term → Top Genes (significant DE genes only) ----
generate_cluster_go_gene_table <- function(contrast_name, volcano_file, go_file, cluster_file, output_file,
                                           top_n_clusters=20, top_n_genes=20, padj_thresh=net_padj_thr, lfc_thresh=net_lfc_thr){
  if (!file.exists(volcano_file) | !file.exists(go_file) | !file.exists(cluster_file)) {
    warning(sprintf("Files missing for %s. Skipping.", contrast_name)); return(NULL)
  }
  volcano_df <- read.csv(volcano_file, stringsAsFactors=FALSE) %>%
    mutate(gene_symbol = ifelse(!is.na(alias_symbol)&nzchar(alias_symbol),
                                alias_symbol, str_remove(final_label,"^cer_"))) %>%
    filter(!is.na(padj)) %>%
    filter(padj < padj_thresh & abs(log2FoldChange) >= lfc_thresh)

  go_df <- read.csv(go_file, stringsAsFactors=FALSE)
  cluster_df <- read.csv(cluster_file, stringsAsFactors=FALSE)
  if (!nrow(go_df) || !nrow(cluster_df)) { warning(sprintf("Empty GO or cluster table for %s.", contrast_name)); return(NULL) }

  top_clusters <- cluster_df %>% arrange(p.adjust) %>% head_n(top_n_clusters)

  records <- list()
  for(i in seq_len(nrow(top_clusters))){
    cluster_name <- top_clusters$Description[i]; cluster_id <- top_clusters$ID[i]
    go_terms <- go_df %>% filter(ID==cluster_id); if(!nrow(go_terms)) next
    for(j in seq_len(nrow(go_terms))){
      genes <- unique(strsplit(go_terms$geneNames[j],"/",fixed=TRUE)[[1]])
      genes <- genes[!is.na(genes) & nzchar(genes)]
      matched <- volcano_df %>% filter(gene_symbol %in% genes) %>% arrange(padj)
      if (!nrow(matched)) next
      matched <- head_n(matched, top_n_genes)
      records[[length(records)+1]] <- data.frame(
        GO_Cluster = cluster_name,
        GO_Term    = go_terms$Description[j],
        Top_DE_Genes = paste0(matched$gene_symbol, " (log2FC=", round(matched$log2FoldChange,2),
                              ", padj=", signif(matched$padj,3), ")", collapse="; "),
        stringsAsFactors=FALSE
      )
    }
  }
  if (length(records)>0){
    result_df <- bind_rows(records)
    write.csv(result_df, file=output_file, row.names=FALSE)
    message(sprintf("Cluster-GO-Gene mapping written: %s", output_file))
    return(result_df)
  } else { warning(sprintf("No matches found for %s", contrast_name)); return(NULL) }
}

# ============================
# Main loop over contrasts
# ============================
for(name in names(contrast_list)){
  contrast_vec <- contrast_list[[name]]
  res <- get_res(contrast_vec, lfc_limit=10) %>% filter(!is.na(log2FoldChange)&!is.na(padj))

  # Volcano
  p_vol <- EnhancedVolcano(
    res, lab=res$plot_label, x='log2FoldChange', y='padj',
    title=paste("Volcano —", species, ":", name),
    subtitle=sprintf("Thresholds: padj<%.2g, |log2FC|≥%.2f. %s",
                     core_padj_thr, core_lfc_thr, condition_note),
    pCutoff=core_padj_thr, FCcutoff=core_lfc_thr, cutoffLineType='dashed',
    cutoffLineCol='black', cutoffLineWidth=0.4, labSize=3,
    pointSize=1.0, col=c("grey30","grey30","royalblue","firebrick2"), colAlpha=0.9
  )
  pdf(file.path(plots_dir, sprintf("Volcano_%s.pdf", name))); print(p_vol); dev.off()
  write.csv(res %>% select(gene_id, orf_id, alias_symbol, log2FoldChange, pvalue, padj, final_label, plot_label),
            file.path(output_dir, sprintf("Volcano_%s.csv", name)), row.names=FALSE)

  # Heatmap (top 50 DE)
  top50 <- head(res %>% filter(padj<core_padj_thr) %>% arrange(padj), 50)
  write.csv(top50, file.path(output_dir, sprintf("Heatmap_top50_%s.csv", name)), row.names=FALSE)
  cols <- colnames(vsd)[colData(vsd)$group %in% contrast_vec[2:3]]
  if (nrow(top50)>1 && length(cols)>=2){
    mat <- assay(vsd)[top50$gene_id, cols, drop=FALSE]
    rownames(mat) <- make.unique(ifelse(!is.na(top50$plot_label)&nzchar(top50$plot_label),
                                        top50$plot_label, gsub("^cer_","", top50$final_label)))
    ht <- pheatmap(mat - rowMeans(mat),
                   annotation_col=as.data.frame(colData(vsd)[cols,c('timepoint','treatment')]),
                   main=paste("Top 50 DE —", species, ":", name),
                   silent=TRUE, annotation_legend=TRUE)
    pdf(file.path(plots_dir, sprintf("Heatmap_%s.pdf", name)), width=6, height=8)
    grid::grid.newpage(); grid::grid.draw(ht$gtable); dev.off()
  }

  # MA
  pdf(file.path(plots_dir, sprintf("MA_%s.pdf", name)))
  res_ma <- res %>% mutate(baseMean=ifelse(is.na(baseMean),0,baseMean)) %>% filter(baseMean>0) %>%
    mutate(DE=abs(log2FoldChange)>=core_lfc_thr & padj<core_padj_thr, col=ifelse(DE,"#ff7f0e","grey80"))
  with(res_ma, plot(baseMean,log2FoldChange,log='x',col=col,pch=20,cex=0.6,
                    main=paste("MA —", species, ":", name)))
  mtext(sprintf("DE highlight: padj<%.2g & |log2FC|≥%.2f. %s", core_padj_thr, core_lfc_thr, condition_note),
        side=3, line=0.5, cex=0.8)
  abline(h=0); legend("topright", legend=c("DE"), col="#ff7f0e", pch=20, pt.cex=0.8, bty="n", title="Gene")
  dev.off()
  write.csv(res_ma, file=file.path(output_dir, sprintf("MA_%s.csv", name)), row.names=FALSE)

  # GO enrichment
  go_out <- run_go(res)
  pdf(file.path(plots_dir, sprintf("GO_%s.pdf", name)), width=8, height=5)
  if (!is.null(go_out)) print(dotplot(go_out$result, showCategory=10) +
                                ggtitle(paste0("GO Enrichment (BP) — ", species, ": ", name)) +
                                labs(subtitle=sprintf("ORA seeded with DE genes (padj<%.2g). %s",
                                                      core_padj_thr, condition_note)))
  else print(ggplot() + ggtitle(paste("GO Enrichment (BP) —", species, ":", name)) +
               labs(subtitle="No significant GO terms"))
  dev.off()

  if (!is.null(go_out)){
    write.csv(go_out$table, file.path(output_dir, sprintf("GO_%s_geneNames.csv", name)), row.names=FALSE)
    plot_go_ora_views(go_out$table, name, topn=20)
    run_go_clusters(go_out$result, name, go_out$entrez_map)
  }

  # GSEA (BP) — x = NES, color = -log10(FDR)
  geneList_df <- res %>%
    filter(!is.na(orf_id), !is.na(log2FoldChange)) %>%
    join_entrez() %>%
    filter(!is.na(ENTREZID)) %>%
    group_by(ENTREZID) %>% summarise(score=mean(log2FoldChange), .groups="drop") %>%
    arrange(desc(score))
  geneList <- sort(setNames(geneList_df$score, geneList_df$ENTREZID), decreasing=TRUE)

  gsea_cer <- tryCatch(gseGO(geneList=geneList, OrgDb=org.Sc.sgd.db, keyType="ENTREZID",
                             ont="BP", minGSSize=10, pvalueCutoff=gsea_fdr_thr, verbose=FALSE),
                       error=function(e){ message("gseGO error (", name, "): ", e$message); NULL })
  pdf(file.path(plots_dir, sprintf("GSEA_%s.pdf", name)), width=8, height=6)
  if (!is.null(gsea_cer) && nrow(as.data.frame(gsea_cer))>0){
    gdf <- as.data.frame(gsea_cer) %>% mutate(negLog10Padj=-log10(p.adjust+1e-300))
    topN <- head(gdf[order(gdf$p.adjust),], min(20, nrow(gdf)))
    p_gsea <- ggplot(topN, aes(x=NES, y=reorder(Description, NES), size=setSize, color=negLog10Padj)) +
      geom_vline(xintercept=0, linetype="dashed") + geom_point() +
      scale_size_continuous(name="Set size") +
      scale_color_gradient(name=expression(-log[10]~"FDR"), low="#d9f0a3", high="#1a9850") +
      labs(title=paste0("GSEA (BP) — ", species, ": ", name),
           subtitle=sprintf("Points shown if FDR<%.2g. Color = -log10(FDR). %s", gsea_fdr_thr, condition_note),
           x="Normalized Enrichment Score (NES)", y=NULL) +
      theme_bw(base_size=11)
    print(p_gsea)
  } else { print(ggplot() + ggtitle(paste("GSEA (BP) —", species, ":", name)) + labs(subtitle="No significant GSEA terms")) }
  dev.off()
  write.csv(if (is.null(gsea_cer)) data.frame() else as.data.frame(gsea_cer),
            file.path(output_dir, sprintf("GSEA_%s.csv", name)), row.names=FALSE)

  # Cluster->GO->Top genes + network/term-projection
  if (!is.null(go_out)){
    volcano_file <- file.path(output_dir, sprintf("Volcano_%s.csv", name))
    go_file      <- file.path(output_dir, sprintf("GO_%s_geneNames.csv", name))
    cluster_file <- file.path(output_dir, sprintf("GO_clusters_%s.csv", name))
    output_file  <- file.path(output_dir, sprintf("Cluster_GO_Genes_%s.csv", name))
    generate_cluster_go_gene_table(name, volcano_file, go_file, cluster_file, output_file,
                                   top_n_clusters=20, top_n_genes=20,
                                   padj_thresh=net_padj_thr, lfc_thresh=net_lfc_thr)
    if (file.exists(output_file)){
      plot_go_network(output_file, volcano_file, name,
                      padj_thresh=net_padj_thr, lfc_thresh=net_lfc_thr, top_n_clusters=20)
      plot_go_term_projection(output_file, name)
    }
  }
}

message("Core analysis done. PDFs in: ", normalizePath(plots_dir))
message("Tables in: ", normalizePath(output_dir))

# ============================
# Volcano comparisons (B minus A) — strict thresholds
# ============================

build_gene_key <- function(df){
  sym <- df$alias_symbol
  lab <- gsub("^cer_","", df$final_label)
  ifelse(!is.na(sym)&nzchar(sym), sym, ifelse(!is.na(lab)&nzchar(lab), lab, df$gene_id))
}

compare_volcano <- function(contrast_A, contrast_B, padj_thr=padj_thr_vc, lfc_thr=lfc_thr_vc, label_top=15){
  fA <- file.path(output_dir, sprintf("Volcano_%s.csv", contrast_A))
  fB <- file.path(output_dir, sprintf("Volcano_%s.csv", contrast_B))
  if (!file.exists(fA) || !file.exists(fB)){ message("Skipping pair (missing CSV): ", contrast_B, " minus ", contrast_A); return(invisible(NULL)) }
  A <- read.csv(fA, stringsAsFactors=FALSE); B <- read.csv(fB, stringsAsFactors=FALSE)
  if (!nrow(A) && !nrow(B)) return(invisible(NULL))

  A$gene_key <- build_gene_key(A); B$gene_key <- build_gene_key(B)
  A2 <- A[,c("gene_key","log2FoldChange","padj")]; colnames(A2) <- c("gene_key","LFC_A","padj_A")
  B2 <- B[,c("gene_key","log2FoldChange","padj")]; colnames(B2) <- c("gene_key","LFC_B","padj_B")
  M <- full_join(A2,B2,by="gene_key") %>%
    mutate(
      deA=!is.na(padj_A)&padj_A<padj_thr&is.finite(LFC_A)&abs(LFC_A)>=lfc_thr,
      deB=!is.na(padj_B)&padj_B<padj_thr&is.finite(LFC_B)&abs(LFC_B)>=lfc_thr,
      upA=deA & LFC_A>=lfc_thr, downA=deA & LFC_A<=-lfc_thr,
      upB=deB & LFC_B>=lfc_thr, downB=deB & LFC_B<=-lfc_thr
    ) %>%
    mutate(
      status = case_when(
        upB & !deA ~ "New Up",      downB & !deA ~ "New Down",
        upA & !deB ~ "Lost Up",     downA & !deB ~ "Lost Down",
        upB & downA ~ "Switch Up (from Down)",
        downB & upA ~ "Switch Down (from Up)",
        upA & upB   ~ "Common Up",  downA & downB ~ "Common Down",
        TRUE ~ "NS"
      ),
      nlog10_p_B=ifelse(is.na(padj_B), NA_real_, -log10(padj_B+1e-300))
    )

  out_csv <- file.path(output_dir, sprintf("VolcanoCompare_%s__minus__%s.csv", contrast_B, contrast_A))
  write.csv(M, out_csv, row.names=FALSE)

  cols_named <- c("New Up"="firebrick2","New Down"="royalblue",
                  "Lost Up"="#fcae91","Lost Down"="#9ecae1",
                  "Switch Up (from Down)"="purple","Switch Down (from Up)"="seagreen",
                  "Common Up"="grey50","Common Down"="grey50","NS"="grey82")

  P <- M %>% filter(is.finite(LFC_B), is.finite(nlog10_p_B))
  if (!nrow(P)) { message("No finite B values for plotting: ", contrast_B, " minus ", contrast_A); return(invisible(out_csv)) }

  P_plot <- P %>% filter(is_standard_symbol(gene_key))
  lab_df <- P_plot %>% filter(status %in% c("New Up","New Down")) %>% arrange(padj_B) %>% slice_head(n=label_top)

  pdf(file.path(plots_dir, sprintf("VolcanoCompare_%s__minus__%s.pdf", contrast_B, contrast_A)), width=7.4, height=6.1)
  p <- ggplot(P_plot, aes(x=LFC_B, y=nlog10_p_B, color=status)) +
    geom_vline(xintercept=c(-lfc_thr, lfc_thr), linetype="dashed", alpha=0.6) +
    geom_hline(yintercept=-log10(padj_thr), linetype="dashed", alpha=0.6) +
    geom_point(alpha=0.9, size=1.6) +
    ggrepel::geom_text_repel(data=lab_df, aes(label=gene_key), size=2.6, box.padding=0.25, max.overlaps=25) +
    scale_color_manual(values=cols_named, drop=FALSE) +
    labs(
      title = paste0("Volcano — ", species, ": ", contrast_B, " minus ", contrast_A),
      subtitle = sprintf("Thresholds: padj<%.2g, |log2FC|≥%.2f (new/lost/switch vs prior contrast). %s",
                         padj_thr, lfc_thr, condition_note),
      x = paste0("log2FC in ", contrast_B), y = paste0("-log10(FDR) in ", contrast_B),
      color = NULL,
      caption = sprintf("B = %s, A = %s. Labels: top 'New' by FDR in B.", contrast_B, contrast_A)
    ) + theme_bw(base_size=10) + theme(legend.position="right")
  print(p); dev.off()

  invisible(out_csv)
}

# ---- Comparison pairs (B minus A) ----
volcano_pairs <- list(
  c("T6_L_vs_T0_L","T24_L_vs_T0_L"),   # timecourse (L)
  c("T6_D_vs_T0_L","T24_D_vs_T0_L"),   # timecourse (D vs T0_L baseline)
  c("T6_L_vs_T6_D","T24_L_vs_T24_D"),  # L vs D at T6 vs at T24
  c("T24_D_vs_T6_D","T24_L_vs_T6_L")   # late vs early within condition (D, L)
)

message("Running volcano comparisons (strict) for ", length(volcano_pairs), " pairs...")
compare_csv_paths <- character(0)
for (pr in volcano_pairs){
  A <- pr[1]; B <- pr[2]
  out <- compare_volcano(contrast_A=A, contrast_B=B, padj_thr=padj_thr_vc, lfc_thr=lfc_thr_vc, label_top=15)
  if (!is.null(out)) compare_csv_paths <- c(compare_csv_paths, out)
}
message("Volcano comparison PDFs in: ", normalizePath(plots_dir))
message("Comparison tables in: ", normalizePath(output_dir))

# ============================
# 6 ALL-VOLCANO COMPARISON HEATMAPS (categories), colored by degree of regulation
# ============================
make_category_heatmaps_all_comparisons <- function(compare_csv_paths, max_genes_per_heatmap=Inf){
  if (length(compare_csv_paths)==0){ message("No compare CSVs to plot."); return(invisible(NULL)) }
  categories <- list(
    "New Up"                = list(fill_from="B", sign="up",   subtitle="fill = log2FC in later contrast (top of ‘–’)"),
    "New Down"              = list(fill_from="B", sign="down", subtitle="fill = |log2FC| in later contrast (top of ‘–’)"),
    "Switch Up (from Down)" = list(fill_from="B", sign="up",   subtitle="fill = log2FC in later contrast (top of ‘–’)"),
    "Switch Down (from Up)" = list(fill_from="B", sign="down", subtitle="fill = |log2FC| in later contrast (top of ‘–’)"),
    "Lost Up"               = list(fill_from="A", sign="up",   subtitle="fill = log2FC in prior contrast (bottom of ‘–’)"),
    "Lost Down"             = list(fill_from="A", sign="down", subtitle="fill = |log2FC| in prior contrast (bottom of ‘–’)")
  )
  pal_up_low <- "#fee5d9"; pal_up_high <- "firebrick2"
  pal_dn_low <- "#deebf7"; pal_dn_high <- "royalblue"

  build_long_for_category <- function(cat_name, info){
    all_rows <- list()
    for (fp in compare_csv_paths){
      comp <- gsub("^VolcanoCompare_(.*)\\.csv$","\\1", basename(fp))
      D <- read.csv(fp, stringsAsFactors=FALSE)
      if (!nrow(D)) next
      D <- D %>% filter(status==cat_name) %>% filter(is_standard_symbol(gene_key))
      if (!nrow(D)) next
      if (info$fill_from=="B"){
        val <- if (info$sign=="up") pmax(0, D$LFC_B) else pmax(0, -D$LFC_B)
      } else {
        val <- if (info$sign=="up") pmax(0, D$LFC_A) else pmax(0, -D$LFC_A)
      }
      all_rows[[length(all_rows)+1]] <- data.frame(Comparison=comp, gene_key=D$gene_key, value=val, stringsAsFactors=FALSE)
    }
    if (length(all_rows)==0) return(NULL)
    do.call(rbind, all_rows)
  }

  for (cat in names(categories)){
    info <- categories[[cat]]
    L <- build_long_for_category(cat, info)
    if (is.null(L) || !nrow(L)){ message("No entries for heatmap: ", cat); next }

    agg <- L %>% group_by(gene_key) %>% summarise(total=sum(value,na.rm=TRUE), .groups="drop")
    ord_genes <- agg %>% arrange(desc(total)) %>% pull(gene_key)
    if (is.finite(max_genes_per_heatmap)) ord_genes <- head(ord_genes, max_genes_per_heatmap)
    comps <- unique(L$Comparison)

    M <- L %>% filter(gene_key %in% ord_genes) %>%
      mutate(gene_key=factor(gene_key, levels=rev(ord_genes)),
             Comparison=factor(Comparison, levels=comps))

    is_up <- identical(info$sign,"up")
    pal_low  <- if (is_up) pal_up_low  else pal_dn_low
    pal_high <- if (is_up) pal_up_high else pal_dn_high

    h <- max(5, 0.12*length(ord_genes) + 2)
    out_pdf <- file.path(plots_dir, sprintf("VolcanoCompare_Heatmap_%s_ALL.pdf", gsub("[() ]","_", cat)))

    pdf(out_pdf, width=7.6, height=h)
    p <- ggplot(M, aes(x=Comparison, y=gene_key, fill=value)) +
      geom_tile(color="white", linewidth=0.05) +
      scale_x_discrete(labels=wrap_comp_label) +
      scale_fill_gradient(low=pal_low, high=pal_high, na.value="white", name="log2FC source") +
      labs(title=paste0(cat, " — ", species),
           subtitle=sprintf("Thresholds: padj<%.2g, |log2FC|≥%.2f. %s\nRows ordered by total intensity across comparisons (descending).",
                            padj_thr_vc, lfc_thr_vc, condition_note),
           x="Comparisons (top of ‘–’ = B, bottom = A)", y="Genes") +
      theme_bw(base_size=9) +
      theme(axis.text.x=element_text(angle=0,hjust=0.5,size=8),
            axis.text.y=element_text(size=7), panel.grid=element_blank(),
            legend.position="right")
    print(p); dev.off()
    message("Wrote: ", normalizePath(out_pdf))
  }
  invisible(NULL)
}
make_category_heatmaps_all_comparisons(compare_csv_paths, max_genes_per_heatmap=Inf)

# ============================
# GSEA comparisons — diverging NES bars (bars from 0) using significant terms in either contrast
# ============================

read_gsea_tbl <- function(contrast){
  fp <- file.path(output_dir, sprintf("GSEA_%s.csv", contrast))
  if (!file.exists(fp)) return(NULL)
  df <- read.csv(fp, stringsAsFactors=FALSE)
  if (!nrow(df)) return(NULL)
  need <- c("Description","NES","p.adjust")
  if (!all(need %in% names(df))) return(NULL)
  df %>% transmute(Description = as.character(Description),
                   NES         = as.numeric(NES),
                   p.adjust    = as.numeric(p.adjust))
}

compare_gsea_bars <- function(contrast_A, contrast_B){
  A <- read_gsea_tbl(contrast_A); B <- read_gsea_tbl(contrast_B)
  if (is.null(A) && is.null(B)) { message("No GSEA tables for: ", contrast_B," vs ",contrast_A); return(invisible(NULL)) }
  A2 <- if (is.null(A)) data.frame(Description=character(), NES_A=numeric(), FDR_A=numeric()) else
    A %>% transmute(Description, NES_A=NES, FDR_A=p.adjust)
  B2 <- if (is.null(B)) data.frame(Description=character(), NES_B=numeric(), FDR_B=numeric()) else
    B %>% transmute(Description, NES_B=NES, FDR_B=p.adjust)
  M <- full_join(A2,B2,by="Description")
  if (!nrow(M)) { message("Empty GSEA after join for: ", contrast_B," vs ",contrast_A); return(invisible(NULL)) }

  keep <- with(M, (is.finite(NES_A)|is.finite(NES_B)) & ((!is.na(FDR_A)&FDR_A<gsea_fdr_thr) | (!is.na(FDR_B)&FDR_B<gsea_fdr_thr)))
  M <- M[keep, , drop=FALSE]
  if (!nrow(M)) { message("No significant terms in either contrast for: ", contrast_B," vs ",contrast_A); return(invisible(NULL)) }

  M$Description <- wrap_text(M$Description, width=48)
  ord <- order(pmax(abs(dplyr::coalesce(M$NES_A,0)), abs(dplyr::coalesce(M$NES_B,0))), decreasing=TRUE)
  M <- M[ord, , drop=FALSE]

  L <- rbind(
    data.frame(Description=M$Description, Contrast=contrast_A, NES=M$NES_A),
    data.frame(Description=M$Description, Contrast=contrast_B, NES=M$NES_B)
  )
  L <- L[is.finite(L$NES), , drop=FALSE]
  if (!nrow(L)) { message("No finite NES after filter: ", contrast_B," vs ",contrast_A); return(invisible(NULL)) }

  L$col_key <- paste0(ifelse(L$NES>=0,"up","down"), "_", ifelse(L$Contrast==contrast_B,"B","A"))
  cols <- c(
    "up_A"   = "#fcbba1",  # light red
    "up_B"   = "#cb181d",  # dark red
    "down_A" = "#9ecae1",  # light blue
    "down_B" = "#08519c"   # dark blue
  )

  h <- max(5, 0.22*length(unique(L$Description)) + 2)
  out_pdf <- file.path(plots_dir, sprintf("GSEA_compare_diverging_%s__vs__%s.pdf", contrast_B, contrast_A))
  pdf(out_pdf, width=8.4, height=h)
  p <- ggplot(L, aes(x=0, xend=NES, y=Description, yend=Description, color=col_key)) +
    geom_vline(xintercept=0, linetype="dashed") +
    geom_segment(linewidth=1.3, lineend="round") +
    geom_point(aes(x=NES, y=Description), size=1.8) +
    scale_color_manual(values=cols, guide="none") +
    labs(title=paste0("GSEA (BP) — ", species, ": ", contrast_B, " vs ", contrast_A),
         subtitle=sprintf("Terms with FDR < %.2g in either contrast. Bars start at 0; red = NES>0, blue = NES<0.\n%s",
                          gsea_fdr_thr, condition_note),
         x="NES (diverging from 0)", y=NULL) +
    theme_bw(base_size=10)
  print(p); dev.off()
  message("Wrote: ", normalizePath(out_pdf))
  invisible(NULL)
}

# Use the SAME pairs as volcano (including the updated order)
gsea_pairs <- volcano_pairs
message("Running GSEA comparisons for ", length(gsea_pairs), " pairs...")
for (pr in gsea_pairs){
  compare_gsea_bars(contrast_A=pr[1], contrast_B=pr[2])
}
message("GSEA comparison PDFs in: ", normalizePath(plots_dir))
