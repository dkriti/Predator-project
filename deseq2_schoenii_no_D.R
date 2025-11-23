# ============================
# End-to-end DE + GO + GSEA pipeline (S. schoenii)
# + VolcanoCompare plots & 6 all-comparison heatmaps
# - All plots show thresholds
# - Labels use valid symbols/ORFs only (no sch_ or g#### placeholders)
# - GO network legend hides NA level
# ============================

# ---- File and directory setup ----
species <- "S. schoenii"
counts_file <- "gene_counts_schoenii.txt"
gtf_file <- "schoenii_annotation.gtf"
blastx_file <- "all_genes_schoneii_out_blastx.best_per_gene.compact.standard.tsv"
output_dir <- "csv_results_schoenii"
plots_dir <- "pdf_plots_schoenii"
dir.create(output_dir, showWarnings = FALSE)
dir.create(plots_dir, showWarnings = FALSE)

# ---- Thresholds (used in subtitles across plots) ----
core_padj_thr <- 0.05   # DE threshold in core volcano/MA and for ORA seeding
core_lfc_thr  <- 1.0
gsea_fdr_thr  <- 0.05   # gseGO p.adjust cutoff
net_padj_thr  <- 0.05   # bipartite network filter
net_lfc_thr   <- 0.5
# VolcanoCompare (stricter)
padj_thr_vc <- 0.01
lfc_thr_vc  <- 1.0

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
ratio_num <- function(x) { parts <- strsplit(x, "/", fixed=TRUE); sapply(parts, function(v) as.numeric(v[1]) / as.numeric(v[2])) }
head_n <- function(df, n) df[seq_len(min(n, nrow(df))), , drop = FALSE]

# "Standard" gene symbol for plotting (CSV keeps everything):
# - Yeast ORF format: Y[A-Z]{2}[0-9]{3}[CW](-suffix)?
# - All-caps letters/digits/hyphen: ^[A-Z0-9][A-Z0-9-]{1,8}$
is_standard_symbol <- function(x) {
  x <- as.character(x)
  orf <- grepl("^Y[A-Z]{2}[0-9]{3}[CW](?:-[A-Z0-9]+)?$", x)
  sym <- grepl("^[A-Z0-9][A-Z0-9-]{1,8}$", x)
  orf | sym
}

# Choose clean plot label: valid symbol > ORF > NA (no sch_*, no g####)
choose_plot_label <- function(alias_symbol, orf_id, final_label) {
  cand1 <- ifelse(!is.na(alias_symbol) & nzchar(alias_symbol), alias_symbol, NA_character_)
  cand1[!is_standard_symbol(cand1)] <- NA_character_
  cand2 <- ifelse(!is.na(orf_id) & nzchar(orf_id), orf_id, NA_character_)
  cand2[!is_standard_symbol(cand2)] <- NA_character_
  lab_no_sch <- gsub("^sch_", "", ifelse(!is.na(final_label), final_label, ""))
  lab_no_sch[!is_standard_symbol(lab_no_sch)] <- NA_character_
  out <- ifelse(!is.na(cand1), cand1,
                ifelse(!is.na(cand2), cand2,
                       ifelse(!is.na(lab_no_sch), lab_no_sch, NA_character_)))
  out
}

# Wrap long comparison labels for heatmaps: "B__minus__A" -> "B\n–\nA" with "vs" wrapped
wrap_comp_label <- function(x) {
  sapply(x, function(s) {
    parts <- strsplit(s, "__minus__", fixed = TRUE)[[1]]
    if (length(parts) != 2) return(s)
    fmt <- function(p) stringr::str_replace_all(p, "_vs_", "\nvs\n")
    paste0(fmt(parts[1]), "\n–\n", fmt(parts[2]))
  })
}

# ---- Load & preprocess ----
counts <- read.table(counts_file, header=TRUE, row.names=1, comment.char="#")
counts <- counts[, 6:ncol(counts)]
colnames(counts) <- gsub("_Aligned.out.sorted.dedup.bam", "", colnames(counts))

sample_names <- colnames(counts)
timepoint <- sub("^(T[0-9]+).*$", "\\1", sample_names)
treatment <- sub("^T[0-9]+R[0-9]+([A-Za-z])$", "\\1", sample_names)
group <- factor(paste(timepoint, treatment, sep="_"),
                levels=c("T0_p","T0_L","T6_L","T24_L"))
replicate <- sub("^T[0-9]+(R[0-9]+).*$", "\\1", sample_names)
samples <- data.frame(
  row.names = sample_names,
  timepoint = factor(timepoint, levels=c("T0","T6","T24")),
  replicate = factor(replicate),
  treatment = factor(treatment, levels=c("p","L")),
  group     = group
)

# Exclude "D"
is_D <- grepl("D$", colnames(counts))
counts <- counts[, !is_D]; samples <- samples[!is_D, ]

# Gene filter
keep <- rowSums(counts >= 10) >= 2
counts <- counts[keep, ]

# Remove all-zero samples if any
zero_samples <- colSums(counts) == 0
if(any(zero_samples)) { counts <- counts[, !zero_samples, drop=FALSE]; samples <- samples[!zero_samples, , drop=FALSE] }

# DESeq2
dds <- DESeqDataSetFromMatrix(countData=counts, colData=samples, design=~group)
dds <- DESeq(dds, fitType="local", sfType="poscounts")
vsd <- vst(dds, blind=FALSE)

# Normalized counts
norm_counts <- counts(dds, normalized=TRUE)
write.csv(as.data.frame(norm_counts), file.path(output_dir, "normalized_counts.csv"), row.names=TRUE)

# ---- Annotation: GTF + BLASTX ----
gtf_raw <- readr::read_tsv(gtf_file, comment = "#", col_names = FALSE, show_col_types = FALSE)
gtf_df <- gtf_raw %>% dplyr::filter(X3 == "gene") %>%
  dplyr::mutate(raw_id = stringr::str_extract(X9, '(?<=gene_id \")[^\"]+'),
                gtf_name = stringr::str_extract(X9, '(?<=gene_name \")[^\"]+')) %>%
  dplyr::select(raw_id, gtf_name) %>% dplyr::distinct()

blastx_raw <- readr::read_tsv(blastx_file, show_col_types = FALSE)
blastx_df <- blastx_raw %>%
  dplyr::transmute(
    gene_id = sch_gene,
    orf_id  = dplyr::if_else(stringr::str_detect(sseqid, "^Y[A-Z]{2}[0-9]{3}[CW](?:-[A-Z0-9]+)?$"), sseqid, NA_character_),
    blast_sym = dplyr::if_else(stringr::str_detect(stitle, "gene_symbol:[^; ]+"),
                        stringr::str_extract(stitle, "(?<=gene_symbol:)[^; ]+"), NA_character_)
  ) %>% dplyr::distinct(gene_id, orf_id, blast_sym)

gene_df <- tibble::tibble(gene_id = rownames(dds))
annot0 <- gene_df %>% dplyr::left_join(gtf_df, by = c("gene_id" = "raw_id")) %>% dplyr::left_join(blastx_df, by = "gene_id")
gtf_used <- unique(stats::na.omit(annot0$gtf_name))
dup_blast_sym <- annot0 %>% dplyr::filter(!is.na(blast_sym)) %>% dplyr::count(blast_sym, name = "n_genes") %>%
  dplyr::filter(n_genes > 1) %>% dplyr::pull(blast_sym)
blast_conflicts_gtf <- intersect(unique(stats::na.omit(annot0$blast_sym)), gtf_used)

annot1 <- annot0 %>% dplyr::mutate(
  blast_sym_clean = dplyr::case_when(
    is.na(blast_sym) ~ NA_character_,
    blast_sym %in% blast_conflicts_gtf ~ NA_character_,
    blast_sym %in% dup_blast_sym ~ NA_character_,
    TRUE ~ blast_sym
  )
)

conflict_log <- dplyr::bind_rows(
  annot0 %>% dplyr::filter(!is.na(blast_sym), blast_sym %in% blast_conflicts_gtf) %>%
    dplyr::transmute(gene_id, conflict_type = "blast_vs_gtf_name", conflicted_label = blast_sym),
  annot0 %>% dplyr::filter(!is.na(blast_sym), blast_sym %in% dup_blast_sym) %>%
    dplyr::transmute(gene_id, conflict_type = "blast_symbol_used_by_multiple_genes", conflicted_label = blast_sym)
)
readr::write_csv(conflict_log, file.path(output_dir, "annotation_label_conflicts_schoenii.csv"))

annot_df <- annot1 %>%
  dplyr::mutate(
    base_label = dplyr::coalesce(gtf_name, blast_sym_clean, gene_id),
    final_label = dplyr::if_else(stringr::str_starts(base_label, "sch_"), base_label, paste0("sch_", base_label)),
    alias_symbol = blast_sym_clean
  ) %>% as.data.frame()
rownames(annot_df) <- annot_df$gene_id
orf_to_symbol <- annot_df %>% dplyr::select(orf_id, alias_symbol) %>% dplyr::distinct()

# ---- Contrasts ----
contrast_list <- list(
  T0_L_vs_T0_p   = c("group","T0_L","T0_p"),
  T6_L_vs_T0_p   = c("group","T6_L","T0_p"),
  T6_L_vs_T0_L   = c("group","T6_L","T0_L"),
  T24_L_vs_T0_p  = c("group","T24_L","T0_p"),
  T24_L_vs_T0_L  = c("group","T24_L","T0_L"),
  T24_L_vs_T6_L  = c("group","T24_L","T6_L")
)

get_res <- function(contrast_vec, lfc_limit=10) {
  res_shrink <- lfcShrink(dds, contrast=contrast_vec, type="ashr")
  as.data.frame(res_shrink) %>%
    tibble::rownames_to_column("gene_id") %>%
    dplyr::left_join(annot_df, by="gene_id") %>%
    dplyr::mutate(plot_label = choose_plot_label(alias_symbol, orf_id, final_label)) %>%
    dplyr::filter(abs(log2FoldChange) <= lfc_limit)
}

# ---- ORA helpers & views (effect size only; seed threshold mentioned) ----
plot_go_ora_views <- function(go_table, contrast_name, topn=20) {
  if (is.null(go_table) || !nrow(go_table)) return(invisible(NULL))
  df <- go_table %>%
    dplyr::mutate(GeneRatio_num = ratio_num(GeneRatio),
                  BgRatio_num   = ratio_num(BgRatio),
                  FoldEnrichment = GeneRatio_num / BgRatio_num,
                  negLog10Padj   = -log10(p.adjust + 1e-300))
  top_eff <- df[order(-df$FoldEnrichment, df$p.adjust), , drop=FALSE] %>% head_n(topn)
  p_eff <- ggplot(top_eff, aes(x = FoldEnrichment, y = reorder(Description, FoldEnrichment),
                               size = Count, color = negLog10Padj)) +
    geom_point() +
    scale_size_continuous(name = "Gene count") +
    scale_color_gradient(name = expression(-log[10]~"FDR"), low = "#bdd7e7", high = "#08519c") +
    labs(title = paste0("GO ORA (BP) — ", species, ": ", contrast_name),
         subtitle = sprintf("Effect size view. ORA seeded with DE genes (padj<%.2g).", core_padj_thr),
         x = "Fold enrichment (GeneRatio/BgRatio)", y = NULL) +
    theme_bw(base_size = 11)
  pdf(file.path(plots_dir, sprintf("GO_ORA_effectsize_%s.pdf", contrast_name)), width = 8, height = 6); print(p_eff); dev.off()
}

run_go <- function(res_df) {
  sig_orfs <- res_df %>% dplyr::filter(padj < core_padj_thr & !is.na(orf_id)) %>% dplyr::pull(orf_id) %>% unique()
  if (length(sig_orfs)==0) return(NULL)
  entrez_map <- AnnotationDbi::select(org.Sc.sgd.db, keys = sig_orfs, keytype = "ORF", columns = c("ORF","ENTREZID"))
  entrez <- stats::na.omit(entrez_map$ENTREZID); if (length(entrez)==0) return(NULL)
  go_result <- enrichGO(gene=entrez, OrgDb=org.Sc.sgd.db, keyType="ENTREZID", ont="BP", pvalueCutoff=core_padj_thr)
  if (is.null(go_result) || nrow(as.data.frame(go_result)) == 0) return(NULL)
  orf_for_entrez <- setNames(entrez_map$ORF, entrez_map$ENTREZID)
  symbol_for_orf <- setNames(orf_to_symbol$alias_symbol, orf_to_symbol$orf_id)
  go_df <- as.data.frame(go_result)
  go_df$geneNames <- sapply(strsplit(go_df$geneID, "/"), function(ids){
    orfs <- orf_for_entrez[ids]; syms <- symbol_for_orf[orfs]
    out <- ifelse(!is.na(syms) & nzchar(syms), syms, orfs); paste(out, collapse="/")
  })
  list(result = go_result, table = go_df, entrez_map = entrez_map)
}

run_go_clusters <- function(go_result, contrast_name, entrez_map) {
  if (is.null(go_result)) return(NULL)
  go_df <- as.data.frame(go_result); if (nrow(go_df) < 2) return(NULL)
  go_simplified <- tryCatch({ clusterProfiler::simplify(go_result, cutoff=0.7, by="p.adjust", select_fun=min) },
                            error = function(e) go_result)
  go_df2 <- as.data.frame(go_simplified)
  orf_for_entrez <- setNames(entrez_map$ORF, entrez_map$ENTREZID)
  symbol_for_orf <- setNames(orf_to_symbol$alias_symbol, orf_to_symbol$orf_id)
  go_df2$geneNames <- sapply(strsplit(go_df2$geneID, "/"), function(ids){
    orfs <- orf_for_entrez[ids]; syms <- symbol_for_orf[orfs]
    out <- ifelse(!is.na(syms) & nzchar(syms), syms, orfs); paste(out, collapse="/")
  })
  write.csv(go_df2, file.path(output_dir, sprintf("GO_clusters_%s.csv", contrast_name)), row.names = FALSE)

  go_sim <- tryCatch({ enrichplot::pairwise_termsim(go_simplified) }, error = function(e) NULL)
  pdf(file.path(plots_dir, sprintf("GO_clusters_%s.pdf", contrast_name)), width = 8, height = 6)
  if (!is.null(go_sim)) print(emapplot(go_sim, showCategory = min(20, nrow(go_df2))) +
                                ggtitle(sprintf("GO Clusters (BP) — %s: %s", species, contrast_name)) +
                                labs(subtitle = sprintf("Seed: DE genes (padj<%.2g).", core_padj_thr)))
  else print(ggplot() + ggtitle('emapplot error (no similarity)'))
  dev.off()

  # Extra: effect-size version of dotplot
  df <- go_df2 %>%
    dplyr::mutate(GeneRatio_num = ratio_num(GeneRatio),
                  BgRatio_num = ratio_num(BgRatio),
                  FoldEnrichment = GeneRatio_num / BgRatio_num,
                  negLog10Padj = -log10(p.adjust + 1e-300))
  df_top <- df[order(-df$FoldEnrichment, df$p.adjust), , drop=FALSE] %>% head_n(20)
  p_eff <- ggplot(df_top, aes(x = FoldEnrichment, y = reorder(Description, FoldEnrichment),
                              size = Count, color = negLog10Padj)) +
    geom_point() + scale_size_continuous(name="Gene count") +
    scale_color_gradient(name = expression(-log[10]~"FDR"), low="#bdd7e7", high="#08519c") +
    labs(title = sprintf("GO Clusters (BP) — %s: %s", species, contrast_name),
         subtitle = sprintf("Effect size view. Seed: DE genes (padj<%.2g).", core_padj_thr),
         x="Fold enrichment (GeneRatio/BgRatio)", y=NULL) +
    theme_bw(base_size=11)
  pdf(file.path(plots_dir, sprintf("GO_clusters_dot_effectsize_%s.pdf", contrast_name)), width=8, height=5); print(p_eff); dev.off()
}

# ---- Bipartite GO network (significant genes only; Up=firebrick2, Down=royalblue) ----
plot_go_network <- function(cluster_go_gene_csv, volcano_file, contrast_name,
                            padj_thresh=net_padj_thr, lfc_thresh=net_lfc_thr, top_n_clusters=20) {
  if (!file.exists(cluster_go_gene_csv)) return(invisible(NULL))
  dat <- read.csv(cluster_go_gene_csv, stringsAsFactors = FALSE)
  if (!nrow(dat)) return(invisible(NULL))

  top_clusters <- unique(dat$GO_Cluster)[1:min(top_n_clusters, length(unique(dat$GO_Cluster)))]
  dat <- dat[dat$GO_Cluster %in% top_clusters, , drop=FALSE]
  if (!nrow(dat)) return(invisible(NULL))

  edges_raw <- dat %>%
    tidyr::separate_rows(Top_DE_Genes, sep="; ") %>%
    dplyr::mutate(gene_symbol = gsub(" \\(log2FC.*", "", Top_DE_Genes)) %>%
    dplyr::select(GO_Cluster, gene_symbol) %>% dplyr::distinct()

  vol <- read.csv(volcano_file, stringsAsFactors = FALSE) %>%
    dplyr::mutate(reg = ifelse(padj < padj_thresh & abs(log2FoldChange) >= lfc_thresh,
                        ifelse(log2FoldChange >= 0, "Up", "Down"), NA_character_))
  vol_sig <- vol %>% dplyr::filter(!is.na(reg)) %>% dplyr::select(alias_symbol, reg) %>%
    dplyr::rename(gene_symbol = alias_symbol)

  edges <- edges_raw %>% dplyr::inner_join(vol_sig, by = "gene_symbol")
  if (!nrow(edges)) { message("No significant edges for network after filtering."); return(invisible(NULL)) }

  terms <- unique(edges$GO_Cluster)
  genes <- unique(edges$gene_symbol)
  verts <- dplyr::bind_rows(
    data.frame(name=terms, is_term=TRUE,  reg=NA_character_, stringsAsFactors=FALSE),
    data.frame(name=genes, is_term=FALSE, reg=vol_sig$reg[match(genes, vol_sig$gene_symbol)], stringsAsFactors=FALSE)
  )

  g <- igraph::graph_from_data_frame(edges[,c("GO_Cluster","gene_symbol")], vertices=verts, directed=FALSE)

  pdf(file.path(plots_dir, sprintf("GO_Network_%s.pdf", contrast_name)), width=10, height=7)
  print(
    ggraph(g, layout="fr") +
      geom_edge_link(alpha=0.25, colour="grey60") +
      geom_node_point(aes(shape = ifelse(is_term, "Term", "Gene"),
                          color = ifelse(is_term, NA, reg)),
                      size = ifelse(igraph::V(g)$is_term, 3, 2.5), show.legend = TRUE) +
      geom_node_text(aes(label=name), repel=TRUE, size=2.6, max.overlaps=100) +
      scale_shape_manual(name = "Node type", values = c("Term"=15, "Gene"=16)) +
      scale_color_manual(
        name = "Regulation", values = c("Down"="royalblue", "Up"="firebrick2"),
        breaks = c("Down","Up"),   # <- remove NA level from legend
        na.value = "black", drop = TRUE
      ) +
      guides(color = guide_legend(order = 1),
             shape = guide_legend(order = 2, override.aes = list(color="black", size=3))) +
      ggtitle(paste("GO Cluster–Gene Network (BP) —", species, ":", contrast_name)) +
      labs(subtitle = sprintf("Nodes: Terms & significant genes. Filter: padj<%.2g & |log2FC|≥%.2f.", padj_thresh, lfc_thresh)) +
      theme_void() + theme(legend.position = "right")
  )
  dev.off()
}

# ---- Term–term projection (built from significant DE genes via cluster CSV) ----
plot_go_term_projection <- function(cluster_go_gene_csv, contrast_name,
                                    top_n_clusters = 20, min_shared = 2) {
  if (!file.exists(cluster_go_gene_csv)) return(invisible(NULL))
  dat <- read.csv(cluster_go_gene_csv, stringsAsFactors = FALSE)
  if (!nrow(dat)) return(invisible(NULL))

  top_clusters <- unique(dat$GO_Cluster)[1:min(top_n_clusters, length(unique(dat$GO_Cluster)))]
  dat <- dat[dat$GO_Cluster %in% top_clusters, , drop=FALSE]
  if (!nrow(dat)) return(invisible(NULL))

  edges <- dat %>% tidyr::separate_rows(Top_DE_Genes, sep="; ") %>%
    dplyr::mutate(gene_symbol = gsub(" \\(log2FC.*", "", Top_DE_Genes)) %>%
    dplyr::select(GO_Cluster, gene_symbol) %>% dplyr::distinct()
  if (!nrow(edges)) return(invisible(NULL))

  terms <- unique(edges$GO_Cluster); genes <- unique(edges$gene_symbol)
  verts <- dplyr::bind_rows(data.frame(name=terms, type=TRUE), data.frame(name=genes, type=FALSE))
  g <- igraph::graph_from_data_frame(edges, vertices=verts, directed=FALSE)
  proj <- igraph::bipartite_projection(g)
  term_g <- if (igraph::vcount(proj$proj1) == length(terms)) proj$proj1 else proj$proj2
  term_g <- igraph::delete_edges(term_g, igraph::E(term_g)[weight < min_shared])
  if (igraph::ecount(term_g) == 0) { message("No term–term edges after filtering."); return(invisible(NULL)) }

  pdf(file.path(plots_dir, sprintf("GO_TermProjection_%s.pdf", contrast_name)), width=9, height=6)
  print(
    ggraph(term_g, layout="fr") +
      geom_edge_link(aes(width = weight), alpha = 0.4, colour = "grey50") +
      scale_edge_width(range = c(0.3, 2.5), name = "Shared DE genes") +
      geom_node_point(size = 4, colour = "#2b8cbe") +
      geom_node_text(aes(label = name), repel = TRUE, size = 3) +
      ggtitle(sprintf("Term–term projection (BP) — %s: %s", species, contrast_name)) +
      labs(subtitle = sprintf("Edges = shared significant DE genes in cluster CSV; min shared = %d.", min_shared)) +
      theme_void()
  )
  dev.off()
}

# ---- Cluster → GO term → Top Genes (significant DE genes only) ----
generate_cluster_go_gene_table <- function(
    contrast_name, volcano_file, go_file, cluster_file, output_file,
    top_n_clusters = 20, top_n_genes = 20, padj_thresh=net_padj_thr, lfc_thresh=net_lfc_thr) {

  if (!file.exists(volcano_file) | !file.exists(go_file) | !file.exists(cluster_file)) {
    warning(sprintf("Files missing for %s. Skipping.", contrast_name)); return(NULL)
  }

  volcano_df <- read.csv(volcano_file, stringsAsFactors = FALSE) %>%
    dplyr::mutate(gene_symbol = ifelse(!is.na(alias_symbol) & nzchar(alias_symbol),
                                alias_symbol, stringr::str_remove(final_label, "^sch_"))) %>%
    dplyr::filter(!is.na(padj)) %>%
    dplyr::filter(padj < padj_thresh & abs(log2FoldChange) >= lfc_thresh)

  go_df <- read.csv(go_file, stringsAsFactors = FALSE)
  cluster_df <- read.csv(cluster_file, stringsAsFactors = FALSE)
  if (!nrow(go_df) || !nrow(cluster_df)) { warning(sprintf("Empty GO or cluster table for %s.", contrast_name)); return(NULL) }

  top_clusters <- cluster_df %>% dplyr::arrange(p.adjust) %>% head_n(top_n_clusters)

  records <- list()
  for (i in seq_len(nrow(top_clusters))) {
    cluster_name <- top_clusters$Description[i]; cluster_id <- top_clusters$ID[i]
    go_terms <- go_df %>% dplyr::filter(ID == cluster_id); if (nrow(go_terms) == 0) next
    for (j in seq_len(nrow(go_terms))) {
      genes <- unique(strsplit(go_terms$geneNames[j], "/", fixed=TRUE)[[1]])
      genes <- genes[!is.na(genes) & nzchar(genes)]
      matched <- volcano_df %>% dplyr::filter(gene_symbol %in% genes) %>% dplyr::arrange(padj)
      if (nrow(matched) == 0) next
      matched <- head_n(matched, top_n_genes)
      records[[length(records)+1]] <- data.frame(
        GO_Cluster = cluster_name,
        GO_Term = go_terms$Description[j],
        Top_DE_Genes = paste0(
          matched$gene_symbol, " (log2FC=", round(matched$log2FoldChange,2),
          ", padj=", signif(matched$padj,3), ")", collapse="; "
        ),
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(records) > 0) {
    result_df <- dplyr::bind_rows(records)
    write.csv(result_df, file = output_file, row.names = FALSE)
    message(sprintf("Cluster-GO-Gene mapping written: %s", output_file))
    return(result_df)
  } else { warning(sprintf("No matches found for %s", contrast_name)); return(NULL) }
}

# ============================
# Main loop over contrasts (core outputs)
# ============================
for(name in names(contrast_list)) {
  contrast_vec <- contrast_list[[name]]
  res <- get_res(contrast_vec, lfc_limit=10) %>% dplyr::filter(!is.na(log2FoldChange)&!is.na(padj))

  # Volcano (labels cleaned; no sch_/g#### placeholders)
  p_vol <- EnhancedVolcano(
    res, lab = res$plot_label, x = 'log2FoldChange', y = 'padj',
    title = paste("Volcano —", species, ":", name),
    subtitle = sprintf("Thresholds: padj<%.2g, |log2FC|≥%.2f (labelled points use valid symbols/ORFs only)",
                       core_padj_thr, core_lfc_thr),
    pCutoff = core_padj_thr, FCcutoff = core_lfc_thr, cutoffLineType = 'dashed',
    cutoffLineCol = 'black', cutoffLineWidth = 0.4, labSize = 3,
    pointSize = 1.0, col = c("grey30", "grey30", "royalblue", "firebrick2"),
    colAlpha = 0.9
  )
  pdf(file.path(plots_dir, sprintf("Volcano_%s.pdf", name))); print(p_vol); dev.off()
  write.csv(res %>% dplyr::select(gene_id, orf_id, alias_symbol, log2FoldChange, pvalue, padj, final_label, plot_label),
            file.path(output_dir, sprintf("Volcano_%s.csv", name)), row.names = FALSE)

  # Heatmap (top 50 DE) — labels cleaned
  top50 <- head(res %>% dplyr::filter(padj<core_padj_thr) %>% dplyr::arrange(padj), 50)
  write.csv(top50, file.path(output_dir, sprintf("Heatmap_top50_%s.csv", name)), row.names=FALSE)
  cols <- colnames(vsd)[colData(vsd)$group %in% contrast_vec[2:3]]
  if (nrow(top50) > 1 && length(cols) >= 2) {
    mat <- assay(vsd)[top50$gene_id, cols, drop=FALSE]
    rownames(mat) <- make.unique(ifelse(!is.na(top50$plot_label) & nzchar(top50$plot_label),
                                        top50$plot_label,
                                        gsub("^sch_", "", top50$final_label)))
    ht <- pheatmap(mat - rowMeans(mat),
                   annotation_col=as.data.frame(colData(vsd)[cols,c('timepoint','treatment')]),
                   main=paste("Top 50 DE —", species, ":", name),
                   silent=TRUE,
                   annotation_legend = TRUE)
    pdf(file.path(plots_dir, sprintf("Heatmap_%s.pdf",name)), width=6, height=8)
    grid::grid.newpage(); grid::grid.draw(ht$gtable); dev.off()
  }

  # MA
  pdf(file.path(plots_dir, sprintf("MA_%s.pdf",name)))
  res_ma <- res %>% dplyr::mutate(baseMean = ifelse(is.na(baseMean), 0, baseMean)) %>%
    dplyr::filter(baseMean>0) %>%
    dplyr::mutate(DE=abs(log2FoldChange)>=core_lfc_thr & padj<core_padj_thr,
                  col=ifelse(DE, "#ff7f0e", "grey80"))
  with(res_ma, plot(baseMean,log2FoldChange,log='x',col=col,pch=20,cex=0.6,
                    main=paste("MA —", species, ":", name)))
  mtext(sprintf("DE highlight: padj<%.2g & |log2FC|≥%.2f", core_padj_thr, core_lfc_thr), side=3, line=0.5, cex=0.8)
  abline(h=0); legend("topright", legend = c("DE"), col = "#ff7f0e", pch = 20, pt.cex = 0.8, bty = "n", title = "Gene")
  dev.off()
  write.csv(res_ma, file = file.path(output_dir, sprintf("MA_%s.csv", name)), row.names = FALSE)

  # GO enrichment
  go_out <- run_go(res)
  pdf(file.path(plots_dir, sprintf("GO_%s.pdf", name)), width = 8, height = 5)
  if (!is.null(go_out)) print(dotplot(go_out$result, showCategory = 10) +
                                ggtitle(paste0("GO Enrichment (BP) — ", species, ": ", name)) +
                                labs(subtitle = sprintf("ORA seeded with DE genes (padj<%.2g).", core_padj_thr)))
  else print(ggplot() + ggtitle(paste("GO Enrichment (BP) —", species, ":", name)) +
               labs(subtitle = "No significant GO terms"))
  dev.off()

  if (!is.null(go_out)) {
    write.csv(go_out$table, file.path(output_dir, sprintf("GO_%s_geneNames.csv", name)), row.names = FALSE)
    plot_go_ora_views(go_out$table, name, topn = 20)
    run_go_clusters(go_out$result, name, go_out$entrez_map)
  }

  # GSEA (x = NES, color = -log10(FDR))
  entrez_map_all <- AnnotationDbi::select(org.Sc.sgd.db, keys = unique(stats::na.omit(res$orf_id)), keytype = "ORF", columns = "ENTREZID")
  geneList_df <- res %>% dplyr::filter(!is.na(orf_id)) %>% dplyr::left_join(entrez_map_all, by = c("orf_id" = "ORF")) %>%
    dplyr::filter(!is.na(ENTREZID), !is.na(log2FoldChange)) %>%
    dplyr::group_by(ENTREZID) %>% dplyr::summarise(score = mean(log2FoldChange), .groups="drop") %>%
    dplyr::arrange(dplyr::desc(score))
  geneList <- sort(setNames(geneList_df$score, geneList_df$ENTREZID), decreasing = TRUE)
  gsea_sch <- tryCatch(gseGO(geneList = geneList, OrgDb = org.Sc.sgd.db, keyType = "ENTREZID",
                             ont = "BP", minGSSize = 10, pvalueCutoff = gsea_fdr_thr, verbose = FALSE),
                       error = function(e) { message("gseGO error (", name, "): ", e$message); NULL })
  pdf(file.path(plots_dir, sprintf("GSEA_%s.pdf", name)), width = 8, height = 6)
  if (!is.null(gsea_sch) && nrow(as.data.frame(gsea_sch)) > 0) {
    gdf <- as.data.frame(gsea_sch) %>% dplyr::mutate(negLog10Padj = -log10(p.adjust + 1e-300))
    topN <- head(gdf[order(gdf$p.adjust), ], min(20, nrow(gdf)))
    p_gsea <- ggplot(topN, aes(x = NES, y = reorder(Description, NES), size = setSize, color = negLog10Padj)) +
      geom_vline(xintercept = 0, linetype = "dashed") + geom_point() +
      scale_size_continuous(name = "Set size") +
      scale_color_gradient(name = expression(-log[10]~"FDR"), low = "#d9f0a3", high = "#1a9850") +
      labs(title = paste0("GSEA (BP) — ", species, ": ", name),
           subtitle = sprintf("Points = leading-edge sets; shown if FDR<%.2g. Color = -log10(FDR).", gsea_fdr_thr),
           x = "Normalized Enrichment Score (NES)", y = NULL) +
      theme_bw(base_size = 11)
    print(p_gsea)
  } else { print(ggplot() + ggtitle(paste("GSEA (BP) —", species, ":", name)) + labs(subtitle = "No significant GSEA terms")) }
  dev.off()
  write.csv(if (is.null(gsea_sch)) data.frame() else as.data.frame(gsea_sch),
            file.path(output_dir, sprintf("GSEA_%s.csv", name)), row.names = FALSE)

  # Cluster->GO->Top genes + network/term-projection
  if (!is.null(go_out)) {
    volcano_file <- file.path(output_dir, sprintf("Volcano_%s.csv", name))
    go_file      <- file.path(output_dir, sprintf("GO_%s_geneNames.csv", name))
    cluster_file <- file.path(output_dir, sprintf("GO_clusters_%s.csv", name))
    output_file  <- file.path(output_dir, sprintf("Cluster_GO_Genes_%s.csv", name))
    generate_cluster_go_gene_table(name, volcano_file, go_file, cluster_file, output_file,
                                   top_n_clusters = 20, top_n_genes = 20,
                                   padj_thresh = net_padj_thr, lfc_thresh = net_lfc_thr)
    if (file.exists(output_file)) {
      plot_go_network(output_file, volcano_file, name,
                      padj_thresh = net_padj_thr, lfc_thresh = net_lfc_thr, top_n_clusters = 20)
      plot_go_term_projection(output_file, name)
    }
  }
}

message("Core analysis done. PDFs in: ", normalizePath(plots_dir))
message("Tables in: ", normalizePath(output_dir))

# ============================
# Volcano comparisons (B minus A) — strict thresholds + plots
# ============================

build_gene_key <- function(df) {
  sym <- df$alias_symbol
  lab <- gsub("^sch_", "", df$final_label)
  ifelse(!is.na(sym) & nzchar(sym), sym,
         ifelse(!is.na(lab) & nzchar(lab), lab, df$gene_id))
}

compare_volcano <- function(contrast_A, contrast_B,
                            padj_thr = padj_thr_vc, lfc_thr = lfc_thr_vc,
                            label_top = 15) {
  fA <- file.path(output_dir, sprintf("Volcano_%s.csv", contrast_A))
  fB <- file.path(output_dir, sprintf("Volcano_%s.csv", contrast_B))
  if (!file.exists(fA) || !file.exists(fB)) {
    message("Skipping pair (missing CSV): ", contrast_B, " minus ", contrast_A)
    return(invisible(NULL))
  }

  A <- read.csv(fA, stringsAsFactors = FALSE)
  B <- read.csv(fB, stringsAsFactors = FALSE)
  if (!nrow(A) && !nrow(B)) return(invisible(NULL))

  A$gene_key <- build_gene_key(A)
  B$gene_key <- build_gene_key(B)

  keep_cols <- c("gene_key","log2FoldChange","padj")
  A2 <- A[, keep_cols]; colnames(A2) <- c("gene_key","LFC_A","padj_A")
  B2 <- B[, keep_cols]; colnames(B2) <- c("gene_key","LFC_B","padj_B")

  M <- dplyr::full_join(A2, B2, by = "gene_key") %>%
    dplyr::mutate(
      deA   = !is.na(padj_A) & padj_A < padj_thr & is.finite(LFC_A) & abs(LFC_A) >= lfc_thr,
      deB   = !is.na(padj_B) & padj_B < padj_thr & is.finite(LFC_B) & abs(LFC_B) >= lfc_thr,
      upA   = deA & LFC_A >=  lfc_thr,
      downA = deA & LFC_A <= -lfc_thr,
      upB   = deB & LFC_B >=  lfc_thr,
      downB = deB & LFC_B <= -lfc_thr
    ) %>%
    dplyr::mutate(
      status = dplyr::case_when(
        upB   & !deA ~ "New Up",
        downB & !deA ~ "New Down",
        upA   & !deB ~ "Lost Up",
        downA & !deB ~ "Lost Down",
        upB   & downA ~ "Switch Up (from Down)",
        downB & upA   ~ "Switch Down (from Up)",
        upA   & upB   ~ "Common Up",
        downA & downB ~ "Common Down",
        TRUE          ~ "NS"
      ),
      nlog10_p_B = ifelse(is.na(padj_B), NA_real_, -log10(padj_B + 1e-300)),
      nlog10_p_A = ifelse(is.na(padj_A), NA_real_, -log10(padj_A + 1e-300)),
      delta_LFC  = ifelse(is.finite(LFC_A) & is.finite(LFC_B), LFC_B - LFC_A, NA_real_)
    )

  # Write full table (CSV keeps all genes; plots hide non-standard symbols)
  out_csv <- file.path(output_dir, sprintf("VolcanoCompare_%s__minus__%s.csv", contrast_B, contrast_A))
  write.csv(M, out_csv, row.names = FALSE)

  # VolcanoCompare plot (only standard symbols)
  cols_named <- c(
    "New Up"                 = "firebrick2",
    "New Down"               = "royalblue",
    "Lost Up"                = "#fcae91",
    "Lost Down"              = "#9ecae1",
    "Switch Up (from Down)"  = "purple",
    "Switch Down (from Up)"  = "seagreen",
    "Common Up"              = "grey50",
    "Common Down"            = "grey50",
    "NS"                     = "grey82"
  )

  P <- M %>% dplyr::filter(is.finite(LFC_B), is.finite(nlog10_p_B))
  if (!nrow(P)) { message("No finite B values for plotting: ", contrast_B, " minus ", contrast_A); return(invisible(out_csv)) }

  P_plot <- P %>% dplyr::filter(is_standard_symbol(gene_key))
  lab_df <- P_plot %>%
    dplyr::filter(status %in% c("New Up","New Down")) %>%
    dplyr::arrange(padj_B) %>%
    dplyr::slice_head(n = label_top)

  pdf(file.path(plots_dir, sprintf("VolcanoCompare_%s__minus__%s.pdf", contrast_B, contrast_A)),
      width = 7.2, height = 6.0)
  p <- ggplot2::ggplot(P_plot, aes(x = LFC_B, y = nlog10_p_B, color = status)) +
    geom_vline(xintercept = c(-lfc_thr, lfc_thr), linetype = "dashed", alpha = 0.6) +
    geom_hline(yintercept = -log10(padj_thr), linetype = "dashed", alpha = 0.6) +
    geom_point(alpha = 0.9, size = 1.6) +
    ggrepel::geom_text_repel(
      data = lab_df, aes(label = gene_key),
      size = 2.6, box.padding = 0.25, max.overlaps = 25
    ) +
    scale_color_manual(values = cols_named, drop = FALSE) +
    labs(
      title = paste0("Volcano — ", species, ": ", contrast_B, " minus ", contrast_A),
      subtitle = sprintf("Thresholds: padj<%.2g, |log2FC|≥%.2f (new/lost/switch vs prior contrast)", padj_thr, lfc_thr),
      x = paste0("log2FC in ", contrast_B),
      y = paste0("-log10(FDR) in ", contrast_B),
      color = NULL,
      caption = sprintf("B = %s, A = %s. Labels show most significant 'New' genes by FDR in B.", contrast_B, contrast_A)
    ) +
    theme_bw(base_size = 10) +
    theme(legend.position = "right")
  print(p); dev.off()

  invisible(out_csv)
}

# ---- EXACT comparison pairs (used for plots & heatmaps) ----
volcano_pairs <- list(
  c("T0_L_vs_T0_p","T6_L_vs_T0_p"),
  c("T6_L_vs_T0_p","T24_L_vs_T0_p"),
  c("T0_L_vs_T0_p","T24_L_vs_T0_p"),
  c("T6_L_vs_T0_L","T24_L_vs_T0_L"),
  c("T0_L_vs_T0_p","T24_L_vs_T0_L"),
  c("T0_L_vs_T0_p","T6_L_vs_T0_L")
)

message("Running volcano comparisons (strict) for ", length(volcano_pairs), " pairs...")
compare_csv_paths <- character(0)
for (pr in volcano_pairs) {
  A <- pr[1]; B <- pr[2]
  out <- compare_volcano(contrast_A = A, contrast_B = B,
                         padj_thr = padj_thr_vc, lfc_thr = lfc_thr_vc, label_top = 15)
  if (!is.null(out)) compare_csv_paths <- c(compare_csv_paths, out)
}
message("Volcano comparison PDFs in: ", normalizePath(plots_dir))
message("Comparison tables in: ", normalizePath(output_dir))

# ============================
# 6 ALL-VOLCANO COMPARISON HEATMAPS (categories), colored by degree of regulation
# ============================
make_category_heatmaps_all_comparisons <- function(compare_csv_paths,
                                                   max_genes_per_heatmap = Inf) {
  if (length(compare_csv_paths) == 0) { message("No compare CSVs to plot."); return(invisible(NULL)) }

  categories <- list(
    "New Up"                = list(fill_from = "B", sign = "up",   subtitle = "fill = log2FC in later contrast (left of '–')"),
    "New Down"              = list(fill_from = "B", sign = "down", subtitle = "fill = |log2FC| in later contrast (left of '–')"),
    "Switch Up (from Down)" = list(fill_from = "B", sign = "up",   subtitle = "fill = log2FC in later contrast (left of '–')"),
    "Switch Down (from Up)" = list(fill_from = "B", sign = "down", subtitle = "fill = |log2FC| in later contrast (left of '–')"),
    "Lost Up"               = list(fill_from = "A", sign = "up",   subtitle = "fill = log2FC in prior contrast (right of '–')"),
    "Lost Down"             = list(fill_from = "A", sign = "down", subtitle = "fill = |log2FC| in prior contrast (right of '–')")
  )

  pal_up_low   <- "#fee5d9"; pal_up_high   <- "firebrick2"
  pal_dn_low   <- "#deebf7"; pal_dn_high   <- "royalblue"

  build_long_for_category <- function(cat_name, info) {
    all_rows <- list()
    for (fp in compare_csv_paths) {
      comp <- gsub("^VolcanoCompare_(.*)\\.csv$", "\\1", basename(fp))
      D <- read.csv(fp, stringsAsFactors = FALSE)
      if (!nrow(D)) next
      D <- D %>% dplyr::filter(status == cat_name) %>%
        dplyr::filter(is_standard_symbol(gene_key))
      if (!nrow(D)) next

      if (info$fill_from == "B") {
        val <- if (info$sign == "up") pmax(0, D$LFC_B) else pmax(0, -D$LFC_B)
        src <- sub("__minus__.*$", "", comp)   # left part = B
      } else {
        val <- if (info$sign == "up") pmax(0, D$LFC_A) else pmax(0, -D$LFC_A)
        src <- sub("^.*__minus__", "", comp)   # right part = A
      }

      all_rows[[length(all_rows)+1]] <- data.frame(
        Comparison = comp, gene_key = D$gene_key,
        value = val, src_contrast = src,
        stringsAsFactors = FALSE
      )
    }
    if (length(all_rows) == 0) return(NULL)
    do.call(rbind, all_rows)
  }

  for (cat in names(categories)) {
    info <- categories[[cat]]
    L <- build_long_for_category(cat, info)
    if (is.null(L) || !nrow(L)) { message("No entries for heatmap: ", cat); next }

    # Order genes by total intensity across comparisons (descending)
    agg <- L %>% dplyr::group_by(gene_key) %>% dplyr::summarise(total = sum(value, na.rm = TRUE), .groups="drop")
    ord_genes <- agg %>% dplyr::arrange(dplyr::desc(total)) %>% dplyr::pull(gene_key)
    if (is.finite(max_genes_per_heatmap)) ord_genes <- head(ord_genes, max_genes_per_heatmap)

    comps <- unique(L$Comparison)
    M <- L %>%
      dplyr::filter(gene_key %in% ord_genes) %>%
      dplyr::mutate(
        gene_key   = factor(gene_key, levels = rev(ord_genes)),
        Comparison = factor(Comparison, levels = comps)
      )

    is_up <- identical(info$sign, "up")
    pal_low  <- if (is_up) pal_up_low  else pal_dn_low
    pal_high <- if (is_up) pal_up_high else pal_dn_high

    # dynamic height; compact theme; wrap x labels
    h <- max(5, 0.12 * length(ord_genes) + 2)
    out_pdf <- file.path(plots_dir, sprintf("VolcanoCompare_Heatmap_%s_ALL.pdf", gsub("[() ]","_", cat)))

    pdf(out_pdf, width = 7.2, height = h)
    p <- ggplot2::ggplot(M, aes(x = Comparison, y = gene_key, fill = value)) +
      geom_tile(color = "white", linewidth = 0.05) +
      scale_x_discrete(labels = wrap_comp_label) +
      scale_fill_gradient(low = pal_low, high = pal_high, na.value = "white",
                          name = "log2FC source") +
      labs(title    = paste0(cat, " — ", species),
           subtitle = sprintf("Thresholds: padj<%.2g, |log2FC|≥%.2f; %s.\nRows ordered by total intensity across comparisons (descending).",
                              padj_thr_vc, lfc_thr_vc, info$subtitle),
           x = "Comparisons (B on top of '–', A below)", y = "Genes") +
      theme_bw(base_size = 9) +
      theme(
        axis.text.x = element_text(angle = 0, hjust = 0.5, size = 8),
        axis.text.y = element_text(size = 7),
        panel.grid  = element_blank(),
        legend.position = "right"
      ) +
      guides(fill = guide_colorbar(title = "Value", title.position = "top"))
    print(p); dev.off()
    message("Wrote: ", normalizePath(out_pdf))
  }

  invisible(NULL)
}

# Build the 6 heatmaps (one per category, all comparisons as columns)
make_category_heatmaps_all_comparisons(compare_csv_paths, max_genes_per_heatmap = Inf)




# ==========================================
# GSEA compare heatmaps (per pair)
# Fill = NES (diverging), dot = FDR < threshold
# All significant terms in either contrast; with wrapped labels
# ==========================================

species_title <- "S. schoenii"
output_dir <- "csv_results_schoenii_T_stringent_padj"   # where GSEA_<contrast>.csv are
plots_dir  <- "pdf_plots_schoenii_T_stringent_padj"
dir.create(plots_dir, showWarnings = FALSE)

# Pairs: (A, B) -> plot titled "B vs A"
gsea_pairs <- list(
  c("T6_L_vs_T0_p", "T24_L_vs_T0_p"),
  c("T6_L_vs_T0_L", "T24_L_vs_T0_L")
)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(readr)
})

# ---- Nicify & wrap contrast labels for the x-axis ----
label_contrast <- function(x) {
  x <- gsub("_vs_", "\nvs\n", x, fixed = TRUE)  # split on vs to two lines
  x <- gsub("_", " ", x, fixed = TRUE)          # underscores -> spaces
  x
}

# ---- Robust reader (works even if ID missing) ----
# Needs: Description, NES, p.adjust. If ID missing, use normalized Description as join key.
read_gsea_tbl <- function(contrast) {
  fp <- file.path(output_dir, sprintf("GSEA_%s.csv", contrast))
  if (!file.exists(fp)) {
    message("Missing GSEA CSV: ", fp)
    return(data.frame())
  }
  df <- read.csv(fp, stringsAsFactors = FALSE, check.names = FALSE)

  has_id   <- "ID"          %in% names(df)
  has_desc <- "Description" %in% names(df)
  has_nes  <- "NES"         %in% names(df)
  has_fdr  <- "p.adjust"    %in% names(df)

  if (!has_desc || !has_nes) {
    message("CSV lacks required columns (Description/NES): ", fp)
    return(data.frame())
  }

  KEY <- if (has_id) df$ID else tolower(gsub("\\s+", " ", trimws(df$Description)))
  if (!has_fdr) df$p.adjust <- NA_real_

  df %>%
    transmute(
      KEY = KEY,
      ID = if ("ID" %in% names(df)) ID else NA_character_,
      Description = Description,
      NES = NES,
      p.adjust = p.adjust
    ) %>%
    filter(!is.na(KEY), !is.na(Description))
}

# ---- Heatmap for a pair (B vs A) with wrapped text everywhere ----
plot_gsea_compare_heatmap <- function(contrast_A, contrast_B,
                                      gsea_fdr_thr = 0.05,
                                      wrap_width_terms = 55,   # wrap width for GO terms
                                      wrap_width_title = 80,   # wrap width for title/subtitle
                                      base_font = 10) {

  A <- read_gsea_tbl(contrast_A)
  B <- read_gsea_tbl(contrast_B)
  if (!nrow(A) && !nrow(B)) {
    message("No GSEA rows for both: ", contrast_B, " vs ", contrast_A)
    return(invisible(NULL))
  }

  # Keep ALL terms significant in either contrast
  A_sig <- A %>% filter(!is.na(p.adjust), p.adjust < gsea_fdr_thr)
  B_sig <- B %>% filter(!is.na(p.adjust), p.adjust < gsea_fdr_thr)
  if (!nrow(A_sig) && !nrow(B_sig)) {
    message("No significant terms in either contrast for: ", contrast_B, " vs ", contrast_A)
    return(invisible(NULL))
  }

  # Join by KEY; keep NES & FDR from both sides
  J <- full_join(
    A_sig %>% transmute(KEY, Description_A = Description, NES_A = NES, FDR_A = p.adjust),
    B_sig %>% transmute(KEY, Description_B = Description, NES_B = NES, FDR_B = p.adjust),
    by = "KEY"
  ) %>%
    mutate(
      Description = coalesce(Description_B, Description_A, KEY),
      # wrap GO term labels for y-axis
      Desc_disp   = str_wrap(Description, width = wrap_width_terms)
    ) %>%
    filter(!is.na(NES_A) | !is.na(NES_B))

  if (!nrow(J)) {
    message("Nothing to plot after selection: ", contrast_B, " vs ", contrast_A)
    return(invisible(NULL))
  }

  # Long (one row per (term, contrast)), include NA for non-sig so tile shows as NA color
  L <- J %>%
    select(Desc_disp, NES_A, NES_B, FDR_A, FDR_B) %>%
    mutate(row_id = row_number()) %>%
    tidyr::pivot_longer(
      cols = c(NES_A, NES_B, FDR_A, FDR_B),
      names_to = c(".value", "which"),
      names_pattern = "(NES|FDR)_(A|B)"
    ) %>%
    mutate(
      Contrast = dplyr::recode(which, A = contrast_A, B = contrast_B),
      sig = !is.na(FDR) & FDR < gsea_fdr_thr
    ) %>%
    select(Desc_disp, Contrast, NES, FDR, sig)

  # Order rows by max |NES| across both contrasts (descending)
  ord <- L %>%
    group_by(Desc_disp) %>%
    summarise(max_absNES = max(abs(NES), na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(max_absNES)) %>%
    pull(Desc_disp)

  L$Desc_disp <- factor(L$Desc_disp, levels = rev(ord))
  # wrap x-axis labels (contrast names)
  L$Contrast  <- factor(L$Contrast, levels = c(contrast_A, contrast_B),
                        labels = label_contrast(c(contrast_A, contrast_B)))

  # Symmetric fill range around 0 for consistent color mapping
  v <- L$NES[is.finite(L$NES)]
  vmax <- if (length(v)) max(abs(v), na.rm = TRUE) else 1
  if (!is.finite(vmax) || vmax == 0) vmax <- 1

  # Figure height scales with number of terms
  n_terms <- length(unique(L$Desc_disp))
  h <- max(5.5, 0.24 * n_terms + 2.2)

  # Wrapped title/subtitle (ASCII to avoid glyph warnings)
  title_txt <- sprintf("%s - GSEA comparison heatmap: %s vs %s",
                       species_title, contrast_B, contrast_A)
  title_wrapped <- str_wrap(title_txt, width = wrap_width_title)

  sub_txt <- paste0(
    "Terms significant in either contrast (FDR < ",
    format(gsea_fdr_thr, digits = 3),
    "). Fill = NES; dot = significant."
  )
  sub_wrapped <- str_wrap(sub_txt, width = wrap_width_title)

  out_pdf <- file.path(
    plots_dir,
    sprintf("GSEA_compare_heatmap_%s__vs__%s.pdf", contrast_B, contrast_A)
  )

  pdf(out_pdf, width = 7.2, height = h)
  p <- ggplot(L, aes(x = Contrast, y = Desc_disp, fill = NES)) +
    geom_tile(color = "white", linewidth = 0.15, na.rm = FALSE) +
    # Dot overlay only where sig == TRUE (per cell)
    geom_point(data = subset(L, isTRUE(sig)),
               shape = 21, size = 1.8, stroke = 0.2,
               fill = "black", color = "black") +
    scale_fill_gradient2(
      low = "#2b8cbe", mid = "white", high = "#e34a33",
      midpoint = 0, limits = c(-vmax, vmax), oob = scales::squish,
      na.value = "grey92", name = "NES"
    ) +
    labs(
      title = title_wrapped,
      subtitle = sub_wrapped,
      x = NULL, y = NULL
    ) +
    theme_bw(base_size = base_font) +
    theme(
      legend.position = "right",
      axis.text.x = element_text(vjust = 1),
      panel.grid = element_blank(),
      plot.title.position = "plot",
      plot.margin = margin(t = 12, r = 14, b = 18, l = 12)
    )
  print(p)
  dev.off()
  message("Wrote: ", normalizePath(out_pdf))
  invisible(NULL)
}

# ---- Run for all pairs ----
for (pr in gsea_pairs) {
  plot_gsea_compare_heatmap(
    contrast_A         = pr[1],
    contrast_B         = pr[2],
    gsea_fdr_thr       = 0.05,  # printed on plot
    wrap_width_terms   = 55,    # tighter wrap for term labels
    wrap_width_title   = 78,    # wrap title/subtitle to fit page
    base_font          = 10
  )
}





