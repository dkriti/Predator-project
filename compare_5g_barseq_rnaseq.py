#!/usr/bin/env python3
"""
Compare BarSeq (5g) with RNA-seq (Live vs Dead) at T6 and T24,
plus an across-time @5g comparison. Also export Top-10 gene tables
for each "strong" quadrant (FD >= +10 or FD <= -10).

Supports BarSeq input as CSV or Excel via --barseq-file.

Outputs
-------
- CSVs: merged tables for each comparison
- CSVs: Top-10 per strong quadrant for each comparison
- PNGs: quadrant scatter plots (clean: no Spearman on figure)
- PNG: quadrants_triptych.png (T6, T24, Δ in one figure)
- PNG: heatmap_topStrong.png (FD & RNA columns; column-wise robust z)
- CSV: qc_correlation_summary.csv with Spearman rho, p, N per comparison

Example
-------
python compare_5g_barseq_rnaseq.py \
  --barseq-file fitness_defect_scores_barseq_marjan.csv \
  --gtf Saccharomyces_cerevisiae.R64-1-1.114.gtf \
  --t6_de MA_T6_L_vs_T6_D.csv \
  --t24_de MA_T24_L_vs_T24_D.csv \
  --c_t6_5g "T6, 5g to T6, 0g - MinimalMedia+10%aa" \
  --c_t24_5g "T24, 5g to T24, 0g - MinimalMedia+10%aa" \
  --c_time_5g "T24, 5g to T6, 5g - MinimalMedia+10%aa" \
  --outdir rnaseq_barseq_5g_comparisons \
  --fd_defect 10 --fd_benefit -10 --label_top 10
"""

import os, re, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from math import erf, sqrt  # for QC p-value

# ---------- Style (you can tweak here) ----------
BASE_S = 6          # small dot size for ALL points (background + strong)
LABEL_FS = 8        # label font size
BASE_ALPHA = 0.25   # background transparency
GUIDE_LW = 1.0

# Consistent colors (match slide deck)
COLOR_BASE = "#c7c7c7"    # light gray for all points
COLOR_DEFECT = "#E74C3C"  # red for FD ≥ +10
COLOR_BENEFIT = "#2AA9FF" # blue for FD ≤ −10

# Label offsets (distance away from the point; scaled by quadrant sign)
LABEL_DX = 0.30
LABEL_DY = 0.60

# ---------- Helpers ----------
def load_gene_map_from_gtf(gtf_path):
    gtf = pd.read_csv(gtf_path, sep="\t", comment="#", header=None,
                      names=["chr","source","feature","start","end","score","strand","frame","attr"])
    genes = gtf[gtf["feature"]=="gene"].copy()
    def parse_attr(attr):
        d = {}
        for m in re.finditer(r'(\S+)\s+"([^"]+)"', str(attr)):
            d[m.group(1)] = m.group(2)
        return pd.Series({"gene_id": d.get("gene_id"),
                          "gene_name": d.get("gene_name"),
                          "gene_biotype": d.get("gene_biotype")})
    out = genes["attr"].apply(parse_attr)
    return out.dropna(subset=["gene_id"]).drop_duplicates(subset=["gene_id"])

def _standardize_columns(df):
    df.columns = [re.sub(r"\s+"," ", str(c).strip()) for c in df.columns]
    return df

def _find_gene_col(cols):
    for cand in ["gene","Gene","GENE","orf","ORF","Orf","GeneID","GeneId","gene_id"]:
        if cand in cols:
            return cand
    lmap = {c.lower(): c for c in cols}
    if "gene" in lmap: return lmap["gene"]
    if "orf" in lmap: return lmap["orf"]
    return cols[0]

def _resolve_contrast_column(cols, requested):
    if requested in cols: return requested
    wanted = re.sub(r"\s+"," ", requested.strip()).lower()
    norm = {re.sub(r"\s+"," ", c.strip()).lower(): c for c in cols}
    if wanted in norm: return norm[wanted]
    raise ValueError(f"BarSeq contrast '{requested}' not found.\nAvailable:\n - " + "\n - ".join(cols))

def load_barseq_fd(barseq_file, contrast, sheet_name=None):
    ext = os.path.splitext(barseq_file)[1].lower()
    if ext in [".xlsx",".xls"]:
        if sheet_name is None: sheet_name = "fitness_defect_scores"
        df = pd.read_excel(barseq_file, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(barseq_file)
    else:
        raise ValueError("Unsupported BarSeq file (use .csv or .xlsx/.xls)")
    df = _standardize_columns(df)
    gene_col = _find_gene_col(df.columns)
    contrast_col = _resolve_contrast_column(df.columns, contrast)
    sub = df[[gene_col, contrast_col]].rename(columns={gene_col:"gene", contrast_col:"FD"}).dropna()
    sub["gene"] = sub["gene"].astype(str)
    return sub

def load_rnaseq_de(de_path, gene_map):
    df = pd.read_csv(de_path)
    df = _standardize_columns(df)
    cand_id = None
    for key in ["gene_id","Geneid","gene","Gene","GeneID","id","ID"]:
        if key in df.columns: cand_id = key; break
    if cand_id is None: cand_id = df.columns[0]
    cand_lfc = None
    for c in df.columns:
        if c.lower().replace(" ","") in ["log2fc","logfc","log2foldchange","log2.foldchange","log2.fold_change"]:
            cand_lfc = c; break
    if cand_lfc is None:
        raise ValueError(f"No log2FC column in {de_path}")
    cand_fdr = None
    for c in df.columns:
        if c.lower().replace(" ","") in ["padj","fdr","adj.p.val","qvalue"]:
            cand_fdr = c; break
    cand_p = None
    for c in df.columns:
        if c.lower().replace(" ","") in ["pvalue","p.val","p"]:
            cand_p = c; break

    out = df.rename(columns={cand_id:"gene_raw", cand_lfc:"log2FC"}).copy()
    if cand_fdr: out = out.rename(columns={cand_fdr:"padj"})
    elif cand_p: out = out.rename(columns={cand_p:"pvalue"})
    gm = gene_map[["gene_id","gene_name"]].dropna().drop_duplicates()
    out = out.merge(gm, left_on="gene_raw", right_on="gene_id", how="left")
    miss = out["gene_id"].isna()
    if miss.any():
        fix = out.loc[miss, ["gene_raw"]].merge(gm, left_on="gene_raw", right_on="gene_name", how="left")
        out.loc[miss, ["gene_id","gene_name"]] = fix[["gene_id","gene_name"]].values
    out["gene"] = out["gene_name"].fillna(out["gene_raw"]).astype(str)
    return out

def merge_fd_de(fd_df, de_df):
    keep = ["gene","log2FC"]
    if "padj" in de_df.columns: keep.append("padj")
    if "pvalue" in de_df.columns and "padj" not in de_df.columns: keep.append("pvalue")
    de_sub = de_df[keep].dropna(subset=["log2FC"]).copy()
    if "padj" in de_sub.columns:
        de_sub = de_sub.sort_values("padj").drop_duplicates(subset=["gene"], keep="first")
    else:
        de_sub = de_sub.drop_duplicates(subset=["gene"], keep="first")
    return fd_df.merge(de_sub, on="gene", how="inner")

def assign_quadrants(df, fd_defect=10.0, fd_benefit=-10.0):
    df = df.dropna(subset=["FD","log2FC"]).copy()
    df["strong_defect"]  = df["FD"] >= fd_defect
    df["strong_benefit"] = df["FD"] <= fd_benefit
    def quad(r):
        if r["log2FC"]>0 and r["FD"]>0:  return "Induced & Required"
        if r["log2FC"]<0 and r["FD"]<0:  return "Repressed & Beneficial"
        if r["log2FC"]>0 and r["FD"]<0:  return "Induced but Beneficial"
        if r["log2FC"]<0 and r["FD"]>0:  return "Repressed but Required"
        return "Other"
    df["quadrant"] = df.apply(quad, axis=1)
    return df

def top_n_per_strong_quadrant(df, n=10, fd_defect_thresh=10.0, fd_benefit_thresh=-10.0):
    x = assign_quadrants(df, fd_defect_thresh, fd_benefit_thresh)
    ir = x[(x["log2FC"]>0) & (x["FD"]>=fd_defect_thresh)].sort_values(["FD","log2FC"], ascending=[False, False])
    rr = x[(x["log2FC"]<0) & (x["FD"]>=fd_defect_thresh)].sort_values(["FD","log2FC"], ascending=[False, True])
    rb = x[(x["log2FC"]<0) & (x["FD"]<=fd_benefit_thresh)].sort_values(["FD","log2FC"], ascending=[True, True])
    ib = x[(x["log2FC"]>0) & (x["FD"]<=fd_benefit_thresh)].sort_values(["FD","log2FC"], ascending=[True, False])
    return {"Induced & Required": ir.head(n),
            "Repressed but Required": rr.head(n),
            "Repressed & Beneficial": rb.head(n),
            "Induced but Beneficial": ib.head(n)}

# ---------- QC: Spearman (off-figure) ----------
def _spearman_rho_p(df):
    """Rank-correlation QC: returns (rho, p, n) for df with columns ['log2FC','FD'] (SciPy-free)."""
    xy = df[["log2FC", "FD"]].dropna().to_numpy()
    n = xy.shape[0]
    if n < 3:
        return float("nan"), float("nan"), n
    # Spearman rho via ranked Pearson
    xr = pd.Series(xy[:, 0]).rank(method="average").to_numpy()
    yr = pd.Series(xy[:, 1]).rank(method="average").to_numpy()
    rho = float(np.corrcoef(xr, yr)[0, 1])
    # Two-sided p from t-transform with normal tail approx
    t = abs(rho) * np.sqrt((n - 2) / max(1e-12, 1 - rho**2))
    p = 2 * (1 - 0.5 * (1 + erf(t / sqrt(2.0))))
    return rho, float(p), n

def write_qc_stats(outdir, name, df):
    """Append a row to qc_correlation_summary.csv for this comparison."""
    rho, p, n = _spearman_rho_p(df)
    path = os.path.join(outdir, "qc_correlation_summary.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("comparison,n,rho,p\n")
        f.write(f"{name},{n},{rho:.6g},{p:.6g}\n")

# ---------- Plot: single quadrant ----------
def plot_quadrant(
    df, title, xlabel, ylabel="BarSeq Fitness Defect (FD)",
    fd_defect_thresh=10.0, fd_benefit_thresh=-10.0,
    label_top=10, save_path=None, top_tables_dir=None, prefix_for_tables="comparison",
):
    x = assign_quadrants(df, fd_defect_thresh, fd_benefit_thresh)

    # Base scatter (single small size for ALL points)
    plt.figure(figsize=(8.2, 6.6))
    plt.scatter(x["log2FC"], x["FD"], s=BASE_S, alpha=BASE_ALPHA, color=COLOR_BASE, zorder=1)

    # Replot strong hits in color (same size as background for consistency)
    sd = x[x["strong_defect"]]
    sb = x[x["strong_benefit"]]
    if len(sd):
        plt.scatter(sd["log2FC"], sd["FD"], s=BASE_S, alpha=0.95, color=COLOR_DEFECT,
                    label=f"FD ≥ {fd_defect_thresh:g}", zorder=2)
    if len(sb):
        plt.scatter(sb["log2FC"], sb["FD"], s=BASE_S, alpha=0.95, color=COLOR_BENEFIT,
                    label=f"FD ≤ {fd_benefit_thresh:g}", zorder=2)

    # Guides
    plt.axvline(0, linestyle="--", linewidth=GUIDE_LW, color="#666666", zorder=0)
    plt.axhline(0, linestyle="--", linewidth=GUIDE_LW, color="#666666", zorder=0)
    plt.axhline(fd_defect_thresh, linestyle=":", linewidth=GUIDE_LW, color="#666666", zorder=0)
    plt.axhline(fd_benefit_thresh, linestyle=":", linewidth=GUIDE_LW, color="#666666", zorder=0)

    # Labels + export Top-N tables a little away from points
    if label_top and label_top > 0:
        picks = top_n_per_strong_quadrant(
            x, n=label_top, fd_defect_thresh=fd_defect_thresh, fd_benefit_thresh=fd_benefit_thresh
        )
        if top_tables_dir: os.makedirs(top_tables_dir, exist_ok=True)
        for qname, sub in picks.items():
            if sub.empty: continue
            xs, ys, genes = sub["log2FC"].values, sub["FD"].values, sub["gene"].values
            col = COLOR_DEFECT if "Required" in qname else COLOR_BENEFIT
            plt.scatter(xs, ys, s=BASE_S, color=col, alpha=0.95, zorder=3)
            for i, (xi, yi, g) in enumerate(zip(xs, ys, genes)):
                sgnx = 1 if xi >= 0 else -1
                sgny = 1 if yi >= 0 else -1
                dx = sgnx * LABEL_DX * (1 + 0.03*i)
                dy = sgny * LABEL_DY * (1 + 0.03*i)
                txt = plt.annotate(
                    g, (xi, yi), xytext=(xi+dx, yi+dy),
                    arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.7, color="#555"),
                    fontsize=LABEL_FS, color=col, zorder=4
                )
                # white halo around text for legibility
                txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
            # export tables
            if top_tables_dir:
                short = qname.lower().replace("&","and").replace(" ","_")
                out_csv = os.path.join(top_tables_dir, f"{prefix_for_tables}__{short}__top{len(sub)}.csv")
                cols = ["gene","FD","log2FC"]
                extra = [c for c in ["padj","pvalue","quadrant","strong_defect","strong_benefit"] if c in sub.columns]
                sub[cols+extra].to_csv(out_csv, index=False)

    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if len(sd) or len(sb):
        plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------- Heatmap over union of strong top hits ----------
def _robust_z(v):
    v = v.astype(float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if mad == 0:
        sd = np.nanstd(v)
        if sd == 0: return v - med
        return (v - med) / (sd + 1e-9)
    return (v - med) / (1.4826 * mad + 1e-9)

def make_topset_heatmap(merged_T6, merged_T24, merged_time, outdir,
                        top_n=10, fd_defect=10.0, fd_benefit=-10.0,
                        fname="heatmap_topStrong.png"):
    # Collect union of strong Top-N across the 3 comparisons
    gene_set = set()
    for df in (merged_T6, merged_T24, merged_time):
        picks = top_n_per_strong_quadrant(df, n=top_n,
                                          fd_defect_thresh=fd_defect,
                                          fd_benefit_thresh=fd_benefit)
        for sub in picks.values():
            gene_set.update(sub["gene"].tolist())
    genes = sorted(gene_set)

    # Build matrix with FD and RNA columns
    t6  = merged_T6.set_index("gene")[["FD","log2FC"]].rename(columns={"FD":"FD_T6","log2FC":"LFC_T6"})
    t24 = merged_T24.set_index("gene")[["FD","log2FC"]].rename(columns={"FD":"FD_T24","log2FC":"LFC_T24"})
    tm  = merged_time.set_index("gene")[["FD","log2FC"]].rename(columns={"FD":"FD_T24minusT6","log2FC":"ΔLFC_T24minusT6"})
    M = pd.DataFrame(index=genes)
    for chunk in (t6, t24, tm):
        M = M.join(chunk, how="left")

    # Robust z-score per column for comparable color scale
    Z = M.apply(_robust_z, axis=0)
    # Order rows by max absolute signal
    row_score = np.nanmax(np.abs(Z.values), axis=1)
    order = np.argsort(-row_score)
    Z = Z.iloc[order]

    # Heatmap
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.35*len(Z))))
    im = ax.imshow(Z.values, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_yticks(range(len(Z.index)))
    ax.set_yticklabels(Z.index, fontsize=8)
    ax.set_xticks(range(len(Z.columns)))
    ax.set_xticklabels(Z.columns, rotation=45, ha="right")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Robust z-score per column", rotation=90)

    ax.set_title("Top strong hits across comparisons (FD & RNA-seq scaled per column)")
    fig.tight_layout()
    out_path = os.path.join(outdir, fname)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    return out_path

# ---------- Triptych: T6, T24, Δ in one figure ----------
def _draw_quadrant_on_ax(ax, df, title, xlabel,
                         fd_defect_thresh=10.0, fd_benefit_thresh=-10.0, label_top=6):
    x = assign_quadrants(df, fd_defect_thresh, fd_benefit_thresh)
    ax.scatter(x["log2FC"], x["FD"], s=BASE_S, alpha=BASE_ALPHA, color=COLOR_BASE, zorder=1)
    sd = x[x["strong_defect"]]; sb = x[x["strong_benefit"]]
    if len(sd): ax.scatter(sd["log2FC"], sd["FD"], s=BASE_S, color=COLOR_DEFECT, alpha=0.95, zorder=2, label=f"FD ≥ {fd_defect_thresh:g}")
    if len(sb): ax.scatter(sb["log2FC"], sb["FD"], s=BASE_S, color=COLOR_BENEFIT, alpha=0.95, zorder=2, label=f"FD ≤ {fd_benefit_thresh:g}")
    ax.axvline(0, ls="--", lw=GUIDE_LW, color="#666"); ax.axhline(0, ls="--", lw=GUIDE_LW, color="#666")
    ax.axhline(fd_defect_thresh, ls=":", lw=GUIDE_LW, color="#666"); ax.axhline(fd_benefit_thresh, ls=":", lw=GUIDE_LW, color="#666")
    ax.set_title(title); ax.set_xlabel(xlabel)
    # light labeling (top 6) to avoid clutter in panel
    if label_top and label_top > 0:
        picks = top_n_per_strong_quadrant(x, n=label_top, fd_defect_thresh=fd_defect_thresh, fd_benefit_thresh=fd_benefit_thresh)
        for qname, sub in picks.items():
            if sub.empty: continue
            xs, ys, genes = sub["log2FC"].values, sub["FD"].values, sub["gene"].values
            col = COLOR_DEFECT if "Required" in qname else COLOR_BENEFIT
            ax.scatter(xs, ys, s=BASE_S, color=col, alpha=0.95, zorder=3)
            for i, (xi, yi, g) in enumerate(zip(xs, ys, genes)):
                sgnx = 1 if xi >= 0 else -1; sgny = 1 if yi >= 0 else -1
                dx = sgnx * (LABEL_DX*0.8) * (1 + 0.03*i); dy = sgny * (LABEL_DY*0.8) * (1 + 0.03*i)
                txt = ax.annotate(g, (xi, yi), xytext=(xi+dx, yi+dy),
                                  arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.7, color="#555"),
                                  fontsize=LABEL_FS, color=col, zorder=4)
                txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

def plot_quadrant_triptych(merged_T6, merged_T24, merged_time, outdir,
                           fd_defect=10.0, fd_benefit=-10.0, label_top=6,
                           fname="quadrants_triptych.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    _draw_quadrant_on_ax(axes[0], merged_T6,  "T6 (L vs D) vs FD (5g→0g)",   "RNA-seq log2FC (T6 L vs D)",
                         fd_defect, fd_benefit, label_top)
    _draw_quadrant_on_ax(axes[1], merged_T24, "T24 (L vs D) vs FD (5g→0g)",  "RNA-seq log2FC (T24 L vs D)",
                         fd_defect, fd_benefit, label_top)
    _draw_quadrant_on_ax(axes[2], merged_time,"Δ(LvsD) vs FD (T24,5g→T6,5g)","Δ log2FC (T24 LvsD − T6 LvsD)",
                         fd_defect, fd_benefit, label_top)
    axes[0].set_ylabel("BarSeq Fitness Defect (FD)")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    out_path = os.path.join(outdir, fname)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    return out_path

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="BarSeq (5g) vs RNA-seq (LvsD) at T6/T24 + across-time @5g; exports Top-10 per strong quadrant.")
    ap.add_argument("--barseq-file", required=False, default="fitness_defect_scores_barseq_marjan.csv")
    ap.add_argument("--barseq-sheet", required=False, default="fitness_defect_scores", help="Excel sheet (ignored for CSV)")
    ap.add_argument("--gtf", required=False, default="Saccharomyces_cerevisiae.R64-1-1.114.gtf")
    ap.add_argument("--t6_de", required=False, default="MA_T6_L_vs_T6_D.csv")
    ap.add_argument("--t24_de", required=False, default="MA_T24_L_vs_T24_D.csv")
    ap.add_argument("--c_t6_5g", required=False, default="T6, 5g to T6, 0g - MinimalMedia+10%aa")
    ap.add_argument("--c_t24_5g", required=False, default="T24, 5g to T24, 0g - MinimalMedia+10%aa")
    ap.add_argument("--c_time_5g", required=False, default="T24, 5g to T6, 5g - MinimalMedia+10%aa")
    ap.add_argument("--outdir", required=False, default="rnaseq_barseq_5g_comparisons")
    ap.add_argument("--fd_defect", type=float, default=10.0)
    ap.add_argument("--fd_benefit", type=float, default=-10.0)
    ap.add_argument("--label_top", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tables_dir = os.path.join(args.outdir, "top_quadrant_tables")
    os.makedirs(tables_dir, exist_ok=True)

    gm = load_gene_map_from_gtf(args.gtf)

    fd_T6_5g   = load_barseq_fd(args.barseq_file, args.c_t6_5g, sheet_name=args.barseq_sheet)
    fd_T24_5g  = load_barseq_fd(args.barseq_file, args.c_t24_5g, sheet_name=args.barseq_sheet)
    fd_5g_time = load_barseq_fd(args.barseq_file, args.c_time_5g, sheet_name=args.barseq_sheet)

    de_T6  = load_rnaseq_de(args.t6_de, gm)
    de_T24 = load_rnaseq_de(args.t24_de, gm)

    # Within-timepoint @ 5g
    merged_T6  = merge_fd_de(fd_T6_5g, de_T6)
    merged_T24 = merge_fd_de(fd_T24_5g, de_T24)

    # Across-time @ 5g: Δ(LvsD) = (T24 LvsD) - (T6 LvsD)
    d6  = de_T6[["gene","log2FC"]].rename(columns={"log2FC":"log2FC_T6"})
    d24 = de_T24[["gene","log2FC"]].rename(columns={"log2FC":"log2FC_T24"})
    delta = d24.merge(d6, on="gene", how="inner")
    delta["log2FC"] = delta["log2FC_T24"] - delta["log2FC_T6"]
    delta = delta[["gene","log2FC"]]
    merged_time = merge_fd_de(fd_5g_time, delta)

    # Save merged tables
    t6_table     = os.path.join(args.outdir, "merged_T6_5g_vs_T6_LvD.csv")
    t24_table    = os.path.join(args.outdir, "merged_T24_5g_vs_T24_LvD.csv")
    across_table = os.path.join(args.outdir, "merged_5g_T24minusT6_FD_vs_deltaLvD.csv")
    merged_T6.to_csv(t6_table, index=False)
    merged_T24.to_csv(t24_table, index=False)
    merged_time.to_csv(across_table, index=False)

    # ---- QC (off-figure): Spearman summary per comparison ----
    write_qc_stats(args.outdir, "T6_within5g", merged_T6)
    write_qc_stats(args.outdir, "T24_within5g", merged_T24)
    write_qc_stats(args.outdir, "AcrossTime5g_T24minusT6", merged_time)
    print("QC correlation summary:", os.path.join(args.outdir, "qc_correlation_summary.csv"))

    # Plots + Top-10 per strong quadrant tables
    t6_fig = os.path.join(args.outdir, "T6_quadrant_5g.png")
    t24_fig = os.path.join(args.outdir, "T24_quadrant_5g.png")
    across_fig = os.path.join(args.outdir, "AcrossTime_quadrant_5g.png")

    plot_quadrant(
        merged_T6, title="T6: RNA-seq (L vs D) vs BarSeq FD (5g→0g)",
        xlabel="RNA-seq log2FC (T6 L vs T6 D)",
        fd_defect_thresh=args.fd_defect, fd_benefit_thresh=args.fd_benefit,
        label_top=args.label_top, save_path=t6_fig,
        top_tables_dir=tables_dir, prefix_for_tables="T6_within5g",
    )
    plot_quadrant(
        merged_T24, title="T24: RNA-seq (L vs D) vs BarSeq FD (5g→0g)",
        xlabel="RNA-seq log2FC (T24 L vs T24 D)",
        fd_defect_thresh=args.fd_defect, fd_benefit_thresh=args.fd_benefit,
        label_top=args.label_top, save_path=t24_fig,
        top_tables_dir=tables_dir, prefix_for_tables="T24_within5g",
    )
    plot_quadrant(
        merged_time, title="Across-time @5g: Δ(LvsD) vs BarSeq FD (T24,5g → T6,5g)",
        xlabel="RNA-seq Δ log2FC: (T24 LvsD) − (T6 LvsD)",
        fd_defect_thresh=args.fd_defect, fd_benefit_thresh=args.fd_benefit,
        label_top=args.label_top, save_path=across_fig,
        top_tables_dir=tables_dir, prefix_for_tables="AcrossTime5g_T24minusT6",
    )

    # Summary visuals: triptych + heatmap
    triptych_path = plot_quadrant_triptych(
        merged_T6, merged_T24, merged_time, args.outdir,
        fd_defect=args.fd_defect, fd_benefit=args.fd_benefit, label_top=6,
        fname="quadrants_triptych.png"
    )
    print("Triptych saved:", triptych_path)

    heatmap_path = make_topset_heatmap(
        merged_T6, merged_T24, merged_time, args.outdir,
        top_n=max(3, args.label_top), fd_defect=args.fd_defect, fd_benefit=args.fd_benefit,
        fname="heatmap_topStrong.png"
    )
    print("Heatmap saved:", heatmap_path)

    print("Saved:")
    print("  Tables:", t6_table, t24_table, across_table, sep="\n           ")
    print("  Figures:", t6_fig, t24_fig, across_fig, sep="\n           ")
    print("  Extras :", triptych_path, heatmap_path, sep="\n           ")
    print("  Top-10 quadrant tables under:", tables_dir)

if __name__ == "__main__":
    main()
