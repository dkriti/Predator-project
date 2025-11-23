#!/usr/bin/env python3
import re
import csv
import argparse

def parse_emapper(emapper_tsv):
    """
    Read an eggNOG-mapper TSV:
      - Skip any leading comments until the header line
      - Strip a leading '#' from the header if present
      - Use the first column (whatever it's named) as the query ID
    Returns: dict of core_id -> annotation dict
    """
    ann = {}
    with open(emapper_tsv) as f:
        # 1) Find the real header line
        header_line = None
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                raise ValueError(f"No header found in {emapper_tsv}")
            # accept lines starting with "#query" or "query"
            if line.lower().lstrip('#').startswith("query"):
                header_line = line
                break

        # 2) Normalize header (strip '#' and split)
        header_line = header_line.lstrip("#").strip()
        fieldnames = header_line.split("\t")

        # 3) Build a DictReader for the remainder of the file
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter="\t")

        # 4) Identify the first column as our query identifier
        query_col = fieldnames[0]

        for row in reader:
            raw_q = row[query_col].strip()
            if not raw_q or raw_q.startswith("#"):
                continue  # skip blank or spurious comment rows

            core = raw_q.split(".")[0]  # e.g. "g4604.t1" -> "g4604"

            # Clean up numeric placeholders like "3.5.2.9"
            name = row.get("Preferred_name", "")
            if name == "-" or re.fullmatch(r"[\d\.]+", name):
                name = ""

            ann[core] = {
                "gene_name":   name,
                "go_terms":    row.get("GOs", ""),
                "kegg_kos":    row.get("KEGG_KOs", ""),
                "cog_category":row.get("COG_category", "")
            }

    return ann

def parse_attrs(attr_str):
    """
    Convert a GTF attribute column into a dict: {key: value, ...}
    """
    attrs = {}
    for field in attr_str.strip().rstrip(";").split(";"):
        if not field.strip():
            continue
        key, val = field.strip().split(" ", 1)
        attrs[key] = val.strip().strip('"')
    return attrs

def format_attrs(attrs):
    """
    Convert an attribute dict back into a GTF-formatted string
    """
    return "; ".join(f'{k} "{v}"' for k, v in attrs.items()) + ";"

def annotate_gtf(gtf_in, gtf_out, ann):
    """
    Read the input GTF, inject eggNOG and default Augustus fields,
    and write the enhanced GTF to gtf_out.
    """
    with open(gtf_in) as fi, open(gtf_out, "w") as fo:
        for line in fi:
            if line.startswith("#"):
                fo.write(line)
                continue

            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                fo.write(line)
                continue

            feature = cols[2]
            attrs = parse_attrs(cols[8])

            # Attempt to extract the numeric part of gene_id: "sch_g5417" -> "g5417"
            gene_id = attrs.get("gene_id", "")
            m = re.search(r'sch_g(\d+)', gene_id)
            core = f"g{m.group(1)}" if m else None

            info = ann.get(core, {})

            # Inject defaults and eggNOG data
            if feature == "gene":
                attrs.setdefault("gene_biotype", "protein_coding")
                attrs.setdefault("gene_source", "augustus")
                if info.get("gene_name"):
                    attrs["gene_name"] = info["gene_name"]
                for key in ("go_terms", "kegg_kos", "cog_category"):
                    if info.get(key):
                        attrs[key] = info[key]

            elif feature == "transcript":
                attrs.setdefault("transcript_biotype", "protein_coding")
                attrs.setdefault("transcript_source", "augustus")
                if info.get("gene_name"):
                    attrs["transcript_name"] = info["gene_name"]

            cols[8] = format_attrs(attrs)
            fo.write("\t".join(cols) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Enhance a GTF file by injecting eggNOG-mapper annotations."
    )
    parser.add_argument("gtf_in",      help="Input GTF file (e.g. from BRAKER)")
    parser.add_argument("emapper_tsv", help="eggNOG-mapper annotations TSV")
    parser.add_argument("gtf_out",     help="Output annotated GTF")
    args = parser.parse_args()

    ann = parse_emapper(args.emapper_tsv)
    annotate_gtf(args.gtf_in, args.gtf_out, ann)
    print(f"Written enhanced GTF to: {args.gtf_out}")

if __name__ == "__main__":
    main()

