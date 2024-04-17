import pandas as pd
import numpy as np
import os
import subprocess
import logging
from multiprocessing import Pool

from utils.argmanager import *
from utils.helpers import *


DEFAULT_CLOSEST_GENE_COUNT = 3
DEFAULT_THREADS = 4

def get_asb_adastra(chunk, sig_adastra_tf, sig_adastra_celltype):
    mean_asb_es_tf_ref = []
    mean_asb_es_tf_alt = []
    asb_tfs = []

    mean_asb_es_celltype_ref = []
    mean_asb_es_celltype_alt = []
    asb_celltypes = []

    for index,row in chunk.iterrows():
        if index % 1000 == 0:
            print(index)

        local_tf_df = sig_adastra_tf.loc[sig_adastra_tf['variant_id'] == row['variant_id']].copy()
        if len(local_tf_df) > 0:
            mean_asb_es_tf_ref.append(local_tf_df['es_mean_ref'].mean())
            mean_asb_es_tf_alt.append(local_tf_df['es_mean_alt'].mean())
            asb_tfs.append(', '.join(local_tf_df['tf'].unique().tolist()))
        else:
            mean_asb_es_tf_ref.append(np.nan)
            mean_asb_es_tf_alt.append(np.nan)
            asb_tfs.append(np.nan)

        local_celltype_df = sig_adastra_celltype.loc[sig_adastra_celltype['variant_id'] == row['variant_id']].copy()
        if len(local_celltype_df) > 0:
            mean_asb_es_celltype_ref.append(local_celltype_df['es_mean_ref'].mean())
            mean_asb_es_celltype_alt.append(local_celltype_df['es_mean_alt'].mean())
            asb_celltypes.append(', '.join(local_celltype_df['celltype'].unique().tolist()))
        else:
            mean_asb_es_celltype_ref.append(np.nan)
            mean_asb_es_celltype_alt.append(np.nan)
            asb_celltypes.append(np.nan)
            
    chunk['adastra_asb_tfs'] = asb_tfs
    chunk['adastra_mean_asb_effect_size_tf_ref'] = mean_asb_es_tf_ref
    chunk['adastra_mean_asb_effect_size_tf_alt'] = mean_asb_es_tf_alt
    chunk['adastra_asb_celltypes'] = asb_celltypes
    chunk['adastra_mean_asb_effect_size_celltype_ref'] = mean_asb_es_celltype_ref
    chunk['adastra_mean_asb_effect_size_celltype_alt'] = mean_asb_es_celltype_alt
    
    return chunk

def main(args = None):

    if args is None:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        args = fetch_annotation_args()

    if args.add_adastra:
        if not args.add_adastra_tf:
            raise ValueError("ADASTRA TF file (-aatf) is required for ADASTRA annotation")    
        if not args.add_adastra_celltype:
            raise ValueError("ADASTRA celltype file (-aact) is required for ADASTRA annotation")

    variant_scores_file = get_summary_output_file(args.summary_output_dir, args.sample_name)
    peak_path = args.peaks
    tss_path = args.closest_genes

    variant_scores = pd.read_table(variant_scores_file)
    tmp_bed_file_path = f"/tmp/{args.sample_name}.variant_table.tmp.bed"

    if args.schema == "bed":
        if variant_scores['pos'].equals(variant_scores['end']):
            variant_scores['pos'] = variant_scores['pos'] - 1
        variant_scores_bed_format = variant_scores[['chr','pos','end','allele1','allele2','variant_id']].copy()
    else:
        variant_scores_bed_format = variant_scores[['chr','pos','allele1','allele2','variant_id']].copy()
        variant_scores_bed_format['pos']  = variant_scores_bed_format.apply(lambda x: int(x.pos)-1, axis = 1)
        variant_scores_bed_format['end']  = variant_scores_bed_format.apply(lambda x: int(x.pos)+len(x.allele1), axis = 1)
        variant_scores_bed_format = variant_scores_bed_format[['chr','pos','end','allele1','allele2','variant_id']]
        variant_scores_bed_format = variant_scores_bed_format.sort_values(["chr","pos","end"])

    variant_scores_bed_format.to_csv(tmp_bed_file_path,\
                                     sep="\t",\
                                     header=None,\
                                     index=False)

    if args.closest_genes:

        logging.info("Annotating with closest genes")
        closest_gene_count = args.closest_gene_count if args.closest_gene_count else DEFAULT_CLOSEST_GENE_COUNT
        closest_gene_path = f"/tmp/{args.sample_name}.closest_genes.tmp.bed"
        gene_bedtools_intersect_cmd = f"bedtools closest -d -t first -k {closest_gene_count} -a {tmp_bed_file_path} -b {tss_path} > {closest_gene_path}"
        _ = subprocess.call(gene_bedtools_intersect_cmd,\
                            shell=True)

        closest_gene_df = pd.read_table(closest_gene_path, header=None)
        os.remove(closest_gene_path)

        logging.debug(f"Closest genes table:\n{closest_gene_df.shape}\n{closest_gene_df.head()}")

        closest_genes = {}
        gene_dists = {}

        for index,row in closest_gene_df.iterrows():
            if not row[5] in closest_genes:
                closest_genes[row[5]] = []
                gene_dists[row[5]] = []
            closest_genes[row[5]].append(row[9])
            gene_dists[row[5]].append(row[10])

        closest_gene_df = closest_gene_df.rename({5:'variant_id'},axis=1)
        closest_gene_df = closest_gene_df[['variant_id']]

        for i in range(closest_gene_count):
            closest_gene_df[f'closest_gene_{i+1}'] = closest_gene_df['variant_id'].apply(lambda x: closest_genes[x][i] if len(closest_genes[x]) > i else '.')
            closest_gene_df[f'gene_distance_{i+1}'] = closest_gene_df['variant_id'].apply(lambda x: gene_dists[x][i] if len(closest_genes[x]) > i else '.')

        closest_gene_df.drop_duplicates(inplace=True)
        variant_scores = variant_scores.merge(closest_gene_df, on='variant_id', how='left')

    if args.peaks:

        logging.info("Annotating with peak overlap")
        peak_intersect_path = f"/tmp/{args.sample_name}.peak_overlap.tmp.bed"
        print(peak_intersect_path)
        peak_bedtools_intersect_cmd = "bedtools intersect -wa -u -a %s -b %s > %s"%(tmp_bed_file_path, peak_path, peak_intersect_path)
        print(peak_bedtools_intersect_cmd)
        _ = subprocess.call(peak_bedtools_intersect_cmd,\
                            shell=True)

        peak_intersect_df = pd.read_table(peak_intersect_path, header=None)
        os.remove(peak_intersect_path)

        logging.debug(f"Peak overlap table:\n{peak_intersect_df.shape}\n{peak_intersect_df.head()}")

        variant_scores['peak_overlap'] = variant_scores['variant_id'].isin(peak_intersect_df[5].tolist())

    if args.r2:
        logging.info("Annotating with r2")
        r2_ld_filepath = args.r2

        r2_tsv_filepath = f"/tmp/{args.sample_name}.r2.tsv"
        with open(r2_ld_filepath, 'r') as r2_ld_file, open(r2_tsv_filepath, mode='w') as r2_tsv_file:
            # temp=r2_tsv_file.name
            for line in r2_ld_file:
                # Process the line
                line = '\t'.join(line.split())
                # Write the processed line to the output file, no need to specify end='' as '\n' is added explicitly
                r2_tsv_file.write(line + '\n')
            r2_tsv_file.flush()
            
        with open(r2_tsv_filepath, 'r') as r2_tsv_file:
            plink_variants = pd.read_table(r2_tsv_file)
            logging.debug(f"Plink variants table:\n{plink_variants.shape}\n{plink_variants.head()}")

            # Get just the lead variants, which is provided by the user.
            lead_variants = variant_scores[['chr', 'pos', 'variant_id']].copy()
            lead_variants['r2'] = 1.0
            lead_variants['lead_variant'] = lead_variants['variant_id']
            logging.debug(f"Lead variants table:\n{lead_variants.head()}\n{lead_variants.shape}")

            # Get just the ld variants.
            plink_ld_variants = plink_variants[['SNP_A','CHR_B','BP_B','SNP_B','R2']].copy()
            plink_ld_variants.columns = ['lead_variant', 'chr', 'pos', 'variant_id', 'r2']
            plink_ld_variants = plink_ld_variants[['chr', 'pos', 'variant_id', 'r2', 'lead_variant']]
            plink_ld_variants['chr'] = 'chr' + plink_ld_variants['chr'].astype(str)
            plink_ld_variants = plink_ld_variants.sort_values(by=['variant_id', 'r2'], ascending=False).drop_duplicates(subset='variant_id')
            logging.debug(f"Plink LD variants table:\n{plink_ld_variants.shape}\n{plink_ld_variants.head()}")

            all_plink_variants = pd.concat([lead_variants, plink_ld_variants])
            all_plink_variants = all_plink_variants[['variant_id', 'r2', 'lead_variant']]
            all_plink_variants = all_plink_variants.sort_values( by=['variant_id', 'r2'], ascending=False)
            logging.debug(f"All plink variants table:\n{all_plink_variants.shape}\n{all_plink_variants.head()}")

            variant_scores = variant_scores.merge(all_plink_variants,
                on=['variant_id'],
                how='left')
            
    if args.add_adastra and args.add_adastra_tf and args.add_adastra_celltype:
        adastra_tf_file = args.add_adastra_tf
        adastra_celltype_file = args.add_adastra_celltype
        sig_adastra_tf = pd.read_table(adastra_tf_file)
        sig_adastra_celltype = pd.read_table(adastra_celltype_file)

        # Modify both to have a variant_id column, since we don't retrieve their rsids. This takes some extra time, might be worth changing later.
        # variant_id should be <chr>:<pos>:<ref>:<alt>
        sig_adastra_tf['variant_id'] = sig_adastra_tf.apply(lambda x: f"{x['#chr']}:{x['pos']}:{x['ref']}:{x['alt']}", axis=1)
        sig_adastra_celltype['variant_id'] = sig_adastra_celltype.apply(lambda x: f"{x['#chr']}:{x['pos']}:{x['ref']}:{x['alt']}", axis=1)

        logging.debug(f"ADASTRA TF table:\n{sig_adastra_tf.shape}\n{sig_adastra_tf.head()}")
        logging.debug(f"ADASTRA celltype table:\n{sig_adastra_celltype.shape}\n{sig_adastra_celltype.head()}")

        n_threads = args.threads if args.threads else DEFAULT_THREADS
        chunk_size = len(variant_scores) // n_threads
        chunks = np.array_split(variant_scores, len(variant_scores) // chunk_size)

        args_for_starmap = [(chunk, sig_adastra_tf, sig_adastra_celltype) for chunk in chunks]

        with Pool(processes=n_threads) as pool:
            results = pool.starmap(get_asb_adastra, args_for_starmap)

        variant_scores = pd.concat(results)

        pool.close()
        pool.join()

        logging.debug(f"ADASTRA annotations added to variant scores:\n{variant_scores.shape}\n{variant_scores.head()}")

    os.remove(tmp_bed_file_path)

    logging.info(f"Final annotation table:\n{variant_scores.shape}\n{variant_scores.head()}")

    out_file = get_annotation_output_file(args.annotation_output_dir, args.sample_name)
    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)

    logging.info(f"Annotation step completed! Output written to: {out_file}")


if __name__ == "__main__":
    main()
