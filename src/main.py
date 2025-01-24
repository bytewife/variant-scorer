import variant_scoring
import variant_shap
import variant_summary_across_folds
import variant_annotation
import variant_viz
import aggregate
import sys
import logging

import argparse
from os.path import isfile
from utils.helpers import setup_logging, MODEL_ID_COL, FOLD_COL, PEAKS_PATH_COL, SCORE_OUT_PATH_PREFIX_COL, SPLIT_PER_CHROMOSOME_COL, SUMMARIZE_OUT_PATH_COL, ANNOTATE_SUMM_OUT_PATH_COL, AGGREGATE_OUT_PATH_COL, ANNOTATE_AGGR_OUT_PATH_COL

class ClosestElementsAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		# Ensure the attribute exists and is a list
		if not hasattr(namespace, 'closest_n_elements_args'):
			setattr(namespace, 'closest_n_elements_args', [])
		
		closest_elements_file, closest_elements_count, closest_elements_label = values
		
		if not isfile(closest_elements_file):
			parser.error(f"BED file '{closest_elements_file}' does not exist.")
		
		# Append the new values as a tuple
		getattr(namespace, 'closest_n_elements_args').append(
			(closest_elements_file, int(closest_elements_count), closest_elements_label)
		)
		setattr(namespace, self.dest, True)

class ClosestElementsInWindowAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		# Ensure the attribute exists and is a list
		if not hasattr(namespace, 'closest_elements_in_window_args'):
			setattr(namespace, 'closest_elements_in_window_args', [])
		
		closest_elements_window_file, closest_elements_window_size, closest_elements_window_label = values
		
		if not isfile(closest_elements_window_file):
			parser.error(f"BED file '{closest_elements_window_file}' does not exist.")
		
		# Append the new values as a tuple
		getattr(namespace, 'closest_elements_in_window_args').append(
			(closest_elements_window_file, int(closest_elements_window_size), closest_elements_window_label)
		)
		setattr(namespace, self.dest, True)

class JoinTSVsAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		valid_directions = {'left', 'right', 'inner', 'outer'}
		
		# Ensure the attribute exists and is a list
		if not hasattr(namespace, 'join_args'):
			setattr(namespace, 'join_args', [])
		
		tsv_file, label, direction = values
		
		if direction not in valid_directions:
			parser.error(f"Invalid direction '{direction}'. Valid directions are {valid_directions}.")
		
		if not isfile(tsv_file):
			parser.error(f"TSV file '{tsv_file}' does not exist.")
		
		# Append the new values as a tuple
		getattr(namespace, 'join_args').append((tsv_file, label, direction))
		setattr(namespace, self.dest, True)

class AdastraAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, 'adastra_tf_file', values[0])
		setattr(namespace, 'adastra_celltype_file', values[1])
		setattr(namespace, self.dest, True)

class PartitionAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		
		# Set max_partitions and cur_partition attributes
		setattr(namespace, 'max_partitions', values[0])
		setattr(namespace, 'cur_partition', values[1])

		# Check if the first integer is greater than the second
		if namespace.max_partitions <= namespace.cur_partition:
			parser.error(f"The number of partitions must be greater than the index of the current partition for {option_string}.")
		if namespace.cur_partition < 0:
			parser.error(f"The partition index must be >=0 for {option_string}.")

def parse_default_value(val):
	# Check if the value can be converted to an int
	if val.isdigit():
		return int(val)
	# Check if the value can be converted to a float
	try:
		float_val = float(val)
		return float_val
	except ValueError:
		pass
	# Check if the value can be converted to a boolean
	if val in ['True', 'False']:
		return val.lower() == 'true'
	# Return the value as a string if no other conversion applies
	return val



shared_annotate_args = {
	# TODO rename this since this is not just for bool. also rename to like: add-label-using-pandas
	("-pd", "--add-annot-using-pandas"): { "nargs": 2, "metavar": ('LABEL', 'EXPRESSION'), "action": "append", "help": '''Generate a new annotation with values equal to the evaluation of a Pandas expression using `pandas.DataFrame.eval(EXPRESSION)`. For example, `--add-annot-using-pandas is_significant "logfc.mean.pval < 0.01 | jsd.mean.pval < 0.01 & active_allele_quantile.mean > 0.05"`. Can be called multiple times to generate multiple annotations.''' },
	("-py", "--add-annot-using-python"): { "nargs": 2, "metavar": ('LABEL', 'EXPRESSION'), "action": "append", "help": '''Generate a new annotation with values equal to the evaluation of a Python expression using `eval(EXPRESSION)`. For example, `--add-annot-using-python is_significant "((df['logfc.mean.pval'] < 0.01 | (df['jsd.mean.pval'] < 0.01)) & (df['active_allele_quantile.mean'] > 0.05)"`. Can be called multiple times to generate multiple annotations.''' },
	("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
	("-si", "--skip-invalid-inputs"): { "action": "store_true", "help": "Skips annotating missing summary files (if annotate-summ) or aggregate output files (if annotate-aggr)." },
	("-erc", "--expected-row-count"): {"type": int, "help": "The number of lines each summarize output file (if annotate-summ) or aggregate output file (if annotate-aggr) is expected to have."},
	("-se", "--skip-existing-outputs"): { "action": "store_true", "help": "Skips creating existing summarize output files." },
	("-ce", "--add-n-closest-elements"): { "nargs": 3, "metavar": ('BED', 'COUNT', 'LABEL'), "action": ClosestElementsAction, "help": "Adds variant annotations for the N closest elements (such as genes) and their distances to the variant. Takes in (1) a bed file containing gene regions, (2) number of closest genes, and (3) output an optional label. Can be used multiple times to generate multiple annotations. For example, `--add-n-closest-elements hg38.genes.bed 5 gene --add-n-closest-elements hg38.lnRNA.bed 3 lnRNA` will output 5`closest_gene_i` and `closest_gene_i_distance` columns, as well as 3 `closest_lnRNA_i` and `closest_lnRNA_i_distance` columns.", "default": False },
	("-cew", "--add-closest-elements-in-window"): { "nargs": 3, "metavar": ('BED', 'WINDOW_SIZE', 'LABEL'), "action": ClosestElementsInWindowAction, "help": "Add variant annotations for the elements (such as genes) existing up to <WINDOWSIZE> bp away from the variant. Takes in (1) a bed file containing gene regions, (2) the window size, and (3) output an optional label. Can be used multiple times to generate multiple annotations. For example, `--add-closest-elements-in-window hg38.genes.bed 100000 genes` will output genes existing within 100000 bp of each variant to their `genes_within_100000_bp` column.", "default": False },
	("-aa", "--add-adastra"): { "nargs": 2, "metavar": ('ADASTRA_TF_FILE', 'ADASTRA_CELLTYPE_FILE'), "action": AdastraAction, "help": "Annotate with ADASTRA. Provide ADASTRA TF and cell type data files.", "default": False },
	("-ifl", "--invalid-file-log"): {"type": str, "metavar": "PATH", "help": "Store the names of invalid files to the path provided. There are no guarantees about thread safety across partitions, so it is recommended to provide a unique path for each partition."},
	("-j", "--join-tsvs"): { "nargs": 3, "metavar": ('TSV', 'LABEL', 'DIRECTION'), "action": JoinTSVsAction, "help": "Add external annotations by joining variant annotations with external TSVs on the specified labels and direction. Valid directions are 'left', 'right', 'inner', 'outer'. Can be used multiple times to generate multiple annotations. For example, `--join-tsvs 1.tsv variant_id left --join-tsvs 2.tsv variant_id outer --join-tsvs 3.tsv hpo left`.", "default": False},
	# ("-th", "--threads"): { "type": int, "help": "The maximum amount of threads to use, where possible." },
	("-pa", "--partition"): { "type": int, "nargs": 2, "help": "Divides the metadata into n (arg 1) partitions and assigns this process to run with the i-th partition (arg 2). This is 0-indexed, so the first partition has index 0. This partition will then split among the processes provided with --multiprocessing-count.", 'default': [1, 0]},
	("-mp", "--multiprocessing-count"): {"type": int, "default": 1, "help": "The number of processes to use for multiprocessing."},
	("-r2", "--r2"): { "type": str, "help": "Adds r2 annotations. Requires a PLINK .ld file." },
	("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}
subcommand_args = {
	"score": {
		"help": "Generate variant effect predictions (i.e. \"scores\") from a given model architecture.",
		"function": variant_scoring.main,
		"args": {
			("-l", "--variant-list"): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
			("-g", "--genome"): {"type": str, "help": "Genome fasta.", "required": True},
			("-m", "--model-path"): {"type": str, "help": "The path of the model file to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
			("-ma", "--model-architecture"): {"type": str, "help": "The name of the model architecture to be used for predictions. Currently supports: ['bpnet', 'chrombpnet', 'chrombpnet-lite'].", "default": "chrombpnet"},
			("-o", "--score-output-path-prefix"): {"type": str, "help": 'A string that will be prefixed to the outputs of the `score` subcommand, to form a valid path to be written to. Should contain information relevant to the project, subcommand, model, and fold, if relevant. Used in this way: "<output-path-prefix><output-file-suffix>". Example usage: `--score-output-path-prefix /projects/score/adipocytes/fold_0/` will output to paths like /projects/score/adipocytes/fold_0/variant_scores.tsv.', "required": True},
			("-sf", "--score-filename"): {"type": str, "help": "The output score filename to use after the output prefix.", 'default': 'variant_scores.tsv'},
			("-psf", "--predictions-suffix"): {"type": str, "help": "The model predictions file path suffix applied after the output prefix.", 'default': 'variant_predictions.tsv'},
			("-s", "--chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes", "required": True},
			("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
			("-sh", "--shuffled-scores"): {"type": str, "help": "An optional path of pre-computed shuffled scores, which will otherwise be generated by the script."},
			("-ps", "--peak-chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes for peak genome."},
			("-pg", "--peak-genome"): {"type": str, "help": "Genome fasta for peaks."},
			("-pc", "--split-per-chromosome"): {"action": "store_true", "help": "Separates variant scores into separate files for each chromosome."},
			("-b", "--bias"): {"type": str, "help": "Bias model to use for variant scoring."},
			("-dm", "--debug-mode"): {"action": "store_true", "help": "Display allele input sequences."},
			("-bs", "--batch-size"): {"type": int, "default": 512, "help": "Batch size to use for the model."},
			("-p", "--peaks"): {"type": str, "help": "Bed file containing peak regions."},
			("-n", "--num-shuf"): {"type": int, "default": 10, "help": "Number of shuffled scores per SNP."},
			("-ts", "--total-shuf"): {"type": int, "help": "Total number of shuffled scores across all SNPs. Overrides --num_shuf."},
			("-mp", "--max-peaks"): {"type": int, "help": "Maximum number of peaks to use for peak percentile calculation."},
			("-c", "--chrom"): {"type": str, "help": "Only score SNPs in selected chromosome."},
			("-r", "--random-seed"): {"type": int, "default": 1234, "help": "Random seed for reproducibility when sampling."},
			("--no-hdf5",): {"action": "store_true", "help": "Prevents saving detailed predictions in hdf5 file during storing, and using those files during the shap and viz steps. Recommended when the variants list is large (>1,000,000)."},
			("-nc", "--num-chunks"): {"type": int, "default": 10, "help": "Number of chunks to divide SNP file into."},
			("-fo", "--forward-only"): {"action": "store_true", "help": "Run variant scoring only on forward sequence."},
			("-st", "--shap-type"): {"nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate."},
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		},
	},
	"summarize": {
		"help": "Summarize variant scores across folds.",
		"function": variant_summary_across_folds.main,
		"args": {
			("-i", "--score-output-path-prefixes"): { "nargs": '+', "help": "A (space-separated) list of variant score file paths prefixes (each for a different fold) to be summarized together, used in the `score` subcommand. Used like so: `--score-output-paths /projects/score/adipocytes/fold_0/ /projects/score/adipocytes/fold_1/ /projects/score/adipocytes/fold_2/ ...`."},
			("-sf", "--score-filename"): {"type": str, "help": "The output score filename to use after the output prefix.", 'default': 'variant_scores.tsv'},
			("-o", "--summarize-output-path"): { "type": str, "help": 'A string representing the output file path of the `summarize` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--summarize-output-path /projects/summarize/adipocytes/mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/mean.variant_scores.tsv.'},
			("-im", "--input-metadata"): {"type": str, "help": f'A TSV file containing at least these columns for this subcommand: "{SCORE_OUT_PATH_PREFIX_COL}", "{SUMMARIZE_OUT_PATH_COL}", and (optionally) {SPLIT_PER_CHROMOSOME_COL} (with values True or False).'},
			("-sc", "--schema"): { "type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
			("-pa", "--partition"): { "type": int, "nargs": 2, "help": "Divides the metadata into n (arg 1) partitions and assigns this process to run with the i-th partition (arg 2). This is 0-indexed, so the first partition has index 0. This partition will then split among the processes provided with --multiprocessing-count.", 'default': [1, 0]},
			("-p", "--peaks"): {"type": str, "help": "Bed file containing peak regions."},
   		("-pc", "--split-per-chromosome"): {"action": "store_true", "help": "Merges all the variants split across chromosomes from the `score` subcommand for each fold, if provided. If using --input-metadata, you may instead create a 'split-per-chromosome' column with values True or False, for more fine-grained control."},
			("-si", "--skip-invalid-scores"): { "action": "store_true", "help": "Skips summarizing missing score output files." },
			("-se", "--skip-existing-outputs"): { "action": "store_true", "help": "Skips creating existing summarize output files." },
			("-mp", "--multiprocessing-count"): {"type": int, "default": 1, "help": "The number of processes to use for multiprocessing."},
			("-ef", "--expected-folds"): {"type": int, "help": "The expected number of folds per model group."},
			("-erc", "--expected-row-count"): {"type": int, "help": "The number of rows each score output file is expected to have (excluding the header)."},
			("-ifl", "--invalid-file-log"): {"type": str, "metavar": "PATH", "help": "Store the names of invalid files to the path provided. There are no guarantees about thread safety across partitions, so it is recommended to provide a unique path for each partition."},
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		},
	},
	"annotate-summ": {
		"help": "Generate useful annotations for each variant based on summarized variant scores specific to each model/celltype. Shares most of the same annotation capabilities as the `annotate-aggr` subcommand, except those that are specific to each model/celltype, such as whether the variant lies in a peak.",
		"function": variant_annotation.main,
		"args": {**{
			("-i", "--summarize-output-path"): { "type": str, "help": 'A string representing the output file path of the `summarize` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--summarize-output-path /projects/summarize/adipocytes/mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/mean.variant_scores.tsv.'},
			("-o", "--annotate-summ-output-path"): { "type": str, "help": 'A string representing the output file path of the `annotate-summ` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--annotate-summ-output-path /projects/annotate_summ/adipocytes/annotated.mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/annotated.mean.variant_scores.tsv. It is recommended to use --input-metadata instead of this flag.'},
			("-im", "--input-metadata"): {"type": str, "help": f'A TSV file containing at least these three required columns for this subcommand: "{MODEL_ID_COL}", "{SUMMARIZE_OUT_PATH_COL}", "{ANNOTATE_SUMM_OUT_PATH_COL}", and "{PEAKS_PATH_COL}" (if using peaks for annotations).'},
			("-p", "--peaks"): {"nargs": "?", "type": str, "const": True, "default": None, "help": "Adds overlapping peaks annotation 'peak_overlap'. If the --input-metadata option isn't provided (with a 'peaks_path' column), this flag requires a path of the bed file containing peak regions."},
		}, **shared_annotate_args},
	},
	"aggregate": {
		"help": "Merge the per-model annotations into a single workable file, as well as generate annotations based on per-model annotations.",
		"function": aggregate.main,
		"args": {
			("-i", "--annotate-summ-output-paths"): { "nargs": '+', "type": str, "help": 'A (space-separated) list of strings representing the output file path of the `annotate` subcommand. Example usage: `--annotate-summ-output-paths /projects/annotate_summ/adipocytes/annotated.mean.variant_scores.tsv /projects/annotate_summ/cardiomyocytes/annotated.mean.variant_scores.tsv ...`. It is recommended to use --input-metadata instead of this flag.'},
			("-im", "--input-metadata"): {"type": str, "help": f'A TSV file containing at least these three required columns for this subcommand: "{MODEL_ID_COL}", "{ANNOTATE_SUMM_OUT_PATH_COL}", and "{AGGREGATE_OUT_PATH_COL}".'},
			("-o", "--aggregate-output-path"): { "type": str, "help": 'A string representing the output file path of the `aggregate` subcommand. Example usage: `--aggregate-output-path /projects/aggregate/adipocytes/aggregate.variant_scores.tsv`. It is recommended to use --input-metadata instead of this flag.'},
			("-si", "--skip-invalid-inputs"): { "action": "store_true", "help": "Skips processing for invalid annotate-summ output files." },
			("-ifl", "--invalid-file-log"): {"type": str, "metavar": "PATH", "help": "Store the names of invalid files to the path provided. There are no guarantees about thread safety across partitions, so it is recommended to provide a unique path for each partition."},
			("-erc", "--expected-row-count"): {"type": int, "help": "The number of lines each summarize output file (if annotate-summ) or aggregate output file (if annotate-aggr) is expected to have."},
			# ("-pa", "--add-aggregate-annot-with-pandas"): { "nargs": 3, "metavar": ("LABEL", "EXPRESSION", "DEFAULT_VALUE_EXPRESSION"), "action": "append", "help": '''Generate a new annotation equal to the evaluation of a Pandas expression using `pandas.DataFrame.eval(EXPRESSION)`, applied on the resulting aggregate variant annotations. The per-model DataFrame is exposed as `cur`. For example, `-pa sum "sum + @cur_df['sum']" 0` generates a new column named "is_significant" with values True or False.`'''},
			("-py", "--add-aggregate-annot-with-python"): { "nargs": 3, "metavar": ("LABEL", "EXPRESSION", "DEFAULT_VALUE_EXPRESSION"), "action": "append", "help": '''Generate a new annotation equal to the evaluation of a Python expression using `eval(EXPRESSION)`, applied on the resulting aggregate variant annotations. The aggregate DataFrame is exposed as `df` and the per-model DataFrame as `cur_df`. For example, `-py sum "df['sum'] + cur_df['sum']" 0` generates a new column named "is_significant" with values True or False.''' },
			("-sort", "--sort-together"): { "nargs": '+', "type": str, "help": '''Sorts multiple columns together, where each column represents a list of elements. These are inputed as strings describing the columns in the order they're given, delimiter, sort_order (optional, defaults to "asc"), and data type (optional, defaults as "str"). Groups of columns to sort together are provided as separate expressions, such as `--sort-together "col1:, :asc:str|col2:; :desc:int" "col3:,:asc:float|col4:;::str"`. Information describing each column and their delimiter and sort order are separated by ':'. Each of of these column specifications are separated by '|'.''' },
			("-id", "--add-temp-model-id"): {'action': 'store_true', "help": f'Temporarily adds a single-value "{MODEL_ID_COL} column to each TSV iterated through, so that the model ID may be used in annotations.".'},
			# ("-id", "--add-temp-model-id"): {"type": bool, "help": f'Temporarily adds a single-value "{MODEL_ID_COL} column to each TSV iterated through, so that the model ID may be used in annotations.".'},
			# LABEL CONDITION (applied to model-specific to get annotations) AGGREGATION_OPERATION (i.e. sum, concat strings. unsure how to do this. prolly with pandas expression again, like df['col1'] + df['col2'])
			# except sometimes you don't even want condition. Instead, you can have one expression, like aggr_df['new_label'] = aggr_df.get('new_label'] + model_df['new_label']
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		}
	},
	"annotate-aggr": {
		"help": "Generate useful annotations for each variant.",
		"function": variant_annotation.main,
		"args": {**{
			("-i", "--aggregate-output-path"): { "type": str, "help": 'A string representing the output file path of the `aggregate` subcommand. Example usage: `--aggregate-output-path /projects/aggregate/adipocytes/aggregate.variant_scores.tsv`. It is recommended to use --input-metadata instead of this flag.'},
			("-o", "--annotate-aggr-output-path"): { "type": str, "help": 'A string representing the output file path of the `annotate-aggr` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--annotate-aggr-output-path /projects/annotate_aggr/adipocytes/annotated.mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/annotated.mean.variant_scores.tsv. It is recommended to use --input-metadata instead of this flag.'},
			("-im", "--input-metadata"): {"type": str, "help": f'A TSV file containing at least these three required columns for this subcommand: "{MODEL_ID_COL}", "{AGGREGATE_OUT_PATH_COL}", and "{ANNOTATE_AGGR_OUT_PATH_COL}".'},
		}, **shared_annotate_args},
	},
	"shap":
	{ 
		"help": "",
		"function": variant_shap.main,
		"args": {
			("-fdir", "--filter-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
			("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<fold>/variant_scores.tsv.", "required": True},
			("-shdir", "--shap-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script. Used like: <shap-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
			("-g", "--genome"): { "type": str, "help": "Genome fasta." , "required": True},
			("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
			("-s", "--chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes.", "required": True},
			("-li", "--lite"): { "action": "store_true", "help": "Models were trained with chrombpnet-lite."},
			("-dm", "--debug-mode"): { "action": "store_true", "help": "Display allele input sequences."},
			("-bs", "--batch-size"): { "type": int, "default": 10000, "help": "Batch size to use for the model."},
			("-c", "--chrom"): { "type": str, "help": "Only score SNPs in selected chromosome."},
			("-shf", "--shap-filenames"): { "nargs": '+', "help": "A list of file names of shap files to be used to overwrite the otherwise generated index filenames, and will be used like so: <shap-dir>/<shap-filename>.{h5,bw} for each file in the list."},
			("-st", "--shap-type"): { "nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate." },
			("--no-hdf5",): {"action": "store_true", "help": "Prevents saving detailed predictions in hdf5 file during storing, and using those files during the shap and viz steps. Recommended when the variants list is large (>1,000,000)."},
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		},
	},
	"viz": {
		"help": "",
		"function": variant_viz.main,
		"args": {
			("-l", "--variant-list"): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
			("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
			("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<fold>/variant_scores.tsv.", "required": True},
			("-fdir", "--filter-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
			("-shdir", "--shap-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script. Used like: <shap-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
			("-vdir", "--viz-dir"): {"type": str, "help": "The directory that will store the visualization outputs. This directory should already exist.", "required": True},
			("-fscf", "--filter-score-filenames"): { "nargs": '+', "help": "A list of file names of filtered variant score files that will be used to overwrite the otherwise generated index filenames, and will be used like so: <filter-dir>/<filtered-score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used."},
			("-st", "--shap-type"): { "nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate." },
			("-predo", "--predictions-override"): { "type": str, "help": "The name of the variant effect predictions' file. Use this if you want to use a name other than the default."},
			("-shapo", "--shap-override"): { "type": str, "help": "The name of the shap output file. Use this if you want to use a name other than the default."},
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		}
	}
}


def get_args():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='subcommand', help='Subcommands for different steps:')

	for subcommand, subcommand_dict in subcommand_args.items():
		args_dict = subcommand_dict['args']
		subparser = subparsers.add_parser(subcommand, help=subcommand_dict['help'])
		for arg_names, kwargs in args_dict.items():
			subparser.add_argument(*arg_names, **kwargs)

	# Parse and handle errors
	try:
		args, _ = parser.parse_known_args()
		if not vars(args) or args.subcommand is None:
			parser.print_help()
			parser.exit()
	except argparse.ArgumentError:
		parser.print_help()
		parser.exit()

	return args


def cli():
	args = get_args()
	setup_logging(logging.getLogger(), args.verbose)
	# logging.basicConfig(format='%(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
	subcommand_args[args.subcommand]['function'](args)