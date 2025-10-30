#!/usr/bin/env python

import numpy as np
import multiprocessing
# from pymethylation_utils.utils import run_epimetheus
from epymetheus import epymetheus
import os
import sys
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re
import polars as pl
from tqdm import tqdm

def read_fasta(path):
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")    
    
    # Check if the file has a valid FASTA extension
    valid_extensions = ['.fasta', '.fa', '.fna', '.gz']
    if not any(path.endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Unsupported file extension. Please provide a FASTA file with one of the following extensions: {', '.join(valid_extensions)}")
    
    # Check if the file is a gzipped FASTA file
    if path.endswith('.gz'):
        with gzip.open(path, "rt") as handle:  # "rt" mode for reading as text
            contigs = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
                    
    else:
        # Read a regular (uncompressed) FASTA file
        with open(path, "r") as handle:
            contigs = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    return contigs

def get_contig_lengths_in_split(assembly, split_contigs):
    contig_lengths = {}
    for c in split_contigs:
        contig = assembly[c]
        contig_len = len(contig.seq)
        contig_lengths[contig.id] = contig_len

    return contig_lengths
    

def calculate_data_split_methylation(
    contigs,
    contig_lengths,
    pileup_path,
    assembly_path,
    motifs,
    threads,
    min_valid_read_coverage,
    min_valid_cov_to_diff_fraction,
    methylation_value
):

    contig_len_df = pl.DataFrame({
                                     "contig": contig_lengths.keys(),
                                     "length": contig_lengths.values(),
                                 })\
        .with_columns(
            half_length = pl.col("length") // 2
        )

    # Convert contigs to list if it's not already
    contigs_list = list(contigs)
    batch_size = 1000

    # Split contigs into batches
    contig_batches = [contigs_list[i:i + batch_size] for i in range(0, len(contigs_list), batch_size)]

    final_result = None

    # Process each batch with progress bar
    for batch in tqdm(contig_batches, desc="Processing contig batches"):
        batch_meth_features = epymetheus.methylation_pattern(
            pileup=pileup_path,
            assembly=assembly_path,
            motifs = motifs,
            output_type=epymetheus.MethylationOutput.Raw,
            contigs=batch,
            threads=threads,
            min_valid_read_coverage=min_valid_read_coverage,
            min_valid_cov_to_diff_fraction=min_valid_cov_to_diff_fraction,
            allow_assembly_pileup_mismatch = True
        )

        batch_meth_features = batch_meth_features\
            .join(other=contig_len_df, on = "contig", how="left")\
            .with_columns(
                pl.when(pl.col("start") < pl.col("half_length")).then(pl.col("contig").cast(pl.String) + "_1").otherwise(pl.col("contig").cast(pl.String) + "_2").alias("contig"),
                pl.when(pl.col("start") < pl.col("half_length")).then(pl.col("start")).otherwise(pl.col("start") - pl.col("half_length")).alias("start")
            )\
            .with_columns(
                (pl.col("n_modified") / pl.col("n_valid_cov")).alias("fraction_mod")
            )

        if methylation_value == epymetheus.MethylationOutput.Median:
            batch_meth_features = batch_meth_features\
                .group_by(["contig", "motif", "mod_type", "mod_position"])\
                .agg(
                    pl.col("fraction_mod").median().alias("methylation_value"),
                    pl.col("n_valid_cov").mean().alias("mean_read_cov"),
                    pl.col("contig").count().alias("n_motif_obs"),
                )
        elif methylation_value == epymetheus.MethylationOutput.WeightedMean:
            batch_meth_features = batch_meth_features\
                .group_by(["contig", "motif", "mod_type", "mod_position"])\
                .agg(
                    pl.col("n_valid_cov").sum().alias("total_cov"),
                    (pl.col("fraction_mod") * pl.col("n_valid_cov")).alias("weighted_sum"),
                    pl.col("n_valid_cov").mean().alias("mean_read_cov"),
                    pl.col("contig").count().alias("n_motif_obs"),
                )\
                .with_columns(
                    (pl.col("weighted_sum") / pl.col("total_cov")).alias("methylation_value")
                )\
                .drop(["weighted_sum", "total_cov"])
        else:
            raise ValueError

        if final_result is None:
            final_result = batch_meth_features
        else:
            final_result = pl.concat([final_result, batch_meth_features])
        del batch_meth_features

    final_result = final_result\
        .sort(["contig", "motif", "mod_type", "mod_position"])
    return final_result



    
    

def check_files_exist(paths=[]):
    """
    Checks if the given files and directories exist.
    
    Parameters:
    paths (list): List of file paths to check.
    directories (list): List of directory paths to check.
    
    Raises:
    FileNotFoundError: If any of the specified files or directories do not exist.
    """
    for f in paths:
        if not os.path.exists(f):
            raise FileNotFoundError(f"The file {f} does not exist.")


def sort_columns(cols):
    mod_columns = sorted([col for col in cols if "methylation_value" in col], key=lambda x: x.split("_")[-2:])
    nomod_columns = sorted([col for col in cols if "motif_present" in col], key=lambda x: x.split("_")[-2:])
    # Interleave the mod and nomod columns
    sorted_columns = [val for pair in zip(mod_columns, nomod_columns) for val in pair]
    return ["contig"] + sorted_columns  # Keep 'contig' as the first column

def create_methylation_matrix(methylation_features):
    """
    Creates a feature matrix with methylation from motifs-scored or methylation features.
    """
    # check if the methylation features have the required columns
    required_columns = ["contig", "motif", "mod_type",  "mod_position", "methylation_value", "motif_present"]
    if not all(col in methylation_features.columns for col in required_columns):
        raise ValueError(f"Missing required columns in methylation features. Required columns: {', '.join(required_columns)}")
    
    # Calculate mean methylation for each motif
    matrix = methylation_features\
        .with_columns(
            motif_mod = pl.col("motif") + "_" + pl.col("mod_type") + "-" + pl.col("mod_position").cast(pl.String)
        )
    
    
    matrix = matrix.select(["contig", "motif_mod", "methylation_value", "motif_present"])\
        .pivot(
            index = "contig",
            on = "motif_mod",
            values = ["methylation_value", "motif_present"],
            aggregate_function = None,
            maintain_order = True
        )\
        .rename(
            lambda column_name: column_name.replace("motif_mod_", "")
        )\
        .fill_null(0)


    new_columns=sort_columns(matrix.columns)
    matrix = matrix.select(new_columns)
    
    return matrix

def filter_must_links(data_split, max_distance):
    import numpy as np

    # Get methylation feature columns
    methylation_cols = [col for col in data_split.columns if col.startswith("methylation_value")]
    motif_present_cols = [col for col in data_split.columns if col.startswith("motif_present")]

    if not methylation_cols:
        return data_split

    # Extract base contig names (remove _1 or _2 suffix)
    data_with_base = data_split.with_columns([
        pl.col("").str.slice(0, pl.col("").str.len_chars() - 2).alias("base_contig"),
        pl.col("").str.slice(-1, 1).alias("split_num")
    ])

    # Get pairs where both _1 and _2 exist
    contig_counts = data_with_base.group_by("base_contig").len()
    complete_pairs = contig_counts.filter(pl.col("len") == 2)["base_contig"]

    # Filter to only complete pairs
    paired_data = data_with_base.filter(pl.col("base_contig").is_in(complete_pairs))

    # Separate _1 and _2 contigs
    all_cols = ["base_contig"] + methylation_cols + motif_present_cols
    contigs_1 = paired_data.filter(pl.col("split_num") == "1").select(all_cols)
    contigs_2 = paired_data.filter(pl.col("split_num") == "2").select(all_cols)

    # Join to get pairs
    pairs = contigs_1.join(contigs_2, on="base_contig", suffix="_2")

    # Calculate euclidean distance only on methylation values where both halves have motif present
    distances = []
    for row in pairs.iter_rows(named=True):
        valid_methylation_vals_1 = []
        valid_methylation_vals_2 = []

        for meth_col in methylation_cols:
            # Find corresponding motif_present column
            motif_present_col = meth_col.replace("methylation_value", "motif_present")

            # Only include methylation values where both halves have the motif present
            if (motif_present_col in row and
                row[motif_present_col] > 0 and
                row[motif_present_col + "_2"] > 0):
                valid_methylation_vals_1.append(row[meth_col])
                valid_methylation_vals_2.append(row[meth_col + "_2"])

        # Calculate distance only if we have valid methylation values
        if len(valid_methylation_vals_1) > 0:
            vals_1 = np.array(valid_methylation_vals_1)
            vals_2 = np.array(valid_methylation_vals_2)
            distance = np.linalg.norm(vals_1 - vals_2)
        else:
            # If no shared motifs, set distance to infinity (will be filtered out)
            distance = np.inf

        distances.append(distance)

    # Add distances to pairs dataframe
    pairs_with_distance = pairs.with_columns(pl.Series("euclidean_distance", distances))

    # Filter pairs with distance <= max_distance
    valid_pairs = pairs_with_distance.filter(pl.col("euclidean_distance") <= max_distance)["base_contig"]

    # Return original data filtered to valid pairs
    return data_split.filter(
        pl.col("").str.slice(0, pl.col("").str.len_chars() - 2).is_in(valid_pairs)
    )

    



def check_data_file_args(logger, data, data_split, args):
    if data and data_split:
        logger.info("Using provided data and data_split files.")
    elif data or data_split:
        logger.error("Missing data or data_split path. Either both should be provided or none.")
        sys.exit(1)
    else:
        logger.info("Using default data and data_split files. Checking output directory...")
        data = os.path.join(args.output, "data.csv")
        data_split = os.path.join(args.output, "data_split.csv")
    return data, data_split
        

def generate_methylation_features(logger, contig_fasta_path, pileup_path, args, data_path = None, data_split_path = None):
    logger.info("Adding Methylation Features")

    if args.methylation_value == "median":
        args.methylation_value = epymetheus.MethylationOutput.Median
    elif args.methylation_value == "weighted-mean":
        args.methylation_value = epymetheus.MethylationOutput.WeightedMean
    else:
        logger.error("Methylation value required. Must be either 'median' or 'weighted-mean'")
        sys.exit(1)
            
    logger.info("Loading data...")
    
    # Check for the data and data_split file
    data_path, data_split_path = check_data_file_args(logger, data_path, data_split_path, args)
        
    paths = [pileup_path, data_path, data_split_path, contig_fasta_path]
    check_files_exist(paths)
    
    # check if output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.motifs_file:
        motifs_file = pl.read_csv(args.motifs_file, separator="\t")

        # Get the unique motifs
        motifs = motifs_file\
            .select(["motif", "mod_position", "mod_type"])\
            .with_columns(
                motif_mod = pl.col("motif") + "_" + pl.col("mod_type") + "_" + pl.col("mod_position").cast(pl.String)
            )\
            .get_column("motif_mod")\
            .unique()
    elif args.motifs:
        motifs = args.motifs
    else:
        logger.error("No motifs provided. Exiting")
        sys.exit(1)

    if len(motifs) == 0:
        logger.error(f"No motifs found")
        sys.exit(1)
        
    # Load the data
    data = pl.read_csv(data_path)
    data = data\
        .rename({"": "contig"})
    data_split = pl.read_csv(data_split_path)
    data_split = data_split\
        .rename({"": "contig"})
    
    
    # Load the assembly file
    assembly = read_fasta(contig_fasta_path)

    # create splitted assembly
    contigs_to_split = data_split.select("contig").to_pandas()
    contigs_to_split = contigs_to_split["contig"].str.rsplit("_",n=1).str[0].unique()

    contig_lengths_for_splitting = get_contig_lengths_in_split(assembly, contigs_to_split)

    number_of_motifs = len(motifs)
    logger.info(f"Motifs found (#{number_of_motifs}): {motifs}")

    # Run methylation utils
    logger.info("Running epimetheus for whole contigs")
    contig_methylation = epymetheus.methylation_pattern(
        pileup = pileup_path,
        assembly = contig_fasta_path,
        motifs = motifs,
        threads = args.num_process,
        min_valid_read_coverage = args.min_valid_read_coverage,
        # batch_size = 1000,
        min_valid_cov_to_diff_fraction = 0.80,
        allow_assembly_pileup_mismatch = True,
        output_type=args.methylation_value
    )

    contig_methylation.write_csv(
        os.path.join(args.output, "contig_methylation_features.tsv"),
        separator = "\t"
    )
    
    logger.info("Running epimetheus for split contigs")
    contig_split_methylation = calculate_data_split_methylation(
        contigs = contigs_to_split,
        contig_lengths=contig_lengths_for_splitting,
        pileup_path=pileup_path,
        assembly_path=contig_fasta_path,
        motifs =motifs,
        min_valid_read_coverage=args.min_valid_read_coverage,
        min_valid_cov_to_diff_fraction=0.80,
        threads=args.num_process,
        methylation_value = args.methylation_value
    )

    if contig_split_methylation.is_empty():
        logger.error("Failed to generate split contig methylation features. No valid results returned.")
        raise RuntimeError("No valid methylation features generated for split contigs")

    
    contig_split_methylation.write_csv(
        os.path.join(args.output, "contig_split_methylation_features.tsv"),
        separator = "\t"
    )


    contig_methylation = contig_methylation\
        .with_columns(
            pl.when(pl.col("n_motif_obs") > 0).then(1).otherwise(0).alias("motif_present")
        )

    contig_split_methylation = contig_split_methylation\
        .with_columns(
            pl.when(pl.col("n_motif_obs") > 0).then(1).otherwise(0).alias("motif_present")
        )
    
    # Methylation number is median of mean methylation. Filtering is applied for too few motif observations.
    # contig_methylation = contig_methylation.filter((pl.col("N_motif_obs") * pl.col("mean_read_cov")) >= 16)
    # contig_split_methylation = contig_split_methylation.filter((pl.col("N_motif_obs") * pl.col("mean_read_cov")) >= 16)
    contig_methylation = contig_methylation.filter((pl.col("n_motif_obs") >= args.min_motif_observations) & (pl.col("mean_read_cov") >= args.min_valid_read_coverage))
    contig_split_methylation = contig_split_methylation.filter((pl.col("n_motif_obs") >= args.min_motif_observations) & (pl.col("mean_read_cov") >= args.min_valid_read_coverage))

    data_split_methylation_matrix = create_methylation_matrix(
        methylation_features = contig_split_methylation
    )
    
    data_split = data_split\
        .join(
            data_split_methylation_matrix,
            on = "contig",
            how = "left"
        )\
        .rename({"contig": ''})\
        .fill_nan(0.0)\
        .fill_null(0.0)
        
    data_split = filter_must_links(data_split, 1)
    
    data_methylation_matrix = create_methylation_matrix(
        methylation_features=contig_methylation
    ).select(data_split_methylation_matrix.columns)
    
    data = data\
        .join(
            data_methylation_matrix,
            on = "contig",
            how = "left"
        )\
        .rename({"contig": ''})\
        .fill_nan(0.0)\
        .fill_null(0.0)
 
    assert data_split_methylation_matrix.columns == data_methylation_matrix.columns, "methylation columns does not match between data_split_methylation_matrix and data_methylation_matrix"

    try:
        logger.info("Writing to data and data_split files...")
        data_split.write_csv(os.path.join(args.output, "data_split.csv"), separator=",", quote_style='never') 
        data.write_csv(os.path.join(args.output, "data.csv"), separator=",", quote_style='never')
    except Exception as e:
        logger.error(f"An error occurred while writing the output: {e}")
        sys.exit(1)
    
    
def generate_methylation_features_multi(
    logger,
    pileup_paths,
    args,
    sample_list = None,
):
    
    # Check if sample names and pileup name match.
    import copy
    pileup_dict = {}

    for p in pileup_paths:
        if not isinstance(p, str) or not os.path.isfile(p):
            raise ValueError(f"Invalid file: {p}")

        base_name = os.path.basename(p).rsplit(".", 1)[0]

        if base_name in pileup_dict:
            logger.error(f"Duplicate entry detected ({base_name}). Exiting")
            sys.exit(1)

        pileup_dict[base_name] = p

    if not sample_list:
        samples_dir = os.path.join(args.output, "samples")
        sample_list = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))]

    missing_samples = set(sample_list) - set(pileup_dict.keys())
    if missing_samples:
        raise ValueError(f"Missing pileup files for samples: {', '.join(missing_samples)}")

    for sample in sample_list:
        logger.info(f"Finding methylation pattern of: {sample}")
        run_args = copy.copy(args)
        run_args.output = os.path.join(args.output, "samples", sample)
        generate_methylation_features(
            logger = logger,
            contig_fasta_path = os.path.join(args.output, "samples", f"{sample}.fa"),
            pileup_path = pileup_dict[sample],
            args = run_args,
            data_path = os.path.join(run_args.output, "data.csv"),
            data_split_path = os.path.join(run_args.output, "data_split.csv")
        )

