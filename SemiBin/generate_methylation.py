#!/usr/bin/env python

import numpy as np
from multiprocessing import get_context
import multiprocessing
from nanomotif.parallel import update_progress_bar
import os
import sys
import gzip
from Bio import SeqIO
import cProfile
import pstats
import argparse


# os.environ['POLARS_MAX_THREADS'] = '1'
import polars as pl

# IUPAC codes dictionary for sequence pattern matching
iupac_dict = {
    "A": "A", "T": "T", "C": "C", "G": "G",
    "R": "[AG]", "Y": "[CT]", 
    "S": "[GC]", "W": "[AT]", 
    "K": "[GT]", "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
    "N": "[ATCG]"
}


def read_fasta(path, contigs):
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")    
    
    # Check if the file has a valid FASTA extension
    valid_extensions = ['.fasta', '.fa', '.fna', '.gz']
    if not any(path.endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Unsupported file extension. Please provide a FASTA file with one of the following extensions: {', '.join(valid_extensions)}")

    # Create a set from the contigs argument for fast lookup
    contigs_set = set(contigs)
    found_contigs = {}
    
    
    # Check if the file is a gzipped FASTA file
    if path.endswith('.gz'):
        with gzip.open(path, "rt") as handle:  # "rt" mode for reading as text
            for record in SeqIO.parse(handle, "fasta"):
                if record.id in contigs_set:
                    found_contigs[record.id] = str(record.seq)
                    # Check if all needed contigs have been found
                    if set(found_contigs.keys()) == contigs_set:
                        break
    else:
        # Read a regular (uncompressed) FASTA file
        with open(path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if record.id in contigs_set:
                    found_contigs[record.id] = str(record.seq)
                    # Check if all needed contigs have been found
                    if set(found_contigs.keys()) == contigs_set:
                        break
                    
    return found_contigs


def check_files_exist(paths=[], directories=[]):
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
    
    for d in directories:
        if not os.path.exists(d):
            raise FileNotFoundError(f"The directory {d} does not exist.")


def load_data(args, logger):
    """Loads data"""
    # try:
    motifs_scored = pl.read_csv(args.motifs_scored, separator="\t")
    # Remove beta initial 1 from n_nomod
    motifs_scored = motifs_scored\
        .with_columns(
            n_nomod = pl.col("n_nomod") - 1
        )
    
    bin_consensus = pl.read_csv(args.bin_motifs, separator="\t")
    
    data = pl.read_csv(args.data)
    data = data\
        .rename({"": "contig"})
    data_split = pl.read_csv(args.data_split)
    data_split = data_split\
        .rename({"": "contig"})

    # except Exception as e:
    #     logger.error(f"An unexpected error occurred: {e}")
    #     sys.exit(1)  # Exit the program with an error code

    logger.info("Data loaded successfully.")
    return motifs_scored, data, data_split, bin_consensus


def get_contigs(data_split):
    """
    Takes the data split and returns a list of unique contigs.
    """
    contigs_in_split = data_split["contig"].str.split("_").map_elements(lambda x: "_".join(x[:-1]), return_dtype = pl.String).to_list()
    contigs_in_split = list(set(contigs_in_split))  # remove duplicates
    contigs_in_split.sort()  # Sort the list in place
    return contigs_in_split

def get_motifs(motifs_scored, bin_consensus, occurence_cutoff=0.9, min_motif_observations = 8):
    """Extracts and returns unique motifs for each contig."""
    motifs_in_bin_consensus = bin_consensus\
        .select(["motif", "mod_position", "mod_type", "n_mod_bin", "n_nomod_bin"])\
        .with_columns(
            motif_mod = pl.col("motif") + "_" + pl.col("mod_position").cast(pl.String) + "_" + pl.col("mod_type"),
            n_motifs = pl.col("n_mod_bin") + pl.col("n_nomod_bin")
        )\
        .filter(pl.col("n_motifs") >= 700)\
        .get_column("motif_mod")\
        .unique()
    
    # filter motifs based on occurence.
    motifs_scored = motifs_scored\
        .with_columns(
            n_motifs = pl.col("n_mod") + pl.col("n_nomod"),
            motif_mod = pl.col("motif") + "_" + pl.col("mod_position").cast(pl.String) + "_" + pl.col("mod_type")
        )\
        .filter(pl.col("motif_mod").is_in(motifs_in_bin_consensus))
    

    # Total contigs in motifs_scored
    total_contigs_in_motifs = motifs_scored.unique(subset=["contig"]).shape[0]
    
    motif_occurences_in_contigs = motifs_scored\
        .filter(pl.col("n_motifs") >= min_motif_observations)\
        .group_by(["motif", "mod_position", "mod_type"])\
        .agg(
            pl.count("contig").alias("motif_distinct_contigs")
        )\
        .with_columns(
            motif_occurences_per = pl.col("motif_distinct_contigs") / total_contigs_in_motifs
        ).sort("motif_distinct_contigs", descending = True)
        
    motifs = motif_occurences_in_contigs\
        .filter(pl.col("motif_occurences_per") >= occurence_cutoff)\
        .select(["motif", "mod_position", "mod_type"]).unique()
    
    motifs_dict = {}
    
    for mod_type in motifs.get_column("mod_type").unique().to_list():
        motif_types = motifs\
            .filter(pl.col("mod_type") == mod_type)\
            .select(["motif", "mod_position", "mod_type"])\
            .unique()\
            .to_dict()
            
            
        motifs_dict[mod_type] = [f"{motif}_{position}_{mod_type}" for motif, position, mod_type in zip(motif_types["motif"], motif_types["mod_position"], motif_types["mod_type"])]
    
    return motifs_dict


def calculate_contig_methylation_pattern(contig, contig_length, motifs, mod_type, contig_meth_pos):
    
    methylation = pl.DataFrame()
    motif_list = motifs[mod_type]
    contig_meth_pos = {key: contig_meth_pos[key] for key in contig_meth_pos.keys() if key in motif_list}
            
    for motif in contig_meth_pos.keys():
        motif_data = contig_meth_pos[motif]
        index_meth_forward = motif_data["index_meth_fwd"]
        index_nonmeth_forward = motif_data["index_nonmeth_fwd"]
        index_meth_reverse = motif_data["index_meth_rev"]
        index_nonmeth_reverse = motif_data["index_nonmeth_rev"]

        n_mod = [
            len(index_meth_forward[index_meth_forward < (contig_length / 2)]) + len(index_meth_reverse[index_meth_reverse < (contig_length / 2)]),
            len(index_meth_forward[index_meth_forward >= (contig_length / 2)]) + len(index_meth_reverse[index_meth_reverse >= (contig_length / 2)])
        ]
        
        n_nomod = [
            len(index_nonmeth_forward[index_nonmeth_forward < (contig_length / 2)]) + len(index_nonmeth_reverse[index_nonmeth_reverse < (contig_length / 2)]),
            len(index_nonmeth_forward[index_nonmeth_forward >= (contig_length / 2)]) + len(index_nonmeth_reverse[index_nonmeth_reverse >= (contig_length / 2)])
        ]
        
        motif_str = motif.split("_")[0]
        
        methylation_tmp = pl.DataFrame({
                "contig": [f"{contig}_1", f"{contig}_2"],
                "motif": [motif_str, motif_str],
                "mod_type": [mod_type, mod_type],
                "mod_position": [motif.split("_")[-2], motif.split("_")[-2]],
                "n_mod": n_mod,
                "n_nomod": n_nomod
            })
        
        methylation = pl.concat([methylation, methylation_tmp])       
    
    return methylation
    
def worker_function(task, motifs, counter, lock):
    """
    
    """
    contig, contig_length, mod_type, contig_meth_pos = task
    
    try:
        result = calculate_contig_methylation_pattern(
            contig = contig, 
            contig_length=contig_length,
            motifs=motifs,
            mod_type=mod_type,
            contig_meth_pos=contig_meth_pos
        )
        with lock:
            counter.value += 1
        return result
    except:
        with lock:
            counter.value += 1
        return None


def data_split_methylation_parallel(contig_lengths, motifs, motif_index_dir, threads=1):
    """
    Calculate methylation pattern for each contig in the data split in parallel.
    """
    # Create and filter tasks: compile only those tasks with existing index files for each contig and modification type.
    with np.load(os.path.join(motif_index_dir, "motif_positions_combined.npz"), allow_pickle=True) as d:
        data = {key: d[key].item() for key in d.files}
        tasks = [
            (contig, contig_lengths[contig], mod_type, data[contig][mod_type])
                for contig in contig_lengths
                for mod_type in motifs
                if mod_type in data.get(contig, {}) and data[contig][mod_type] is not None
        ]
    
    # tasks = [(contig, contig_lengths[contig], mod_type, os.path.join(motif_index_dir, f"{contig}_{mod_type}_motifs_positions.npz")) for contig in contig_lengths for mod_type in motifs if os.path.exists(os.path.join(motif_index_dir, f"{contig}_{mod_type}_motifs_positions.npz"))]

    # Create a progress manager
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    # Create a pool of workers
    pool = get_context("spawn").Pool(processes=threads)

    # Create a process for the progress bar
    progress_bar_process = multiprocessing.Process(target=update_progress_bar, args=(counter, len(tasks), True))
    progress_bar_process.start()

    # Put them workers to work
    results = pool.starmap(worker_function, [(
        task, 
        motifs,
        counter,
        lock
        ) for task in tasks])
    results = [result for result in results if result is not None] #TODO: Check if this is necessary

    # Close the pool
    pool.close()
    pool.join()

    # Close the progress bar
    progress_bar_process.join()
    
    methylation_pattern = pl.concat(results)
    methylation_pattern = methylation_pattern\
            .sort(["mod_type", "motif", "contig"])
    
    return methylation_pattern
    

def sort_columns(cols):
    mod_columns = sorted([col for col in cols if "n_mod" in col], key=lambda x: x.split("_")[-2:])
    nomod_columns = sorted([col for col in cols if "n_nomod" in col], key=lambda x: x.split("_")[-2:])
    # Interleave the mod and nomod columns
    sorted_columns = [val for pair in zip(mod_columns, nomod_columns) for val in pair]
    return ["contig"] + sorted_columns  # Keep 'contig' as the first column


def create_methylation_matrix(methylation_features, motifs=None, min_motif_observations = 8):
    """
    Creates a feature matrix with methylation from motifs-scored or methylation features.
    """
    # check if the methylation features have the required columns
    required_columns = ["contig", "motif",  "mod_type", "mod_position", "n_mod", "n_nomod"]
    if not all(col in methylation_features.columns for col in required_columns):
        raise ValueError(f"Missing required columns in methylation features. Required columns: {', '.join(required_columns)}")
    
    # Calculate mean methylation for each motif
    matrix = methylation_features\
        .with_columns(
            motif_mod = pl.col("motif") + "_" + pl.col("mod_type") + "-" + pl.col("mod_position").cast(pl.String),
            n_motifs = pl.col("n_mod") + pl.col("n_nomod")
        )\
        .filter(pl.col("n_motifs") >= min_motif_observations)
    
    if motifs:
        matrix = matrix.filter(pl.col("motif_mod").is_in(motifs))
    
    matrix = matrix.select(["contig", "motif_mod", "n_mod", "n_nomod"])\
        .pivot(
            index = "contig",
            columns = "motif_mod",
            values = pl.selectors.starts_with("n_"),
            aggregate_function = None,
            maintain_order = True
        )\
        .rename(
            lambda column_name: column_name.replace("motif_mod_", "")
        )\
        .fill_null(0)


        new_columns=sort_columns(matrix)
        matrix = matrix.select(new_columns)
    
    return matrix


def add_must_links(data, data_split, must_links):
    """
    Processes the must_links file and concatenates the filtered data to the main DataFrame.
    
    Parameters:
    data (pl.DataFrame): The main data DataFrame.
    must_links (pl.DataFrame): DataFrame containing must link pairs with columns ['ml_1', 'ml_2'].
    
    Returns:
    pl.DataFrame: Updated data DataFrame with concatenated must link pairs.
    """
    for ml_1, ml_2 in zip(must_links["ml_1"], must_links["ml_2"]):
        ml_1_data = data.filter(pl.col("") == ml_1).drop(pl.selectors.matches(".*(mean|var).*"))
        ml_2_data = data.filter(pl.col("") == ml_2).drop(pl.selectors.matches(".*(mean|var).*"))
        
        
        if ml_1_data.shape[0] == 0 or ml_2_data.shape[0] == 0:
            print(f"Must link {ml_1} or {ml_2} not found in data")
            continue
        
        assert ml_1_data.shape[0] == 1, f"Must link {ml_1} should have only one row."
        assert ml_2_data.shape[0] == 1, f"Must link {ml_2} should have only one row."
        ml = pl.concat([ml_1_data, ml_2_data], rechunk = True)
        data_split.extend(ml)
    
    return data_split

def check_data_file_args(logger, args):
    if args.data and args.data_split:
        logger.info("Using provided data and data_split files.")
    elif args.data or args.data_split:
        logger.error("Missing data or data_split path. Either both should be provided or none.")
        sys.exit(1)
    else:
        logger.info("Using default data and data_split files. Checking output directory.")
        args.data = os.path.join(args.output, "data.csv")
        args.data_split = os.path.join(args.output, "data_split.csv")
    return args
        


def generate_methylation_features(logger, args):
    logger.info("Adding Methylation Features")    
    logger.info("Loading data...")
    
    # Check for the data and data_split file
    args = check_data_file_args(logger, args)
        
        
    paths = [args.motifs_scored, args.data, args.data_split, args.contig_fasta, args.bin_motifs]
    directories = [args.motif_index_dir]

    check_files_exist(paths, directories)
    
    # check if output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load the data
    logger.info("Loading methylation data...")
    motifs_scored, data, data_split, bin_consensus = load_data(args, logger)
    
    # Get the unique contigs from the data split
    contigs = get_contigs(data_split)
    
    # Load the assembly file
    contig_sequences = read_fasta(args.contig_fasta, contigs)
    
    # Get lenths of the contigs
    contig_lengths = {contig: len(sequence) for contig, sequence in contig_sequences.items()}
    
    # Get the unique motifs
    motifs = get_motifs(
        motifs_scored = motifs_scored, 
        bin_consensus = bin_consensus,
        occurence_cutoff = args.motif_occurence_cutoff,
        min_motif_observations=args.min_motif_observations
    )

    if len(motifs) == 0:
        logger.error(f"No motifs found with --motif-occurence-cutoff {args.motif_occurence_cutoff}, --min-motif-observations {args.min_motif_observations}")
        sys.exit(1)
    number_of_motifs = sum([len(motifs[mod_type]) for mod_type in motifs])
    logger.info(f"Motifs found (#{number_of_motifs}): {motifs}")
    
    # Create methylation matrix for contig_splits
    logger.info(f"Calculating methylation pattern for each contig split using {args.num_process} threads.")
    contig_split_methylation = data_split_methylation_parallel(contig_lengths, motifs, args.motif_index_dir, threads=args.num_process)
    
    data_split_methylation_matrix = create_methylation_matrix(
        methylation_features = contig_split_methylation,
        min_motif_observations = args.min_motif_observations
    )
    
    # extract motfis from the data
    motifs_in_contig_split = contig_split_methylation\
        .with_columns(
            motif = pl.col("motif") + "_" + pl.col("mod_type") + "-" + pl.col("mod_position").cast(pl.String)
        )\
        .get_column("motif")\
        .unique()\
        .to_list()
    
    data_split = data_split\
        .join(
            data_split_methylation_matrix,
            on = "contig",
            how = "left"
        )\
        .rename({"contig": ''})\
        .fill_nan(0.0)\
        .fill_null(0.0)
        
    # Create methylation matrix for all data
    motifs_scored_matrix = create_methylation_matrix(
        methylation_features = motifs_scored, 
        motifs = motifs_in_contig_split,
        min_motif_observations = args.min_motif_observations
    ).select(data_split_methylation_matrix.columns)
    
    data = data\
        .join(
            motifs_scored_matrix,
            on = "contig",
            how = "left"
        )\
        .rename({"contig": ''})\
        .fill_null(0.0)\
        .fill_nan(0.0)
    
    try:
        logger.info("Writing to data and data_split files...")
        data_split.write_csv(os.path.join(args.output, "data_split.csv"), separator=",", quote_style='never') 
        data.write_csv(os.path.join(args.output, "data.csv"), separator=",", quote_style='never')
    except Exception as e:
        print(f"An error occurred while writing the output: {e}")
        sys.exit(1)
    
    
    
