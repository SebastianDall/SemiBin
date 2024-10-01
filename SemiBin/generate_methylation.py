#!/usr/bin/env python

import numpy as np
from multiprocessing import get_context
import multiprocessing
from nanomotif.parallel import update_progress_bar
from nanomotif.candidate import Motif
import os
import sys
import gzip
from Bio import SeqIO
import re


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

def find_motif_indexes(contig, motif):
    # Find the indices of the motif in the sequence
    regex_motif = re.compile(motif.from_iupac())
    return np.array([m.start() for m in re.finditer(regex_motif, str(contig))]) + motif.mod_position

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
    pileup = pl.scan_csv(args.pileup, separator = "\t", has_header = False, new_columns = [
                             "contig", "start", "end", "mod_type", "score", "strand", "start2", "end2", "color", "N_valid_cov", "percent_modified", "N_modified", "N_canonical", "N_other_mod", "N_delete", "N_fail", "N_diff", "N_nocall"
                         ])
    bin_consensus = pl.read_csv(args.bin_motifs, separator="\t")
    
    data = pl.read_csv(args.data)
    data = data\
        .rename({"": "contig"})
    data_split = pl.read_csv(args.data_split)
    data_split = data_split\
        .rename({"": "contig"})

    logger.info("Data loaded successfully.")
    return pileup, data, data_split, bin_consensus


def find_motif_read_methylation(contig, pileup, motifs, perform_split = False):
    data_split_read_meth = None if not perform_split else pl.DataFrame()

    data_read_meth = pl.DataFrame()

    
    for motif_tup in motifs:
        motif, mod_type = motif_tup
        fwd_indexes = find_motif_indexes(contig.seq, motif)
        rev_indexes = find_motif_indexes(contig.seq, motif.reverse_compliment())

        p_fwd = pileup.filter(pl.col("strand") == "+", pl.col("mod_type") == mod_type, pl.col("start").is_in(fwd_indexes))
        p_rev = pileup.filter(pl.col("strand") == "-", pl.col("mod_type") == mod_type, pl.col("start").is_in(rev_indexes))

        p_con = pl.concat([p_fwd, p_rev])
        if p_con.shape[0] == 0:
            continue

        p_read_meth_counts = p_con\
            .with_columns(
                motif_read_mean = pl.col("N_modified") / pl.col("N_valid_coverage")
            )\
            .group_by("contig", "mod_type")\
            .agg([
                 pl.col("motif_read_mean").median().alias("median")
             ]).with_columns(
                pl.lit(motif.string).alias("motif"),
                pl.lit(motif.mod_position).alias("mod_position"),
                pl.lit(1).alias("motif_present")
            )\
            .drop("motif_read_med")

        data_read_meth= pl.concat([data_read_meth, p_read_meth_counts])


        if perform_split:
            length = len(contig.seq)

            p_fwd_split = p_fwd.with_columns(
                pl.when(pl.col("start") <= (length / 2)).then(pl.lit(contig.id + "_1")).otherwise(pl.lit(contig.id + "_2")).alias("contig")
            )
            p_rev_split = p_rev.with_columns(
                pl.when(pl.col("start") <= (length / 2)).then(pl.lit(contig.id + "_1")).otherwise(pl.lit(contig.id + "_2")).alias("contig")
            )

            p_split_con = pl.concat([p_fwd_split, p_rev_split])
            if p_split_con.shape[0] == 0:
                continue

            p_split_read_meth_counts = p_split_con\
            .with_columns(
                motif_read_mean = pl.col("N_modified") / pl.col("N_valid_coverage")
            )\
            .group_by("contig", "mod_type")\
            .agg([
                 pl.col("motif_read_mean").median().alias("median")
             ]).with_columns(
                pl.lit(motif.string).alias("motif"),
                pl.lit(motif.mod_position).alias("mod_position"),
                pl.lit(1).alias("motif_present")
            )\
            .drop("motif_read_med")
            
            data_split_read_meth= pl.concat([data_split_read_meth, p_split_read_meth_counts])

    return data_read_meth, data_split_read_meth

        

    


def worker_function(task, motifs, counter, lock):
    """
    
    """
    pileup, contig, perform_split = task
    
    try:
        contig_meth, contig_split_meth = find_motif_read_methylation(
            contig = contig,
            pileup = pileup,
            motifs = motifs,
            perform_split = perform_split
        )
        with lock:
          counter.value += 1
        return contig_meth, contig_split_meth
    except:
        with lock:
          counter.value += 1
        return None, None


def find_read_methylation(contigs, pileup, assembly, motifs, logger, threads=1):
    """
    Calculate methylation pattern for each contig in the data split in parallel.
    """
    logger.info("Creating tasks")
    tasks = []
    for contig in contigs.keys():
        subpileup = pileup.filter(pl.col("contig") == contig)
        task = (subpileup, assembly[contig], contigs[contig])
        tasks.append(task)
    logger.info("Tasks done")
    
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
    contig_meth_results = [result[0] for result in results if result[0] is not None]
    contig_split_meth_results = [result[1] for result in results if result[1] is not None]
        
    contig_meth_results = pl.concat(contig_meth_results)
    contig_split_meth_results = pl.concat(contig_split_meth_results)
    # Close the pool
    pool.close()
    pool.join()

    # Close the progress bar
    progress_bar_process.join()
      
    return contig_meth_results, contig_split_meth_results
    

def create_methylation_matrix(methylation_features):
    """
    Creates a feature matrix with methylation from motifs-scored or methylation features.
    """
    # check if the methylation features have the required columns
    required_columns = ["contig", "motif", "mod_type",  "mod_position", "median", "motif_present"]
    if not all(col in methylation_features.columns for col in required_columns):
        raise ValueError(f"Missing required columns in methylation features. Required columns: {', '.join(required_columns)}")
    
    # Calculate mean methylation for each motif
    matrix = methylation_features\
        .with_columns(
            motif_mod = pl.col("motif") + "_" + pl.col("mod_type") + "-" + pl.col("mod_position").cast(pl.String)
        )
    
    
    matrix = matrix.select(["contig", "motif_mod", "n_mod", "n_nomod"])\
        .pivot(
            index = "contig",
            columns = "motif_mod",
            values = ["median", "motif_present"],
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
        
        
    paths = [args.pileup, args.data, args.data_split, args.contig_fasta, args.bin_motifs]
    directories = []

    check_files_exist(paths, directories)
    
    # check if output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load the data
    logger.info("Loading methylation data...")
    pileup, data, data_split, bin_consensus = load_data(args, logger)
    
    
    # Load the assembly file
    assembly = read_fasta(args.contig_fasta)
        
    # Get the unique motifs
    motifs = bin_consensus\
        .select(["motif", "mod_position", "mod_type", "n_mod_bin", "n_nomod_bin"])\
        .with_columns(
            motif_mod = pl.col("motif") + "_" + pl.col("mod_position").cast(pl.String) + "_" + pl.col("mod_type"),
            n_motifs = pl.col("n_mod_bin") + pl.col("n_nomod_bin")
        )\
        .filter(pl.col("n_motifs") >= args.min_motif_observations_bin)\
        .unique(["motif_mod"])

    if len(motifs) == 0:
        logger.error(f"No motifs found")
        sys.exit(1)
    
    motif_list = [(Motif(row[0], row[1]), row[2])for row in motifs.unique(["motif_mod"]).iter_rows()]
    
    number_of_motifs = len(motif_list)
    logger.info(f"Motifs found (#{number_of_motifs}): {motif_list}")

    contigs_in_split = data_split.select("contig").to_pandas()
    contigs_in_split = contigs_in_split["contig"].str.rsplit("_",n=1).str[0].unique()

    contigs = {}
    for c in data.get_column("contig"):
        if c in contigs_in_split:
            contigs[c] = True
        else:
            contigs[c] = False

    pileup = pileup.select(["contig", "start", "strand","mod_type", "N_modified", "N_valid_cov"]).collect()
    
    # Create methylation matrix for contig_splits
    logger.info(f"Calculating methylation pattern for each contig split using {args.num_process} threads.")
    contig_methylation, contig_split_methylation = find_read_methylation(contigs, pileup, assembly, motif_list, threads=args.num_process, logger = logger)
    
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
 
    try:
        logger.info("Writing to data and data_split files...")
        data_split.write_csv(os.path.join(args.output, "data_split.csv"), separator=",", quote_style='never') 
        data.write_csv(os.path.join(args.output, "data.csv"), separator=",", quote_style='never')
    except Exception as e:
        print(f"An error occurred while writing the output: {e}")
        sys.exit(1)
    
    
    
