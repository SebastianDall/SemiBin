import polars as pl

def worker_function(contig, methylation_binary, counter, lock):
    """
    Perform a comparison of methylation patterns for a given contig and methylation binary.

    Args:
        contig (str): The contig to compare methylation patterns for.
        methylation_binary (str): The methylation binary to compare against.
        counter (Value): A shared counter object to keep track of the number of worker functions completed.
        lock (Lock): A lock object to ensure thread safety when updating the counter.

    Returns:
        The result of the comparison of methylation patterns for the given contig and methylation binary.
        If an exception occurs during the comparison, None is returned.

    """
    try:
        result = compare_methylation_pattern(
            contig=contig,
            methylation_binary=methylation_binary
        )
        with lock:
            counter.value += 1
        return result
    except:
        with lock:
            counter.value += 1
        return None

def compare_methylation_pattern_multiprocessed(methylation_binary, threads=1):
    """
    Compare the methylation pattern of contigs in parallel using multiple processes.

    Args:
        methylation_binary (DataFrame): A DataFrame containing methylation data.
        threads (int, optional): The number of processes to use for parallel execution. Defaults to 1.

    Returns:
        DataFrame: A DataFrame containing the comparison scores for each contig.

    Raises:
        Exception: If an error occurs during multiprocessing.

    """
    from multiprocessing import get_context
    import multiprocessing
    from nanomotif.parallel import update_progress_bar
    
    # Create batches of contigs
    
    
    contigs = methylation_binary.get_column("contig").unique()
    
    # Create a progress manager
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Create a pool of workers
    pool = get_context("spawn").Pool(processes=threads)

    # Create a process for the progress bar
    progress_bar_process = multiprocessing.Process(target=update_progress_bar, args=(counter, len(contigs), True))
    progress_bar_process.start()

    try:
        results = pool.starmap(worker_function, [
            (contig, methylation_binary, counter, lock) for contig in contigs
        ])
        results = [result for result in results if result is not None]
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        results = []
    finally:
        pool.close()
        pool.join()
        
        progress_bar_process.join()
    
    comparison_score = pl.DataFrame()
    for result in results:
        comparison_score = pl.concat([comparison_score, result])
    
    return comparison_score



def compare_methylation_pattern(contig, methylation_binary):
    """
    Compare the methylation pattern of a given contig with other contigs in a binary methylation dataset.
    
    Args:
        contig (str): The contig to compare.
        methylation_binary (polars.DataFrame): The binary methylation dataset.
        
    Returns:
        polars.DataFrame: A DataFrame containing the comparison results, including the sum of mismatches and the number of comparisons for each pair of contigs.
    """
    import polars as pl
    
    methylation_contig = methylation_binary.filter(pl.col("contig") == contig)
    methylation_sample = methylation_binary\
        .filter(pl.col("contig") != contig)\
        .rename({
            "contig": "contig_compare",
            "is_methylated": "is_methylated_compare",
        })
        
    
    methylation_compare = methylation_contig\
        .join(
            methylation_sample,
            on = ["motif_mod"],
            how = "inner"
        )
    
    # Add the new column to the dataframe
    methylation_compare = methylation_compare.with_columns(
        pl.when(
            pl.col("is_methylated") == pl.col("is_methylated_compare")).then(0).otherwise(1).alias("motif_comparison_score")
    )
    
    # methylation_compare = methylation_compare.with_columns(motif_comparison_score.alias("motif_comparison_score"))
    assert methylation_compare.null_count().get_column("motif_comparison_score")[0] == 0, "There are null values in the motif_comparison_score column"
    
    result = methylation_compare\
        .group_by(["contig", "contig_compare"])\
        .agg(
            pl.sum("motif_comparison_score").alias("sum_mismatches"),
            pl.count("motif_comparison_score").alias("n_comparisons")
        )\
        .sort(["contig", "contig_compare"])
    
    return result

