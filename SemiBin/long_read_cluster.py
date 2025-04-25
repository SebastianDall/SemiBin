import os
import torch
import numpy as np
import polars as pl
from .utils import cal_num_bins, get_marker, write_bins, normalize_kmer_motif_features
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from collections import defaultdict
from scipy.sparse import save_npz
####
from sklearn.preprocessing import normalize
import time
import functools
from sklearn.cluster._kmeans import euclidean_distances, stable_cumsum, KMeans, check_random_state, row_norms
from typing import List, Union, Optional, Dict
import gzip
import scipy.sparse as sp
import multiprocessing
import hnswlib
import sys
import leidenalg
from igraph import Graph
import pandas as pd
from .unitem_markers import Markers
from .unitem_profile import Profile, make_sure_path_exists
# from .unitem_common import read_bins
from .filter_small_bins import filter_small_bins
import copy
####

def get_best_bin(results_dict, contig_to_marker, namelist, contig_dict, minfasta):

    # There is room for improving the loop below to avoid repeated computation
    # but it runs very fast in any case
    for max_contamination in [0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        max_F1 = 0
        weight_of_max = 1e9
        max_bin = None

        for res_labels in results_dict.values():
            res = defaultdict(list)
            for label, name in zip(res_labels, namelist):
                if label != -1:
                    res[label].append(name)
            for bin_contig in res.values():
                cur_weight = sum(len(contig_dict[contig]) for contig in bin_contig)
                if cur_weight < minfasta:
                    continue
                marker_list = []
                for contig in bin_contig:
                    marker_list.extend(contig_to_marker[contig])
                if len(marker_list) == 0:
                    continue
                recall = len(set(marker_list)) / 107
                contamination = (len(marker_list) - len(set(marker_list))) / len(marker_list)
                if contamination <= max_contamination:
                    F1 = 2 * recall * (1 - contamination) / (
                                recall + (1 - contamination))
                    if F1 > max_F1:
                        max_F1 = F1
                        weight_of_max = cur_weight
                        max_bin = bin_contig
                    elif F1 == max_F1 and cur_weight <= weight_of_max:
                        weight_of_max = cur_weight
                        max_bin = bin_contig
        if max_F1 > 0: # if there is a bin with F1 > 0
            return max_bin


def cluster_long_read(logger, model, data, device, is_combined,
            n_sample, out, contig_dict, *, binned_length, args,
            minfasta, features_data):
    import pandas as pd
    from .utils import norm_abundance
    contig_list = data.index.tolist()

    if features_data['motif']:
        train_data_motif_present = data[features_data['motif_present']].values

    if not is_combined:
        if not features_data["motif"]:
            train_data_input = data.values[:, features_data["kmer"]]

        else:
            train_data_input = data[features_data["kmer"] + features_data["motif"]].values
            train_data_input, _ = normalize_kmer_motif_features(train_data_input, train_data_input)
            # train_data_input = np.concatenate((train_data_input, train_data_motif_present), axis = 1)
    else:
        train_data_input = data
        if norm_abundance(train_data_input, features_data):
            train_data_seq = train_data_input[features_data["kmer"] + features_data["motif"]].values
            if features_data["motif"]:
                train_data_seq, _ = normalize_kmer_motif_features(train_data_seq, train_data_seq)
                # train_data_seq = np.concatenate((train_data_seq, train_data_motif_present), axis = 1)

            train_data_depth = train_data_input[features_data["depth"]].values
            from sklearn.preprocessing import normalize
            train_data_depth = normalize(train_data_depth, axis=1, norm='l1')
            train_data_input = np.concatenate((train_data_seq, train_data_depth), axis=1)

    # Get the embeddings
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(train_data_input).to(device)
        embedding = model.embedding(x.float()).detach().cpu().numpy()

    ####################################################################################################################
    # The COMEBIN method doesn't use the log term
    # Lines 344-347
    length_weight = np.array([len(contig_dict[name]) for name in contig_list])
    # ---
    # # Define the contig weights as the log10 of contig lengths, it is used in the DBSCAN algorithm
    # length_weight = np.array([len(contig_dict[name]) for name in contig_list])
    # if len(features_data["motif"]) > 0:
    #     length_weight = np.log10(length_weight)
    ####################################################################################################################

    # Get the embeddings
    if not is_combined:
        depth = data[features_data["depth"]].values.astype(np.float32)
        mean_index = [2 * temp for temp in range(n_sample)]
        depth = depth[:, mean_index]
        embedding_new = np.concatenate((embedding, np.log(depth)), axis=1)
    else:
        embedding_new = embedding

    # We don't need the following lines
    '''
    # Create a DataFrame to save the embeddings
    embedding_df = pd.DataFrame(embedding_new, index=contig_list)

    # Save the embeddings to a CSV file
    embedding_df.to_csv(os.path.join(out, 'embedding_space.csv'))
    '''

    ####################################################################################################################
    # The COMEBIN method has normalization option and the default is set to False so I added but not used.
    # Lines 354 - 357
    not_l2normaize = True
    if not not_l2normaize:
        normalize(embedding_new, norm='l2', axis=1)
    ####################################################################################################################


    ####################################################################################################################
    num_threads = args.num_process

    # Here minimum contig length, args.min_len, is only used in the file names so the actual value is not important
    seed_num_backup = gen_seed(
        logger, args.contig_fasta, num_threads, args.min_len, marker_name="bacar_marker", quarter="2quarter"
    )
    ####################################################################################################################


    ####################################################################################################################
    cluster_num = 0  # Default value used in the COMEBIN method
    contig_file = args.contig_fasta
    namelist = contig_list
    output_path = os.path.join(out, 'cluster_res')
    os.makedirs(os.path.join(out, 'cluster_res'), exist_ok=True)
    norm_embeddings = embedding_new  # Define the embeddings
    seed_file = args.contig_fasta + f".bacar_marker.2quarter_lencutoff_{args.min_len}.seed"

    seed_namelist = pd.read_csv(seed_file, header=None, sep='\t', usecols=range(1)).values[:, 0]
    seed_num = len(np.unique(seed_namelist))

    mode = 'weight_seed_kmeans'
    logger.info(
        "Run weighted seed k-means for obtaining the SCG information of the contigs within "
        "a manageable time during the final step."
    )
    bin_nums = [seed_num]

    if cluster_num:
        bin_nums.append(cluster_num)

    logger.info("Bin_numbers:\t" + str(bin_nums))
    for k in bin_nums:
        logger.info(k)
        seed_kmeans_full(logger, contig_file, namelist, output_path, norm_embeddings, k, mode, length_weight, seed_file)
    ####################################################################################################################


    ####################################################################################################################
    num_workers = args.num_process
    seed_file = seed_file

    ##### #########  hnswlib_method
    parameter_list = [1, 5, 10, 30, 50, 70, 90, 110]
    bandwidth_list = [0.05, 0.1, 0.15, 0.2, 0.3]
    partgraph_ratio_list = [50, 100, 80]
    max_edges_list = [100]
    for max_edges in max_edges_list:
        # Run the fast approximate nearest neighbor search
        # |--> ef parameter controls index search speed/build speed tradeoff
        p = fit_hnsw_index(logger, norm_embeddings, num_workers, ef=max_edges * 10)
        seed_bacar_marker_idx = gen_seed_idx(seed_file, contig_id_list=namelist)
        initial_list = list(np.arange(len(namelist)))
        is_membership_fixed = [i in seed_bacar_marker_idx for i in initial_list]

        time_start = time.time()
        ann_neighbor_indices, ann_distances = p.knn_query(norm_embeddings, max_edges + 1, num_threads=num_workers)
        # ann_distances is l2 distance's square
        time_end = time.time()
        logger.info('knn query time cost:\t' + str(time_end - time_start) + "s")

        with multiprocessing.Pool(num_workers) as multiprocess:
            for partgraph_ratio in partgraph_ratio_list:
                for bandwidth in bandwidth_list:
                    for para in parameter_list:
                        output_file = os.path.join(output_path, 'Leiden_bandwidth_' + str(
                            bandwidth) + '_res_maxedges' + str(max_edges) + 'respara_' + str(
                            para) + '_partgraph_ratio_' + str(partgraph_ratio) + '.tsv')

                        if not (os.path.exists(output_file)):
                            multiprocess.apply_async(run_leiden, (
                            logger, output_file, namelist, ann_neighbor_indices, ann_distances, length_weight, max_edges,
                            norm_embeddings,
                            bandwidth, 'l2', initial_list, is_membership_fixed,
                            para, partgraph_ratio))

            multiprocess.close()
            multiprocess.join()
        logger.info('multiprocess Done')
    ####################################################################################################################


    ####################################################################################################################
    seed_num = seed_num_backup
    num_threads = args.num_process
    bac_mg_table = False
    ar_mg_table = False
    contig_file = contig_file
    output_folder = out #os.path.join(out, 'cluster_res')
    os.makedirs(output_folder, exist_ok=True)
    run_get_final_result(logger, bac_mg_table, ar_mg_table, contig_file, output_folder, seed_num, num_threads, ignore_kmeans_res=True)
    ####################################################################################################################
    '''
    import tempfile
    with tempfile.TemporaryDirectory() as tdir:
        cfasta = os.path.join(tdir, 'concatenated.fna')
        with open(cfasta, 'wt') as concat_out:
            for h in contig_list:
                concat_out.write(f'>{h}\n{contig_dict[h]}\n')

            # :TODO
            # Estimate number of bins from a FASTA file
            seeds = cal_num_bins(
                cfasta,
                binned_length,
                args.num_process,
                output=out,
                orf_finder=args.orf_finder,
                prodigal_output_faa=args.prodigal_output_faa)

            # :TODO
            contig2marker = get_marker(
                os.path.join(out, 'markers.hmmout'), orf_finder=args.orf_finder,
                min_contig_len=binned_length, fasta_path=cfasta, contig_to_marker=True
            )

    output_bin_path = os.path.join(out, 'output_bins')

    logger.debug('Running DBSCAN.')
    dist_matrix = kneighbors_graph(
        embedding_new,
        n_neighbors=min(200, embedding_new.shape[0] - 1),
        mode='distance',
        p=2,
        n_jobs=args.num_process
    )

    # Save distance matrix
    save_npz(os.path.join(out, "dist_matrix.npz"), dist_matrix)

    # Run DBSCAN
    DBSCAN_results_dict = {}
    eps_values = [0.01] + [round(x, 2) for x in np.arange(0.05, 0.6, 0.05)]
    for eps_value in eps_values:
        dbscan = DBSCAN(eps=eps_value, min_samples=5, n_jobs=args.num_process, metric='precomputed')
        dbscan.fit(dist_matrix, sample_weight=length_weight)
        labels = dbscan.labels_
        DBSCAN_results_dict[eps_value] = labels.tolist()

    
    # Prepare DataFrame for saving results
    results_df = pd.DataFrame({'Contig': contig_list})

    # Add cluster labels for each eps value to the DataFrame
    for eps_value in eps_values:
        results_df[f'Cluster_Label_eps_{eps_value}'] = DBSCAN_results_dict[eps_value]

    # Save results to CSV
    results_df.to_csv(os.path.join(out,'dbscan_results_multiple_eps.csv'), index=False)
    results_df = pl.DataFrame(results_df)
    
    logger.debug('Integrating results.')

    extracted = []
    initial_cluster_dict = {k: v for k, v in DBSCAN_results_dict.items() if 0.01 <= k <= 0.55}
    # while the sum of contigs is higher than the smallest bin allowed continue to find bins
    while sum(len(contig_dict[contig]) for contig in contig_list) >= minfasta:
        # If there is only one contig left, add it to the extracted list
        if len(contig_list) == 1:
            extracted.append(contig_list)
            break
        
        
        max_bin = get_best_bin(initial_cluster_dict, contig2marker, contig_list, contig_dict, minfasta)

        if not max_bin:
            break

        extracted.append(max_bin)
        for temp in max_bin:
            temp_index = contig_list.index(temp)
            contig_list.pop(temp_index)
            for eps_value in DBSCAN_results_dict:
                DBSCAN_results_dict[eps_value].pop(temp_index)
        
        
    contig2ix = {}
    for i, cs in enumerate(extracted):
        for c in cs:
            contig2ix[c] = i
    namelist = data.index.tolist()
    contig_labels = [contig2ix.get(c, -1) for c in namelist]
    written = write_bins(namelist, contig_labels,
                    output_bin_path, contig_dict,
                    minfasta=minfasta,
                    output_tag=args.output_tag,
                    output_compression=args.output_compression)
    logger.info(f'Number of bins: {len(written)}')
    written.to_csv(os.path.join(out, 'bins_info.tsv'), index=False,
                   sep='\t')
    pd.DataFrame({'contig': namelist, 'bin': contig_labels}).to_csv(
            os.path.join(out, 'contig_bins.tsv'), index=False, sep='\t')
    '''
    logger.info('Finished binning.')


# The following codes were taken from the COMEBIN implementation
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def gen_seed(
        logger, contig_file: str, threads: int, contig_length_threshold: int, marker_name: str = "marker",
        quarter: str = "3quarter"
):
    """
    Generate seed sequences from contigs using FragGeneScan, HMMsearch, and custom markers.

    :param contig_file: Path to the input contig file.
    :param threads: The number of threads to use for processing.
    :param contig_length_threshold: The contig length threshold.
    :param marker_name: The marker name (default: "marker").
    :param quarter: The quarter identifier (default: "3quarter").
    :return: The number of candidate seeds generated.
    """
    fragScanURL = 'run_FragGeneScan.pl'

    hmmExeURL = 'hmmsearch'
    markerExeURL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auxiliary', 'test_getmarker_' + quarter + '.pl')
    markerURL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auxiliary', marker_name + '.hmm')
    seedURL = contig_file + "." + marker_name + "." + quarter + "_lencutoff_" + str(contig_length_threshold) + ".seed"
    fragResultURL = contig_file + ".frag.faa"
    hmmResultURL = contig_file + '.' + marker_name + ".hmmout"

    if not (os.path.exists(fragResultURL)):
        fragCmd = fragScanURL + " -genome=" + contig_file + " -out=" + contig_file + ".frag -complete=0 -train=complete -thread=" + str(
            threads) + " 1>" + contig_file + ".frag.out 2>" + contig_file + ".frag.err"
        logger.info("exec cmd: " + fragCmd)
        os.system(fragCmd)

    if os.path.exists(fragResultURL):
        if not (os.path.exists(hmmResultURL)):
            hmmCmd = hmmExeURL + " --domtblout " + hmmResultURL + " --cut_tc --cpu " + str(
                threads) + " " + markerURL + " " + fragResultURL + " 1>" + hmmResultURL + ".out 2>" + hmmResultURL + ".err"
            logger.info("exec cmd: " + hmmCmd)
            os.system(hmmCmd)

        if os.path.exists(hmmResultURL):
            if not (os.path.exists(seedURL)):
                markerCmd = markerExeURL + " " + hmmResultURL + " " + contig_file + " " + str(
                    contig_length_threshold) + " " + seedURL
                logger.info("exec cmd: " + markerCmd)
                os.system(markerCmd)

            if os.path.exists(seedURL):
                candK = file_len(seedURL)
            else:
                logger.info("markerCmd failed! Not exist: " + markerCmd)
                candK = 0
        else:
            logger.info("Hmmsearch failed! Not exist: " + hmmResultURL)
            sys.exit()
    else:
        logger.info("FragGeneScan failed! Not exist: " + fragResultURL)
        sys.exit()
    return candK

def seed_kmeans_full(
        logger, contig_file: str, namelist: List[str], out_path: str, X_mat: np.ndarray, bin_number: int,
        prefix: str, length_weight: np.ndarray, seed_bacar_marker_url: str
):
    """
    Perform weighted seed-kmeans clustering with specified parameters.

    Parameters:
    :param contig_file: The path to the contig file.
    :param namelist: A list of contig names.
    :param out_path: The output path for saving results.
    :param X_mat: The input data matrix for clustering.
    :param bin_number: The number of bins (clusters) to create.
    :param prefix: A prefix to be added to the output file names.
    :param length_weight: The weights for contig lengths.
    :param seed_bacar_marker_url: The path to the seed markers used for initialization.

    :return: None

    This function performs weighted seed-based k-means clustering on the input data using specified parameters and saves the results.
    """
    out_path = os.path.join(out_path, prefix)
    seed_bacar_marker_idx = gen_seed_idx(seed_bacar_marker_url, contig_id_list=namelist)
    time_start = time.time()
    # run seed-kmeans; length weight
    output_temp = out_path + '_k_' + str(bin_number) + '_result.tsv'
    if not (os.path.exists(output_temp)):
        km = KMeans(
            n_clusters=bin_number, random_state=7, algorithm="full",
            init=functools.partial(partial_seed_init, seed_idx=seed_bacar_marker_idx)
        )
        km.fit(X_mat, sample_weight=length_weight)
        idx = km.labels_
        save_result(idx, output_temp, namelist)

        gen_bins_from_tsv(contig_file, output_temp, output_temp+'_bins')

        time_end = time.time()
        logger.info("Running weighted seed-kmeans cost:\t"+str(time_end - time_start) + 's.')

def gen_seed_idx(seedURL: str, contig_id_list: List[str]) -> List[int]:
    """
    Generate a list of indices corresponding to seed contig IDs from a given URL.

    :param seedURL: The URL or path to the file containing seed contig names.
    :param contig_id_list: List of all contig IDs to match with the seed contig names.
    :return: List[int]
    """
    seed_list = []
    with open(seedURL) as f:
        for line in f:
            if line.rstrip('\n') in contig_id_list:
                seed_list.append(line.rstrip('\n'))
    name_map = dict(zip(contig_id_list, range(len(contig_id_list))))
    seed_idx = [name_map[seed_name] for seed_name in seed_list]
    return seed_idx

# change from sklearn.cluster.kmeans
def partial_seed_init(X, n_clusters: int, random_state, seed_idx, n_local_trials=None) -> np.ndarray:
    """
    Partial initialization of KMeans centers with seeds from seed_idx.

    Parameters:
    :param X: Features.
    :param n_clusters: The number of clusters.
    :param random_state: Determines random number generation for centroid initialization. Use an int for reproducibility.
    :param seed_idx: Indices of seed points for initialization.
    :param n_local_trials: The number of local seeding trials. Default is None.

    Returns:
    :return centers (ndarray): The initialized cluster centers.

    This function initializes a KMeans clustering by partially seeding the centers with provided seeds.
    It is a modification of the KMeans initialization algorithm.
    """
    random_state = check_random_state(random_state)
    x_squared_norms = row_norms(X, squared=True)

    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly

    center_id = seed_idx[0]

    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)

    for c, center_id in enumerate(seed_idx[1:], 1):
        if sp.issparse(X):
            centers[c] = X[center_id].toarray()
        else:
            centers[c] = X[center_id]
        closest_dist_sq = np.minimum(closest_dist_sq,
                                     euclidean_distances(
                                         centers[c, np.newaxis], X, Y_norm_squared=x_squared_norms,
                                         squared=True))
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(len(seed_idx), n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

def gen_bins_from_tsv(fastafile, resultfile, outputdir):
    # read fasta file
    print("Processing file:\t{}".format(fastafile))
    sequences = {}
    if fastafile.endswith("gz"):
        with gzip.open(fastafile, 'r') as f:
            for line in f:
                line = str(line, encoding="utf-8")
                if line.startswith(">"):
                    if " " in line:
                        seq, others = line.split(' ', 1)
                        sequences[seq] = ""
                    else:
                        seq = line.rstrip("\n")
                        sequences[seq] = ""
                else:
                    sequences[seq] += line.rstrip("\n")
    else:
        with open(fastafile, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    if " " in line:
                        seq, others = line.split(' ', 1)
                        sequences[seq] = ""
                    else:
                        seq = line.rstrip("\n")
                        sequences[seq] = ""
                else:
                    sequences[seq] += line.rstrip("\n")
    print("Reading Map:\t{}".format(resultfile))
    dic = {}
    with open(resultfile, "r") as f:
        for line in f:
            contig_name, cluster_name = line.strip().split('\t')
            try:
                dic[cluster_name].append(contig_name)
            except:
                dic[cluster_name] = []
                dic[cluster_name].append(contig_name)
    print("Writing bins:\t{}".format(outputdir))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    bin_name = 0
    for _, cluster in dic.items():
        binfile = os.path.join(outputdir, "{}.fa".format(bin_name))
        with open(binfile, "w") as f:
            for contig_name in cluster:
                contig_name = ">" + contig_name
                try:
                    sequence = sequences[contig_name]
                except:
                    bin_name += 1
                    continue
                f.write(contig_name + "\n")
                f.write(sequence + "\n")
                bin_name += 1

def save_result(result, filepath, namelist):
    filedir, filename = os.path.split(filepath)
    if not filename:
        filename = "result.tsv"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    f = open(filepath, 'w')
    for contigIdx in range(len(result)):
        f.write(namelist[contigIdx] + "\t" + str(result[contigIdx].item(0)) + "\n")
    f.close()

def fit_hnsw_index(
        logger, features, num_threads, ef: int = 100, M: int = 16, space: str = 'l2', save_index_file: bool = False
) -> hnswlib.Index:
    """
    Fit an HNSW index with the given features using the HNSWlib library; Convenience function to create HNSW graph.

    :param logger: The logger object for logging messages.
    :param features: A list of lists containing the embeddings.
    :param ef: The ef parameter to tune the HNSW algorithm (default: 100). |--> controls index search speed/build speed tradeoff
    :param M: The M parameter to tune the HNSW algorithm (default: 16).
    :param space: The space in which the index operates (default: 'l2'). |--> this is the metric
    :param save_index_file: The path to save the HNSW index file (optional).

    :return: The HNSW index created using the given features.

    This function fits an HNSW index to the provided features, allowing efficient similarity search in high-dimensional spaces.
    """

    time_start = time.time()
    num_elements = len(features)
    labels_index = np.arange(num_elements)
    EMBEDDING_SIZE = len(features[0])

    # Declaring index
    # possible space options are l2, cosine or ip
    p = hnswlib.Index(space=space, dim=EMBEDDING_SIZE)

    # Initing index - the maximum number of elements should be known
    p.init_index(max_elements=num_elements, ef_construction=ef, M=M)

    # Element insertion
    int_labels = p.add_items(features, labels_index, num_threads=num_threads)

    # Controlling the recall by setting ef
    # ef should always be > k
    p.set_ef(ef)

    # If you want to save the graph to a file
    if save_index_file:
        p.save_index(save_index_file)
    time_end = time.time()
    logger.info('Time cost:\t' +str(time_end - time_start) + "s")
    return p

def run_leiden(
        logger, output_file: str, namelist: List[str], ann_neighbor_indices: np.ndarray, ann_distances: np.ndarray,
        length_weight: List[float], max_edges: int, norm_embeddings: np.ndarray,
        bandwidth: float = 0.1, lmode: str = 'l2', initial_list: Optional[List[Union[int, None]]] = None,
        is_membership_fixed: Optional[bool] = None, resolution_parameter: float = 1.0, partgraph_ratio: int = 50
):
    """
    Run Leiden community detection algorithm and save the results.

    :param output_file: The path to the output file.
    :param namelist: A list of contig names.
    :param ann_neighbor_indices: Array of ANN neighbor indices.
    :param ann_distances: Array of ANN distances.
    :param length_weight: List of length weights.
    :param max_edges: Maximum number of edges.
    :param norm_embeddings: Array of normalized embeddings.
    :param bandwidth: Bandwidth parameter (default: 0.1).
    :param lmode: Distance mode ('l1' or 'l2', default: 'l2').
    :param initial_list: Initial membership list (default: None).
    :param is_membership_fixed: Whether membership is fixed (default: None).
    :param resolution_parameter: Resolution parameter (default: 1.0).
    :param partgraph_ratio: Partition graph ratio (default: 50).

    :return: None
    """

    sources = np.repeat(np.arange(len(norm_embeddings)), max_edges)
    targets_indices = ann_neighbor_indices[:,1:]
    targets = targets_indices.flatten()
    wei = ann_distances[:,1:]
    wei = wei.flatten()

    dist_cutoff = np.percentile(wei, partgraph_ratio)
    save_index = wei <= dist_cutoff

    sources = sources[save_index]
    targets = targets[save_index]
    wei = wei[save_index]

    if lmode == 'l1':
        wei = np.sqrt(wei)
        wei = np.exp(-wei / bandwidth)

    if lmode == 'l2':
        wei = np.exp(-wei / bandwidth)

    index = sources > targets
    sources = sources[index]
    targets = targets[index]
    wei = wei[index]
    vcount = len(norm_embeddings)
    edgelist = list(zip(sources, targets))
    g = Graph(vcount, edgelist)

    res = leidenalg.RBERVertexPartition(g,
                                        weights=wei, initial_membership = initial_list,
                                        resolution_parameter = resolution_parameter,node_sizes=length_weight)

    optimiser = leidenalg.Optimiser()
    optimiser.optimise_partition(res, is_membership_fixed=is_membership_fixed,n_iterations=-1)

    part = list(res)


    contig_labels_dict ={}
    # dict of communities
    numnode = 0
    rang = []
    for ci in range(len(part)):
        rang.append(ci)
        numnode = numnode+len(part[ci])
        for id in part[ci]:
            contig_labels_dict[namelist[id]] = 'group'+str(ci)

    logger.info(output_file)
    f = open(output_file, 'w')
    for contigIdx in range(len(contig_labels_dict)):
        f.write(namelist[contigIdx] + "\t" + str(contig_labels_dict[namelist[contigIdx]]) + "\n")
    f.close()

def get_binstats(bin_contig_names, markers):
    _, comp, cont = markers.bin_quality(bin_contig_names)

    return comp, cont

def read_bins_nosequences(bin_dirs):
    """Read sequences in bins."""

    bins = defaultdict(lambda: defaultdict(set))

    contigs_in_bins = defaultdict(lambda: {})
    for method_id, bin_dir in bin_dirs.items():
        Header = pd.read_csv(bin_dir, sep='\t', nrows=1)
        cluster_ids = pd.read_csv(bin_dir, sep='\t',header=None, usecols=range(1, Header.shape[1])).values[:, 0]
        namelist = pd.read_csv(bin_dir, sep='\t', header=None, usecols=range(1)).values[:, 0]

        for i in range(len(namelist)):
            bins[method_id][cluster_ids[i]].add(namelist[i])
            contigs_in_bins[namelist[i]][method_id] = cluster_ids[i]

    return bins, contigs_in_bins

def get_bin_quality(orig_bins: Dict[str, Dict[int, List[int]]], methods_sorted: List[str], markers: List[int]):
    """
    Calculate the quality of each bin in the original bins and determine the best method.

    :param orig_bins: A dictionary of original bins with method IDs as keys and bin IDs as sub-keys.
    :param methods_sorted: A list of method IDs sorted in a specific order.
    :param markers: A list of markers used for bin quality calculations.

    :return: A tuple containing a dictionary of bin quality information and the best method.
    """
    bin_quality_dict = defaultdict(lambda: {})
    sum_list = []
    sumcont5_list = []

    for method_id in methods_sorted:
        num_5010 = 0
        num_7010 = 0
        num_9010 = 0
        num_505 = 0
        num_705 = 0
        num_905 = 0

        for bin_id in orig_bins[method_id]:
            comp, cont = get_binstats(orig_bins[method_id][bin_id], markers)
            if comp > 50 and cont < 10:
                num_5010 += 1
            if comp > 70 and cont < 10:
                num_7010 += 1
            if comp > 90 and cont < 10:
                num_9010 += 1
            if comp > 50 and cont < 5:
                num_505 += 1
            if comp > 70 and cont < 5:
                num_705 += 1
            if comp > 90 and cont < 5:
                num_905 += 1

        bin_quality_dict[method_id]['num_5010'] = num_5010
        bin_quality_dict[method_id]['num_7010'] = num_7010
        bin_quality_dict[method_id]['num_9010'] = num_9010
        bin_quality_dict[method_id]['num_505'] = num_505
        bin_quality_dict[method_id]['num_705'] = num_705
        bin_quality_dict[method_id]['num_905'] = num_905
        bin_quality_dict[method_id]['sum'] = num_5010 + num_7010 + num_9010 + num_505 + num_705 + num_905
        bin_quality_dict[method_id]['sum_cont5'] = num_505 + num_705 + num_905

        sum_list.append(bin_quality_dict[method_id]['sum'] )
        sumcont5_list.append(bin_quality_dict[method_id]['sum_cont5'])

    sum_max = max(sum_list)
    sum_max_method = []

    sumcont5_remain = []

    for i in range(len(methods_sorted)):
        if sum_list[i] == sum_max:
            sum_max_method.append(methods_sorted[i])
            sumcont5_remain.append(sumcont5_list[i])
    if len(sum_max_method) == 1:
        best_method = sum_max_method[0]
    else:
        best_method = sum_max_method[sumcont5_remain.index(max(sumcont5_remain))]

    return bin_quality_dict, best_method

# update for checkm marker
def savecontigs_with_high_bin_quality(
        orig_bins: Dict[str, Dict[int, List[int]]], best_method: str, markers: List[int], outpath: str
):
    """
    Save contigs with high bin quality to text files based on specified criteria.

    :param orig_bins: A dictionary of original bins with method IDs as keys and bin IDs as sub-keys.
    :param best_method: The best method to consider.
    :param markers: A list of markers used for bin quality calculations.
    :param outpath: The path to save the output text files.
    :return: None
    """
    bin_count_5010 = 0
    bin_count_5005 = 0
    with open(outpath+'/'+best_method+'5010_res.txt','w') as f1:
        with open(outpath+'/'+best_method+'5005_res.txt','w') as f2:
            for bin_id in orig_bins[best_method]:
                comp, cont = get_binstats(orig_bins[best_method][bin_id], markers)
                if comp > 50 and cont < 10:
                    for key in orig_bins[best_method][bin_id]:
                        f1.write(key+'\t'+str(bin_count_5010)+'\n')
                    bin_count_5010 += 1

                if comp > 50 and cont < 5:
                    for key in orig_bins[best_method][bin_id]:
                        f2.write(key+'\t'+str(bin_count_5005)+'\n')
                    bin_count_5005 += 1
def write_estimated_bin_quality(bin_quality_dict, output_file):
    fout = open(os.path.join(output_file), 'w')
    fout.write('Binning_method\tnum_5010\tnum_7010\tnum_9010\tnum_505\tnum_705\tnum_905\tsum\tsum_cont5\n')
    for method_id in bin_quality_dict:
        fout.write(method_id + '\t' + str(bin_quality_dict[method_id]['num_5010']) + '\t'
                   + str(bin_quality_dict[method_id]['num_7010']) + '\t'
                   + str(bin_quality_dict[method_id]['num_9010']) + '\t'
                   + str(bin_quality_dict[method_id]['num_505']) + '\t'
                   + str(bin_quality_dict[method_id]['num_705']) + '\t'
                   + str(bin_quality_dict[method_id]['num_905']) + '\t'
                   + str(bin_quality_dict[method_id]['sum']) + '\t'
                   + str(bin_quality_dict[method_id]['sum_cont5']) + '\n')

    fout.close()

def estimate_bins_quality_nobins(bac_mg_table, ar_mg_table, res_path, ignore_kmeans_res = False):
    """
    Estimate the quality of bins based on SCG information.

    :param bac_mg_table: The path to the marker gene table for bacteria.
    :param ar_mg_table: The path to the marker gene table for archaea.
    :param res_path: The path to the result files.
    :param ignore_kmeans_res: Whether to ignore K-means results (default: False).

    :return: The best method based on estimated bin quality.
    """
    markers = Markers()

    # bin_dirs = get_bin_dirs(bin_dirs_file)
    filenames = os.listdir(res_path)
    namelist = []
    for filename in filenames:
        if filename.endswith('.tsv'):
            if ignore_kmeans_res:
                if not filename.startswith('weight'):
                    namelist.append(filename)
            else:
                namelist.append(filename)

    namelist.sort()

    bin_dirs = {}
    for res in namelist:
        bin_dirs[res] =  (res_path + res)

    bins, contigs = read_bins_nosequences(bin_dirs)

    methods_sorted = sorted(bins.keys())
    contig_lens = {cid: len(contigs[cid]) for cid in contigs}
    orig_bins = copy.deepcopy(bins)

    gene_tables = markers.marker_gene_tables(bac_mg_table, ar_mg_table)

    bin_quality_dict, best_method = get_bin_quality(orig_bins, methods_sorted, markers)

    savecontigs_with_high_bin_quality(orig_bins, best_method, markers, res_path)

    output_file = res_path + 'estimate_res.txt'
    write_estimated_bin_quality(bin_quality_dict, output_file)
    return best_method

def run_get_final_result(
        logger, bac_mg_table, ar_mg_table, contig_file, output_path, seed_num: int, num_threads: int = 40,
        res_name: Optional[str] = None, ignore_kmeans_res: bool = True
):
    """
    Run the final step to get the best clustering result based on estimated bin quality.

    :param seed_num: The seed number.
    :param num_threads: The number of threads to use (default: 40).
    :param res_name: The name of the result (default: None).
    :param ignore_kmeans_res: Whether to ignore K-means results (default: True).
    """
    logger.info("Seed_num:\t" + str(seed_num))

    if not (bac_mg_table and ar_mg_table):

        logger.info("Run unitem profile:\t" + str(seed_num))
        bin_dirs = {}

        if res_name==None:
            res_name = 'weight_seed_kmeans_k_' + str(seed_num) + '_result.tsv'
        bin_dirs[res_name] = (output_path + '/cluster_res/' + res_name + '_bins', 'fa')

        output_dir = output_path + '/cluster_res/unitem_profile'
        if not (os.path.exists(output_dir)):
            make_sure_path_exists(output_dir)
            profile = Profile(num_threads)
            profile.run(bin_dirs, output_dir)

        bac_mg_table = output_dir + '/binning_methods/' + res_name + '/checkm_bac/marker_gene_table.tsv'
        ar_mg_table = output_dir + '/binning_methods/' + res_name + '/checkm_ar/marker_gene_table.tsv'

    else:
        logger.info("bac_mg_table and ar_mg_table was given")

    best_method = estimate_bins_quality_nobins(
        bac_mg_table, ar_mg_table, output_path + '/cluster_res/', ignore_kmeans_res=ignore_kmeans_res
    )

    logger.info('Final result:\t'+output_path + '/cluster_res/'+best_method)
    filter_small_bins(logger, contig_file, output_path + '/cluster_res/'+best_method, args=None)