# Adapted from https://github.com/BinPro/CONCOCT/blob/develop/scripts/fasta_to_features.py
from itertools import product
from Bio import SeqIO
from collections import OrderedDict


def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC", repeat=kmer_len):
        kmer = ''.join(kmer)
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[''.join(rev_compl)] = counter
            counter += 1
    return kmer_hash, counter


def generate_kmer_features_from_fasta(
        fasta_file, length_threshold, kmer_len, split=False, split_threshold=0):
    import numpy as np
    import pandas as pd
    def seq_list():
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            if not split:
                yield (seq_record.id, seq_record.seq)
            elif len(seq_record) >= split_threshold:
                half = int(len(seq_record.seq) / 2)
                yield (seq_record.id + '_1', seq_record.seq[:half])
                yield (seq_record.id + '_2', seq_record.seq[half:])

    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    composition_d = OrderedDict()
    for h, seq in seq_list():
        if len(seq) <= length_threshold:
            continue
        norm_seq = str(seq).upper()
        kmers = [kmer_dict[norm_seq[i:i+kmer_len]]
                for i in range(len(norm_seq) - kmer_len + 1)]
        kmers.append(nr_features - 1)
        composition_v = np.bincount(np.array(kmers, dtype=np.int64))
        composition_v[-1] -= 1
        composition_d[h] = composition_v
    df = pd.DataFrame.from_dict(composition_d, orient='index', dtype=float)

    df = df.apply(lambda x: x + 1e-5)
    df = df.div(df.sum(axis=1), axis=0)
    return df
