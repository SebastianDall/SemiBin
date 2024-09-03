import pytest
from SemiBin.generate_methylation import *
import unittest
from unittest.mock import patch, MagicMock
import os


class SetupArgs:
    def __init__(self):
        self.pileup = "test/methylation_data/pileup.bed"
        self.contig_fasta = "test/methylation_data/assembly.fasta"
        self.bin_motifs = "test/methylation_data/bin-motifs.tsv"
        self.data = "test/methylation_data/data.csv"
        self.data_split = "test/methylation_data/data_split.csv"
        self.num_process = 1
        self.min_motif_observations = 500
        self.min_motif_methylation = 0.5
        self.min_valid_read_coverage= 8
        self.output = "test_output"

@pytest.fixture
def data():
    args = SetupArgs()
    logger = MagicMock()
    pileup, data, data_split, bin_consensus = load_data(args, logger)
    
    
    assembly = read_fasta(args.contig_fasta)
    
    # Get the unique motifs
    motifs = bin_consensus\
        .select(["motif", "mod_position", "mod_type", "n_mod_bin", "n_nomod_bin"])\
        .with_columns(
            motif_mod = pl.col("motif") + "_" + pl.col("mod_position").cast(pl.String) + "_" + pl.col("mod_type"),
            n_motifs = pl.col("n_mod_bin") + pl.col("n_nomod_bin")
        )\
        .filter(pl.col("n_motifs") >= args.min_motif_observations)\
        .unique(["motif_mod"])

    if len(motifs) == 0:
        logger.error(f"No motifs found")
        sys.exit(1)
    
    motif_list = [(Motif(row[0], row[1]), row[2])for row in motifs.unique(["motif", "mod_position"]).iter_rows()]
    contigs_in_split = data_split.select("contig").to_pandas()
    contigs_in_split = contigs_in_split["contig"].str.rsplit("_",n=1).str[0].unique()

    contigs = {}
    for c in data.get_column("contig"):
        if c in contigs_in_split:
            contigs[c] = True
        else:
            contigs[c] = False
    
    return {
        "args": args,
        "data_split": data_split,
        "data": data,
        "pileup": pileup,
        "bin_consensus": bin_consensus,
        "assembly": assembly,
        "contigs": contigs,
        "motif_list": motif_list
    }


# def test_get_motifs(data):
#     """
#     Test get_motifs function at different cutoffs.
#     """
#     bin_consensus = data["bin_consensus"]
#     data = data["motifs_scored"]
    
#     motifs_all = get_motifs(data, bin_consensus, occurence_cutoff=0)
#     motifs_all_len = len([motif for motifs in motifs_all.values() for motif in motifs])
    
#     motifs_90 = get_motifs(data, bin_consensus, occurence_cutoff=0.9)
#     motifs_90_len = len([motif for motifs in motifs_90.values() for motif in motifs])
    
#     assert motifs_all_len > motifs_90_len, "All motifs should be greater than 90% cutoff."
#     assert motifs_all_len == 17, "All motifs should be 17."
    



def test_find_read_methylation(data):
    contigs = data["contigs"]
    pileup = data["pileup"]
    motif_list = data["motif_list"]
    assembly = data["assembly"]
    args = SetupArgs()
    
    contig_methylation, contig_split_methylation = find_read_methylation(contigs, pileup, assembly, motif_list, threads=args.num_process)

    
    contig_split_methylation = contig_split_methylation\
        .with_columns(
            motif = pl.col("motif") + "_" + pl.col("motif_type") + "-" + pl.col("mod_position").cast(pl.String)
        )
        
        
    contig_methylation = contig_methylation\
        .with_columns(
            motif = pl.col("motif") + "_" + pl.col("motif_type") + "-" + pl.col("mod_position").cast(pl.String)
        )\
        .filter(pl.col("contig") == "contig_10")
    
    
    motif = "RGATCY_a-1"
    
    split_n_modified = contig_split_methylation\
        .filter(pl.col("motif") == motif)\
        .get_column("sum_N_modified").to_list()
        
    scored_n_modified = contig_methylation\
        .filter(pl.col("motif") == motif)\
        .get_column("sum_N_modified").to_list()
    
    
    assert sum(split_n_modified) == sum(scored_n_modified)
    
    split_n_valid_cov= contig_split_methylation\
        .filter(pl.col("motif") == motif)\
        .get_column("sum_N_valid_cov").to_list()
    
    scored_n_valid_cov= contig_methylation\
        .filter(pl.col("motif") == motif)\
        .get_column("sum_N_valid_cov").to_list()

    assert sum(split_n_valid_cov) == sum(scored_n_valid_cov)




class TestCheckFilesExist(unittest.TestCase):

    @patch('os.path.exists')
    def test_all_files_exist(self, mock_exists):
        # Setup the mock to return True for all paths
        mock_exists.return_value = True
        
        # args = SetupArgs('motifs_scored.txt', 'data.txt', 'data_split.txt', 'assembly.fasta', 'motif_index_dir')
        args = SetupArgs()
        # No exception should be raised if all files exist
        paths = [args.pileup, args.data, args.data_split, args.contig_fasta]
        try:
            check_files_exist(paths)
        except FileNotFoundError:
            self.fail("FileNotFoundError raised unexpectedly!")

    @patch('os.path.exists')
    def test_file_does_not_exist(self, mock_exists):
        # Setup the mock to return False when checking for the first missing file
        def side_effect(arg):
            return arg != 'data.txt'
        
        mock_exists.side_effect = side_effect
        
        # args = SetupArgs('motifs_scored.txt', 'data.txt', 'data_split.txt', 'assembly.fasta', 'motif_index_dir')
        args = SetupArgs()
        paths = ["data.txt"]
        with self.assertRaises(FileNotFoundError) as context:
            check_files_exist(paths)
        
        # Check if the error message is correct
        self.assertIn('The file data.txt does not exist.', str(context.exception))

    @patch('os.path.exists')
    def test_directory_does_not_exist(self, mock_exists):
        # Assume all files exist but the directory does not
        def side_effect(arg):
            if arg == 'motif_index_dir':
                return False
            return True

        mock_exists.side_effect = side_effect
        
        dirs = ['motif_index_dir']
        with self.assertRaises(FileNotFoundError) as context:
            check_files_exist(directories = dirs)
        
        # Check if the correct exception for the directory is raised
        self.assertIn('The directory motif_index_dir does not exist.', str(context.exception))



def test_generate_methylation_features():
    args = SetupArgs()
    
    # create a mock logger
    logger = MagicMock()
    
    generate_methylation_features(logger, args)
    
    assert os.path.exists(os.path.join(args.output, "data_split.csv")), "data_split.csv should be created."
    assert os.path.exists(os.path.join(args.output, "data.csv")), "data.csv should be created."
    
    # Cleanup
    os.remove(os.path.join(args.output, "data_split.csv"))
    os.remove(os.path.join(args.output, "data.csv"))
    os.rmdir(args.output)



class TestCheckFilesAndLog(unittest.TestCase):

    @patch('sys.exit')
    def test_check_data_files_missing_data_split(self, mock_exit):
        args = SetupArgs()
        args.data_split = None
        logger = MagicMock()
        
        check_data_file_args(logger, args)
        
        # Ensure sys.exit(1) was called
        mock_exit.assert_called_once_with(1)
        
        # Ensure the correct error message was logged
        logger.error.assert_called_with("Missing data or data_split path. Either both should be provided or none.")

    @patch('sys.exit')
    def test_check_data_files_missing_data(self, mock_exit):
        args = SetupArgs()
        args.data = ""
        logger = MagicMock()
        
        check_data_file_args(logger, args)
        
        # Ensure sys.exit(1) was called
        mock_exit.assert_called_once_with(1)
        
        # Ensure the correct error message was logged
        logger.error.assert_called_with("Missing data or data_split path. Either both should be provided or none.")


    def test_check_data_files_default_files_missing(self):
        args = SetupArgs()
        args.data = None
        args.data_split = None
        logger = MagicMock()
        
        check_data_file_args(logger, args)
        
        # Ensure the correct error message was logged
        logger.info.assert_called_with("Using default data and data_split files. Checking output directory.")




if __name__ == '__main__':
    unittest.main()
