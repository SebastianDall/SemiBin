import shutil
import pytest
import polars as pl
import pandas as pd
from SemiBin.generate_methylation import *
import unittest
from unittest.mock import patch, MagicMock
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO



class SetupArgs:
    def __init__(self):
        self.pileup = "test/methylation_data/geobacillus-plasmids.pileup.bed.gz"
        self.contig_fasta = "test/methylation_data/geobacillus-plasmids.assembly.fasta"
        self.motifs_file = "test/methylation_data/bin-motifs.tsv"
        self.data = "test/methylation_data/data.csv"
        self.data_split = "test/methylation_data/data_split.csv"
        self.num_process = 1
        self.min_valid_read_coverage= 1
        self.min_motif_observations=3
        self.methylation_value = "median"
        self.output = "test/methylation_data/test_output"

@pytest.fixture
def data():
    args = SetupArgs()
    logger = MagicMock()
    
    data = pl.read_csv(args.data)
    data = data\
        .rename({"": "contig"})
    data_split = pl.read_csv(args.data_split)
    data_split = data_split\
        .rename({"": "contig"})
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    assembly = read_fasta(args.contig_fasta)
    
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
        "assembly": assembly,
        "contigs": contigs,
    }


def test_generate_methylation_features(tmp_path, data):
    logger = MagicMock()
    args = SetupArgs()

    args.output = str(tmp_path)

    data_before = pl.read_csv(args.data)

    shutil.copy(args.data, args.output)
    shutil.copy(args.data_split, args.output)
   
    generate_methylation_features(logger, args.contig_fasta, args.pileup, args)

    assert os.path.exists(os.path.join(args.output, "data.csv"))
    assert os.path.exists(os.path.join(args.output, "contig_methylation_features.tsv"))

    data_after = pl.read_csv(os.path.join(args.output, "data.csv"))
    assert len(data_before.columns) != len(data_after.columns)


def test_split_contigs(tmp_path, data):
    args = data["args"]
    args.output = str(tmp_path)
    contigs = ["contig_2", "contig_3"]
    lengths = get_contig_lengths_in_split(data["assembly"], contigs)

    motifs = ["GATC_a_1", "GATC_m_3"]

    current_output = calculate_data_split_methylation(
        contigs=contigs,
        contig_lengths=lengths,
        pileup_path=args.pileup,
        assembly_path=args.contig_fasta,
        motifs=motifs,
        threads=1,
        min_valid_read_coverage=3,
        min_valid_cov_to_diff_fraction=0.8,
        methylation_value=epymetheus.MethylationOutput.Median
    )

    current_output = current_output.to_pandas()

    schema = {
        "contig": pl.String,
        "motif": pl.String,
        "mod_type": pl.String,
        "mod_position": pl.UInt64,
        "methylation_value": pl.Float64,
        "mean_read_cov": pl.Float64,
        "n_motif_obs": pl.UInt32,
        
    }
    expected_output = pl.read_csv(
        os.path.join("test/methylation_data/expected_contig_methylation.tsv"),
        separator = "\t",
        schema = schema
    )
    expected_output = expected_output\
        .sort([
            "contig",
            "motif",
            "mod_type",
            "mod_position",
            # "split_num"
        ])\
        .to_pandas()
        
    pd.testing.assert_frame_equal(expected_output, current_output)


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


class TestFilterMustLinks(unittest.TestCase):
    def setUp(self):
        # Create test data with contig pairs and methylation features
        self.test_data = pl.DataFrame({
            "": ["contig_1_1", "contig_1_2", "contig_2_1", "contig_2_2", "contig_3_1", "contig_3_2"],
            "methylation_value_GATC_a_1": [0.1, 0.2, 0.8, 0.9, 0.5, 0.5],
            "motif_present_GATC_a_1": [1, 1, 1, 1, 1, 1],
            "methylation_value_CCGG_m_3": [0.3, 0.4, 0.7, 0.0, 0.2, 0.8],
            "motif_present_CCGG_m_3": [1, 1, 1, 0, 1, 1],
            "methylation_value_CCWGG_m_0": [0.0, 0.0, 0.7, 0.0, 0.2, 0.8],
            "motif_present_CCWGG_m_0": [1, 1, 1, 1, 1, 1],
            "other_feature": [1, 2, 3, 4, 5, 6]
        })

    def test_filter_must_links_removes_distant_pairs(self):
        # Test with max_distance that should filter out contig_2 pair (distance ~0.17)
        # but keep contig_1 pair (distance ~0.14) and contig_3 pair (distance ~0.6)
        max_distance = 0.5
        result = filter_must_links(self.test_data, max_distance)
        print(result)

        # Should keep contig_1 pair (close) but remove contig_2 and contig_3 pairs (distant)
        expected_contigs = ["contig_1_1", "contig_1_2"]
        result_contigs = result.get_column("").to_list()

        self.assertEqual(len(result_contigs), 2)
        self.assertIn("contig_1_1", result_contigs)
        self.assertIn("contig_1_2", result_contigs)

    def test_filter_must_links_keeps_all_close_pairs(self):
        # Test with large max_distance that should keep all pairs
        max_distance = 1.0
        result = filter_must_links(self.test_data, max_distance)

        # Should keep all contigs
        self.assertEqual(len(result), len(self.test_data))

    def test_filter_must_links_removes_all_distant_pairs(self):
        # Test with very small max_distance that should remove all pairs
        max_distance = 0.01
        result = filter_must_links(self.test_data, max_distance)

        # Should remove all contigs
        self.assertEqual(len(result), 0)

    def test_filter_must_links_no_methylation_columns(self):
        # Test with data that has no methylation columns
        test_data_no_meth = pl.DataFrame({
            "": ["contig_1_1", "contig_1_2"],
            "other_feature": [1, 2]
        })

        result = filter_must_links(test_data_no_meth, 0.5)

        # Should return original data unchanged
        self.assertTrue(result.equals(test_data_no_meth))

class TestCheckFilesAndLog(unittest.TestCase):

    @patch('sys.exit')
    def test_check_data_files_missing_data_split(self, mock_exit):
        args = SetupArgs()
        args.data_split = None
        logger = MagicMock()
        
        check_data_file_args(logger, args.data, args.data_split, args)
        
        # Ensure sys.exit(1) was called
        mock_exit.assert_called_once_with(1)
        
        # Ensure the correct error message was logged
        logger.error.assert_called_with("Missing data or data_split path. Either both should be provided or none.")

    @patch('sys.exit')
    def test_check_data_files_missing_data(self, mock_exit):
        args = SetupArgs()
        args.data = ""
        logger = MagicMock()
        
        check_data_file_args(logger, args.data, args.data_split, args)
        
        # Ensure sys.exit(1) was called
        mock_exit.assert_called_once_with(1)
        
        # Ensure the correct error message was logged
        logger.error.assert_called_with("Missing data or data_split path. Either both should be provided or none.")


    def test_check_data_files_default_files_missing(self):
        args = SetupArgs()
        args.data = None
        args.data_split = None
        logger = MagicMock()
        
        check_data_file_args(logger, args.data, args.data_split, args)
        
        # Ensure the correct error message was logged
        logger.info.assert_called_with("Using default data and data_split files. Checking output directory...")


if __name__ == '__main__':
    unittest.main()
