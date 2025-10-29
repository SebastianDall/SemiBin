#!/usr/bin/env python3

from Bio import SeqIO
import random
import sys

def create_fake_bam(fasta_file, coverage_dict, output_prefix="fake_reads"):
    """
    Create a fake BAM file with specified coverage for each contig

    Args:
        fasta_file: Path to FASTA assembly file
        coverage_dict: Dictionary mapping contig names to desired coverage
        output_prefix: Prefix for output files
    """

    # Read the assembly
    with open(fasta_file, 'r') as f:
        records = list(SeqIO.parse(f, 'fasta'))

    sam_file = f"{output_prefix}.sam"

    # Write SAM file
    with open(sam_file, 'w') as sam:
        # Write header
        sam.write('@HD\tVN:1.6\tSO:coordinate\n')
        for record in records:
            sam.write(f'@SQ\tSN:{record.id}\tLN:{len(record.seq)}\n')

        # Generate fake reads with specified coverage
        read_id = 1
        read_length = 150

        for record in records:
            contig_name = record.id
            seq_len = len(record.seq)

            # Get coverage for this contig (default to 1x if not specified)
            coverage = coverage_dict.get(contig_name, 1.0)

            # Calculate number of reads needed
            num_reads = int((seq_len * coverage) / read_length)

            print(f"Contig: {contig_name}, Length: {seq_len}, Coverage: {coverage}x, Reads: {num_reads}")

            for i in range(num_reads):
                # Random position along the contig
                pos = random.randint(1, max(1, seq_len - read_length + 1))

                # Create a fake read sequence (just repeat A's for simplicity)
                read_seq = 'A' * read_length
                qual = 'I' * read_length  # High quality scores

                # Write SAM line: QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL
                sam.write(f'read_{read_id}\t0\t{contig_name}\t{pos}\t60\t{read_length}M\t*\t0\t0\t{read_seq}\t{qual}\n')
                read_id += 1

    print(f"Created {sam_file} with {read_id-1} total reads")
    return sam_file

if __name__ == "__main__":
    # Define coverage for specific contigs
    coverage_settings = {
        'contig_2': 50.0,
        'contig_3': 12.0,
        'contig_4': 50.0,
        'contig_5': 12.0,
        # Any other contigs will get 1x coverage by default
    }

    # Create the fake BAM
    fasta_file = "geobacillus-plasmids.assembly.bin.fasta"
    sam_file = create_fake_bam(fasta_file, coverage_settings)

    print(f"\nTo convert to BAM and sort:")
    print(f"samtools view -bS {sam_file} | samtools sort -o fake_reads.bam")
    print(f"samtools index fake_reads.bam")
