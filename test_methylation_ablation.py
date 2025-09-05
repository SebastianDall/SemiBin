#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import argparse
import subprocess

def create_ablation_datasets(data_file, data_split_file, output_dir):
    """
    Create different versions of the dataset for ablation testing:
    1. Original (all features)
    2. No methylation (k-mer + coverage only)  
    3. Only motif presence (k-mer + coverage + motif_present, no methylation values)
    4. Only methylation values (k-mer + coverage + methylation, no motif_present)
    """
    
    # Load data
    data = pd.read_csv(data_file)
    data_split = pd.read_csv(data_split_file)
    
    # Identify feature types
    methylation_cols = [col for col in data.columns if 'median' in col]
    motif_presence_cols = [col for col in data.columns if 'motif_present' in col]
    other_cols = [col for col in data.columns if col not in methylation_cols + motif_presence_cols]
    
    print(f"Feature analysis:")
    print(f"  Other features (k-mer/coverage): {len(other_cols)}")
    print(f"  Methylation value features: {len(methylation_cols)}")
    print(f"  Motif presence features: {len(motif_presence_cols)}")
    
    # Create output directories for each condition
    conditions = {
        'original': other_cols + methylation_cols + motif_presence_cols,
        'no_methylation': other_cols,
        'motif_presence_only': other_cols + motif_presence_cols,
        'methylation_values_only': other_cols + methylation_cols
    }
    
    for condition, columns in conditions.items():
        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        # Save filtered datasets
        data[columns].to_csv(os.path.join(condition_dir, 'data.csv'), index=False)
        data_split[columns].to_csv(os.path.join(condition_dir, 'data_split.csv'), index=False)
        
        print(f"Created {condition} dataset with {len(columns)} features")
    
    return conditions

def run_semibin_comparison(base_dir, conditions, contig_fasta, bam_files, output_base, environment='human_gut'):
    """
    Run SemiBin with different feature combinations and compare results
    """
    results = {}
    
    for condition in conditions.keys():
        print(f"\nRunning SemiBin with {condition} features...")
        
        condition_data_dir = os.path.join(base_dir, condition)
        condition_output = os.path.join(output_base, f'semibin_{condition}')
        
        # Construct SemiBin command
        cmd = [
            'SemiBin2', 'single_easy_bin',
            '-i', contig_fasta,
            '-b'] + bam_files + [
            '-o', condition_output,
            '--environment', environment,
            '--data', os.path.join(condition_data_dir, 'data.csv'),
            '--data-split', os.path.join(condition_data_dir, 'data_split.csv')
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                # Parse results
                results[condition] = parse_semibin_results(condition_output)
                print(f"✓ {condition} completed successfully")
            else:
                print(f"✗ {condition} failed:")
                print(f"  STDERR: {result.stderr}")
                results[condition] = {'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            print(f"✗ {condition} timed out")
            results[condition] = {'error': 'timeout'}
        except Exception as e:
            print(f"✗ {condition} error: {e}")
            results[condition] = {'error': str(e)}
    
    return results

def parse_semibin_results(output_dir):
    """
    Parse SemiBin output to extract key metrics
    """
    try:
        # Look for bins directory
        bins_dir = os.path.join(output_dir, 'output_bins')
        if not os.path.exists(bins_dir):
            return {'error': 'No bins directory found'}
        
        # Count bins
        bin_files = [f for f in os.listdir(bins_dir) if f.endswith('.fa')]
        n_bins = len(bin_files)
        
        # Calculate total binned sequence length
        total_binned_length = 0
        for bin_file in bin_files:
            bin_path = os.path.join(bins_dir, bin_file)
            with open(bin_path, 'r') as f:
                for line in f:
                    if not line.startswith('>'):
                        total_binned_length += len(line.strip())
        
        return {
            'n_bins': n_bins,
            'total_binned_length': total_binned_length,
            'avg_bin_size': total_binned_length / n_bins if n_bins > 0 else 0
        }
        
    except Exception as e:
        return {'error': f'Failed to parse results: {e}'}

def analyze_ablation_results(results, output_dir):
    """
    Analyze and compare ablation test results
    """
    print("\nABLATION TEST RESULTS:")
    print("=" * 50)
    
    # Create results dataframe
    results_data = []
    for condition, metrics in results.items():
        if 'error' not in metrics:
            results_data.append({
                'condition': condition,
                'n_bins': metrics['n_bins'],
                'total_binned_length': metrics['total_binned_length'],
                'avg_bin_size': metrics['avg_bin_size']
            })
        else:
            print(f"❌ {condition}: {metrics['error']}")
    
    if results_data:
        df = pd.DataFrame(results_data)
        print(df.to_string(index=False))
        
        # Save results
        df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)
        
        # Calculate relative performance
        if 'original' in df['condition'].values:
            baseline = df[df['condition'] == 'original']
            baseline_bins = baseline['n_bins'].iloc[0]
            baseline_length = baseline['total_binned_length'].iloc[0]
            
            print(f"\nRELATIVE TO ORIGINAL:")
            print("-" * 30)
            for _, row in df.iterrows():
                if row['condition'] != 'original':
                    bin_change = ((row['n_bins'] - baseline_bins) / baseline_bins) * 100
                    length_change = ((row['total_binned_length'] - baseline_length) / baseline_length) * 100
                    print(f"{row['condition']}:")
                    print(f"  Bins: {bin_change:+.1f}%")
                    print(f"  Binned length: {length_change:+.1f}%")
    
    return results_data

def main():
    parser = argparse.ArgumentParser(description='Test methylation feature ablation')
    parser.add_argument('--data', required=True, help='Path to data.csv')
    parser.add_argument('--data_split', required=True, help='Path to data_split.csv') 
    parser.add_argument('--contig_fasta', required=True, help='Path to contig FASTA file')
    parser.add_argument('--bam', required=True, nargs='+', help='BAM file(s)')
    parser.add_argument('--output_dir', default='ablation_test_output', help='Output directory')
    parser.add_argument('--environment', default='human_gut', help='SemiBin environment')
    parser.add_argument('--skip_semibin', action='store_true', help='Only create datasets, skip SemiBin runs')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Create ablation datasets
    print("Creating ablation datasets...")
    conditions = create_ablation_datasets(args.data, args.data_split, args.output_dir)
    
    if not args.skip_semibin:
        # Run SemiBin comparisons
        print("Running SemiBin comparisons...")
        results = run_semibin_comparison(
            args.output_dir, 
            conditions,
            args.contig_fasta,
            args.bam,
            os.path.join(args.output_dir, 'semibin_runs'),
            args.environment
        )
        
        # Analyze results
        analyze_ablation_results(results, args.output_dir)
    else:
        print(f"Datasets created in {args.output_dir}")
        print("Run with --skip_semibin=False to perform SemiBin comparisons")

if __name__ == "__main__":
    main()