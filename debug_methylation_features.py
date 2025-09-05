#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def analyze_feature_distributions(data_file, sample_type, output_dir):
    """
    Analyze feature distributions and identify potential issues
    """
    print(f"Analyzing {sample_type} data from {data_file}")
    
    # Load data
    data = pd.read_csv(data_file)
    
    # Separate feature types
    kmer_cols = [col for col in data.columns if not ('median' in col or 'motif_present' in col)]
    methylation_cols = [col for col in data.columns if 'median' in col]
    motif_presence_cols = [col for col in data.columns if 'motif_present' in col]
    
    print(f"Features found:")
    print(f"  K-mer/coverage features: {len(kmer_cols)}")
    print(f"  Methylation features: {len(methylation_cols)}")
    print(f"  Motif presence features: {len(motif_presence_cols)}")
    
    # 1. CONTIG LENGTH ANALYSIS
    print("\n1. CONTIG ANALYSIS")
    print(f"   Total contigs: {len(data)}")
    
    # 2. METHYLATION SPARSITY ANALYSIS
    print("\n2. METHYLATION SPARSITY ANALYSIS")
    if methylation_cols:
        methylation_data = data[methylation_cols]
        zero_fraction = (methylation_data == 0).sum(axis=1) / len(methylation_cols)
        
        print(f"   Contigs with 0% methylation features: {(zero_fraction == 1.0).sum()} ({(zero_fraction == 1.0).mean()*100:.1f}%)")
        print(f"   Contigs with >90% zero methylation: {(zero_fraction > 0.9).sum()} ({(zero_fraction > 0.9).mean()*100:.1f}%)")
        print(f"   Contigs with >50% zero methylation: {(zero_fraction > 0.5).sum()} ({(zero_fraction > 0.5).mean()*100:.1f}%)")
        print(f"   Mean methylation sparsity: {zero_fraction.mean()*100:.1f}%")
        
        # Plot sparsity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(zero_fraction, bins=50, alpha=0.7)
        plt.title(f'Methylation Sparsity Distribution - {sample_type}')
        plt.xlabel('Fraction of Zero Methylation Features')
        plt.ylabel('Number of Contigs')
        plt.savefig(f'{output_dir}/methylation_sparsity_{sample_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. MOTIF PRESENCE ANALYSIS
    print("\n3. MOTIF PRESENCE ANALYSIS")
    if motif_presence_cols:
        motif_data = data[motif_presence_cols]
        motif_presence_per_contig = motif_data.sum(axis=1)
        
        print(f"   Contigs with no motifs detected: {(motif_presence_per_contig == 0).sum()} ({(motif_presence_per_contig == 0).mean()*100:.1f}%)")
        print(f"   Mean motifs per contig: {motif_presence_per_contig.mean():.2f}")
        print(f"   Max motifs per contig: {motif_presence_per_contig.max()}")
        
        # Plot motif presence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(motif_presence_per_contig, bins=min(50, motif_presence_per_contig.max()+1), alpha=0.7)
        plt.title(f'Motifs per Contig Distribution - {sample_type}')
        plt.xlabel('Number of Motifs Present')
        plt.ylabel('Number of Contigs')
        plt.savefig(f'{output_dir}/motif_presence_{sample_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. FEATURE CORRELATION ANALYSIS
    print("\n4. FEATURE CORRELATION ANALYSIS")
    if methylation_cols and len(kmer_cols) > 10:  # Need some k-mer features for comparison
        # Sample some k-mer features for correlation analysis
        sample_kmer_cols = kmer_cols[1:11]  # Skip first column (usually contig name)
        sample_meth_cols = methylation_cols[:10] if len(methylation_cols) > 10 else methylation_cols
        
        # Calculate correlations between methylation and k-mer features
        corr_data = data[sample_kmer_cols + sample_meth_cols].corr()
        
        # Extract cross-correlations (k-mer vs methylation)
        kmer_meth_corr = corr_data.loc[sample_kmer_cols, sample_meth_cols]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(kmer_meth_corr, annot=False, cmap='coolwarm', center=0)
        plt.title(f'K-mer vs Methylation Feature Correlations - {sample_type}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_correlations_{sample_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Max cross-correlation: {abs(kmer_meth_corr).max().max():.3f}")
        print(f"   Mean absolute cross-correlation: {abs(kmer_meth_corr).mean().mean():.3f}")
    
    # 5. METHYLATION VALUE DISTRIBUTIONS
    print("\n5. METHYLATION VALUE DISTRIBUTIONS")
    if methylation_cols:
        non_zero_meth = data[methylation_cols].replace(0, np.nan)
        
        print(f"   Non-zero methylation values:")
        print(f"     Mean: {non_zero_meth.mean().mean():.3f}")
        print(f"     Std: {non_zero_meth.std().mean():.3f}")
        print(f"     Min: {non_zero_meth.min().min():.3f}")
        print(f"     Max: {non_zero_meth.max().max():.3f}")
        
        # Plot distribution of non-zero methylation values
        all_non_zero_values = non_zero_meth.values.flatten()
        all_non_zero_values = all_non_zero_values[~np.isnan(all_non_zero_values)]
        
        if len(all_non_zero_values) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(all_non_zero_values, bins=50, alpha=0.7)
            plt.title(f'Non-Zero Methylation Values Distribution - {sample_type}')
            plt.xlabel('Methylation Value')
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/methylation_values_{sample_type}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    return {
        'sample_type': sample_type,
        'n_contigs': len(data),
        'n_kmer_features': len(kmer_cols),
        'n_methylation_features': len(methylation_cols),
        'methylation_sparsity': zero_fraction.mean() if methylation_cols else None,
        'contigs_no_motifs': (motif_presence_per_contig == 0).sum() if motif_presence_cols else None,
    }

def compare_sample_types(results_list, output_dir):
    """
    Compare results across sample types
    """
    df = pd.DataFrame(results_list)
    
    print("\nCOMPARISON ACROSS SAMPLE TYPES:")
    print("=" * 50)
    print(df.to_string(index=False))
    
    # Save comparison table
    df.to_csv(f'{output_dir}/sample_comparison.csv', index=False)
    
    # Identify problematic patterns
    print("\nPOTENTIAL ISSUES IDENTIFIED:")
    print("-" * 30)
    
    if 'methylation_sparsity' in df.columns:
        high_sparsity = df[df['methylation_sparsity'] > 0.8]
        if not high_sparsity.empty:
            print(f"High methylation sparsity (>80% zeros): {list(high_sparsity['sample_type'])}")
    
    if 'contigs_no_motifs' in df.columns:
        high_no_motifs = df[df['contigs_no_motifs'] / df['n_contigs'] > 0.5]
        if not high_no_motifs.empty:
            print(f"High fraction of contigs without motifs (>50%): {list(high_no_motifs['sample_type'])}")

def main():
    parser = argparse.ArgumentParser(description='Debug methylation features across sample types')
    parser.add_argument('--fecal_data', help='Path to fecal sample data.csv')
    parser.add_argument('--wwtp_data', help='Path to WWTP sample data.csv') 
    parser.add_argument('--soil_data', help='Path to soil sample data.csv')
    parser.add_argument('--output_dir', default='methylation_debug_output', help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Analyze each sample type
    results = []
    
    if args.fecal_data:
        result = analyze_feature_distributions(args.fecal_data, 'Fecal', args.output_dir)
        results.append(result)
    
    if args.wwtp_data:
        result = analyze_feature_distributions(args.wwtp_data, 'WWTP', args.output_dir)
        results.append(result)
        
    if args.soil_data:
        result = analyze_feature_distributions(args.soil_data, 'Soil', args.output_dir)
        results.append(result)
    
    if len(results) > 1:
        compare_sample_types(results, args.output_dir)
    
    print(f"\nDebug results and plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()