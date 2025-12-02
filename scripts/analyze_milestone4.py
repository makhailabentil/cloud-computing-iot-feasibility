#!/usr/bin/env python3
"""
Milestone 4: Comprehensive Performance Analysis

This script aggregates all evaluation results from Milestones 1, 2, and 3,
performs comprehensive analysis, and generates final comparison reports.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime

def load_milestone2_results():
    """Load Milestone 2 evaluation results (all 151 participants)."""
    results_file = Path("results/evaluation/milestone2_evaluation_ALL_PARTICIPANTS.json")
    if not results_file.exists():
        print(f"⚠️  Milestone 2 results not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    evaluations = data.get('evaluations', [])
    df = pd.DataFrame(evaluations)
    
    print(f"✅ Loaded {len(df)} Milestone 2 evaluations")
    return df

def load_milestone3_results():
    """Load Milestone 3 evaluation results."""
    results = {}
    
    # Adaptive compression
    adaptive_file = Path("results/evaluation/milestone3_adaptive_results.json")
    if adaptive_file.exists():
        with open(adaptive_file, 'r') as f:
            results['adaptive'] = json.load(f)
        print(f"✅ Loaded adaptive compression results")
    
    # Hybrid methods
    hybrid_file = Path("results/evaluation/milestone3_hybrid_results.json")
    if hybrid_file.exists():
        with open(hybrid_file, 'r') as f:
            results['hybrid'] = json.load(f)
        print(f"✅ Loaded hybrid compression results")
    
    # Multi-axis strategies
    multi_axis_file = Path("results/evaluation/milestone3_multi_axis_results.json")
    if multi_axis_file.exists():
        with open(multi_axis_file, 'r') as f:
            results['multi_axis'] = json.load(f)
        print(f"✅ Loaded multi-axis compression results")
    
    # Compressed analytics
    analytics_file = Path("results/evaluation/milestone3_analytics_results.json")
    if analytics_file.exists():
        with open(analytics_file, 'r') as f:
            results['analytics'] = json.load(f)
        print(f"✅ Loaded compressed analytics results")
    
    return results

def analyze_basic_algorithms(df):
    """Analyze basic compression algorithms (Delta, RLE, Quantization)."""
    if df is None or len(df) == 0:
        return None
    
    analysis = {}
    
    for algo in ['delta_encoding', 'run_length', 'quantization']:
        algo_df = df[df['algorithm'] == algo]
        if len(algo_df) == 0:
            continue
        
        analysis[algo] = {
            'count': len(algo_df),
            'compression_ratio': {
                'mean': float(algo_df['compression_ratio'].mean()),
                'std': float(algo_df['compression_ratio'].std()),
                'min': float(algo_df['compression_ratio'].min()),
                'max': float(algo_df['compression_ratio'].max()),
                'median': float(algo_df['compression_ratio'].median())
            },
            'reconstruction_error': {
                'mean': float(algo_df['reconstruction_error'].mean()),
                'max': float(algo_df['reconstruction_error'].max()),
                'median': float(algo_df['reconstruction_error'].median())
            },
            'compression_time_ms': {
                'mean': float(algo_df['compression_time'].mean() * 1000),
                'std': float(algo_df['compression_time'].std() * 1000),
                'median': float(algo_df['compression_time'].median() * 1000)
            },
            'memory_usage_mb': {
                'mean': float(algo_df['memory_usage_mb'].mean()),
                'max': float(algo_df['memory_usage_mb'].max())
            },
            'snr_db': {
                'mean': float(algo_df['snr_db'].replace([np.inf, -np.inf], np.nan).mean()),
                'min': float(algo_df['snr_db'].replace([np.inf, -np.inf], np.nan).min()),
                'max': float(algo_df['snr_db'].replace([np.inf, -np.inf], np.nan).max())
            }
        }
    
    return analysis

def analyze_adaptive_compression(m3_results):
    """Analyze activity-aware adaptive compression results."""
    if 'adaptive' not in m3_results:
        return None
    
    adaptive_data = m3_results['adaptive']
    analysis = {
        'total_segments': 0,
        'activities': defaultdict(int),
        'algorithms_used': defaultdict(int),
        'compression_by_activity': defaultdict(list),
        'compression_by_algorithm': defaultdict(list)
    }
    
    for participant_data in adaptive_data:
        for result in participant_data.get('results', []):
            analysis['total_segments'] += 1
            activity = result.get('activity', 'unknown')
            algorithm = result.get('algorithm', 'unknown')
            compression_ratio = result.get('compression_ratio', 0)
            
            analysis['activities'][activity] += 1
            analysis['algorithms_used'][algorithm] += 1
            analysis['compression_by_activity'][activity].append(compression_ratio)
            analysis['compression_by_algorithm'][algorithm].append(compression_ratio)
    
    # Calculate statistics
    for activity in analysis['compression_by_activity']:
        ratios = analysis['compression_by_activity'][activity]
        analysis['compression_by_activity'][activity] = {
            'mean': float(np.mean(ratios)),
            'min': float(np.min(ratios)),
            'max': float(np.max(ratios)),
            'count': len(ratios)
        }
    
    for algorithm in analysis['compression_by_algorithm']:
        ratios = analysis['compression_by_algorithm'][algorithm]
        analysis['compression_by_algorithm'][algorithm] = {
            'mean': float(np.mean(ratios)),
            'min': float(np.min(ratios)),
            'max': float(np.max(ratios)),
            'count': len(ratios)
        }
    
    return analysis

def analyze_hybrid_methods(m3_results):
    """Analyze hybrid compression methods."""
    if 'hybrid' not in m3_results:
        return None
    
    hybrid_data = m3_results['hybrid']
    analysis = {}
    
    # Handle list of participant results
    if isinstance(hybrid_data, list):
        # Extract all method results from all participants
        method_stats = defaultdict(lambda: {'ratios': [], 'errors': []})
        
        for participant_data in hybrid_data:
            if isinstance(participant_data, dict) and 'results' in participant_data:
                for result in participant_data['results']:
                    if isinstance(result, dict):
                        # Extract each hybrid method
                        for method_name in ['delta_quantization', 'delta_rle', 'quantization_delta']:
                            if method_name in result:
                                method_result = result[method_name]
                                if isinstance(method_result, dict):
                                    ratio = method_result.get('compression_ratio', 0)
                                    error = method_result.get('reconstruction_error', 0)
                                    if ratio > 0:
                                        method_stats[method_name]['ratios'].append(ratio)
                                        method_stats[method_name]['errors'].append(error)
        
        # Calculate statistics for each method
        for method_name, stats in method_stats.items():
            if stats['ratios']:
                analysis[method_name] = {
                    'compression_ratio': {
                        'mean': float(np.mean(stats['ratios'])),
                        'min': float(np.min(stats['ratios'])),
                        'max': float(np.max(stats['ratios'])),
                        'count': len(stats['ratios'])
                    },
                    'reconstruction_error': {
                        'mean': float(np.mean(stats['errors'])),
                        'max': float(np.max(stats['errors'])),
                        'count': len(stats['errors'])
                    }
                }
    elif isinstance(hybrid_data, dict):
        # If it's a dict, process each method
        for method_name, method_data in hybrid_data.items():
            if isinstance(method_data, list):
                ratios = [r.get('compression_ratio', 0) for r in method_data if isinstance(r, dict)]
                errors = [r.get('reconstruction_error', 0) for r in method_data if isinstance(r, dict)]
                
                if ratios:
                    analysis[method_name] = {
                        'compression_ratio': {
                            'mean': float(np.mean(ratios)),
                            'min': float(np.min(ratios)),
                            'max': float(np.max(ratios)),
                            'count': len(ratios)
                        },
                        'reconstruction_error': {
                            'mean': float(np.mean(errors)),
                            'max': float(np.max(errors)),
                            'count': len(errors)
                        }
                    }
    
    return analysis

def analyze_multi_axis_strategies(m3_results):
    """Analyze multi-axis compression strategies."""
    if 'multi_axis' not in m3_results:
        return None
    
    multi_axis_data = m3_results['multi_axis']
    analysis = {}
    
    # Handle both dict and list structures
    if isinstance(multi_axis_data, list):
        # If it's a list, process all items
        ratios = [r.get('compression_ratio', 0) for r in multi_axis_data if isinstance(r, dict)]
        errors = [r.get('reconstruction_error', 0) for r in multi_axis_data if isinstance(r, dict)]
        
        if ratios:
            analysis['multi_axis_strategies'] = {
                'compression_ratio': {
                    'mean': float(np.mean(ratios)),
                    'min': float(np.min(ratios)),
                    'max': float(np.max(ratios)),
                    'count': len(ratios)
                },
                'reconstruction_error': {
                    'mean': float(np.mean(errors)),
                    'max': float(np.max(errors)),
                    'count': len(errors)
                }
            }
    elif isinstance(multi_axis_data, dict):
        # If it's a dict, process each strategy
        for strategy_name, strategy_data in multi_axis_data.items():
            if isinstance(strategy_data, list):
                ratios = [r.get('compression_ratio', 0) for r in strategy_data if isinstance(r, dict)]
                errors = [r.get('reconstruction_error', 0) for r in strategy_data if isinstance(r, dict)]
                
                if ratios:
                    analysis[strategy_name] = {
                        'compression_ratio': {
                            'mean': float(np.mean(ratios)),
                            'min': float(np.min(ratios)),
                            'max': float(np.max(ratios)),
                            'count': len(ratios)
                        },
                        'reconstruction_error': {
                            'mean': float(np.mean(errors)),
                            'max': float(np.max(errors)),
                            'count': len(errors)
                        }
                    }
    
    return analysis

def generate_comparison_table(basic_analysis):
    """Generate comprehensive comparison table."""
    if basic_analysis is None:
        return None
    
    table_data = []
    for algo_name, stats in basic_analysis.items():
        algo_display = {
            'delta_encoding': 'Delta Encoding',
            'run_length': 'Run-Length Encoding',
            'quantization': 'Quantization (8-bit)'
        }.get(algo_name, algo_name)
        
        table_data.append({
            'Algorithm': algo_display,
            'Compression Ratio': f"{stats['compression_ratio']['mean']:.2f}×",
            'Reconstruction Error': f"{stats['reconstruction_error']['mean']:.6f}",
            'Compression Time (ms)': f"{stats['compression_time_ms']['mean']:.2f}",
            'Memory Usage (MB)': f"{stats['memory_usage_mb']['mean']:.3f}",
            'SNR (dB)': f"{stats['snr_db']['mean']:.1f}" if not np.isnan(stats['snr_db']['mean']) else 'N/A',
            'Evaluations': stats['count']
        })
    
    return pd.DataFrame(table_data)

def generate_recommendations(basic_analysis, adaptive_analysis, hybrid_analysis):
    """Generate use case recommendations."""
    recommendations = []
    
    if basic_analysis:
        # Basic algorithm recommendations
        recommendations.append({
            'Use Case': 'General-purpose lossless compression',
            'Recommendation': 'Delta Encoding',
            'Reason': f"Consistent {basic_analysis['delta_encoding']['compression_ratio']['mean']:.2f}× compression, perfect reconstruction, fastest processing"
        })
        
        recommendations.append({
            'Use Case': 'Maximum compression with acceptable quality loss',
            'Recommendation': 'Quantization (8-bit)',
            'Reason': f"{basic_analysis['quantization']['compression_ratio']['mean']:.2f}× compression with minimal error"
        })
        
        recommendations.append({
            'Use Case': 'Rest/sleep periods (low activity)',
            'Recommendation': 'Run-Length Encoding',
            'Reason': 'Up to 2,222× compression for repetitive data during rest periods'
        })
    
    if adaptive_analysis:
        recommendations.append({
            'Use Case': 'Variable activity patterns',
            'Recommendation': 'Activity-Aware Adaptive Compression',
            'Reason': 'Automatically selects optimal algorithm based on detected activity (Delta for active, RLE for rest)'
        })
    
    if hybrid_analysis:
        best_hybrid = max(hybrid_analysis.items(), 
                         key=lambda x: x[1]['compression_ratio']['mean'] if isinstance(x[1], dict) and 'compression_ratio' in x[1] else 0)
        recommendations.append({
            'Use Case': 'Maximum compression with multi-stage processing',
            'Recommendation': f'Hybrid: {best_hybrid[0]}',
            'Reason': f"Combines multiple algorithms for {best_hybrid[1]['compression_ratio']['mean']:.2f}× compression"
        })
    
    return pd.DataFrame(recommendations)

def generate_final_report(basic_analysis, adaptive_analysis, hybrid_analysis, multi_axis_analysis):
    """Generate comprehensive final report."""
    report = []
    report.append("# Comprehensive Performance Analysis Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("=" * 80)
    
    # Executive Summary
    report.append("\n## Executive Summary\n")
    report.append("This report aggregates performance analysis from Milestones 1, 2, and 3, ")
    report.append("providing comprehensive comparison of all compression algorithms and methods ")
    report.append("evaluated on the CAPTURE-24 dataset.\n")
    
    # Basic Algorithms
    if basic_analysis:
        report.append("\n## 1. Basic Compression Algorithms\n")
        report.append("### Performance Summary\n")
        
        for algo_name, stats in basic_analysis.items():
            algo_display = {
                'delta_encoding': 'Delta Encoding',
                'run_length': 'Run-Length Encoding',
                'quantization': 'Quantization (8-bit)'
            }.get(algo_name, algo_name)
            
            report.append(f"\n#### {algo_display}\n")
            report.append(f"- **Compression Ratio**: {stats['compression_ratio']['mean']:.2f}× (range: {stats['compression_ratio']['min']:.2f}× - {stats['compression_ratio']['max']:.2f}×)\n")
            report.append(f"- **Reconstruction Error**: {stats['reconstruction_error']['mean']:.6f}\n")
            report.append(f"- **Compression Time**: {stats['compression_time_ms']['mean']:.2f} ms (median: {stats['compression_time_ms']['median']:.2f} ms)\n")
            report.append(f"- **Memory Usage**: {stats['memory_usage_mb']['mean']:.3f} MB\n")
            if not np.isnan(stats['snr_db']['mean']):
                report.append(f"- **SNR**: {stats['snr_db']['mean']:.1f} dB\n")
            report.append(f"- **Total Evaluations**: {stats['count']}\n")
    
    # Adaptive Compression
    if adaptive_analysis:
        report.append("\n## 2. Activity-Aware Adaptive Compression\n")
        report.append(f"- **Total Segments Evaluated**: {adaptive_analysis['total_segments']}\n")
        report.append("\n### Activity Distribution\n")
        for activity, count in adaptive_analysis['activities'].items():
            report.append(f"- **{activity.capitalize()}**: {count} segments\n")
        
        report.append("\n### Compression Performance by Activity\n")
        for activity, stats in adaptive_analysis['compression_by_activity'].items():
            if isinstance(stats, dict):
                report.append(f"- **{activity.capitalize()}**: {stats['mean']:.2f}× (range: {stats['min']:.2f}× - {stats['max']:.2f}×, {stats['count']} segments)\n")
    
    # Hybrid Methods
    if hybrid_analysis:
        report.append("\n## 3. Hybrid Compression Methods\n")
        for method_name, stats in hybrid_analysis.items():
            if isinstance(stats, dict) and 'compression_ratio' in stats:
                report.append(f"\n### {method_name.replace('_', ' ').title()}\n")
                report.append(f"- **Compression Ratio**: {stats['compression_ratio']['mean']:.2f}× (range: {stats['compression_ratio']['min']:.2f}× - {stats['compression_ratio']['max']:.2f}×)\n")
                report.append(f"- **Reconstruction Error**: {stats['reconstruction_error']['mean']:.6f}\n")
    
    # Multi-Axis Strategies
    if multi_axis_analysis:
        report.append("\n## 4. Multi-Axis Compression Strategies\n")
        for strategy_name, stats in multi_axis_analysis.items():
            if isinstance(stats, dict) and 'compression_ratio' in stats:
                report.append(f"\n### {strategy_name.replace('_', ' ').title()}\n")
                report.append(f"- **Compression Ratio**: {stats['compression_ratio']['mean']:.2f}×\n")
                report.append(f"- **Reconstruction Error**: {stats['reconstruction_error']['mean']:.6f}\n")
    
    # Recommendations
    report.append("\n## 5. Recommendations\n")
    report.append("\n### Use Case Guidance\n")
    
    recommendations = generate_recommendations(basic_analysis, adaptive_analysis, hybrid_analysis)
    for _, rec in recommendations.iterrows():
        report.append(f"\n**{rec['Use Case']}**: {rec['Recommendation']}\n")
        report.append(f"- *Reason*: {rec['Reason']}\n")
    
    report.append("\n" + "=" * 80)
    report.append("\n## Conclusion\n")
    report.append("\nThis comprehensive analysis demonstrates the effectiveness of lightweight ")
    report.append("compression algorithms for IoT sensor data. Key findings:\n")
    report.append("\n1. **Delta Encoding** provides consistent, lossless compression suitable for general use\n")
    report.append("2. **Quantization** offers high compression with minimal quality loss\n")
    report.append("3. **Run-Length Encoding** excels during rest/sleep periods (up to 2,222× compression)\n")
    report.append("4. **Adaptive compression** automatically optimizes algorithm selection based on activity\n")
    report.append("5. **Hybrid methods** can achieve higher compression ratios through multi-stage processing\n")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Milestone 4 analysis')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for reports')
    parser.add_argument('--format', choices=['json', 'markdown', 'both'], default='both',
                       help='Output format')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Milestone 4: Comprehensive Performance Analysis")
    print("=" * 80)
    print()
    
    # Load results
    print("Loading evaluation results...")
    m2_df = load_milestone2_results()
    m3_results = load_milestone3_results()
    
    # Perform analysis
    print("\nPerforming comprehensive analysis...")
    basic_analysis = analyze_basic_algorithms(m2_df)
    adaptive_analysis = analyze_adaptive_compression(m3_results)
    hybrid_analysis = analyze_hybrid_methods(m3_results)
    multi_axis_analysis = analyze_multi_axis_strategies(m3_results)
    
    # Generate comparison table
    comparison_table = generate_comparison_table(basic_analysis)
    
    # Generate final report
    final_report = generate_final_report(basic_analysis, adaptive_analysis, hybrid_analysis, multi_axis_analysis)
    
    # Save results
    if args.format in ['json', 'both']:
        analysis_data = {
            'basic_algorithms': basic_analysis,
            'adaptive_compression': adaptive_analysis,
            'hybrid_methods': hybrid_analysis,
            'multi_axis_strategies': multi_axis_analysis,
            'generated': datetime.now().isoformat()
        }
        
        output_file = output_dir / 'milestone4_comprehensive_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✅ Saved JSON analysis: {output_file}")
    
    if args.format in ['markdown', 'both']:
        output_file = output_dir / 'milestone4_comprehensive_report.md'
        with open(output_file, 'w') as f:
            f.write(final_report)
        print(f"✅ Saved Markdown report: {output_file}")
        
        if comparison_table is not None:
            table_file = output_dir / 'milestone4_comparison_table.md'
            with open(table_file, 'w') as f:
                f.write("# Algorithm Comparison Table\n\n")
                f.write(comparison_table.to_markdown(index=False))
            print(f"✅ Saved comparison table: {table_file}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()

