#!/usr/bin/env python3
"""
ESP32 Streaming Data Visualization Tool

This script provides comprehensive visualization of ESP32 streaming data including:
- Real-time performance metrics
- Compression algorithm comparisons
- Time-series analysis
- Upload size and latency trends
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import time
import argparse
from datetime import datetime
import numpy as np
from collections import defaultdict

# Try to import plotly for interactive charts (optional)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not available. Install with 'pip install plotly' for interactive charts.")


class ESP32Visualizer:
    """Visualization tool for ESP32 streaming data."""
    
    def __init__(self, csv_path=None):
        """Initialize visualizer with CSV file path."""
        if csv_path is None:
            # Try to find the CSV file
            possible_locations = [
                Path("stream_compression_results.csv"),
                Path("src/edge_gateway/stream_compression_results.csv"),
                Path(".") / "stream_compression_results.csv"
            ]
            for loc in possible_locations:
                if loc.exists():
                    csv_path = loc
                    break
        
        self.csv_path = Path(csv_path) if csv_path else None
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load data from CSV file."""
        if self.csv_path is None or not self.csv_path.exists():
            print(f"⚠️  CSV file not found at {self.csv_path}")
            print("   Make sure ESP32 streaming has been run and data has been collected.")
            self.df = pd.DataFrame()
            return
        
        try:
            self.df = pd.read_csv(self.csv_path)
            if 'timestamp' in self.df.columns:
                # Convert timestamp to datetime for better plotting
                self.df['datetime'] = pd.to_datetime(self.df['timestamp'], format='%H:%M:%S', errors='coerce')
            print(f"✅ Loaded {len(self.df)} records from {self.csv_path}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            self.df = pd.DataFrame()
    
    def reload_data(self):
        """Reload data from CSV (useful for real-time updates)."""
        self.load_data()
    
    def plot_performance_summary(self, save_path=None):
        """Create a summary dashboard with key metrics."""
        if self.df.empty:
            print("⚠️  No data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ESP32 Streaming Performance Summary', fontsize=16, fontweight='bold')
        
        # 1. Upload size by algorithm
        ax1 = axes[0, 0]
        if 'algorithm' in self.df.columns and 'upload_bytes' in self.df.columns:
            algo_sizes = self.df.groupby('algorithm')['upload_bytes'].mean()
            colors = ['#667eea', '#764ba2', '#f093fb']
            bars = ax1.bar(algo_sizes.index, algo_sizes.values, color=colors[:len(algo_sizes)])
            ax1.set_title('Average Upload Size by Algorithm', fontweight='bold')
            ax1.set_ylabel('Bytes')
            ax1.set_xlabel('Algorithm')
            ax1.grid(axis='y', alpha=0.3)
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)} B', ha='center', va='bottom')
        
        # 2. Upload time by algorithm
        ax2 = axes[0, 1]
        if 'algorithm' in self.df.columns and 'upload_time_ms' in self.df.columns:
            algo_times = self.df.groupby('algorithm')['upload_time_ms'].mean()
            bars = ax2.bar(algo_times.index, algo_times.values, color=colors[:len(algo_times)])
            ax2.set_title('Average Upload Time by Algorithm', fontweight='bold')
            ax2.set_ylabel('Time (ms)')
            ax2.set_xlabel('Algorithm')
            ax2.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f} ms', ha='center', va='bottom')
        
        # 3. Uploads over time
        ax3 = axes[1, 0]
        if 'segment_index' in self.df.columns:
            uploads_by_segment = self.df.groupby('segment_index').size()
            ax3.plot(uploads_by_segment.index, uploads_by_segment.values, 
                    marker='o', linewidth=2, markersize=6, color='#667eea')
            ax3.set_title('Uploads per Segment', fontweight='bold')
            ax3.set_xlabel('Segment Index')
            ax3.set_ylabel('Number of Uploads')
            ax3.grid(alpha=0.3)
        
        # 4. Size distribution
        ax4 = axes[1, 1]
        if 'upload_bytes' in self.df.columns:
            ax4.hist(self.df['upload_bytes'], bins=20, color='#764ba2', alpha=0.7, edgecolor='black')
            ax4.set_title('Upload Size Distribution', fontweight='bold')
            ax4.set_xlabel('Upload Size (bytes)')
            ax4.set_ylabel('Frequency')
            ax4.grid(axis='y', alpha=0.3)
            ax4.axvline(self.df['upload_bytes'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {int(self.df["upload_bytes"].mean())} B')
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved summary plot to {save_path}")
        else:
            plt.show()
    
    def plot_time_series(self, save_path=None):
        """Plot time-series data for uploads."""
        if self.df.empty:
            print("⚠️  No data to visualize")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('ESP32 Streaming Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Group by segment for time series
        if 'segment_index' in self.df.columns:
            segment_data = self.df.groupby(['segment_index', 'algorithm']).agg({
                'upload_bytes': 'mean',
                'upload_time_ms': 'mean'
            }).reset_index()
            
            algorithms = self.df['algorithm'].unique()
            colors = {'Delta': '#667eea', 'RLE': '#764ba2', 'Quant': '#f093fb'}
            
            # 1. Upload size over time
            ax1 = axes[0]
            for algo in algorithms:
                algo_data = segment_data[segment_data['algorithm'] == algo]
                ax1.plot(algo_data['segment_index'], algo_data['upload_bytes'], 
                        marker='o', label=algo, linewidth=2, 
                        color=colors.get(algo, '#000000'))
            ax1.set_title('Upload Size Over Time', fontweight='bold')
            ax1.set_xlabel('Segment Index')
            ax1.set_ylabel('Upload Size (bytes)')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # 2. Upload time over time
            ax2 = axes[1]
            for algo in algorithms:
                algo_data = segment_data[segment_data['algorithm'] == algo]
                ax2.plot(algo_data['segment_index'], algo_data['upload_time_ms'], 
                        marker='s', label=algo, linewidth=2,
                        color=colors.get(algo, '#000000'))
            ax2.set_title('Upload Time Over Time', fontweight='bold')
            ax2.set_xlabel('Segment Index')
            ax2.set_ylabel('Upload Time (ms)')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # 3. Total uploads per segment
            ax3 = axes[2]
            uploads_per_segment = self.df.groupby('segment_index').size()
            ax3.bar(uploads_per_segment.index, uploads_per_segment.values, 
                   color='#667eea', alpha=0.7, edgecolor='black')
            ax3.set_title('Total Uploads per Segment', fontweight='bold')
            ax3.set_xlabel('Segment Index')
            ax3.set_ylabel('Number of Uploads')
            ax3.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved time series plot to {save_path}")
        else:
            plt.show()
    
    def plot_algorithm_comparison(self, save_path=None):
        """Compare algorithms side-by-side."""
        if self.df.empty:
            print("⚠️  No data to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
        
        algorithms = self.df['algorithm'].unique()
        colors = {'Delta': '#667eea', 'RLE': '#764ba2', 'Quant': '#f093fb'}
        
        # 1. Average upload size
        ax1 = axes[0]
        algo_sizes = self.df.groupby('algorithm')['upload_bytes'].mean()
        bars = ax1.bar(algo_sizes.index, algo_sizes.values, 
                      color=[colors.get(a, '#000000') for a in algo_sizes.index])
        ax1.set_title('Average Upload Size', fontweight='bold')
        ax1.set_ylabel('Bytes')
        ax1.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)} B', ha='center', va='bottom')
        
        # 2. Average upload time
        ax2 = axes[1]
        algo_times = self.df.groupby('algorithm')['upload_time_ms'].mean()
        bars = ax2.bar(algo_times.index, algo_times.values,
                      color=[colors.get(a, '#000000') for a in algo_times.index])
        ax2.set_title('Average Upload Time', fontweight='bold')
        ax2.set_ylabel('Time (ms)')
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} ms', ha='center', va='bottom')
        
        # 3. Total uploads
        ax3 = axes[2]
        algo_counts = self.df['algorithm'].value_counts()
        bars = ax3.bar(algo_counts.index, algo_counts.values,
                      color=[colors.get(a, '#000000') for a in algo_counts.index])
        ax3.set_title('Total Uploads', fontweight='bold')
        ax3.set_ylabel('Count')
        ax3.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved algorithm comparison to {save_path}")
        else:
            plt.show()
    
    def plot_interactive_dashboard(self):
        """Create an interactive Plotly dashboard."""
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available. Install with 'pip install plotly'")
            return
        
        if self.df.empty:
            print("⚠️  No data to visualize")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Upload Size by Algorithm', 'Upload Time by Algorithm',
                          'Uploads Over Time', 'Size Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        algorithms = self.df['algorithm'].unique()
        colors = {'Delta': '#667eea', 'RLE': '#764ba2', 'Quant': '#f093fb'}
        
        # 1. Upload size by algorithm
        algo_sizes = self.df.groupby('algorithm')['upload_bytes'].mean()
        fig.add_trace(
            go.Bar(x=algo_sizes.index, y=algo_sizes.values,
                  marker_color=[colors.get(a, '#000000') for a in algo_sizes.index],
                  name='Avg Size'),
            row=1, col=1
        )
        
        # 2. Upload time by algorithm
        algo_times = self.df.groupby('algorithm')['upload_time_ms'].mean()
        fig.add_trace(
            go.Bar(x=algo_times.index, y=algo_times.values,
                  marker_color=[colors.get(a, '#000000') for a in algo_times.index],
                  name='Avg Time'),
            row=1, col=2
        )
        
        # 3. Uploads over time
        if 'segment_index' in self.df.columns:
            segment_data = self.df.groupby(['segment_index', 'algorithm'])['upload_bytes'].mean().reset_index()
            for algo in algorithms:
                algo_data = segment_data[segment_data['algorithm'] == algo]
                fig.add_trace(
                    go.Scatter(x=algo_data['segment_index'], y=algo_data['upload_bytes'],
                             mode='lines+markers', name=algo,
                             line=dict(color=colors.get(algo, '#000000'))),
                    row=2, col=1
                )
        
        # 4. Size distribution
        fig.add_trace(
            go.Histogram(x=self.df['upload_bytes'], nbinsx=20,
                        marker_color='#764ba2', name='Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="ESP32 Streaming Interactive Dashboard",
            height=800,
            showlegend=True
        )
        
        fig.show()
    
    def print_statistics(self):
        """Print summary statistics."""
        if self.df.empty:
            print("⚠️  No data available")
            return
        
        print("\n" + "="*60)
        print("ESP32 Streaming Statistics")
        print("="*60)
        print(f"\nTotal Uploads: {len(self.df)}")
        
        if 'upload_bytes' in self.df.columns:
            print(f"\nUpload Size:")
            print(f"  Mean: {self.df['upload_bytes'].mean():.2f} bytes")
            print(f"  Median: {self.df['upload_bytes'].median():.2f} bytes")
            print(f"  Min: {self.df['upload_bytes'].min():.2f} bytes")
            print(f"  Max: {self.df['upload_bytes'].max():.2f} bytes")
        
        if 'upload_time_ms' in self.df.columns:
            print(f"\nUpload Time:")
            print(f"  Mean: {self.df['upload_time_ms'].mean():.2f} ms")
            print(f"  Median: {self.df['upload_time_ms'].median():.2f} ms")
            print(f"  Min: {self.df['upload_time_ms'].min():.2f} ms")
            print(f"  Max: {self.df['upload_time_ms'].max():.2f} ms")
        
        if 'algorithm' in self.df.columns:
            print(f"\nBy Algorithm:")
            for algo in self.df['algorithm'].unique():
                algo_df = self.df[self.df['algorithm'] == algo]
                print(f"\n  {algo}:")
                print(f"    Count: {len(algo_df)}")
                if 'upload_bytes' in algo_df.columns:
                    print(f"    Avg Size: {algo_df['upload_bytes'].mean():.2f} bytes")
                if 'upload_time_ms' in algo_df.columns:
                    print(f"    Avg Time: {algo_df['upload_time_ms'].mean():.2f} ms")
        
        if 'segment_index' in self.df.columns:
            print(f"\nSegments Processed: {self.df['segment_index'].nunique()}")
            print(f"Segments Range: {self.df['segment_index'].min()} - {self.df['segment_index'].max()}")
        
        print("\n" + "="*60 + "\n")
    
    def real_time_monitor(self, interval=2):
        """Monitor data in real-time (updates every few seconds)."""
        print("🔄 Starting real-time monitor (Ctrl+C to stop)...")
        print(f"   Updating every {interval} seconds\n")
        
        try:
            while True:
                self.reload_data()
                if not self.df.empty:
                    # Clear screen (works on most terminals)
                    print("\033[2J\033[H", end="")
                    self.print_statistics()
                    print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"\nNext update in {interval} seconds... (Ctrl+C to stop)")
                else:
                    print("⏳ Waiting for data...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visualize ESP32 streaming data')
    parser.add_argument('--csv', type=str, help='Path to CSV file (auto-detected if not specified)')
    parser.add_argument('--summary', action='store_true', help='Show performance summary')
    parser.add_argument('--timeseries', action='store_true', help='Show time series plots')
    parser.add_argument('--comparison', action='store_true', help='Show algorithm comparison')
    parser.add_argument('--interactive', action='store_true', help='Show interactive Plotly dashboard')
    parser.add_argument('--stats', action='store_true', help='Print statistics')
    parser.add_argument('--monitor', action='store_true', help='Real-time monitoring mode')
    parser.add_argument('--save', type=str, help='Save plots to directory (instead of showing)')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = ESP32Visualizer(csv_path=args.csv)
    
    if args.monitor:
        viz.real_time_monitor()
        return
    
    if args.stats or args.all:
        viz.print_statistics()
    
    if args.summary or args.all:
        save_path = f"{args.save}/summary.png" if args.save else None
        if save_path:
            Path(args.save).mkdir(parents=True, exist_ok=True)
        viz.plot_performance_summary(save_path=save_path)
    
    if args.timeseries or args.all:
        save_path = f"{args.save}/timeseries.png" if args.save else None
        if save_path:
            Path(args.save).mkdir(parents=True, exist_ok=True)
        viz.plot_time_series(save_path=save_path)
    
    if args.comparison or args.all:
        save_path = f"{args.save}/comparison.png" if args.save else None
        if save_path:
            Path(args.save).mkdir(parents=True, exist_ok=True)
        viz.plot_algorithm_comparison(save_path=save_path)
    
    if args.interactive:
        viz.plot_interactive_dashboard()
    
    # If no specific option, show summary by default
    if not any([args.summary, args.timeseries, args.comparison, args.interactive, args.stats, args.all]):
        viz.print_statistics()
        viz.plot_performance_summary()


if __name__ == '__main__':
    main()

