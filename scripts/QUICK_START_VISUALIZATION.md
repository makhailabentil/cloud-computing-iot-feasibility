# Quick Start: ESP32 Visualization

## Generate Sample Data (For Testing Without ESP32)

If you don't have the ESP32 hardware, you can generate sample data to test the visualizations:

```bash
# Generate sample data (30 segments, creates stream_compression_results.csv)
python scripts/generate_sample_esp32_data.py

# Generate more segments
python scripts/generate_sample_esp32_data.py --segments 100

# Use different participant ID
python scripts/generate_sample_esp32_data.py --participant P002
```

This creates realistic sample data that mimics ESP32 streaming behavior, allowing you to test all visualization features.

## Option 1: Python Script (Recommended for Analysis)

```bash
# Basic usage - shows summary
python scripts/visualize_esp32.py

# Real-time monitoring while ESP32 is streaming
python scripts/visualize_esp32.py --monitor

# Generate all plots and save to files
python scripts/visualize_esp32.py --all --save results/plots
```

## Option 2: Web Dashboard (Best for Real-Time)

1. Start your Flask server:
   ```bash
   cd src/edge_gateway
   python server_capture24.py
   ```

2. Open your browser to:
   ```
   http://localhost:5001/visualize
   ```

3. Click "Enable Auto-Refresh" to see live updates!

## What You'll See

### Python Script
- **Summary Dashboard**: Key metrics, algorithm comparison, distributions
- **Time Series**: Trends over time for size and latency
- **Statistics**: Detailed breakdown by algorithm
- **Real-time Monitor**: Live updates every 2 seconds

### Web Dashboard
- **Live Statistics**: Total uploads, average size/time, segments
- **Interactive Charts**: Plotly charts that update automatically
- **Algorithm Comparison**: Side-by-side performance metrics
- **Auto-refresh**: Updates every 2 seconds when enabled

## Tips

- **For presentations**: Use `--all --save` to generate PNG files
- **For monitoring**: Use web dashboard with auto-refresh enabled
- **For analysis**: Use Python script with `--stats` and `--timeseries`
- **For interactive exploration**: Use `--interactive` (requires plotly)

## Requirements

- Python packages: `pandas`, `matplotlib` (already in requirements.txt)
- Optional: `plotly` for interactive charts (`pip install plotly`)

