# ESP32 Streaming Data Visualization Guide

This guide explains how to visualize data from your ESP32 streaming setup.

## Quick Start

### 0. Generate Sample Data (For Testing Without ESP32)

If you don't have the ESP32 hardware, you can generate sample data to test the visualizations:

```bash
# Generate sample data (50 segments by default)
python scripts/generate_sample_esp32_data.py

# Generate more segments for better visualization
python scripts/generate_sample_esp32_data.py --segments 100

# Use different participant ID
python scripts/generate_sample_esp32_data.py --participant P002
```

This creates `stream_compression_results.csv` with realistic sample data that mimics ESP32 streaming behavior.

### 1. Basic Usage

After running your ESP32 (`main.py`) or generating sample data, visualize the collected data:

```bash
# Show summary statistics and plots
python scripts/visualize_esp32.py

# Show only statistics
python scripts/visualize_esp32.py --stats

# Generate all visualizations
python scripts/visualize_esp32.py --all

# Real-time monitoring (updates every 2 seconds)
python scripts/visualize_esp32.py --monitor
```

### 2. Visualization Options

#### Performance Summary
```bash
python scripts/visualize_esp32.py --summary
```
Shows:
- Average upload size by algorithm
- Average upload time by algorithm
- Uploads per segment
- Upload size distribution

#### Time Series Analysis
```bash
python scripts/visualize_esp32.py --timeseries
```
Shows:
- Upload size trends over time
- Upload time trends over time
- Total uploads per segment

#### Algorithm Comparison
```bash
python scripts/visualize_esp32.py --comparison
```
Side-by-side comparison of all three algorithms (Delta, RLE, Quantization).

#### Interactive Dashboard (Plotly)
```bash
python scripts/visualize_esp32.py --interactive
```
**Note:** Requires `pip install plotly` for interactive charts.

### 3. Save Plots to Files

```bash
# Save all plots to a directory
python scripts/visualize_esp32.py --all --save results/plots
```

This will create:
- `results/plots/summary.png`
- `results/plots/timeseries.png`
- `results/plots/comparison.png`

### 4. Real-Time Monitoring

Monitor your ESP32 streaming in real-time:

```bash
python scripts/visualize_esp32.py --monitor
```

This will:
- Update every 2 seconds
- Show current statistics
- Display latest upload counts
- Press Ctrl+C to stop

## Example Workflow

1. **Start the Flask server:**
   ```bash
   cd src/edge_gateway
   python server_capture24.py
   ```

2. **Run ESP32 main.py** (on your ESP32 device)

3. **In another terminal, start visualization:**
   ```bash
   # Real-time monitoring
   python scripts/visualize_esp32.py --monitor
   
   # Or generate static plots
   python scripts/visualize_esp32.py --all --save results/esp32_plots
   ```

## Data Location

The visualization tool automatically looks for `stream_compression_results.csv` in:
1. `stream_compression_results.csv` (current directory)
2. `src/edge_gateway/stream_compression_results.csv`
3. `.` (root directory)

Or specify a custom path:
```bash
python scripts/visualize_esp32.py --csv path/to/your/data.csv
```

## What Gets Visualized

The tool visualizes data from the CSV file which contains:
- `timestamp`: Time of upload
- `participant`: Participant ID (e.g., "P001")
- `segment_index`: Segment number
- `axis`: Accelerometer axis (x, y, z)
- `algorithm`: Compression algorithm (Delta, RLE, Quant)
- `upload_bytes`: Size of compressed upload
- `upload_time_ms`: Upload latency

## Tips

1. **For real-time monitoring:** Use `--monitor` while ESP32 is streaming
2. **For reports:** Use `--all --save` to generate all plots
3. **For quick stats:** Use `--stats` to see summary without plots
4. **For presentations:** Use `--interactive` for Plotly charts (can export to HTML)

## Troubleshooting

**"CSV file not found"**
- Make sure ESP32 has uploaded at least one segment
- Check that the Flask server is running and receiving uploads
- Verify the CSV file exists in one of the expected locations

**"No data to visualize"**
- The CSV file exists but is empty
- Wait for ESP32 to complete at least one upload cycle

**Plotly not working**
- Install with: `pip install plotly`
- Or use matplotlib-based visualizations (default)

