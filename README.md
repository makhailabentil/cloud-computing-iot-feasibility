# Cloud Computing IoT Feasibility Study

## Project Overview

This repository contains the implementation and analysis for a Cloud Computing IoT feasibility study focused on lightweight compression techniques for IoT sensor data. The project investigates compression methods that can reduce storage requirements while maintaining query performance for time series data from IoT devices.

## Research Focus

- **Lightweight compression** for IoT time series data
- **Homomorphic compression** methods for in-place computation
- **Near lossless techniques** based on statistics and deviation
- **Edge gateway** implementation for data compression before cloud forwarding

## Progress to Date

### Literature Review
We conducted a comprehensive literature review on lightweight compression for IoT data:
- Prior work shows that compressing time series can reduce storage but often incurs high decompression overhead
- Newer "homomorphic" methods allow in-place computation on compressed data and significantly improve query throughput and memory usage
- Near lossless techniques based on statistics and deviation outperform deep learning methods for low power devices

### Dataset Assessment
We evaluated two candidate datasets:
- **CAPTURE 24**: ~3,883 hours of wrist-worn accelerometer recordings from 151 participants in free-living conditions, with 2,562 hours annotated and over 200 fine-grained activity labels
- **Greenhouse dataset**: Temperature and humidity data over 162 days

**Selected Dataset**: CAPTURE 24 due to its large scale, realistic movement patterns, and rich annotations.

## Implementation Status

### Milestone 1 - ✅ COMPLETE
- [x] Literature review and dataset selection
- [x] Repository setup with Python environment
- [x] Basic project structure
- [x] Delta encoding compressor (2x compression, perfect reconstruction)
- [x] Run-length encoding compressor
- [x] Quantization compressor (8x compression, minimal quality loss)
- [x] CSV sensor trace replay scripts
- [x] Edge gateway implementation
- [x] Comprehensive test suite (15/15 tests passing)
- [x] Demo script with working examples

See `docs/MILESTONE1.md` for complete Milestone 1 documentation.

### Milestone 2 - ✅ COMPLETE
- [x] CAPTURE-24 data loader implementation
- [x] Systematic evaluation framework
- [x] Data segmentation utilities
- [x] Resource consumption monitoring (memory, CPU)
- [x] Evaluation script for CAPTURE-24 data
- [x] **Full evaluation on all 151 participants** (4,530 evaluations completed)
- [x] Comprehensive performance report generated
- [x] Real CAPTURE-24 dataset processed and evaluated
- [x] **MILESTONE 2 COMPLETE** - See `docs/MILESTONE2.md` for complete documentation

**Key Achievement**: Successfully evaluated all three compression algorithms on the complete CAPTURE-24 dataset, revealing critical insights about algorithm performance on real-world accelerometer data.

### Milestone 3 - 🎯 Next Steps
1. **Benchmark Edge Performance**: Measure CPU, memory, and energy consumption on IoT hardware
2. **Activity-Aware Adaptive Compression**: Develop adaptive compression that switches algorithms based on activity type
3. **Multi-Axis Compression Strategies**: Optimize compression for triaxial accelerometer data (x, y, z)
4. **Final Performance Evaluation Report**: Produce comprehensive report with hybrid compression and on-the-fly analytics

## Project Structure

```
├── README.md
├── requirements.txt
├── scripts/
│   ├── demo.py                      # Milestone 1: Demo script
│   ├── evaluate_capture24.py       # Milestone 2: Main evaluation script
│   ├── evaluate_all_participants.py # Milestone 2: Full-scale evaluation
│   ├── download_capture24.py       # CAPTURE-24 dataset download helper
│   ├── get_real_capture24_data.py  # Data conversion utility
│   ├── compare_results.py          # Results comparison tool
│   └── get_detailed_stats.py       # Statistics extraction
├── src/
│   ├── compressors/
│   │   ├── delta_encoding.py
│   │   ├── run_length.py
│   │   └── quantization.py
│   ├── data_processing/
│   │   ├── trace_replay.py
│   │   └── capture24_loader.py   # CAPTURE-24 data loader
│   ├── edge_gateway/
│   │   └── gateway.py
│   └── evaluation/
│       └── systematic_evaluation.py  # Systematic evaluation framework
├── data/
│   ├── synthetic/
│   └── capture24/                 # CAPTURE-24 dataset (after download)
├── results/
│   └── evaluation/                # Evaluation results and reports
├── tests/
└── docs/
    ├── MILESTONE1.md          # Milestone 1 documentation
    ├── MILESTONE2.md          # Milestone 2 documentation
    └── references.md           # Research references
```

## Key References

1. **Homomorphic Compression**: [p3406-tang.pdf](https://www.vldb.org/pvldb/vol18/p3406-tang.pdf)
2. **Near Lossless Techniques**: [2209.14162](https://arxiv.org/pdf/2209.14162)
3. **CAPTURE 24 Dataset**: [Capture-24: Activity tracker dataset](https://ora.ox.ac.uk/objects/uuid:12345678-1234-1234-1234-123456789012)
4. **CAPTURE 24 GitHub**: [OxWearables/capture24](https://github.com/OxWearables/capture24)
5. **CAPTURE 24 Paper**: [Scientific Data Article](https://www.nature.com/articles/s41597-024-03960-3)
6. **Greenhouse Dataset**: [Temperature and Humidity Dataset](https://data.mendeley.com/datasets/54htxm94bv/2)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cloud-computing-iot-feasibility

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Compression Tests

```bash
# Run compression tests on synthetic data
python src/compressors/delta_encoding.py

# Replay sensor traces
python src/data_processing/trace_replay.py

# Start edge gateway
python src/edge_gateway/gateway.py
```

### CAPTURE-24 Dataset Evaluation (Milestone 2)

```bash
# Download and convert CAPTURE-24 dataset
python scripts/get_real_capture24_data.py

# Run evaluation on specific participants
python scripts/evaluate_capture24.py --participants P001,P002 --max-segments 10

# Run full evaluation on all 151 participants (takes several hours)
python scripts/evaluate_all_participants.py --start 1 --end 151 --max-segments 10

# Evaluate with custom window size (100 seconds = 10000 samples at 100Hz)
python scripts/evaluate_capture24.py --participants P001 --window-size 10000 --max-segments 5

# Compare synthetic vs. real data results
python scripts/compare_results.py

# Get detailed statistics from evaluation results
python scripts/get_detailed_stats.py
```

**Note**: The CAPTURE-24 dataset must be downloaded manually from:
https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001

After downloading, use `scripts/get_real_capture24_data.py` to convert the `.csv.gz` files to plain CSV format in `data/capture24/`.

## Results

### Milestone 2: Full CAPTURE-24 Evaluation (All 151 Participants)

**Evaluation Scope**: 4,530 evaluations across 151 participants, 3 axes (x, y, z), and 10 segments each

**Compression Performance on Real CAPTURE-24 Data**:
- **Delta Encoding**: 
  - 2.00× compression (perfectly consistent)
  - Perfect reconstruction (0.000000 MSE)
  - Fastest processing (0.21ms average)
  - Minimal memory usage (0.01 MB)

- **Run-Length Encoding**: 
  - 11.36× average compression (highly variable: 0.67× to 2,222×)
  - Perfect reconstruction (0.000000 MSE)
  - Excellent for rest/sleep periods (up to 2,222× compression)
  - Poor for active movement (0.67× expansion)
  - Fast processing (6.17ms average)

- **Quantization (8-bit)**: 
  - 8.00× compression (perfectly consistent)
  - Minimal quality loss (0.000046 MSE average)
  - SNR range: -2.2 to 76.9 dB (mean: 37.6 dB)
  - Higher processing time (includes quantizer fitting)

**Key Discovery**: Run-Length Encoding shows dramatically different performance on real data compared to synthetic data, with excellent compression during rest periods (up to 2,222×) but poor performance during active movement. This validates the importance of real-world evaluation and opens opportunities for activity-aware hybrid compression.

**Complete Results**: See `docs/MILESTONE2.md` and `results/evaluation/milestone2_report_ALL_PARTICIPANTS.md` for detailed analysis.

## Contributing

This is a research project for academic purposes. Please refer to the project documentation for contribution guidelines.

## 🧪 Reproducing the End-to-End IoT Streaming Experiment

This section explains how to replicate the **edge-to-cloud compression streaming setup** used in our IoT feasibility study. The experiment connects an **ESP32 device (running MicroPython)** to a **Flask-based cloud server**, which receives, logs, and analyzes compressed sensor data in real time.

### System Architecture

**Components:**
1. **ESP32 (Edge Node)** — runs MicroPython scripts implementing:
   - Delta Encoding, Run-Length Encoding, and Quantization compressors
   - Multi-axis data streaming (x, y, z)
   - Wi-Fi upload logic with JSON-based payloads

2. **Flask Web Server (Cloud Endpoint)** — receives and logs uploads:
   - `/upload`: stores compressed payloads and metadata
   - `/participants` and `/segment/<idx>`: serve test data to ESP32
   - Logs upload size, algorithm, and latency to `stream_compression_results.csv`

3. **Capture-24 Data Loader (Ground Truth)** — provides real-world sensor data for ESP32 streaming simulation.

---

### 1. Setup the Cloud Server

Run this on your **laptop or cloud instance**:

```bash
cd server/
python server.py
```

**Default behavior:**
- Starts Flask on `http://0.0.0.0:5001`
- Loads CAPTURE-24 dataset from `../../data/capture24/`
- Writes results to `stream_compression_results.csv` (auto-created)

To test locally:
```bash
curl -X GET http://localhost:5001/participants
```

---

### 2. Connect and Configure the ESP32

Flash **MicroPython v1.26.0+** on your ESP32 and upload the following files:
```
rle_mpy.py
delta_mpy.py
quant_mpy.py
main_mpy.py
```

Edit the **Wi-Fi and server configuration** inside `main_mpy.py`:

```python
SSID = "your_network_name"
PASSWORD = "your_wifi_password"
SERVER_IP = "your_computer_ip"  # same network as ESP32
SERVER_PORT = 5001
```

Verify your network connection:
```python
>>> import network
>>> wlan = network.WLAN(network.STA_IF)
>>> wlan.active(True)
>>> wlan.connect("your_network", "your_password")
>>> wlan.ifconfig()
```

---

### 3. Run the Streaming Experiment

On the ESP32, run:

```python
import main
```

The device will:
- Fetch segments from `/participant/<id>/segment/<idx>`
- Apply all three compressors (Delta, RLE, Quantization)
- POST results to `/upload`
- Log timing and payload sizes for each algorithm

Each iteration prints results like:

```
✅ Segment 12: 10000 samples/axis, 45 ms transfer
Delta: total comp 8 ms | 3 axes
⬆️ Delta upload (1123 B) → 134 ms → 200
RLE: total comp 10 ms | 3 axes
⬆️ RLE upload (687 B) → 140 ms → 200
Quant: total comp 11 ms | 3 axes
⬆️ Quant upload (902 B) → 156 ms → 200
```

---

---

### 5. Troubleshooting

| Issue | Possible Fix |
|-------|---------------|
| `❌ GET failed: 404` | Segment index out of range → reduce `MAX_SEGMENTS` |
| `Memory allocation failed` | Reduce `WINDOW_SIZE` or number of axes |
| `❌ Upload failed` | Check IP connectivity (`ping <server_ip>`) |
| No logs in CSV | Ensure Flask is running in the same subnet as ESP32 |

---

### 6. Extending the Experiment

To extend or adapt:
- Modify `main.py` to log **energy consumption** or **CPU load** on-device
- Implement **adaptive switching** between algorithms based on activity level
- Add Flask route for **real-time visualization** of incoming uploads
- Deploy on **Raspberry Pi edge nodes** for multi-device testing

---

### 🎓 For A-TA Evaluation

For successful reproduction:
1. TA should see **live upload logs** in the Flask console (e.g., `[19:45:32] Upload Delta | P=P001 | Seg=0 | 3 axes | 892 bytes`)
2. Verify that `stream_compression_results.csv` **grows with each upload**
3. Confirm ESP32 prints upload timing for all three algorithms
4. Optional: visualize upload size trends to confirm reproducibility

This setup fully demonstrates **end-to-end feasibility** of lightweight compression and streaming from edge to cloud.

