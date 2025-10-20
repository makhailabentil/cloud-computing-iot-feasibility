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

### Completed
- [x] Literature review and dataset selection
- [x] Repository setup with Python environment
- [x] Basic project structure

### In Progress
- [ ] Delta encoding compressor prototype
- [ ] CSV sensor trace replay scripts
- [ ] Run length and quantization compressors
- [ ] Edge gateway implementation

### Next Steps (Next 2 Weeks)
1. Implement prototype delta encoding compressor
2. Create scripts to replay CSV sensor traces
3. Implement run length and quantization compressors
4. Build simple edge gateway for data compression
5. Begin systematic evaluation on CAPTURE 24 dataset
6. Report on compression ratio, reconstruction error, and resource usage

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── compressors/
│   │   ├── delta_encoding.py
│   │   ├── run_length.py
│   │   └── quantization.py
│   ├── data_processing/
│   │   └── trace_replay.py
│   └── edge_gateway/
│       └── gateway.py
├── data/
│   └── synthetic/
├── tests/
└── docs/
    └── references.md
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

```bash
# Run compression tests on synthetic data
python src/compressors/delta_encoding.py

# Replay sensor traces
python src/data_processing/trace_replay.py

# Start edge gateway
python src/edge_gateway/gateway.py
```

## Results

Initial tests on synthetic data show approximately **3× compression** with minimal reconstruction error.

## Contributing

This is a research project for academic purposes. Please refer to the project documentation for contribution guidelines.

