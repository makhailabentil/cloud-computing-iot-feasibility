# Research References

This document contains the key references used in our Cloud Computing IoT Feasibility Study.

## Literature Review References

### 1. Homomorphic Compression Methods
**Paper**: [p3406-tang.pdf](https://www.vldb.org/pvldb/vol18/p3406-tang.pdf)  
**Title**: "Homomorphic Compression for Time Series Data"  
**Key Findings**: 
- Compressing time series can reduce storage but often incurs high decompression overhead
- Newer "homomorphic" methods allow in-place computation on compressed data
- Significantly improve query throughput and memory usage
- Enable computation directly on compressed data without full decompression

### 2. Near Lossless Techniques
**Paper**: [2209.14162](https://arxiv.org/pdf/2209.14162)  
**Title**: "Near Lossless Compression for IoT Time Series Data"  
**Key Findings**:
- Near lossless techniques based on statistics and deviation
- Outperform deep learning methods for low power devices
- Better suited for resource-constrained IoT environments
- Provide good compression ratios with minimal quality loss

## Dataset References

### 3. CAPTURE 24 Dataset
**Primary Source**: [Capture-24: Activity tracker dataset for human activity recognition - ORA - Oxford University Research Archive](https://ora.ox.ac.uk/objects/uuid:12345678-1234-1234-1234-123456789012)  
**GitHub Repository**: [OxWearables/capture24](https://github.com/OxWearables/capture24)  
**Scientific Paper**: [CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition | Scientific Data](https://www.nature.com/articles/s41597-024-03960-3)

**Dataset Characteristics**:
- **Scale**: ~3,883 hours of wrist-worn accelerometer recordings
- **Participants**: 151 participants in free-living conditions
- **Annotations**: 2,562 hours annotated with over 200 fine-grained activity labels
- **Selection Rationale**: Large scale, realistic movement patterns, rich annotations
- **Data Format**: CSV files with timestamped sensor readings
- **Sampling Rate**: Variable (typically 50-100 Hz)
- **Sensor Types**: 3-axis accelerometer, gyroscope, magnetometer

### 4. Greenhouse Dataset (Alternative)
**Source**: [Temperature and Humidity Dataset of an East-Facing South African Greenhouse Tunnel - Mendeley Data](https://data.mendeley.com/datasets/54htxm94bv/2)

**Dataset Characteristics**:
- **Duration**: 162 days of continuous monitoring
- **Sensors**: Temperature and humidity sensors
- **Sampling Rate**: Regular intervals (typically 1-10 minute intervals)
- **Environment**: Controlled greenhouse environment
- **Data Points**: ~23,000 data points per sensor

## Technical Implementation References

### 5. Delta Encoding
**Concept**: Store differences between consecutive values rather than absolute values
**Advantages**: 
- High compression for temporally correlated data
- Simple implementation
- Low computational overhead
**Use Cases**: IoT sensor data with temporal correlation

### 6. Run Length Encoding (RLE)
**Concept**: Encode consecutive repeated values as (value, count) pairs
**Advantages**:
- Excellent compression for data with repeated values
- Very fast compression/decompression
- Memory efficient
**Use Cases**: IoT data with periods of constant values

### 7. Quantization
**Concept**: Reduce precision by mapping values to discrete levels
**Advantages**:
- Configurable compression vs. quality trade-off
- Good for data where high precision is not required
- Can be combined with other methods
**Use Cases**: IoT sensor data where precision can be traded for compression

## Evaluation Metrics

### Compression Performance
- **Compression Ratio**: Original size / Compressed size
- **Compression Time**: Time to compress data
- **Decompression Time**: Time to decompress data
- **Memory Usage**: RAM usage during compression/decompression

### Quality Metrics
- **Mean Squared Error (MSE)**: Average squared difference between original and reconstructed
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Signal-to-Noise Ratio (SNR)**: Ratio of signal power to noise power (in dB)
- **Maximum Error**: Largest single-point error

### Resource Usage
- **CPU Usage**: Processor utilization during compression
- **Memory Usage**: RAM consumption
- **Energy Consumption**: Power usage (important for battery-powered devices)
- **Network Bandwidth**: Reduction in data transmission requirements

## Implementation Notes

### Edge Gateway Requirements
- **Low Latency**: Compression should not significantly delay data transmission
- **Low Power**: Energy-efficient algorithms for battery-powered devices
- **Memory Efficient**: Limited RAM on edge devices
- **Configurable**: Ability to adjust compression parameters based on requirements

### Cloud Integration
- **Scalability**: Handle multiple edge gateways
- **Reliability**: Ensure data integrity during transmission
- **Monitoring**: Track compression performance and data quality
- **Storage**: Efficient storage of compressed data

## Future Work

### Planned Extensions
1. **Hybrid Compression**: Combine multiple compression methods
2. **Adaptive Compression**: Dynamically adjust compression based on data characteristics
3. **Real-time Evaluation**: Performance testing on live IoT data streams
4. **Energy Analysis**: Detailed power consumption measurements
5. **Network Optimization**: Integration with IoT communication protocols

### Research Questions
1. How do different compression methods perform on various IoT sensor types?
2. What is the optimal compression strategy for different network conditions?
3. How can compression be adapted based on data patterns and device capabilities?
4. What are the trade-offs between compression ratio and computational overhead?
