#!/usr/bin/env python3
"""Get detailed statistics from evaluation results."""

import json
import pandas as pd
from pathlib import Path

# Load real data results
data = json.load(open('results/evaluation/milestone2_evaluation_ALL_PARTICIPANTS.json'))
df = pd.DataFrame(data[0] if isinstance(data, list) else list(data.values())[0])

print("=== DETAILED STATISTICS ===\n")
for alg in ['delta_encoding', 'run_length', 'quantization']:
    subset = df[df['algorithm'] == alg]
    print(f"{alg.upper().replace('_', ' ')}:")
    print(f"  Compression Ratio: {subset['compression_ratio'].mean():.2f}x (std: {subset['compression_ratio'].std():.2f}, min: {subset['compression_ratio'].min():.2f}x, max: {subset['compression_ratio'].max():.2f}x)")
    print(f"  Reconstruction Error: {subset['reconstruction_error'].mean():.6f} (MSE)")
    print(f"  Compression Time: {subset['compression_time'].mean()*1000:.2f}ms (std: {subset['compression_time'].std()*1000:.2f}ms)")
    if alg == 'quantization':
        print(f"  SNR: {subset['snr_db'].mean():.1f} dB (range: {subset['snr_db'].min():.1f} to {subset['snr_db'].max():.1f} dB)")
    print()

