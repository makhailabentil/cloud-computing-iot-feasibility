#!/usr/bin/env python3
"""Compare Milestone 1 (synthetic) vs Milestone 2 (real data) results."""

import json
import pandas as pd
from pathlib import Path

# Load real data results
data = json.load(open('results/evaluation/milestone2_evaluation_ALL_PARTICIPANTS.json'))
df = pd.DataFrame(data[0] if isinstance(data, list) else list(data.values())[0])

print("=" * 70)
print("COMPARISON: Milestone 1 (Synthetic) vs Milestone 2 (Real CAPTURE-24)")
print("=" * 70)

print("\n1. DELTA ENCODING:")
print("   Milestone 1 (Synthetic): 2.0x compression, perfect reconstruction, ~0.1ms")
delta = df[df['algorithm'] == 'delta_encoding']
print(f"   Milestone 2 (Real):     {delta['compression_ratio'].mean():.2f}x compression, perfect, {delta['compression_time'].mean()*1000:.1f}ms")
print("   -> Very similar! Delta encoding works consistently on both data types")

print("\n2. RUN-LENGTH ENCODING:")
print("   Milestone 1 (Synthetic): 0.67x-1.13x (data expansion), perfect, ~4ms")
rle = df[df['algorithm'] == 'run_length']
print(f"   Milestone 2 (Real):     {rle['compression_ratio'].mean():.2f}x (std: {rle['compression_ratio'].std():.2f}), perfect, {rle['compression_time'].mean()*1000:.1f}ms")
print(f"                            Range: {rle['compression_ratio'].min():.2f}x to {rle['compression_ratio'].max():.2f}x")
print("   -> MAJOR DIFFERENCE! Real accelerometer data has:")
print("     - Periods of rest/sleep (highly repetitive -> excellent RLE compression)")
print("     - Active movement (continuous signals -> poor RLE compression)")
print("     - This explains the high variance (std: 54.92)")

print("\n3. QUANTIZATION:")
print("   Milestone 1 (Synthetic): 8.0x compression, minimal loss (SNR 40-50 dB), ~46ms")
quant = df[df['algorithm'] == 'quantization']
print(f"   Milestone 2 (Real):     {quant['compression_ratio'].mean():.2f}x, MSE: {quant['reconstruction_error'].mean():.6f}, {quant['compression_time'].mean()*1000:.1f}ms")
print(f"                            SNR range: {quant['snr_db'].min():.1f} to {quant['snr_db'].max():.1f} dB")
print("   -> Similar compression ratio, but:")
print("     - Real data shows wider SNR range (32-52 dB vs 40-50 dB)")
print("     - Processing time in summary seems inflated (likely includes fitting time)")

print("\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("The biggest difference is Run-Length Encoding:")
print("- Synthetic data: Continuous signals -> poor RLE performance (0.67x-1.13x)")
print("- Real data: Mixed patterns (rest + activity) -> variable RLE (0.67x to 91x+)")
print("- This validates that real-world data has different characteristics than synthetic!")
print("=" * 70)

