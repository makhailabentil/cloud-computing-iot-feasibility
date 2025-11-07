import time
import math
import random

class DeltaEncodingCompressor:
    @staticmethod
    def compress(data):
        deltas = []
        for i in range(1, len(data)):
            deltas.append(data[i] - data[i-1])
        first = data[0]
        comp_ratio = len(data) / len(deltas) if len(deltas) > 0 else 1
        return deltas, first, comp_ratio

    @staticmethod
    def decompress(deltas, first):
        recon = [first]
        for d in deltas:
            recon.append(recon[-1] + d)
        return recon