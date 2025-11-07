import time
import math
import random

class QuantizationCompressor:
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
    def compress(self, data):
        mn = min(data); mx = max(data)
        step = (mx - mn) / (2**self.n_bits)
        q = [int((x - mn) / step) for x in data]
        return q, {'min': mn, 'max': mx, 'step': step}
    def decompress(self, q, meta):
        mn, step = meta['min'], meta['step']
        return [mn + i*step for i in q]