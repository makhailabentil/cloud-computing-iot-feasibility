import time
import math
import random

class RunLengthCompressor:
    def compress(self, data):
        runs = []
        val = data[0]
        count = 1
        for i in range(1, len(data)):
            if data[i] == val:
                count += 1
            else:
                runs.append((val, count))
                val = data[i]
                count = 1
        runs.append((val, count))
        ratio = len(data) / len(runs)
        return runs, len(data), ratio

    def decompress(self, runs, length):
        out = []
        for val, count in runs:
            out += [val]*count
        return out[:length]