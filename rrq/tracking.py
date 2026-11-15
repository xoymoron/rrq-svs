"""Codec token diversity diagnostics."""
import numpy as np


def token_metrics(codes):
    result = {}
    for book, values in enumerate(codes, 1):
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        result.update({f'codebook_{book}_unique': int(len(counts)),
                       f'codebook_{book}_top_fraction': float(probabilities.max()),
                       f'codebook_{book}_entropy_bits': float(-(probabilities * np.log2(probabilities)).sum())})
    return result
