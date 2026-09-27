"""Copy task (Graves et al., 2014), following myTorch's CopyData, tokenized.

myTorch uses binary vectors of num_bits with the last bit reserved for the delimiter; here each vector
is one token, so there are 2 ** (num_bits - 1) data symbols.
Tokens: data [0, S), delimiter S, blank S + 1.
Input (length 2 * (L + 1)):
    L random symbols, the delimiter, then blanks
Target:
    the L symbols followed by the delimiter over the second half, which is the only part in the loss.
L is sampled uniformly from [min_len, max_len] once per batch.
"""
import torch

from .base import Task


class Copy(Task):
    def __init__(self, num_bits=8, min_len=1, max_len=20):
        self.num_bits = num_bits
        self.min_len = min_len
        self.max_len = max_len
        self.num_symbols = 2 ** (num_bits - 1)
        self.delimiter = self.num_symbols
        self.blank = self.num_symbols + 1
        self.input_size = self.output_size = self.num_symbols + 2

    def sample(self, batch_size, generator):
        length = int(torch.randint(self.min_len, self.max_len + 1, (1,), generator=generator))
        data = torch.randint(0, self.num_symbols, (batch_size, length + 1), generator=generator)
        data[:, -1] = self.delimiter
        x = torch.full((batch_size, 2 * (length + 1)), self.blank)
        y = torch.full_like(x, self.blank)
        x[:, :length + 1] = data
        y[:, length + 1:] = data
        mask = torch.zeros(batch_size, 2 * (length + 1), dtype=torch.bool)
        mask[:, length + 1:] = True
        return x, y, mask
