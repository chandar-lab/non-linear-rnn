"""Adding task, following myTorch's AddingData (a variant of Arjovsky et al., 2016), tokenized.

As in myTorch, values and markers come one after the other in time (not as two channels). Values are
num_levels digit tokens instead of reals in U(0, 1), and the target is their exact sum as a class.
Tokens: digits [0, num_levels), unmarked = num_levels, marked = num_levels + 1.
Input (length 2 * seq_len + 1):
    seq_len random digits, then seq_len marker tokens (exactly one marked in each half), then unmarked
Target:
    the sum of the two marked digits (class in [0, 2 * num_levels - 2]) at the last position, which is the
    only position in the loss.
"""
import math
import torch

from .base import Task


class Adding(Task):
    def __init__(self, seq_len=100, num_levels=10):
        self.seq_len = seq_len
        self.num_levels = num_levels
        self.input_size = num_levels + 2
        self.output_size = 2 * num_levels - 1

    def sample(self, batch_size, generator):
        k, half, unmarked = self.seq_len, self.seq_len // 2, self.num_levels
        rows = torch.arange(batch_size)
        values = torch.randint(0, self.num_levels, (batch_size, k), generator=generator)
        first = torch.randint(0, half, (batch_size,), generator=generator)
        second = torch.randint(0, half, (batch_size,), generator=generator) + half
        x = torch.full((batch_size, 2 * k + 1), unmarked)
        x[:, :k] = values
        x[rows, k + first] = unmarked + 1
        x[rows, k + second] = unmarked + 1
        y = torch.zeros(batch_size, 2 * k + 1, dtype=torch.long)
        y[:, -1] = values[rows, first] + values[rows, second]
        mask = torch.zeros(batch_size, 2 * k + 1, dtype=torch.bool)
        mask[:, -1] = True
        return x, y, mask

    @property
    def baseline_loss(self):
        # Entropy of the sum of two independent uniform digits (predicting its distribution without memory).
        n = self.num_levels
        probs = [(n - abs(s - (n - 1))) / n ** 2 for s in range(2 * n - 1)]
        return -sum(p * math.log(p) for p in probs)
