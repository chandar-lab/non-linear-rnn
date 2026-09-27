"""Repeat copy task (Graves et al., 2014), following myTorch's RepeatCopyData, tokenized.

Same symbols as the copy task (2 ** (num_bits - 1) data symbols). myTorch feeds the repeat count as a
normalized real value; here each count in [min_repeat, max_repeat] has its own token.
Tokens: data [0, S), delimiter S, blank S + 1, repeat counts S + 2 + (R - min_repeat).
Input (length (L + 1) * (R + 1) + 1):
    L random symbols, the delimiter, the repeat-count token, then blanks
Target:
    the L symbols + delimiter repeated R times, starting right after the repeat-count step; only that
    part is in the loss.
L and R are sampled uniformly from [min_len, max_len] and [min_repeat, max_repeat] once per batch.
"""
import torch

from .base import Task


class RepeatCopy(Task):
    def __init__(self, num_bits=8, min_len=1, max_len=10, min_repeat=1, max_repeat=10):
        self.num_bits = num_bits
        self.min_len = min_len
        self.max_len = max_len
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self.num_symbols = 2 ** (num_bits - 1)
        self.delimiter = self.num_symbols
        self.blank = self.num_symbols + 1
        self.input_size = self.num_symbols + 2 + (max_repeat - min_repeat + 1)
        self.output_size = self.num_symbols + 2

    def sample(self, batch_size, generator):
        length = int(torch.randint(self.min_len, self.max_len + 1, (1,), generator=generator))
        repeats = int(torch.randint(self.min_repeat, self.max_repeat + 1, (1,), generator=generator))
        total = (length + 1) * (repeats + 1) + 1
        data = torch.randint(0, self.num_symbols, (batch_size, length + 1), generator=generator)
        data[:, -1] = self.delimiter
        x = torch.full((batch_size, total), self.blank)
        y = torch.full_like(x, self.blank)
        x[:, :length + 1] = data
        x[:, length + 1] = self.num_symbols + 2 + repeats - self.min_repeat
        y[:, length + 2:] = data.repeat(1, repeats)
        mask = torch.zeros(batch_size, total, dtype=torch.bool)
        mask[:, length + 2:] = True
        return x, y, mask
