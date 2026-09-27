"""Associative recall task (Graves et al., 2014), following myTorch's AssociativeRecallData, tokenized.

myTorch uses binary vectors of num_bits with the last two bits reserved for delimiters; here each vector
is one token, so there are 2 ** (num_bits - 2) data symbols. An item is block_len symbols followed by
an item delimiter.
Tokens: data [0, S), item delimiter S, query marker S + 1, blank S + 2.
Input (length (block_len + 1) * (L + 2) + 1):
    L items, a query marker, a copy of one item's symbols (the query), a query marker, then blanks
Target:
    the symbols of the item that followed the query item, then a query marker as the end token; only
    that part is in the loss.
L is sampled uniformly from [min_len, max_len] once per batch. The query item is sampled per example
(myTorch uses one per batch) and is never the last item.
"""
import torch

from .base import Task


class AssociativeRecall(Task):
    def __init__(self, num_bits=8, min_len=2, max_len=6, block_len=3):
        assert min_len >= 2, "need at least two items so the query item has a successor"
        self.num_bits = num_bits
        self.min_len = min_len
        self.max_len = max_len
        self.block_len = block_len
        self.num_symbols = 2 ** (num_bits - 2)
        self.delimiter = self.num_symbols
        self.marker = self.num_symbols + 1
        self.blank = self.num_symbols + 2
        self.input_size = self.output_size = self.num_symbols + 3

    def sample(self, batch_size, generator):
        n_items = int(torch.randint(self.min_len, self.max_len + 1, (1,), generator=generator))
        stride = self.block_len + 1
        total = stride * (n_items + 2) + 1
        items = torch.randint(0, self.num_symbols, (batch_size, n_items, stride), generator=generator)
        items[:, :, -1] = self.delimiter
        rows = torch.arange(batch_size)
        key = torch.randint(0, n_items - 1, (batch_size,), generator=generator)

        x = torch.full((batch_size, total), self.blank)
        y = torch.full_like(x, self.blank)
        x[:, :stride * n_items] = items.flatten(1)
        x[:, stride * n_items] = self.marker
        x[:, stride * n_items + 1:stride * (n_items + 1)] = items[rows, key, :self.block_len]
        x[:, stride * (n_items + 1)] = self.marker
        y[:, stride * (n_items + 1) + 1:stride * (n_items + 2)] = items[rows, key + 1, :self.block_len]
        y[:, -1] = self.marker
        mask = torch.zeros(batch_size, total, dtype=torch.bool)
        mask[:, stride * (n_items + 1) + 1:] = True
        return x, y, mask
