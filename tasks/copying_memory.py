"""Copying memory task (Arjovsky et al., 2016), following myTorch's CopyingMemoryData.

Token layout (digit_range = num_noise_digits + num_digits + 1):
    noise digits:  [0, num_noise_digits)
    data digits:   [num_noise_digits, num_noise_digits + num_digits)
    marker:        digit_range - 1
Input (length 2 * seq_len + time_lag):
    seq_len data digits, then noise, with the marker at position seq_len + time_lag - 1
Target:
    0 everywhere except the last seq_len positions, which hold the data digits.
The time lag is sampled uniformly from [time_lag_min, time_lag_max] once per batch.
The loss covers every position (as in myTorch) unless mask_recall_only is set.
"""
import math
import torch

from .base import Task


class CopyingMemory(Task):
    def __init__(self, seq_len=10, time_lag_min=100, time_lag_max=100, num_digits=8, num_noise_digits=1,
                 mask_recall_only=False):
        self.seq_len = seq_len
        self.time_lag_min = time_lag_min
        self.time_lag_max = time_lag_max
        self.num_digits = num_digits
        self.num_noise_digits = num_noise_digits
        self.mask_recall_only = mask_recall_only
        self.marker = num_noise_digits + num_digits
        self.input_size = num_noise_digits + num_digits + 1
        self.output_size = num_noise_digits + num_digits

    def sample(self, batch_size, generator):
        k = self.seq_len
        time_lag = int(torch.randint(self.time_lag_min, self.time_lag_max + 1, (1,), generator=generator))
        length = 2 * k + time_lag
        x = torch.randint(0, self.num_noise_digits, (batch_size, length), generator=generator)
        data = torch.randint(self.num_noise_digits, self.marker, (batch_size, k), generator=generator)
        x[:, :k] = data
        x[:, k + time_lag - 1] = self.marker
        y = torch.zeros(batch_size, length, dtype=torch.long)
        y[:, k + time_lag:] = data
        mask = torch.ones(batch_size, length, dtype=torch.bool)
        if self.mask_recall_only:
            mask[:, :k + time_lag] = False
        return x, y, mask

    def metrics(self, logits, y, mask):
        correct = logits[:, -self.seq_len:].argmax(-1) == y[:, -self.seq_len:]
        return {
            "recall_acc": correct.float().mean().item(),
            "seq_acc": correct.all(dim=-1).float().mean().item(),
        }

    def format_sample(self, x, y, mask, logits):
        recall = torch.zeros_like(mask)
        recall[-self.seq_len:] = True
        return super().format_sample(x, y, recall, logits)

    @property
    def baseline_loss(self):
        # Output blanks until the recall phase, then guess uniformly over the data digits.
        if self.mask_recall_only:
            return math.log(self.num_digits)
        mean_length = 2 * self.seq_len + (self.time_lag_min + self.time_lag_max) / 2
        return self.seq_len * math.log(self.num_digits) / mean_length
