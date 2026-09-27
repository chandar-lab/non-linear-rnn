"""Denoising task (Jing et al., 2017), following myTorch's DenoisingData.

Same token layout as the copying memory task. Input (length seq_len + time_lag + 1):
    time_lag noise positions with the seq_len data digits scattered at random (ordered) positions among
    them, then the marker, then seq_len noise positions
Target:
    the data digits, in order, over the last seq_len positions; the loss only covers those positions.
The time lag is sampled uniformly from [time_lag_min, time_lag_max] once per batch.
"""
import math
import torch

from .base import Task


class Denoising(Task):
    def __init__(self, seq_len=10, time_lag_min=100, time_lag_max=100, num_digits=8, num_noise_digits=1):
        assert time_lag_min >= seq_len, "the data digits must fit in the noise region"
        self.seq_len = seq_len
        self.time_lag_min = time_lag_min
        self.time_lag_max = time_lag_max
        self.num_digits = num_digits
        self.num_noise_digits = num_noise_digits
        self.marker = num_noise_digits + num_digits
        self.input_size = num_noise_digits + num_digits + 1
        self.output_size = num_noise_digits + num_digits

    def sample(self, batch_size, generator):
        k = self.seq_len
        time_lag = int(torch.randint(self.time_lag_min, self.time_lag_max + 1, (1,), generator=generator))
        length = k + time_lag + 1
        data = torch.randint(self.num_noise_digits, self.marker, (batch_size, k), generator=generator)
        x = torch.randint(0, self.num_noise_digits, (batch_size, length), generator=generator)
        positions = torch.rand(batch_size, time_lag, generator=generator).argsort(-1)[:, :k].sort(-1).values
        x[:, :time_lag].scatter_(1, positions, data)
        x[:, time_lag] = self.marker
        x[:, time_lag + 1:] = 0
        y = torch.zeros(batch_size, length, dtype=torch.long)
        y[:, -k:] = data
        mask = torch.zeros(batch_size, length, dtype=torch.bool)
        mask[:, -k:] = True
        return x, y, mask

    @property
    def baseline_loss(self):
        return math.log(self.num_digits)
