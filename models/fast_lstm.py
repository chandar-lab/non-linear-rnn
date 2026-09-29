"""LSTM with the fast forget gate (Ohno et al., 2023, "Fast Saturating Gate for Learning Long Time Scales
with Recurrent Neural Networks", https://arxiv.org/abs/2210.01348).

The forget gate is f = sigmoid(sinh(z)) instead of sigmoid(z): it saturates doubly exponentially, which
the paper shows mitigates gradient vanishing when learning long time scales. Everything else is a
standard LSTM. The forget bias is initialized so the gate starts where a sigmoid LSTM with bias
forget_bias would, i.e. sigmoid(sinh(b_f)) = sigmoid(forget_bias), b_f = asinh(forget_bias); the paper
uses forget_bias = 1.

Layers are stacked with pre-norm residual connections: u <- u + proj(cell(norm(u))), output = norm(u).
"""
import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight

class FastLSTMCell(nn.Module):
    def __init__(self, embed_size, hidden_size, forget_bias=1.0):
        super().__init__()
        self.Wx = nn.Linear(embed_size, 4 * hidden_size)
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        with torch.no_grad():
            self.Wx.bias[hidden_size:2 * hidden_size].fill_(math.asinh(forget_bias))

    def forward(self, x, h, c):
        i, f, g, o = (self.Wx(x) + self.Wh(h)).chunk(4, dim=-1)
        # Clamping doesn't change the value (sigmoid(sinh(20)) is exactly 1 in fp32) but keeps sinh / cosh
        # finite, so the backward pass can't produce 0 * inf = NaN for very large pre-activations.
        f = torch.sigmoid(torch.sinh(f.clamp(-20, 20)))
        c = f * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class FastLSTM(nn.Module):
    """Multi-layer, batch-first fast-gate LSTM. State is (hidden, cell), each (num_layers, batch, hidden_size)."""
    def __init__(self, embed_size, num_layers, hidden_size=None, forget_bias=1.0, chunk_size=16, compile=False):
        super().__init__()
        hidden_size = hidden_size or embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.cells = nn.ModuleList([FastLSTMCell(embed_size, hidden_size, forget_bias) for _ in range(num_layers)])
        self.in_norms = nn.ModuleList([RMSNorm(embed_size) for _ in range(num_layers)])
        self.out_projs = nn.ModuleList([
            nn.Identity() if hidden_size == embed_size else nn.Linear(hidden_size, embed_size, bias=False)
            for _ in range(num_layers)
        ])
        self.out_norm = RMSNorm(embed_size)
        # See models/nru.py for why time steps are compiled in chunks.
        if compile:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self.run_chunk = torch.compile(self._run_chunk, dynamic=False)
        else:
            self.run_chunk = self._run_chunk

    def _run_chunk(self, x, hidden, cell_state):
        hidden, cell_state = list(hidden.unbind(0)), list(cell_state.unbind(0))
        outputs = []
        for t in range(x.size(1)):
            u = x[:, t]
            for i, cell in enumerate(self.cells):
                # hidden[i], cell_state[i] = cell(self.in_norms[i](u), hidden[i], cell_state[i])
                hidden[i], cell_state[i] = cell(u, hidden[i], cell_state[i])
                # u = u + self.out_projs[i](hidden[i])
                u = self.out_projs[i](hidden[i])
            # outputs.append(self.out_norm(u))
            outputs.append(u)
        return torch.stack(outputs, dim=1), torch.stack(hidden), torch.stack(cell_state)

    def forward(self, x, state=None):
        if state is None:
            state = (x.new_zeros(self.num_layers, x.size(0), self.hidden_size),
                     x.new_zeros(self.num_layers, x.size(0), self.hidden_size))
        hidden, cell_state = state
        outputs = []
        for start in range(0, x.size(1), self.chunk_size):
            out, hidden, cell_state = self.run_chunk(x[:, start:start + self.chunk_size], hidden, cell_state)
            outputs.append(out)
        return torch.cat(outputs, dim=1), (hidden, cell_state)

class FastLSTMModel(nn.Module):
    def __init__(self, vocab_size, output_size, embed_size, num_layers, hidden_size=None, forget_bias=1.0,
                 chunk_size=16, compile=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = FastLSTM(embed_size, num_layers, hidden_size, forget_bias, chunk_size, compile)
        self.head = nn.Linear(embed_size, output_size, bias=False)

    def forward(self, idx, state=None):
        x, state = self.rnn(self.embedding(idx), state)
        return self.head(x), state
