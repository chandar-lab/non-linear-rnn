"""Gated Recurrent Unit, reset-after variant (same equations as torch.nn.GRU).

Layers are stacked with pre-norm residual connections: u <- u + proj(cell(norm(u))), output = norm(u).
"""
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

class GRUCell(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.Wx = nn.Linear(embed_size, 3 * hidden_size)
        self.Wh = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        xr, xz, xn = self.Wx(x).chunk(3, dim=-1)
        hr, hz, hn = self.Wh(h).chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h

class GRU(nn.Module):
    """Multi-layer, batch-first GRU. State is hidden, (num_layers, batch, hidden_size)."""
    def __init__(self, embed_size, num_layers, hidden_size=None, chunk_size=16, compile=False):
        super().__init__()
        hidden_size = hidden_size or embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.cells = nn.ModuleList([GRUCell(embed_size, hidden_size) for _ in range(num_layers)])
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

    def _run_chunk(self, x, hidden):
        hidden = list(hidden.unbind(0))
        outputs = []
        for t in range(x.size(1)):
            u = x[:, t]
            for i, cell in enumerate(self.cells):
                # hidden[i] = cell(self.in_norms[i](u), hidden[i])
                hidden[i] = cell(u, hidden[i])
                # u = u + self.out_projs[i](hidden[i])
                u = self.out_projs[i](hidden[i])
            # outputs.append(self.out_norm(u))
            outputs.append(u)
        return torch.stack(outputs, dim=1), torch.stack(hidden)

    def forward(self, x, state=None):
        hidden = x.new_zeros(self.num_layers, x.size(0), self.hidden_size) if state is None else state
        outputs = []
        for start in range(0, x.size(1), self.chunk_size):
            out, hidden = self.run_chunk(x[:, start:start + self.chunk_size], hidden)
            outputs.append(out)
        return torch.cat(outputs, dim=1), hidden

class GRUModel(nn.Module):
    def __init__(self, vocab_size, output_size, embed_size, num_layers, hidden_size=None,
                 chunk_size=16, compile=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = GRU(embed_size, num_layers, hidden_size, chunk_size, compile)
        self.head = nn.Linear(embed_size, output_size, bias=False)

    def forward(self, idx, state=None):
        x, state = self.rnn(self.embedding(idx), state)
        return self.head(x), state
