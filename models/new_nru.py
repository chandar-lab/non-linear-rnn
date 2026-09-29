"""Non-saturating Recurrent Unit (Chandar et al., 2019), modified version from the preliminary notebook.

The memory is updated additively with non-negative write (w) and erase (e) vectors, each built per head
as a low-rank outer product p q^T of size (num_heads, memory_size), gated by non-negative scalars a and b.
Layers are stacked with pre-norm residual connections: u <- u + cell(norm(u)), output = norm(u).
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight

class NewNRUCell(nn.Module):
    def __init__(self, embed_size, memory_size, num_heads):
        super().__init__()
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.rank = 8
        self.Wi = nn.Linear(embed_size, embed_size)
        self.Wh = nn.Linear(embed_size, embed_size)
        self.Wc = nn.Linear(memory_size, embed_size)
        feature_size = 2 * embed_size + memory_size
        self.f_ab = nn.Linear(feature_size, 2 * num_heads)  # write / erase strengths
        # Use low rank weights for f_w and f_e
        self.f_w_a = nn.Parameter(torch.empty(self.rank, feature_size))
        self.f_w_b = nn.Parameter(torch.empty(self.num_heads * self.memory_size, self.rank))
        self.b_w = nn.Parameter(torch.zeros(self.num_heads * self.memory_size))
        self.f_e_a = nn.Parameter(torch.empty(self.rank, feature_size))
        self.f_e_b = nn.Parameter(torch.empty(self.num_heads * self.memory_size, self.rank))
        self.b_e = nn.Parameter(torch.zeros(self.num_heads * self.memory_size))
        self.norm_h = RMSNorm(embed_size)
        self.norm_m = RMSNorm(memory_size)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.f_w_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.f_w_b, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.f_e_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.f_e_b, a=math.sqrt(5))
    
    def forward(self, x, h, m):
        h = F.relu(self.Wi(x) + self.Wh(h) + self.Wc(m))
        features = torch.cat([x, self.norm_h(h), self.norm_m(m)], dim=-1)
        a, b = F.relu(self.f_ab(features)).unsqueeze(-1).chunk(2, dim=1)
        w = F.relu(F.linear(features, self.f_w_b @ self.f_w_a) + self.b_w).view(-1, self.num_heads, self.memory_size)
        e = F.relu(F.linear(features, self.f_e_b @ self.f_e_a) + self.b_e).view(-1, self.num_heads, self.memory_size)
        m = m + (a * w - b * e).mean(dim=1)
        return h, m

class NewNRU(nn.Module):
    """Multi-layer, batch-first NRU. State is (hidden, memory), each (num_layers, batch, size)."""
    def __init__(self, embed_size, num_layers, memory_size=50, num_heads=2, chunk_size=16, compile=False):
        super().__init__()
        self.embed_size = embed_size
        self.memory_size = memory_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.cells = nn.ModuleList([NewNRUCell(embed_size, memory_size, num_heads) for _ in range(num_layers)])
        self.in_norms = nn.ModuleList([RMSNorm(embed_size) for _ in range(num_layers)])
        self.out_norm = RMSNorm(embed_size)
        # Compiling a fixed-size chunk of time steps keeps cross-step fusion without the very slow
        # lowering of one monolithic graph over the whole sequence. Each distinct (batch, chunk length)
        # shape compiles once, so allow enough cached graphs for variable-length tasks.
        if compile:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self.run_chunk = torch.compile(self._run_chunk, dynamic=False)
        else:
            self.run_chunk = self._run_chunk

    def _run_chunk(self, x, hidden, memory):
        hidden, memory = list(hidden.unbind(0)), list(memory.unbind(0))
        outputs = []
        for t in range(x.size(1)):
            u = x[:, t]
            for i, cell in enumerate(self.cells):
                hidden[i], memory[i] = cell(self.in_norms[i](u), hidden[i], memory[i])
                u = u + hidden[i]
            outputs.append(self.out_norm(u))
        return torch.stack(outputs, dim=1), torch.stack(hidden), torch.stack(memory)

    def forward(self, x, state=None):
        if state is None:
            state = (x.new_zeros(self.num_layers, x.size(0), self.embed_size),
                     x.new_zeros(self.num_layers, x.size(0), self.memory_size))
        hidden, memory = state
        outputs = []
        for start in range(0, x.size(1), self.chunk_size):
            out, hidden, memory = self.run_chunk(x[:, start:start + self.chunk_size], hidden, memory)
            outputs.append(out)
        return torch.cat(outputs, dim=1), (hidden, memory)

class NewNRUModel(nn.Module):
    def __init__(self, vocab_size, output_size, embed_size, num_layers, memory_size=50, num_heads=2,
                 chunk_size=16, compile=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = NewNRU(embed_size, num_layers, memory_size, num_heads, chunk_size, compile)
        self.head = nn.Linear(embed_size, output_size, bias=False)

    def forward(self, idx, state=None):
        x, state = self.rnn(self.embedding(idx), state)
        return self.head(x), state
