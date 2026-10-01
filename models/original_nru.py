"""Non-saturating Recurrent Unit (Chandar et al., 2019, https://arxiv.org/abs/1902.06704), original version.

    h_t = ReLU(W_i x_t + W_h h_{t-1} + W_c m_{t-1})
    m_t = m_{t-1} + mean_i(a_i v^w_i - b_i v^e_i)
The strengths a, b and the directions v^w, v^e are computed from [x_t, h_t, m_{t-1}]. Each set of
num_heads directions uses the outer product trick: a linear map to 2 * sqrt(num_heads * memory_size)
values p, q, then vec(p q^T) reshaped to (num_heads, memory_size), followed by ReLU and L5 normalization
over the memory dim (as in the paper and the official code, which also averages over heads).
Layers are stacked plainly: each layer's hidden state is the next layer's input; the last layer's
hidden state is the output. No residual connections or normalization layers.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class OriginalNRUCell(nn.Module):
    def __init__(self, embed_size, memory_size, num_heads):
        super().__init__()
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.rank = math.isqrt(num_heads * memory_size)
        assert self.rank ** 2 == num_heads * memory_size, "num_heads * memory_size must be a perfect square"
        self.Wi = nn.Linear(embed_size, embed_size)
        self.Wh = nn.Linear(embed_size, embed_size)
        self.Wc = nn.Linear(memory_size, embed_size)
        feature_size = 2 * embed_size + memory_size
        self.f_ab = nn.Linear(feature_size, 2 * num_heads)  # write / erase strengths
        self.f_w = nn.Linear(feature_size, 2 * self.rank)   # write direction factors p, q
        self.f_e = nn.Linear(feature_size, 2 * self.rank)   # erase direction factors p, q

    def _directions(self, pq):
        p, q = pq.chunk(2, dim=-1)
        v = (p.unsqueeze(-1) * q.unsqueeze(-2)).reshape(-1, self.num_heads, self.memory_size)
        return F.normalize(F.relu(v), p=5, dim=-1)

    def forward(self, x, h, m):
        h = F.relu(self.Wi(x) + self.Wh(h) + self.Wc(m))
        features = torch.cat([x, h, m], dim=-1)
        a, b = F.relu(self.f_ab(features)).unsqueeze(-1).chunk(2, dim=1)
        w = self._directions(self.f_w(features))
        e = self._directions(self.f_e(features))
        m = m + (a * w - b * e).mean(dim=1)
        return h, m

class OriginalNRU(nn.Module):
    """Multi-layer, batch-first NRU. State is (hidden, memory), each (num_layers, batch, size)."""
    def __init__(self, embed_size, num_layers, memory_size=64, num_heads=4, chunk_size=16, compile=False):
        super().__init__()
        self.embed_size = embed_size
        self.memory_size = memory_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.cells = nn.ModuleList([OriginalNRUCell(embed_size, memory_size, num_heads) for _ in range(num_layers)])
        # See models/new_nru.py for why time steps are compiled in chunks.
        if compile:
            self.run_chunk = torch.compile(self._run_chunk, dynamic=False, options={"triton.cudagraphs": False})
        else:
            self.run_chunk = self._run_chunk

    def _run_chunk(self, x, hidden, memory):
        hidden, memory = list(hidden.unbind(0)), list(memory.unbind(0))
        outputs = []
        for t in range(x.size(1)):
            u = x[:, t]
            for i, cell in enumerate(self.cells):
                hidden[i], memory[i] = cell(u, hidden[i], memory[i])
                u = hidden[i]
            outputs.append(u)
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

class OriginalNRUModel(nn.Module):
    def __init__(self, vocab_size, output_size, embed_size, num_layers, memory_size=64, num_heads=4,
                 chunk_size=16, compile=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = OriginalNRU(embed_size, num_layers, memory_size, num_heads, chunk_size, compile)
        self.head = nn.Linear(embed_size, output_size, bias=False)

    def forward(self, idx, state=None):
        x, state = self.rnn(self.embedding(idx), state)
        return self.head(x), state
