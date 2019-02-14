import math

import torch
from torch import nn


class ConstantPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx):
        super(ConstantPositionalEmbedding, self).__init__()

        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx

        self.register_buffer('_position_embedding', None)

    @classmethod
    def get_embedding(cls, seq_len, embedding_dim):

        half_dim = embedding_dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(seq_len, -1)

        if embedding_dim % 2:
            emb = torch.cat([emb, torch.zeros(seq_len, 1)], dim=1)

        return emb

    def forward(self, x):
        batch_size, seq_len = x.size()

        if self._position_embedding is None or seq_len > self._position_embedding.size(0):
            self._position_embedding = PositionalEmbedding.get_embedding(seq_len, self._embedding_dim)

        content_mask = x.ne(self._padding_idx).long()
        positions = content_mask * torch.arange(seq_len).unsqueeze(0)

        return self._position_embedding.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).to(x.device)
