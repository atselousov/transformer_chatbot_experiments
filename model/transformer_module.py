#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm


class ConstantPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.register_buffer('_embedding', ConstantPositionalEmbedding.get_embedding(1024, self.embedding_dim))

    @staticmethod
    def get_embedding(n_embeddings, embedding_dim, device=None):
        n_embeddings += 1  # 0 is the padding

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
        emb = torch.arange(n_embeddings, dtype=torch.float, device=device).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(n_embeddings, -1)
        emb[0, :] = 0

        if embedding_dim % 2:
            emb = torch.cat([emb, torch.zeros(n_embeddings, 1, dtype=torch.float, device=device)], dim=1)

        return emb

    def forward(self, positions):
        batch_size, seq_length = positions.shape

        if seq_length >= self._embedding.shape[0]:
            self._embedding = ConstantPositionalEmbedding.get_embedding(seq_length,
                                                                        self.embedding_dim,
                                                                        self._embedding.device)

        positions = positions.view(-1)
        pos_embeddings = self._embedding.index_select(0, positions)
        pos_embeddings = pos_embeddings.view(batch_size, seq_length, -1)

        return pos_embeddings


class CombinedEmbedding(nn.Module):
    def __init__(self, n_embeddings, n_pos_embeddings, embedding_dim, padding_idx=None,
                 constant_pos_embedding=False, sparse=False):
        super().__init__()

        self.tok_padding_idx = padding_idx
        self.pos_padding_idx = 0

        self.tok_embedding = nn.Embedding(n_embeddings, embedding_dim,
                                          padding_idx=self.tok_padding_idx, sparse=sparse)
        if constant_pos_embedding:
            self.pos_embedding = ConstantPositionalEmbedding(embedding_dim)
        else:
            self.pos_embedding = nn.Embedding(n_pos_embeddings + 1, embedding_dim,
                                              padding_idx=self.pos_padding_idx, sparse=sparse)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embedding.weight, std=0.02)
        if isinstance(self.pos_embedding, nn.Embedding):
            nn.init.normal_(self.pos_embedding.weight, std=0.01)

    def forward(self, x, add_length=0):
        assert x.dim() == 2 or x.dim() == 3

        if x.dim() == 2:
            x = x.unsqueeze(2)  # additional embeddings

        padding_mask = x[:, :, 0].eq(self.tok_padding_idx)
        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long) + add_length
        positions.masked_fill_(padding_mask, self.pos_padding_idx)

        x = self.tok_embedding(x)
        x = x.sum(dim=2)
        x += self.pos_embedding(positions)

        return x, padding_mask


class MultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        nd, ns = size
        max_size = max(nd, ns)
        if not hasattr(cls, '_future_mask') or \
                cls._future_mask.device != device or \
                any(s < max_size for s in cls._future_mask.shape):
            cls._future_mask = torch.triu(torch.ones(max_size, max_size, dtype=torch.uint8, device=device), 1)

        mask = cls._future_mask[ns-nd:ns, :ns]  # future mask when we already may have past pre-computed values: take a slice at the end of the mask

        return mask

    def __init__(self, n_features, n_heads, dropout, future_mask=True):
        super().__init__()

        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.future_mask = future_mask
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, apply_future_mask=True, padding_mask=None):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        if apply_future_mask:
            future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
            w.masked_fill_(future_mask, float('-inf'))

        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        mask = (w == float('-inf')).all(dim=-1)

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        w.masked_fill_(mask.unsqueeze(-1), 0)

        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, query, key, value, padding_mask, past_key_value=None, past_query=None):
        qkv_same = (query.data_ptr() == key.data_ptr() == value.data_ptr())
        kv_same = (key.data_ptr() == value.data_ptr())

        if qkv_same:
            query, key, value = self.qkv_proj(query).split(self.n_features, dim=-1)
            if past_key_value is not None:  # we have already computed part of key, value of this
                past_key, past_value = past_key_value
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            apply_future_mask = self.future_mask  # self-attention

        elif kv_same:
            if past_query is not None:  # we have already computed this query
                query = past_query
            else:
                q_w, q_b = self.qkv_proj.weight[:self.n_features, :], self.qkv_proj.bias[:self.n_features]
                query = F.linear(query, q_w, q_b)

            if past_key_value is not None:  # we have already computed key, value of this
                key, value = past_key_value
            else:
                kv_w, kv_b = self.qkv_proj.weight[self.n_features:, :], self.qkv_proj.bias[self.n_features:]
                key, value = F.linear(key, kv_w, kv_b).split(self.n_features, dim=-1)

            apply_future_mask = False
        else:
            assert False

        saved_key_value = (key, value)
        saved_query = query

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, apply_future_mask, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x, saved_key_value, saved_query  # we can reuse: key/value for next forward steps, query for next attention ops


class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, in_features, middle_features, dropout):
        super().__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = self.layer_1(x)
        x = FeedForward.gelu(x)
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class GatedResidual(nn.Module):
    """ A gated residual layer: see https://arxiv.org/abs/1810.03581
    """
    def __init__(self, in_features):
        super().__init__()
        self.linear_additional = nn.Linear(in_features, 1, bias=True)
        self.linear_residual = nn.Linear(in_features, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.linear_additional.weight, std=0.02)
        nn.init.normal_(self.linear_residual.weight, std=0.02)

        self.linear_additional.bias.data[:] = 5
        self.linear_residual.bias.data[:] = 0

    def forward(self, additional_x, residual_x):
        gate = torch.sigmoid(self.linear_additional(additional_x) + self.linear_residual(residual_x))
        return gate * residual_x + (1 - gate) * additional_x


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout=0, attn_dropout=0, ff_dropout=0, normalize_before=False,
                 successive_attention=False, shared_attention=True, future_mask=True, context_size=0):
        super().__init__()

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout, future_mask)
        self.context_attns = [self.attn if shared_attention else copy.deepcopy(self.attn) for _ in range(context_size)]
        if not shared_attention:
            self.context_attns = nn.ModuleList(self.context_attns)
        self.gated_residual = GatedResidual(n_features) if successive_attention else None
        self.attn_norm = LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def _process_attn(self, x, padding_mask, contexts, layer_past):
        residual = x

        if self.normalize_before:
            x = self.attn_norm(x)

        inputs = [(x, padding_mask)] + contexts
        result_attns = []
        saved_kv, query = [], None
        if layer_past is None:
            layer_past = [None] * len(inputs)

        for i, attn_past_kv in zip(range(len(inputs)), layer_past):
            c, m = inputs[i]
            attn = self.attn if i == 0 else self.context_attns[i - 1]
            a, key_value, query = attn(x, c, c, m, past_key_value=attn_past_kv, past_query=query)
            
            saved_kv.append(key_value)
            result_attns.append(a)

        if self.gated_residual is not None:
            for i, a in enumerate(result_attns):
                a = self.dropout(a)
                x = residual + a if i == 0 else self.gated_residual(a, x)
        else:
            a = sum(result_attns, 0) / len(result_attns)
            a = self.dropout(a)
            x = residual + a

        if not self.normalize_before:
            x = self.attn_norm(x)

        return x, saved_kv

    def _process_ff(self, x):
        residual = x

        if self.normalize_before:
            x = self.ff_norm(x)

        x = self.ff(x)
        x = self.dropout(x)
        x = residual + x

        if not self.normalize_before:
            x = self.ff_norm(x)

        return x

    def forward(self, x, padding_mask, contexts, layer_past=None):
        # contexts = [(context1, padding_mask1), ...]
        x, saved_kv = self._process_attn(x, padding_mask, contexts, layer_past)
        x = self._process_ff(x)

        return x, saved_kv


class TransformerModule(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embedding_dim, 
                 padding_idx, n_heads, dropout=0, embedding_dropout=0, attn_dropout=0, ff_dropout=0,
                 constant_pos_embedding=False, sparse_embedding=False, normalize_before=False,
                 successive_attention=False, shared_attention=True, context_size=0):
        super().__init__()

        self.embedding = CombinedEmbedding(n_embeddings=n_embeddings,
                                           n_pos_embeddings=n_pos_embeddings,
                                           embedding_dim=embedding_dim,
                                           padding_idx=padding_idx,
                                           constant_pos_embedding=constant_pos_embedding,
                                           sparse=sparse_embedding)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        base_block = TransformerBlock(n_features=embedding_dim,
                                      n_heads=n_heads,
                                      dropout=dropout,
                                      attn_dropout=attn_dropout,
                                      ff_dropout=ff_dropout,
                                      normalize_before=normalize_before,
                                      successive_attention=successive_attention,
                                      shared_attention=shared_attention,
                                      context_size=context_size)
        self.layers = nn.ModuleList([copy.deepcopy(base_block) for _ in range(n_layers)])
        self.final_norm = LayerNorm(embedding_dim) if normalize_before else lambda x: x

    def forward(self, x, enc_contexts=[], past=None):
        # x.dim() == 3 if we have additional embeddings else x.dim() == 2

        if past is None:  # past store previously computed keys/values for the current generated sequence
            past_length, past = 0, [None] * len(self.layers)
        else:
            past_length = past[0][0][0].size(-2)  # layer 0, attn ops 0, key (0)

        x, padding_mask = self.embedding(x, past_length)
        x = self.embedding_dropout(x)

        saved_keys_values = []
        for layer, layer_past in zip(self.layers, past):
            x, saved_kv = layer(x, padding_mask, enc_contexts, layer_past=layer_past)
            saved_keys_values.append(saved_kv)

        x = self.final_norm(x)

        return x, padding_mask, saved_keys_values
