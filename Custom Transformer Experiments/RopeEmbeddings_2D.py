from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import math


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

class RotaryPositionalEmbeddings2D(nn.Module):
    """
    Implements 2D Rotary Positional Embeddings (RoPE) for 2D inputs.
    Args:
        dim (int): Embedding dimension. Usually this is set to the dimension of each head in the attention module.
        base (int): The base for the geometric progression used to compute the rotation angles.
        max_grid_size (int): Maximum grid size (for both rows and columns). This value sets the limit on how big the 2D grid can be.
    """
    def __init__(self, dim: int, max_grid_size: int = 30, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_grid_size = max_grid_size
        self._rope_init()

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        # Compute theta for rows and columns, both having the same dimension
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_grid_size)

    def build_rope_cache(self, max_grid_size: int = 30) -> None:
        # Create position indexes for rows and columns
        row_idx = torch.arange(max_grid_size, dtype=self.theta.dtype, device=self.theta.device)
        col_idx = torch.arange(max_grid_size, dtype=self.theta.dtype, device=self.theta.device)

        # Compute outer products for row and column positions with theta
        row_theta = torch.einsum("i,j -> ij", row_idx, self.theta).float()
        col_theta = torch.einsum("i,j -> ij", col_idx, self.theta).float()

        # Cache both cosine and sine components for row and column positional encodings
        row_cache = torch.stack([torch.cos(row_theta), torch.sin(row_theta)], dim=-1)
        col_cache = torch.stack([torch.cos(col_theta), torch.sin(col_theta)], dim=-1)

        self.register_buffer("row_cache", row_cache, persistent=False)
        self.register_buffer("col_cache", col_cache, persistent=False)

    def forward(self, x: Tensor, grid_sizes) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with shape [b, s, n_h, h_d].
            grid_sizes (Optional[Tensor]): Tensor of shape [b, 2] where each element
                                           represents (rows, cols) for each input in the batch.
        Returns:
            Tensor: Output tensor with RoPE applied on both row and column dimensions.
        """
        batch_size, seq_len, n_heads, head_dim = x.size()
        
        assert grid_sizes is not None, "grid_sizes must be provided"

        # Apply RoPE for each item in the batch
        for b in range(batch_size):
            rows, cols = grid_sizes[b]
            row_cache = self.row_cache[:rows]
            col_cache = self.col_cache[:cols]

            # Reshape input tensor for 2D positional embedding
            x_shaped = x[b].reshape(rows, cols, n_heads, head_dim // 2, 2)

            # Apply rotary embeddings for rows and columns
            for i in range(rows):
                for j in range(cols):
                    x_shaped[i, j] = torch.stack(
                        [
                            x_shaped[i, j, :, :, 0] * row_cache[i, :, 0] - x_shaped[i, j, :, :, 1] * row_cache[i, :, 1],
                            x_shaped[i, j, :, :, 1] * row_cache[i, :, 0] + x_shaped[i, j, :, :, 0] * row_cache[i, :, 1],
                        ], -1
                    )
                    x_shaped[i, j] = torch.stack(
                        [
                            x_shaped[i, j, :, :, 0] * col_cache[j, :, 0] - x_shaped[i, j, :, :, 1] * col_cache[j, :, 1],
                            x_shaped[i, j, :, :, 1] * col_cache[j, :, 0] + x_shaped[i, j, :, :, 0] * col_cache[j, :, 1],
                        ], -1
                    )

            # Flatten the last dimension and assign back to the original x tensor
            x[b] = x_shaped.flatten(2)

        return x




class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 512) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, *, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class DynamicPositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(DynamicPositionalEncoding2D, self).__init__()
        max_grid_size = 30
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.max_seq_length = max_seq_length  # Including /s and /e tokens

        # Precompute positional encodings for rows and columns up to max_grid_size
        row_position = torch.arange(0, max_grid_size, dtype=torch.float).unsqueeze(1)
        col_position = torch.arange(0, max_grid_size, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * -(math.log(10000.0) / d_model))

        pe_row = torch.zeros(max_grid_size, d_model // 2)
        pe_col = torch.zeros(max_grid_size, d_model // 2)

        pe_row[:, 0::2] = torch.sin(row_position * div_term)
        pe_row[:, 1::2] = torch.cos(row_position * div_term)

        pe_col[:, 0::2] = torch.sin(col_position * div_term)
        pe_col[:, 1::2] = torch.cos(col_position * div_term)

        self.register_buffer('pe_row', pe_row)
        self.register_buffer('pe_col', pe_col)
        
    def forward(self, x, grid_sizes):
        """
        x: input tensor (batch_size, sequence_length, d_model)
        grid_sizes: tensor (batch_size, 2) where each element is (rows, cols)
        """
        batch_size, seq_length, _ = x.size()

        # Initialize positional encoding for the batch
        pe = torch.zeros(batch_size, self.max_seq_length, self.d_model, device=x.device)

        for b in range(batch_size):
            rows, cols = grid_sizes[b]
            grid_pe = torch.zeros(rows * cols, self.d_model, device=x.device)

            # Efficiently populate the positional encoding for the grid
            grid_pe[:, :self.d_model // 2] = self.pe_row[:rows].repeat_interleave(cols, dim=0)
            grid_pe[:, self.d_model // 2:] = self.pe_col[:cols].repeat(rows, 1)

            # Assign to pe tensor/ Is redundant given the last line of :seq_length
            pe[b, 1:1 + min(rows * cols, seq_length - 1)] = grid_pe[:min(rows * cols, seq_length - 1)]

        # Add positional encoding to the input tensor
        x = x + pe[:, :seq_length]

        return x


# Update the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # RoPE is applied here
        #self.rope = RotaryPositionalEmbeddings(dim=self.d_k, max_seq_len=max_seq_len)
        self.rope = RotaryPositionalEmbeddings2D(dim=self.d_k)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, grid_sizes,mask=None):
        # Split into multiple heads
        Q = self.split_heads(self.W_q(Q))  # Shape: (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(self.W_k(K))  # Shape: (batch_size, num_heads, seq_length, d_k)
        V = self.split_heads(self.W_v(V))  # Shape: (batch_size, num_heads, seq_length, d_k)
        
        # Reshape to [batch_size, seq_length, num_heads, d_k]
        Q = Q.transpose(1, 2)  # Now Q is [batch_size, seq_length, num_heads, d_k]
        K = K.transpose(1, 2)  # Now K is [batch_size, seq_length, num_heads, d_k]

        # Apply RoPE to Q and K
        Q = self.rope(Q,grid_sizes)
        K = self.rope(K,grid_sizes)

        # Reshape back to [batch_size, num_heads, seq_length, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads back
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, max_seq_len):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, max_seq_len)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, grid_sizes,mask):
        attn_output = self.self_attn(x, x, x,grid_sizes, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, max_seq_len):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, max_seq_len)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, max_seq_len)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, grid_sizes,tgt_mask):
        attn_output = self.self_attn(x, x, x, grid_sizes,tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, grid_sizes,src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, is_2D,src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout):
        super(Transformer, self).__init__()
        self.shared_embedding = nn.Embedding(src_vocab_size, d_model)
        self.is_2D = is_2D
        if is_2D:
            #self.positional_encoding = DynamicPositionalEncoding2D(d_model, max_seq_len)
            print("Using 2D  ROPE")
        else:
            print("USING ROPE")
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, max_seq_len) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, max_seq_len) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(tgt.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, grid_sizes):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # if self.is_2D:
            
        #     src_embedded = self.dropout(self.positional_encoding(self.shared_embedding(src),grid_sizes))
        #     tgt_embedded = self.dropout(self.positional_encoding(self.shared_embedding(tgt),grid_sizes))
        # else:
        
        src_embedded = self.dropout(self.shared_embedding(src))
        tgt_embedded = self.dropout(self.shared_embedding(tgt))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, grid_sizes,src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask,grid_sizes ,tgt_mask)
        
        output = self.fc(dec_output)
        return output
