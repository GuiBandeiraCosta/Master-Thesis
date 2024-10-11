import torch
import torch.nn as nn
import math

import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
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
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
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
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
       
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



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




class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x



class LearnedPositionalEncoding1D(nn.Module):
    def __init__(self, max_seq_length, d_model):
        super(LearnedPositionalEncoding1D, self).__init__()
        # Positional embeddings are trainable parameters
        self.positional_embeddings = nn.Embedding(max_seq_length, d_model)
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        position_embeddings = self.positional_embeddings(positions)
        return x + position_embeddings


class Transformer(nn.Module):
    def __init__(self, is_2D, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, max_grid_size=30):
        super(Transformer, self).__init__()
        self.shared_embedding = nn.Embedding(src_vocab_size, d_model)
        self.is_2D = is_2D
        
        # Choose between learned positional encoding based on whether input is 1D or 2D
        if is_2D:
            #self.positional_encoding = LearnedPositionalEncoding2D(max_grid_size, d_model)
            print("Using Learned 2D Positional Encoding")
        else:
            self.positional_encoding = LearnedPositionalEncoding1D(max_seq_length, d_model)
            print("Using Learned 1D Positional Encoding")
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(tgt.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, grid_sizes=None):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
       
        if self.is_2D:
            src_embedded = self.dropout(self.positional_encoding(self.shared_embedding(src), grid_sizes))
            tgt_embedded = self.dropout(self.positional_encoding(self.shared_embedding(tgt), grid_sizes))
        else:
            src_embedded = self.dropout(self.positional_encoding(self.shared_embedding(src)))
            tgt_embedded = self.dropout(self.positional_encoding(self.shared_embedding(tgt)))
        
        enc_output = src_embedded
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc(dec_output)
        return output

    

