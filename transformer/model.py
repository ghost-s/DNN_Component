import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p = 0.):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, query, key, value, attn_mask = None):

        embed_size = query.shape[-1]
        scaling_factor = torch.sqrt(torch.tensor(embed_size, dtype=torch.float32))
        attn = torch.matmul(query, torch.transpose(key, -1, -2) / scaling_factor)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)

        attn = self.softmax(attn)

        attn = self.dropout(attn)

        output = torch.matmul(attn, value)

        return output, attn


# 文本补全后的部分不应该拥有注意力权重
def get_attn_pad_mask(seq_q, seq_k, pad_val):

    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape

    pad_attn_mask = (seq_k == pad_val)
    pad_attn_mask = pad_attn_mask.unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand([batch_size, len_q, len_k])

    return pad_attn_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, dropout_p = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        self.W_O = nn.Linear(d_k * n_heads, d_model)
        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def forward(self, query, key, value, attn_mask):

        batch_size = query.shape[0]

        q_s = self.W_Q(query).reshape(batch_size, -1, self.n_heads, self.d_k)
        k_s = self.W_K(key).reshape(batch_size, -1, self.n_heads, self.d_k)
        v_s = self.W_V(value).reshape(batch_size, -1, self.n_heads, self.d_k)

        q_s = q_s.permute(0, 2, 1, 3)
        k_s = k_s.permute(0, 2, 1, 3)
        v_s = v_s.permute(0, 2, 1, 3)

        attn_mask = attn_mask.unsqueeze(1)
        attn_mask = attn_mask.repeat(1, self.n_heads, 1, 1)
        context, attn = self.attention(q_s, k_s, v_s, attn_mask)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * self.d_k)

        output = self.W_O(context)

        return output, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        self.pe = torch.zeros([max_len, d_model], dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)
        angle = torch.pow(10000.0, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model)
        self.pe[:, 0::2] = torch.sin(pos/angle)
        self.pe[:, 1::2] = torch.cos(pos/angle)

    def forward(self, x):
        batch_size = x.shape[0]
        pe = self.pe.unsqueeze(0)
        pe = pe.expand(batch_size, -1, -1)

        x = x + pe[:, :x.shape[1], :]
        return self.dropout(x)


class PoswiseFeedForward(nn.Module):
    def __init__(self, d_ff, d_model, dropout_p=0.):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.linear2(x)
        return output


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout_p=0.):
        super().__init__()
        self.layer_norm = nn.LayerNorm((d_model,), eps=1e-5)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, residual):
        return self.layer_norm(self.dropout(x) + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.):
        super().__init__()
        d_k = d_model // n_heads
        if d_k * n_heads != d_model:
            raise ValueError(f"The 'd_model' {d_model} can not be divisible into 'n_heads' {n_heads}.")
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, n_heads, dropout_p)
        self.pos_ffn = PoswiseFeedForward(d_ff, d_model, dropout_p)
        self.add_norm1 = AddNorm(d_model, dropout_p)
        self.add_norm2 = AddNorm(d_model, dropout_p)

    def forward(self, enc_inputs, enc_self_attn_mask):
        residual = enc_inputs
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.add_norm1(enc_outputs, residual)
        residual = enc_outputs
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = self.add_norm2(enc_outputs, residual)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout_p)
        self.layers = nn.Sequential(*[EncoderLayer(d_model, n_heads, d_ff, dropout_p) for _ in range(n_layers)])
        self.scaling_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

    def forward(self, enc_inputs, src_pad_idx):
        enc_outputs = self.src_emb(enc_inputs.to(dtype=torch.int32))
        enc_outputs = self.pos_emb(enc_outputs * self.scaling_factor)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, src_pad_idx)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attn


def get_attn_subsequent_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape

    ones = torch.ones([batch_size, len_q, len_k], dtype=torch.float32)
    return torch.triu(ones, diagonal=1)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.):
        super().__init__()
        d_k = d_model // n_heads
        if d_k * n_heads != d_model:
            raise ValueError(f"The 'd_model' {d_model} can not be divisible into 'n_heads' {n_heads}.")
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, n_heads, dropout_p)
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, n_heads, dropout_p)
        self.pos_ffn = PoswiseFeedForward(d_ff, d_model, dropout_p)
        self.add_norm1 = AddNorm(d_model, dropout_p)
        self.add_norm2 = AddNorm(d_model, dropout_p)
        self.add_norm3 = AddNorm(d_model, dropout_p)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        residual = dec_inputs
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.add_norm1(dec_outputs, residual)
        residual = dec_outputs
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.add_norm2(dec_outputs, residual)
        residual = dec_outputs
        dec_outputs = self.pos_ffn(dec_outputs)
        dec_outputs = self.add_norm3(dec_outputs, residual)

        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.):
        super().__init__()
        self.trg_emb = nn.Embedding(trg_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout_p)
        self.layers = nn.Sequential(*[DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.projection = nn.Linear(d_model, trg_vocab_size)
        self.scaling_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

    def forward(self, dec_inputs, enc_inputs, enc_outputs, src_pad_idx, trg_pad_idx):
        """
        dec_inputs: [batch_size, trg_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        """
        dec_outputs = self.trg_emb(dec_inputs.to(torch.int32))
        dec_outputs = self.pos_emb(dec_outputs * self.scaling_factor)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, trg_pad_idx)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs, dec_inputs)
        dec_self_attn_mask = (dec_self_attn_pad_mask + dec_self_attn_subsequent_mask) > 0

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, src_pad_idx)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        dec_outputs = self.projection(dec_outputs)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_inputs, dec_inputs, src_pad_idx, trg_pad_idx):
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, trg_len]
        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, src_pad_idx)

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, src_pad_idx, trg_pad_idx)

        dec_logits = dec_outputs.view((-1, dec_outputs.shape[-1]))

        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns