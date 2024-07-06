import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
from tqdm import tqdm
import os
import re
import numpy as np


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


class Multi30K(Dataset):
    def __init__(self, path):
        self.data = self._load(path)

    def _load(self, path):
        def tokenize(text):
            text = text.rstrip()
            return [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', text)]

        members = {i.split('.')[-1]: i for i in os.listdir(path)}
        de_path = os.path.join(path, members['de'])
        en_path = os.path.join(path, members['en'])
        with open(de_path, 'r', encoding='utf-8') as de_file:
            de = de_file.readlines()[:-1]
            de = [tokenize(i) for i in de]
        with open(en_path, 'r', encoding='utf-8') as en_file:
            en = en_file.readlines()[:-1]
            en = [tokenize(i) for i in en]

        return list(zip(de, en))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


train_dataset, valid_dataset, test_dataset = Multi30K("datasets/train"), Multi30K("datasets/valid"), Multi30K("datasets/test")


class Vocab:

    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, word_count_dict, min_freq=1):
        self.word2idx = {}
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx

        filted_dict = {
            w: c
            for w, c in word_count_dict.items() if c >= min_freq
        }
        for w, _ in filted_dict.items():
            self.word2idx[w] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.bos_idx = self.word2idx['<bos>']
        self.eos_idx = self.word2idx['<eos>']
        self.pad_idx = self.word2idx['<pad>']
        self.unk_idx = self.word2idx['<unk>']

    def _word2idx(self, word):
        """单词映射至数字索引"""
        if word not in self.word2idx:
            return self.unk_idx
        return self.word2idx[word]

    def _idx2word(self, idx):
        """数字索引映射至单词"""
        if idx not in self.idx2word:
            raise ValueError('input index is not in vocabulary.')
        return self.idx2word[idx]

    def encode(self, word_or_list):
        """将单个单词或单词数组映射至单个数字索引或数字索引数组"""
        if isinstance(word_or_list, list):
            return [self._word2idx(i) for i in word_or_list]
        return self._word2idx(word_or_list)

    def decode(self, idx_or_list):
        """将单个数字索引或数字索引数组映射至单个单词或单词数组"""
        if isinstance(idx_or_list, list):
            return [self._idx2word(i) for i in idx_or_list]
        return self._idx2word(idx_or_list)

    def __len__(self):
        return len(self.word2idx)


def build_vocab(dataset):
    de_words, en_words = [], []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)

    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))

    return Vocab(de_count_dict, min_freq=2), Vocab(en_count_dict, min_freq=2)


de_vocab, en_vocab = build_vocab(train_dataset)


class Iterator():
    """创建数据迭代器"""
    def __init__(self, dataset, de_vocab, en_vocab, batch_size, max_len=32, drop_reminder=False):
        self.dataset = dataset
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab

        self.batch_size = batch_size
        self.max_len = max_len
        self.drop_reminder = drop_reminder

        length = len(self.dataset) // batch_size
        self.len = length if drop_reminder else length + 1  # 批量数量

    def __iter__(self):
        def pad(idx_list, vocab, max_len):
            """统一序列长度，并记录有效长度"""
            idx_pad_list, idx_len = [], []
            for i in idx_list:
                if len(i) > max_len - 2:
                    idx_pad_list.append(
                        [vocab.bos_idx] + i[:max_len-2] + [vocab.eos_idx]
                    )
                    idx_len.append(max_len)
                else:
                    idx_pad_list.append(
                        [vocab.bos_idx] + i + [vocab.eos_idx] + [vocab.pad_idx] * (max_len - len(i) - 2)
                    )
                    idx_len.append(len(i) + 2)
            return idx_pad_list, idx_len

        def sort_by_length(src, trg):
            """对德/英语的字段长度进行排序"""
            data = zip(src, trg)
            data = sorted(data, key=lambda t: len(t[0]), reverse=True)
            return zip(*list(data))

        def encode_and_pad(batch_data, max_len):
            """将批量中的文本数据转换为数字索引，并统一每个序列的长度"""
            src_data, trg_data = zip(*batch_data)
            src_idx = [self.de_vocab.encode(i) for i in src_data]
            trg_idx = [self.en_vocab.encode(i) for i in trg_data]

            src_idx, trg_idx = sort_by_length(src_idx, trg_idx)
            src_idx_pad, src_len = pad(src_idx, self.de_vocab, max_len)
            trg_idx_pad, _ = pad(trg_idx, self.en_vocab, max_len)

            return src_idx_pad, src_len, trg_idx_pad

        for i in range(self.len):
            if i == self.len - 1 and not self.drop_reminder:
                batch_data = self.dataset[i * self.batch_size:]
            else:
                batch_data = self.dataset[i * self.batch_size: (i+1) * self.batch_size]

            src_idx, src_len, trg_idx = encode_and_pad(batch_data, self.max_len)
            yield torch.tensor(src_idx, dtype=torch.int32), \
                torch.tensor(src_len, dtype=torch.int32), \
                torch.tensor(trg_idx, dtype=torch.int32)

    def __len__(self):
        return self.len


train_iterator = Iterator(train_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=True)
valid_iterator = Iterator(valid_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=False)
test_iterator = Iterator(test_dataset, de_vocab, en_vocab, batch_size=1, max_len=32, drop_reminder=False)


src_vocab_size = len(de_vocab)
trg_vocab_size = len(en_vocab)
src_pad_idx = de_vocab.pad_idx
trg_pad_idx = en_vocab.pad_idx

d_model = 512
d_ff = 2048
n_layers = 6
n_heads = 8

encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
decoder = Decoder(trg_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
model = Transformer(encoder, decoder)

loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def forward(enc_inputs, dec_inputs, model, loss_fn):
    """前向网络
    enc_inputs: [batch_size, src_len]
    dec_inputs: [batch_size, trg_len]
    """
    logits, _, _, _ = model(enc_inputs, dec_inputs[:, :-1], src_pad_idx, trg_pad_idx)

    logits = logits.view(-1, logits.size(-1))  # 重新调整形状以匹配目标
    targets = dec_inputs[:, 1:].contiguous().view(-1).long()
    loss = loss_fn(logits, targets)

    return loss


def forward_with_grad(enc_inputs, dec_inputs, model, loss_fn, optimizer):
    optimizer.zero_grad()
    loss = forward(enc_inputs, dec_inputs, model, loss_fn)
    loss.backward()
    optimizer.step()
    return loss


def train(iterator, model, loss_fn, optimizer, epoch=0):
    model.train()
    num_batches = len(iterator)
    total_loss = 0
    total_steps = 0

    with tqdm(total=num_batches) as t:
        t.set_description(f'Epoch: {epoch}')
        for src, src_len, trg in iterator:
            loss = forward_with_grad(src, trg, model, loss_fn, optimizer)
            total_loss += loss.item()
            total_steps += 1
            curr_loss = total_loss / total_steps
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps


def evaluate(iterator, model, loss_fn):
    model.eval()  # 设置模型为评估模式
    num_batches = len(iterator)
    total_loss = 0
    total_steps = 0

    with torch.no_grad():  # 禁用梯度计算
        with tqdm(total=num_batches) as t:
            for src, _, trg in iterator:
                loss = forward(src, trg, model, loss_fn)
                total_loss += loss.item()
                total_steps += 1
                curr_loss = total_loss / total_steps
                t.set_postfix({'loss': f'{curr_loss:.2f}'})
                t.update(1)

    return total_loss / total_steps


num_epochs = 10
best_valid_loss = float('inf')
ckpt_file_name = './transformer_checkpoint.pth'  # 推荐使用.pth或.pt作为文件扩展名

for i in range(num_epochs):
    train_loss = train(train_iterator, model, loss_fn, optimizer, i)
    valid_loss = evaluate(valid_iterator, model, loss_fn)

    print(f'Epoch {i}: Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        # 保存模型的状态字典
        torch.save(model.state_dict(), ckpt_file_name)
        print(f"Model saved to {ckpt_file_name} with validation loss: {best_valid_loss:.4f}")


encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
decoder = Decoder(trg_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
new_model = Transformer(encoder, decoder)
ckpt_file_name = './transformer_checkpoint.pth'  # Path to your checkpoint file

# Load the checkpoint
checkpoint = torch.load(ckpt_file_name)

# Load the state dictionary into the model
new_model.load_state_dict(checkpoint)

def inference(sentence, max_len=32):
    """模型推理：输入一个德语句子，输出翻译后的英文句子
    enc_inputs: [batch_size(1), src_len]
    """
    new_model.eval()
    # 对输入句子进行分词
    if isinstance(sentence, str):
        tokens = [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', sentence.rstrip())]
    else:
        tokens = [token.lower() for token in sentence]

    # 补充起始、终止占位符，统一序列长度
    if len(tokens) > max_len - 2:
        src_len = max_len
        tokens = ['<bos>'] + tokens[:max_len - 2] + ['<eos>']
    else:
        src_len = len(tokens) + 2
        tokens = ['<bos>'] + tokens + ['<eos>'] + ['<pad>'] * (max_len - src_len)

    # 将德语单词转换为数字索引，并进一步转换为tensor
    # enc_inputs: [1, src_len]
    indexes = de_vocab.encode(tokens)
    enc_inputs = torch.tensor(indexes, dtype=torch.float32).unsqueeze(0)

    # 将输入送入encoder，获取信息
    enc_outputs, _ = new_model.encoder(enc_inputs, src_pad_idx)

    dec_inputs = torch.tensor([[en_vocab.bos_idx]], dtype=torch.float32)

    # 初始化decoder输入，此时仅有句首占位符<pad>
    # dec_inputs: [1, 1]
    max_len = enc_inputs.shape[1]
    for _ in range(max_len):
        dec_outputs, _, _ = new_model.decoder(dec_inputs, enc_inputs, enc_outputs, src_pad_idx, trg_pad_idx)
        dec_logits = dec_outputs.view((-1, dec_outputs.shape[-1]))

        # 找到下一个词的概率分布，并输出预测
        dec_logits = dec_logits[-1, :]
        pred = dec_logits.argmax(axis=0).unsqueeze(0).unsqueeze(0)
        pred = pred.to(dtype=torch.float32)
        # 更新dec_inputs
        dec_inputs = torch.concat((dec_inputs, pred), axis=1)
        # 如果出现<eos>，则终止循环
        if int(pred.numpy()[0, 0]) == en_vocab.eos_idx:
            break
    # 将数字索引转换为英文单词
    trg_indexes = [int(i) for i in dec_inputs.view(-1).numpy()]
    eos_idx = trg_indexes.index(en_vocab.eos_idx) if en_vocab.eos_idx in trg_indexes else -1
    trg_tokens = en_vocab.decode(trg_indexes[1:eos_idx])

    return trg_tokens

example_idx = 10

src = test_dataset[example_idx][0]
trg = test_dataset[example_idx][1]
pred_trg = inference(src)

print(f'src = {src}')
print(f'trg = {trg}')
print(f"predicted trg = {pred_trg}")

