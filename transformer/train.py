import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
from model import Encoder, Decoder, Transformer
from tqdm import tqdm
import os
import re


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



# num_epochs = 10
# best_valid_loss = float('inf')
# ckpt_file_name = './transformer_checkpoint.pth'  # 推荐使用.pth或.pt作为文件扩展名
#
# for i in range(num_epochs):
#     train_loss = train(train_iterator, model, loss_fn, optimizer, i)
#     valid_loss = evaluate(valid_iterator, model, loss_fn)
#
#     print(f'Epoch {i}: Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         # 保存模型的状态字典
#         torch.save(model.state_dict(), ckpt_file_name)
#         print(f"Model saved to {ckpt_file_name} with validation loss: {best_valid_loss:.4f}")
