import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
from model import Transformer
from tqdm import tqdm
import os
import re

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