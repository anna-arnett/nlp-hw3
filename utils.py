import math
import os
import re

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Punctuation, Whitespace


class BPETokenizer:
    def __init__(self, lang: str, vocab_size: int):
        self.lang = lang
        path = f'tokenizer.{lang}.json'
        if not os.path.exists(path):
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])
            self.train(vocab_size)
        else:
            self.tokenizer = Tokenizer.from_file(path)

    def train(self, vocab_size: int):
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=['<BOS>', '<EOS>', '<UNK>']
        )
        self.tokenizer.train([f'data/train.{self.lang}.txt'], trainer)
        self.tokenizer.save(f'tokenizer.{self.lang}.json')

    def numberize(self, sym: str) -> int:
        return self.tokenizer.token_to_id(sym)

    def denumberize(self, num: int) -> str:
        return self.tokenizer.id_to_token(num)

    def tokenize(self, sequence: str) -> list[int]:
        return self.tokenizer.encode(sequence).ids

    def detokenize(self, nums: list[int]) -> str:
        return re.sub(r'\s([?!.,:;])', r'\1', self.tokenizer.decode(nums))

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim: int, max_len=5000):
        super(PositionalEncoding, self).__init__()
        enc = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000) / embed_dim))
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('enc', enc.unsqueeze(0))

    def forward(self, x):
        return x + self.enc[:, : x.size(1)]


class LayerNorm(torch.nn.Module):
    def __init__(self, size: int, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(size))
        self.beta = torch.nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(variance + self.eps) + self.beta


def translate(src_encs, model, tokenizer, max_len=256) -> list[int]:
    BOS = tokenizer.numberize('<BOS>')
    EOS = tokenizer.numberize('<EOS>')
    path = torch.full((max_len,), BOS, device=src_encs.device)
    for i in range(1, max_len):
        tgt_encs = model.decode(src_encs, path[:i].unsqueeze(0))
        logits = model.out_proj(tgt_encs[:, -1])
        path[i] = logits.log_softmax(dim=-1).argmax(dim=-1)
        if path[i] == EOS:
            return path[: i + 1].tolist()
    return path.tolist()


def download_dataset(dataset: str, src_lang: str, tgt_lang: str):
    ds = load_dataset(dataset)
    os.makedirs('data', exist_ok=True)
    for split in ('train', 'validation', 'test'):
        split_name = 'dev' if split == 'validation' else split
        src_path = os.path.join('data', f'{split_name}.{src_lang}.txt')
        tgt_path = os.path.join('data', f'{split_name}.{tgt_lang}.txt')
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            with open(src_path, 'w') as src_f, open(tgt_path, 'w') as tgt_f:
                for sent in ds[split]:
                    src_f.write(sent[src_lang] + '\n')
                    tgt_f.write(sent[tgt_lang] + '\n')
