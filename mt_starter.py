import torch
import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER
import math
import random

from utils import BPETokenizer, LayerNorm, PositionalEncoding, download_dataset, translate


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(Embedding, self).__init__()
        # TODO: initialze weights for initial embedding and positional encodings.
        # ! TIP use torch.nn.Embedding and PositionalEncoding from utils.
        self.tok = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)

    def forward(self, x):
        # TODO: Apply initial embedding weights to x first, then positional encoding.
        return self.pos(self.tok(x))


class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(FeedForward, self).__init__()
        # TODO: initialize weights for W1 and W2 with appropriate dimension mappings
        # ! TIP use torch.nn.Linear
        self.w1 = torch.nn.Linear(embed_dim, ff_dim)
        self.w2 = torch.nn.Linear(ff_dim, embed_dim)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        # TODO Apply the FFN equation.
        # ! TIP use torch.nn.ReLU
        return self.w2(self.act(self.w1(x)))


class MaskedAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MaskedAttention, self).__init__()
        # TODO: initialize weights for Wq, Wk, Wv, Wo
        # ! TIP use torch.nn.Linear
        self.wq = torch.nn.Linear(embed_dim, embed_dim)
        self.wk = torch.nn.Linear(embed_dim, embed_dim)
        self.wv = torch.nn.Linear(embed_dim, embed_dim)
        self.wo = torch.nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, q, k, v, mask=None):
        # TODO: Build up to the attention equation (including softmax and Wo)
        # ! TIP function arguments: q is the x used with Wq, k is the x used with Wk, and v is the x used with Wv
        # ! TIP if mask is not None: logits = logits.masked_fill(mask == 0, -torch.inf)
        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        KT = K.transpose(-2, -1)
        logits = (Q @ KT) / self.scale

        if mask is not None:
            logits = logits.masked_fill(mask == 0, -torch.inf)

        probs = torch.softmax(logits, dim=-1)
        ctx = probs @ V
        out = self.wo(ctx)
        return out 


class Encoder(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(Encoder, self).__init__()
        # TODO: initialize weights for:
            # layer normalization (provided in utils)
            # self-attention (use MaskedAttention class)
            # feedforward (use FeedForward class)
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MaskedAttention(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm_out = LayerNorm(embed_dim)

    def forward(self, src_embs):
        # TODO: Pass src_embs through each module of the encoder block.
        # ! TIP Apply LayerNorm *before* each residual connection, but remember that in the residual connection (old + new), old is pre-LayerNorm.
        # ! TIP Residual connection is implemented simply with the + operator.
        # ! HINT For example, for the FFN, this would look like: encs = encs + self.ff(self.norm(encs)))
        x = src_embs
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=None)
        x = x + self.ff(self.norm2(x))
        return self.norm_out(x)


class Decoder(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(Decoder, self).__init__()
        # TODO: initialize weights for:
            # layer normalization (provided in utils)
            # self-attention (use MaskedAttention class)
            # cross-attention (use MaskedAttention class)
            # feedforward (use FeedForward class)
        self.norm_self1 = LayerNorm(embed_dim)
        self.self_attn = MaskedAttention(embed_dim)

        self.norm_cross = LayerNorm(embed_dim)
        self.cross_attn = MaskedAttention(embed_dim)

        self.norm_ff = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm_out = LayerNorm(embed_dim)

    def forward(self, src_encs, tgt_embs):
        seq_len, device = tgt_embs.size(1), tgt_embs.device
        causal_mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device)).bool()
        # TODO: Pass tgt_embs through each module of the decoder block.
        # ! TIP Same tips and hints as in Encoder
        # ! TIP Decoder self-attention operates on tgt_encs (remember to pass in the mask!).
        # ! TIP Cross-attention operates on tgt_encs (for query) AND src_encs (for key and value). Only tgt_encs gets LayerNorm-ed in this case. No mask for cross-attention.
        x = tgt_embs
        x = x + self.self_attn(self.norm_self1(x), self.norm_self1(x), self.norm_self1(x), mask=causal_mask)
        x = x + self.cross_attn(self.norm_cross(x), src_encs, src_encs, mask=None)
        x = x + self.ff(self.norm_ff(x))
        return self.norm_out(x)


class Model(torch.nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int, ff_dim: int):
        super(Model, self).__init__()
        # TODO: initialize weights for: 
            # source embeddings
            # target embeddings
            # encoder
            # decoder
            # out_proj
        self.src_embed = Embedding(src_vocab_size, embed_dim)
        self.tgt_embed = Embedding(tgt_vocab_size, embed_dim)

        self.encoder = Encoder(embed_dim, ff_dim)
        self.decoder = Decoder(embed_dim, ff_dim)

        self.out_proj = torch.nn.Linear(embed_dim, tgt_vocab_size)

    def encode(self, src_nums):
        # TODO: get source embeddings and apply encoder
        src_embs = self.src_embed(src_nums)
        src_encs = self.encoder(src_embs)
        return src_encs

    def decode(self, src_encs, tgt_nums):
        # TODO: get target embeddings and apply decoder
        tgt_embs = self.tgt_embed(tgt_nums)
        tgt_encs = self.decoder(src_encs, tgt_embs)
        return tgt_encs

    def forward(self, src_nums, tgt_nums):
        # TODO: call encode() and decode(), pass into out_proj
        src_encs = self.encode(src_nums)
        tgt_encs = self.decode(src_encs, tgt_nums[:, :-1])
        logits = self.out_proj(tgt_encs)
        return logits
    
    
def add_bos_eos(tokenizer: BPETokenizer, sent: str) -> list[int]:
    BOS = tokenizer.numberize('<BOS>')
    EOS = tokenizer.numberize('<EOS>')
    return [BOS] + tokenizer.tokenize(sent) + [EOS]

def batchify_pairs(pairs, device):
    for src_nums, tgt_nums in pairs:
        yield (
            torch.tensor(src_nums, dtype=torch.long, device=device).unsqueeze(0),
            torch.tensor(tgt_nums, dtype=torch.long, device=device).unsqueeze(0),
        )

def detok_strip_special(tokenizer: BPETokenizer, ids: list[int]) -> str:
    BOS = tokenizer.numberize('<BOS>')
    EOS = tokenizer.numberize('<EOS>')
    if ids and ids[0] == BOS:
        ids = ids[1:]
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tokenizer.detokenize(ids)


def main():

    ### DATA AND TOKENIZATION
    src_lang, tgt_lang = 'de', 'en'
    download_dataset('bentrevett/multi30k', src_lang, tgt_lang)

    # TODO: tokenize splits with BPETokenizer (see utils)
        # keep separate tokenizers for German (de) and English (en)
        # specify vocabulary size 10000
        # tokenizer.tokenize() produces a numberized token sequence as a list
        # use denumberize() to see the tokens as text
        # remember to add BOS and EOS
        # ! HINT src_nums = [src_tokenizer.numberize('<BOS>')] + src_tokenizer.tokenize(src_sent) + [src_tokenizer.numberize('<EOS>')]
    # TODO: assemble the train, dev, and test data to pass into the model, each as a list of tuples (src_nums, tgt_nums) corresponding to each parallel sentence
    src_tokenizer = BPETokenizer(src_lang, 10_000)
    tgt_tokenizer = BPETokenizer(tgt_lang, 10_000)

    def load_split(name):
        with open(f"data/{name}.{src_lang}.txt", "r") as fsrc, open(f"data/{name}.{tgt_lang}.txt", "r") as ftgt:
            src_sents = [line.rstrip("\n") for line in fsrc]
            tgt_sents = [line.rstrip("\n") for line in ftgt]
        pairs = []
        for s, t in zip(src_sents, tgt_sents):
            src_nums = add_bos_eos(src_tokenizer, s)
            tgt_nums = add_bos_eos(tgt_tokenizer, t)
            pairs.append((src_nums, tgt_nums))
        return pairs, tgt_sents
    
    train_pairs, _ = load_split("train")
    dev_pairs, _ = load_split("dev")
    test_pairs, test_tgt_raw = load_split("test")

    '''# ---- Debug switches -----
    DEBUG_N_TRAIN = 1_000
    DEBUG_N_TEST = 100

    train_pairs = train_pairs[:DEBUG_N_TRAIN]
    dev_pairs = dev_pairs[:DEBUG_N_TEST]
    test_pairs = test_pairs[:DEBUG_N_TEST]'''

    print("\n=== First 10 tokenized English test sentences (with BOS/EOS) ===")
    for i in range(10):
        nums = add_bos_eos(tgt_tokenizer, test_tgt_raw[i])
        toks = [tgt_tokenizer.denumberize(n) for n in nums]
        print(f"{i:02d} INDICES: {nums}")
        print(f"{i:02d} TOKENS: {toks}")

    ### TRAINING

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO: declare model, loss function, and optimizer
    # ! TIP use torch.nn.CrossEntropyLoss and torch.optim.Adam with embed_dim = 512, ff_dim = 1024, and lr = 3e-4
    # ! TIP put the model on the device!
    embed_dim = 512
    ff_dim = 1024
    lr = 3e-4

    model = Model(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        embed_dim=embed_dim,
        ff_dim=ff_dim,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(5):
        # TODO: implement training loop (similar to RNN/LSTM)
        # ! TIP use the tqdm.tqdm() iterator for a progress bar
        # ! TIP remember to shuffle the training data and put do model.train()
        # ! HINT src_nums = torch.tensor(src_nums, device=device).unsqueeze(0) -- do the same for tgt_nums
        # ! HINT per-sentence loss: train_loss += loss.item() * num_tgt_tokens
        # ! HINT per-epoch average loss: train_loss /= total_tokens
        model.train()
        random.shuffle(train_pairs)
        train_loss = 0.0
        total_tgt_tokens = 0

        for src_nums, tgt_nums in tqdm.tqdm(batchify_pairs(train_pairs, device), total=len(train_pairs), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            logits = model(src_nums, tgt_nums)
            gold = tgt_nums[:, 1:]

            N, Tm1, V = logits.size()
            loss = criterion(logits.view(N * Tm1, V), gold.reshape(-1))
            loss.backward()
            optimizer.step()

            num_toks = gold.numel()
            train_loss += loss.item() * num_toks
            total_tgt_tokens += num_toks 
        
        avg_loss = train_loss / max(1, total_tgt_tokens)
        print(f"Epoch {epoch}: avg train loss = {avg_loss:.4f}")

    ### SAVING
    # state_dict = torch.load(f'model.{src_lang}-{tgt_lang}.pth', map_location=device)
    # model.load_state_dict(state_dict)
    torch.save(model.state_dict(), f"model.{src_lang}-{tgt_lang}.pth")
    
    ### TRANSLATE
    # TODO: translate test set with translate() (see utils)
    # ! TIP: remember to do model.eval() and with torch.no_grad()
    # ! TIP: looping over the test set, keep appending to a predictions list and a references list. Use tgt_tokenizer.detokenize() to produce the string that should be appended to those lists.
    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for src_nums, tgt_nums in batchify_pairs(test_pairs, device):
            src_encs = model.encode(src_nums)
            pred_path = translate(src_encs, model, tgt_tokenizer)

            hyp = detok_strip_special(tgt_tokenizer, pred_path)
            ref = detok_strip_special(tgt_tokenizer, tgt_nums.squeeze(0).tolist())
            hypotheses.append(hyp)
            references.append(ref)

    ### EVALUATE
    # TODO: compute evaluation metrics BLEU, chrF, and TER for the test set
    # ! TIP use {metric}.corpus_score(hypotheses, [references]).score
    bleu, chrf, ter = BLEU(), CHRF(), TER()
    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    chrf_score = chrf.corpus_score(hypotheses, [references]).score
    ter_score = ter.corpus_score(hypotheses, [references]).score

    print("\n=== Test Set Scores ===")
    print(f"BLEU: {bleu_score:.2f}")
    print(f"ChrF: {chrf_score:.2f}")
    print(f"TER: {ter_score:.2f}")


if __name__ == '__main__':
    main()
