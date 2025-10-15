import torch
import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER

from utils import BPETokenizer, LayerNorm, PositionalEncoding, download_dataset, translate


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(Embedding, self).__init__()
        # TODO: initialze weights for initial embedding and positional encodings.
        # ! TIP use torch.nn.Embedding and PositionalEncoding from utils.

    def forward(self, x):
        # TODO: Apply initial embedding weights to x first, then positional encoding.
        raise NotImplementedError()


class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(FeedForward, self).__init__()
        # TODO: initialize weights for W1 and W2 with appropriate dimension mappings
        # ! TIP use torch.nn.Linear

    def forward(self, x):
        # TODO Apply the FFN equation.
        # ! TIP use torch.nn.ReLU
        raise NotImplementedError()


class MaskedAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MaskedAttention, self).__init__()
        # TODO: initialize weights for Wq, Wk, Wv, Wo
        # ! TIP use torch.nn.Linear

    def forward(self, q, k, v, mask=None):
        # TODO: Build up to the attention equation (including softmax and Wo)
        # ! TIP function arguments: q is the x used with Wq, k is the x used with Wk, and v is the x used with Wv
        # ! TIP if mask is not None: logits = logits.masked_fill(mask == 0, -torch.inf)
        raise NotImplementedError()


class Encoder(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(Encoder, self).__init__()
        # TODO: initialize weights for:
            # layer normalization (provided in utils)
            # self-attention (use MaskedAttention class)
            # feedforward (use FeedForward class)

    def forward(self, src_embs):
        # TODO: Pass src_embs through each module of the encoder block.
        # ! TIP Apply LayerNorm *before* each residual connection, but remember that in the residual connection (old + new), old is pre-LayerNorm.
        # ! TIP Residual connection is implemented simply with the + operator.
        # ! HINT For example, for the FFN, this would look like: encs = encs + self.ff(self.norm(encs)))
        raise NotImplementedError()


class Decoder(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(Decoder, self).__init__()
        # TODO: initialize weights for:
            # layer normalization (provided in utils)
            # self-attention (use MaskedAttention class)
            # cross-attention (use MaskedAttention class)
            # feedforward (use FeedForward class)

    def forward(self, src_encs, tgt_embs):
        seq_len, device = tgt_embs.size(1), tgt_embs.device
        causal_mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device)).bool()
        # TODO: Pass tgt_embs through each module of the decoder block.
        # ! TIP Same tips and hints as in Encoder
        # ! TIP Decoder self-attention operates on tgt_encs (remember to pass in the mask!).
        # ! TIP Cross-attention operates on tgt_encs (for query) AND src_encs (for key and value). Only tgt_encs gets LayerNorm-ed in this case. No mask for cross-attention.
        raise NotImplementedError()


class Model(torch.nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int, ff_dim: int):
        super(Model, self).__init__()
        # TODO: initialize weights for: 
            # source embeddings
            # target embeddings
            # encoder
            # decoder
            # out_proj

    def encode(self, src_nums):
        # TODO: get source embeddings and apply encoder
        raise NotImplementedError()

    def decode(self, src_encs, tgt_nums):
        # TODO: get target embeddings and apply decoder
        raise NotImplementedError()

    def forward(self, src_nums, tgt_nums):
        # TODO: call encode() and decode(), pass into out_proj
        raise NotImplementedError()


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
    raise NotImplementedError()

    ### TRAINING

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO: declare model, loss function, and optimizer
    # ! TIP use torch.nn.CrossEntropyLoss and torch.optim.Adam with embed_dim = 512, ff_dim = 1024, and lr = 3e-4
    # ! TIP put the model on the device!
    raise NotImplementedError()

    for epoch in range(5):
        # TODO: implement training loop (similar to RNN/LSTM)
        # ! TIP use the tqdm.tqdm() iterator for a progress bar
        # ! TIP remember to shuffle the training data and put do model.train()
        # ! HINT src_nums = torch.tensor(src_nums, device=device).unsqueeze(0) -- do the same for tgt_nums
        # ! HINT per-sentence loss: train_loss += loss.item() * num_tgt_tokens
        # ! HINT per-epoch average loss: train_loss /= total_tokens
        raise NotImplementedError()

    ### SAVING
    # state_dict = torch.load(f'model.{src_lang}-{tgt_lang}.pth', map_location=device)
    # model.load_state_dict(state_dict)
    
    ### TRANSLATE
    # TODO: translate test set with translate() (see utils)
    # ! TIP: remember to do model.eval() and with torch.no_grad()
    # ! TIP: looping over the test set, keep appending to a predictions list and a references list. Use tgt_tokenizer.detokenize() to produce the string that should be appended to those lists.
    raise NotImplementedError()

    ### EVALUATE
    # TODO: compute evaluation metrics BLEU, chrF, and TER for the test set
    # ! TIP use {metric}.corpus_score(hypotheses, [references]).score
    bleu, chrf, ter = BLEU(), CHRF(), TER()
    raise NotImplementedError()


if __name__ == '__main__':
    main()
