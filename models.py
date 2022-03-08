"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import masked_softmax
import torch_scatter


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):
    def __init__(self, word_vectors, hidden_size, num_head, max_len, drop_prob=0.):
        super(QANet, self).__init__()
        self.drop_prob = drop_prob

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        self.ln_emb = nn.LayerNorm(hidden_size)

        self.enc = layers.QANetStack(hidden_size=hidden_size,
                                     num_layers=1,
                                     num_head=num_head,
                                     kernel_size=7,
                                     cnn_layer=4,
                                     max_len=max_len,
                                     gain=0.5,
                                     s4=False,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.out = layers.QANetOutput(hidden_size=hidden_size,
                                      num_head=num_head,
                                      max_len=max_len,
                                      s4=False,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        c_emb = self.ln_emb(c_emb)
        q_emb = self.ln_emb(q_emb)

        c_enc = self.enc(c_emb, c_mask)    # (batch_size, c_len, hidden_size)
        q_enc = self.enc(q_emb, q_mask)    # (batch_size, q_len, hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        out = self.out(att, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANetChar(nn.Module):
    def __init__(self, word_vectors, char_emb_dim, hidden_size, num_head, max_len, s4=False, attn=True, drop_prob=0.):
        super(QANetChar, self).__init__()
        self.drop_prob = drop_prob

        self.emb = layers.CharEmbedding(word_vectors=word_vectors,
                                        char_emb_dim=char_emb_dim,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)
        self.ln_emb = nn.LayerNorm(hidden_size)

        self.enc = layers.QANetStack(hidden_size=hidden_size,
                                     num_layers=1,
                                     num_head=num_head,
                                     kernel_size=7,
                                     cnn_layer=4,
                                     max_len=max_len,
                                     gain=0.5,
                                     s4=s4,
                                     attn=attn,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.out = layers.QANetOutput(hidden_size=hidden_size,
                                      num_head=num_head,
                                      max_len=max_len,
                                      s4=s4,
                                      attn=attn,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)
        c_emb = self.ln_emb(c_emb)
        q_emb = self.ln_emb(q_emb)

        c_enc = self.enc(c_emb, c_mask)    # (batch_size, c_len, hidden_size)
        q_enc = self.enc(q_emb, q_mask)    # (batch_size, q_len, hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        out = self.out(att, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

    def reset_s4(self):
        self.enc.reset_s4()
        self.out.reset_s4()


class S4Char(nn.Module):
    def __init__(self, word_vectors, char_emb_dim, hidden_size, max_len, drop_prob=0.):
        super(S4Char, self).__init__()
        self.drop_prob = drop_prob

        self.emb = layers.CharEmbedding(word_vectors=word_vectors,
                                        char_emb_dim=char_emb_dim,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)
        self.ln_emb = nn.LayerNorm(hidden_size)

        self.enc = layers.S4Stack(hidden_size=hidden_size,
                                  num_layers=6,
                                  max_len=max_len,
                                  gain=0.5,
                                  drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.out = layers.S4Output(hidden_size=hidden_size,
                                   max_len=max_len,
                                   depth=12,
                                   drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, cc_poss, qc_poss):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs, cc_poss)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs, qc_poss)         # (batch_size, q_len, hidden_size)
        c_emb = self.ln_emb(c_emb)
        q_emb = self.ln_emb(q_emb)

        c_enc = self.enc(c_emb, c_mask)    # (batch_size, c_len, hidden_size)
        q_enc = self.enc(q_emb, q_mask)    # (batch_size, q_len, hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        out = self.out(att, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

    def reset_s4(self):
        self.enc.reset_s4()
        self.out.reset_s4()


class S4CharLevel(nn.Module):
    def __init__(self, word_vectors, char_emb_dim, hidden_size, max_len, drop_prob=0.):
        super(S4CharLevel, self).__init__()
        self.drop_prob = drop_prob

        self.emb = layers.CharEmbeddingChar(word_vectors=word_vectors,
                                            char_emb_dim=char_emb_dim,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
        self.ln_emb = nn.LayerNorm(hidden_size)
        self.ln_enc = nn.LayerNorm(hidden_size)
        self.enc_c = layers.S4Stack(hidden_size=hidden_size,
                                    num_layers=6,
                                    max_len=2048,
                                    gain=0.5,
                                    drop_prob=drop_prob)
        self.enc = layers.S4Stack(hidden_size=hidden_size,
                                  num_layers=4,
                                  max_len=max_len,
                                  gain=0.5,
                                  drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.out = layers.S4OutputSimple(hidden_size=hidden_size,
                                         max_len=max_len,
                                         depth=12,
                                         drop_prob=drop_prob)

    def encode(self, w_idxs, c_idxs, c_poss):
        w_mask = torch.zeros_like(w_idxs) != w_idxs
        c_mask_comb = torch.zeros_like(c_poss) != c_poss
        c_mask_layer = torch.zeros_like(c_idxs) != c_idxs

        emb = self.emb(w_idxs, c_idxs, c_poss)         # (batch_size, c_len, hidden_size)
        emb = self.ln_emb(emb)
        emb = self.enc_c(emb, c_mask_layer)
        emb = emb * c_mask_comb.unsqueeze(-1)
        # denom = torch_scatter.scatter(torch.ones_like(c_idxs), c_poss, dim_size=w_idxs.shape[1], reduce="sum").unsqueeze(-1)
        # emb = torch_scatter.scatter(emb, c_poss, dim=1, dim_size=w_idxs.shape[1], reduce="sum") / (1e-3 + denom.sqrt())
        emb = torch_scatter.scatter(emb, c_poss, dim=1, dim_size=w_idxs.shape[1], reduce="max")
        emb = self.ln_enc(emb)

        enc = self.enc(emb, w_mask)    # (batch_size, c_len, hidden_size)
        return enc

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, cc_poss, qc_poss):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_enc = self.encode(cw_idxs, cc_idxs, cc_poss)
        q_enc = self.encode(qw_idxs, qc_idxs, qc_poss)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        out = self.out(att, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

    def reset_s4(self):
        self.enc.reset_s4()
        self.out.reset_s4()
        self.enc_c.reset_s4()


class Transformer(nn.Module):
    def __init__(self, word_vectors, char_emb_dim, hidden_size, num_head, max_len, drop_prob=0.):
        super(Transformer, self).__init__()
        self.drop_prob = drop_prob

        self.emb = layers.CharEmbedding(word_vectors=word_vectors,
                                        char_emb_dim=char_emb_dim,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)
        self.ln_emb = nn.LayerNorm(hidden_size)

        self.enc = layers.TransformerStack(hidden_size=hidden_size,
                                           num_layers=8,
                                           num_head=num_head,
                                           kernel_size=5,
                                           cnn_layer=2,
                                           max_len=max_len,
                                           gain=0.5,
                                           drop_prob=drop_prob)
        self.linear_1 = nn.Linear(hidden_size, 1)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.ln_emb(self.emb(cw_idxs, cc_idxs))
        q_emb = self.ln_emb(self.emb(qw_idxs, qc_idxs))

        c_enc, q_enc = self.enc(c_emb, q_emb, c_mask, q_mask)

        logits_1 = self.linear_1(c_enc)
        logits_2 = self.linear_2(c_enc)

        log_p1 = masked_softmax(logits_1.squeeze(), c_mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), c_mask, log_softmax=True)

        return log_p1, log_p2


def get_model(args, word_vectors):
    if args.model.lower() == "bidaf":
        model = BiDAF(word_vectors=word_vectors,
                      hidden_size=args.hidden_size,
                      drop_prob=args.drop_prob)
    elif args.model.lower() == "qanet":
        model = QANet(word_vectors=word_vectors,
                      hidden_size=args.hidden_size,
                      num_head=args.num_head,
                      max_len=args.max_len,
                      drop_prob=args.drop_prob)
    elif args.model.lower() == "qanet-char":
        model = QANetChar(word_vectors=word_vectors,
                          char_emb_dim=args.char_emb_dim,
                          hidden_size=args.hidden_size,
                          num_head=args.num_head,
                          max_len=args.max_len,
                          drop_prob=args.drop_prob)
    elif args.model.lower() == "qanet-cnn":
        model = QANetChar(word_vectors=word_vectors,
                          char_emb_dim=args.char_emb_dim,
                          hidden_size=args.hidden_size,
                          num_head=args.num_head,
                          max_len=args.max_len,
                          attn=False,
                          drop_prob=args.drop_prob)
    elif args.model.lower() == "s4":
        model = QANetChar(word_vectors=word_vectors,
                          char_emb_dim=args.char_emb_dim,
                          hidden_size=args.hidden_size,
                          num_head=args.num_head,
                          max_len=args.max_len,
                          s4=True,
                          attn=False,
                          drop_prob=args.drop_prob)
    elif args.model.lower() == "pure-s4":
        model = S4Char(word_vectors=word_vectors,
                       char_emb_dim=args.char_emb_dim,
                       hidden_size=args.hidden_size,
                       max_len=args.max_len,
                       drop_prob=args.drop_prob)
    elif args.model.lower() == "s4-char":
        model = S4CharLevel(word_vectors=word_vectors,
                            char_emb_dim=args.char_emb_dim,
                            hidden_size=args.hidden_size,
                            max_len=args.max_len,
                            drop_prob=args.drop_prob)
    elif args.model.lower() == "transformer":
        model = Transformer(word_vectors=word_vectors,
                            char_emb_dim=args.char_emb_dim,
                            hidden_size=args.hidden_size,
                            num_head=args.num_head,
                            max_len=args.max_len,
                            drop_prob=args.drop_prob)
    else:
        raise NotImplementedError(f"unknown model type {args.model}")

    return model
