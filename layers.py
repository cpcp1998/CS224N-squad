"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class CharEmbedding(nn.Module):
    def __init__(self, word_vectors, char_emb_dim, hidden_size, drop_prob):
        super(CharEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding(257, char_emb_dim) # 256 for UNK
        self.char_embed.weight.data.normal_(0, word_vectors.std() / 4)
        self.proj = nn.Linear(word_vectors.size(1)+char_emb_dim, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w, c):
        emb_w = self.embed(w)   # (batch_size, seq_len, embed_size)
        emb_c = self.char_embed(torch.minimum(c, torch.full_like(c, 256)))
        emb_c, _ = emb_c.max(dim=-2)
        emb = torch.cat((emb_w, emb_c), dim=-1)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(SinusoidalPositionEmbedding, self).__init__()
        assert hidden_size % 2 == 0
        position = torch.arange(0, max_len).unsqueeze(1)
        freq = torch.exp(torch.arange(0, hidden_size, 2) / hidden_size * (-math.log(10000.)))
        emb = torch.cat((
            torch.sin(position * freq),
            torch.cos(position * freq),
        ), dim=1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        seq_len = x.size(1)
        return self.emb[:seq_len].unsqueeze(0)


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class SelfAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 attention_size,
                 num_head,
                 drop_prob=0.):
        super(SelfAttention, self).__init__()

        assert attention_size % num_head == 0
        self.num_head = num_head
        self.head_size = attention_size // num_head

        self.attn_dropout = nn.Dropout(drop_prob)
        self.dropout = nn.Dropout(drop_prob)

        self.q_linear = nn.Linear(hidden_size, attention_size)
        self.k_linear = nn.Linear(hidden_size, attention_size)
        self.v_linear = nn.Linear(hidden_size, attention_size)
        self.o_linear = nn.Linear(attention_size, hidden_size)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.num_head, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_head, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_head, self.head_size).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(self.head_size)
        mask = mask.unsqueeze(1).unsqueeze(2)
        attn = attn * mask - 1e5 * (~mask)
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        o = torch.matmul(attn, v)
        o = o.transpose(1, 2).reshape(batch_size, seq_len, -1)
        o = self.o_linear(o)

        o = self.dropout(o)

        return o


class FeedForward(nn.Module):
    def __init__(self,
                 hidden_size,
                 ff_size,
                 drop_prob=0.):
        super(FeedForward, self).__init__()

        self.dropout = nn.Dropout(drop_prob)

        self.linear_1 = nn.Linear(hidden_size, ff_size)
        self.linear_2 = nn.Linear(ff_size, hidden_size)

    def forward(self, x, mask):
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 kernel_size,
                 drop_prob=0.):
        super(CNN, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same')

    def forward(self, x, mask):
        x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        return self.dropout(x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, gain, transform):
        super(ResidualBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.transform = transform
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.gain = gain

    def forward(self, x, mask):
        y = self.ln_1(x)
        y = self.transform(y, mask)
        y = self.ln_2(y)
        return x + y * self.gain


class QANetBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_head,
                 kernel_size,
                 cnn_layer,
                 max_len,
                 gain,
                 drop_prob=0.):
        super(QANetBlock, self).__init__()
        self.pos_emb = SinusoidalPositionEmbedding(hidden_size, max_len)
        self.cnn = nn.ModuleList([ResidualBlock(hidden_size, gain,
                                                CNN(hidden_size, kernel_size, drop_prob))
                                  for _ in range(cnn_layer)])
        self.attn = ResidualBlock(hidden_size, gain,
                                  SelfAttention(hidden_size, hidden_size, num_head, drop_prob))
        self.ff = ResidualBlock(hidden_size, gain,
                                FeedForward(hidden_size, hidden_size, drop_prob))

    def forward(self, x, mask):
        x = x + self.pos_emb(x)
        for cnn in self.cnn:
            x = cnn(x, mask)
        x = self.attn(x, mask)
        x = self.ff(x, mask)
        return x


class QANetStack(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_head,
                 num_layers,
                 kernel_size,
                 cnn_layer,
                 max_len,
                 gain,
                 drop_prob=0.):
        super(QANetStack, self).__init__()
        self.blocks = nn.ModuleList([QANetBlock(hidden_size, num_head, kernel_size, cnn_layer, max_len, gain, drop_prob)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class QANetOutput(nn.Module):
    def __init__(self, hidden_size, num_head, max_len, drop_prob):
        super(QANetOutput, self).__init__()
        self.input = nn.Linear(4 * hidden_size, hidden_size)
        self.mod = QANetStack(hidden_size=hidden_size,
                              num_layers=7,
                              num_head=num_head,
                              kernel_size=5,
                              cnn_layer=2,
                              max_len=max_len,
                              gain=0.5,
                              drop_prob=drop_prob)
        self.linear_1 = nn.Linear(2*hidden_size, 1)
        self.linear_2 = nn.Linear(2*hidden_size, 1)

    def forward(self, att, mask):
        att = self.input(att)
        m0 = self.mod(att, mask)
        m1 = self.mod(m0, mask)
        m2 = self.mod(m1, mask)
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.linear_1(torch.cat((m0, m1), dim=-1))
        logits_2 = self.linear_2(torch.cat((m0, m2), dim=-1))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
