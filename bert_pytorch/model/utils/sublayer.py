import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    # Residual connection
    def __init__(self, size, dropout):
        # size: hidden = word_embedding_dim = word vector dimension
        # size: input vector matrix, size(-1) = hidden
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 这层的目的是对上一层sublayer的输出值进行norm, dropout, residual connection
        # x: input vector matrix, size(-1) = hidden

        # 这里的sublayer可能是
        # 1. MultiHeadAttention
        # 2. FeedForward
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
