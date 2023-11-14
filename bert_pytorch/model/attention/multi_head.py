import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        # d_model是输入的维度
        # h是多头的个数
        assert d_model % h == 0

        # q: query, k: key, v: value
        # d_v = q.size(-1)
        # d_k = k.size(-1)
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # Q_i = W_Q^i @ q
        # K_i = W_K^i @ k
        # V_i = W_V^i @ v
        # i = 1, 2, ..., h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Q_i, K_i, V_i, i = 1, 2, ..., h
        # Q_i, K_i, V_i are the same shape with input query, key, value
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # TODO: 不明白那里用到了多个头，view能够变成多个头吗？
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # 把前面的多头的输出拼接，输入到一个Linear layer中，就相当于是乘上了一个权重
        # 最终的输出和输入x的维度是一样的
        return self.output_linear(x)
