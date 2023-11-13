import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # attention: softmax(QK^T / sqrt(d_k))
        # 它的使用方法：用它去乘以V，得到每个value对应的得分（每个Token对应的得分）
        # 看哪个token的得分高一点，就重要一点
        # query最后两个维度是: [seq_len, d_k]
        # key最后两个维度分别是：[seq_len, d_k]
        # matmul计算matmul(A, B)需要保证A.size(-1) == B.size(-2)
        # d_k, query的词向量的维度
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        # NOTE: 让我不理解的一点是：几乎所有的网络层都有dropout, 
        # 还是说只有这种attention这种复杂的网络层需要dropout?
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
