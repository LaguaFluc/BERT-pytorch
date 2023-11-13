import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        # NOTE: 不知道为什么取3，不是有前面一句话，后面一句话吗？，这个应该只有两个
        super().__init__(3, embed_size, padding_idx=0)
