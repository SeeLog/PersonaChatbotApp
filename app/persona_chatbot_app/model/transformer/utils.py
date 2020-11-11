import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np



def attention(query: torch.tensor, key: torch.tensor, value: torch.tensor, mask=None, dropout=None) -> torch.tensor:
    "Compute Simple 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def greedy_decode(model, src: torch.tensor, src_mask: torch.tensor, max_len: torch.tensor, start_symbol: int):
    """
    Greedy Decoding

    :param model: encode()とdecode()が実装されたモデル
    :param src: Source Tensor
    :param src_mask: Source Mask
    :param max_len: Max Length
    :param start_symbol: SOSシンボル

    :return: Greedy DecodingされたTensor
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys



def load_checkpoint(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model.to(device)
