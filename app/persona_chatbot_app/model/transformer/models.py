import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from .layers import Encoder, Decoder, EncoderLayer, DecoderLayer
from .sublayers import Embeddings, MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward
from .auto_encoder import AutoEncoder



def make_target_persona_model(src_vocab: int, tgt_vocab: int, num_layers: int = 6, d_model: int = 512, d_ff: int = 2048, head_num: int = 8, dropout: float = 0.1, ae_dims=[256, 128, 64, 32, 16]):
    """
    モデルを作成します．

    :param src_vocab: SourceのVocabサイズ
    :param tgt_vocab: TargetのVocabサイズ，-1でSourceとシェアする
    :param num_layers: Transformerの層の数, default: 6
    :param d_model: モデルの隠れ層のサイズ, default: 512
    :param d_ff: FeedForwardの隠れ層のサイズ, default: 2048
    :param head_num: MultiHeadAttentionの分割数, default: 8
    :param dropout: Dropout率, default: 0.1
    :param ae_dims: AutoEncoderの次元遷移

    :return: 作成されたTransformerによるPersonaEncoderDecoderモデル
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(head_num, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    src_emb = Embeddings(d_model, src_vocab)
    if tgt_vocab == -1:
        tgt_emb = src_emb
        tgt_vocab = src_vocab
    else:
        tgt_emb = Embeddings(d_model, tgt_emb)

    rev_dim = list(reversed(ae_dims))

    model = TargetPersonaEncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), num_layers),
        nn.Sequential(src_emb, c(position)),
        nn.Sequential(tgt_emb, c(position)),
        Generator(d_model, tgt_vocab),
        AutoEncoder(ae_dims, rev_dim))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: Embeddings, tgt_embed: Embeddings, generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: torch.tensor, tgt: torch.tensor, src_mask: torch.tensor, tgt_mask: torch.tensor) -> torch.tensor:
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask) -> torch.tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask) -> torch.tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class TargetPersonaEncoderDecoder(nn.Module):
    """
    Targetにペルソナ入力が可能なEncoderDecoder w/Transformer
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: Embeddings, tgt_embed: Embeddings, generator: Generator, auto_encoder: AutoEncoder):
        super(TargetPersonaEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.auto_encoder = auto_encoder

    def forward(self, src: torch.tensor, tgt: torch.tensor, persona_tensor: torch.tensor, src_mask: torch.tensor, tgt_mask: torch.tensor) -> torch.tensor:
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, persona_tensor)

    def encode(self, src, src_mask) -> torch.tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, persona_tensor) -> torch.tensor:
        decoded_persona = self.auto_encoder(persona_tensor)
        return self.decoder(self.tgt_embed(tgt) + decoded_persona, memory, src_mask, tgt_mask)

    def decode_from_encoded_persona(self, memory, src_mask, tgt, tgt_mask, persona_tensor) -> torch.tensor:
        decoded_persona = self.auto_encoder.decode(persona_tensor)
        return self.decoder(self.tgt_embed(tgt) + decoded_persona, memory, src_mask, tgt_mask)
