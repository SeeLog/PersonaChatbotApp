import torch
import torch.nn as nn

from typing import List


class AutoEncoder(nn.Module):
    def __init__(self, encoder_dims: List[int], decoder_dims: List[int]) -> None:
        '''
        AutoEncoder
        :param encoder_dims: Encoder dimensions, e.g. [256, 128, 64, 32, 16]
        :param decoder_dims: Decoder dimensions, e.g. [16, 32, 64, 128, 256]
        '''
        super(AutoEncoder, self).__init__()

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        encoder_list = []
        decoder_list = []

        for i, input_dim in enumerate(encoder_dims):
            if i < len(encoder_dims) - 2:
                encoder_list.append(nn.Linear(input_dim, encoder_dims[i + 1]))
                encoder_list.append(nn.ReLU(True))
                # 他の活性化関数試すなら下に変えてみるといいかも
                # encoder_list.append(nn.Tanh())
                # encoder_list.append(nn.LeakyReLU(True))
            else:
                # 最終層は活性化関数を入れない
                encoder_list.append(nn.Linear(input_dim, encoder_dims[i + 1]))
                break

        for i, input_dim in enumerate(decoder_dims):
            if i < len(decoder_dims) - 2:
                decoder_list.append(nn.Linear(input_dim, decoder_dims[i + 1]))
                decoder_list.append(nn.ReLU(True))
            else:
                # 最終層は活性化関数を入れない
                decoder_list.append(nn.Linear(input_dim, decoder_dims[i + 1]))
                break

        self.encoder = nn.ModuleList(encoder_list)
        self.decoder = nn.ModuleList(decoder_list)

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        順方向計算

        :param x: 入力テンソル

        :return: 復元されたテンソル
        '''
        return self.decode(self.encode(x))

    def encode(self, x: torch.tensor) -> torch.tensor:
        '''
        でかい次元のテンソルをエンコードします

        :param x: 入力テンソル
        '''
        for module in self.encoder:
            x = module(x)
        return x

    def decode(self, x: torch.tensor) -> torch.tensor:
        '''
        圧縮されたテンソルから元のテンソルの復元を試みます

        :param x: 入力テンソル
        '''
        for module in self.decoder:
            x = module(x)

        return x
