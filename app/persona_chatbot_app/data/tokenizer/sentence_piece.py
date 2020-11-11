import sentencepiece as sp
from typing import List

from .tokenizer_base import TokenizerBase


class SentencepieceTokenizer(TokenizerBase):
    '''
    Sentencepieceを用いてトークナイズを行います．
    '''

    def __init__(self, path: str) -> None:
        '''
        コンストラクタです．
        モデルのパスを渡す必要があります．

        :param path: Sentencepiece学習済みモデルのパス
        '''
        super().__init__()

        self.spm = sp.SentencePieceProcessor()
        self.spm.load(path)

    def tokenize(self, text: str) -> List[str]:
        return self.spm.EncodeAsPieces(self._remove(text))

    def _remove(self, text: str) -> str:
        return text.replace("\r", "").replace("\n", "").strip()

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)
