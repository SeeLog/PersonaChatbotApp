import MeCab

from typing import List
from ..tokenizer.tokenizer_base import TokenizerBase


class SimpleMecabTokenizer(TokenizerBase):
    '''
    MeCabの標準辞書でサクッと形態素解析してtokenizeする
    '''
    def __init__(self):
        super().__init__()
        self.mecab = MeCab.Tagger("-Owakati")

    def tokenize(self, sentence: str) -> List[str]:
        '''
        実際のトークナイズする．

        :param sentence: 文，str

        :return: トークナイズされた文，List[str]
        '''
        return self._remove(self.mecab.parse(sentence)).split(" ")

    def __call__(self, sentence: str) -> List[str]:
        '''
        tokenizeをコールします．

        :param sentence: 文，str

        :return: トークナイズされた文，List[str]
        '''
        return self.tokenize(sentence)

    def _remove(self, text: str) -> str:
        return text.replace("\r", "").replace("\n", "").strip()
