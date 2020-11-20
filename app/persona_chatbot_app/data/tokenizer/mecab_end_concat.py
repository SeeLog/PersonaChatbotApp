# import MeCab
from subprocess import Popen, PIPE

from typing import List
from ..tokenizer.tokenizer_base import TokenizerBase


TYPE = "助詞-終助詞"


class MecabEndConcatTokenizer(TokenizerBase):
    '''
    MeCabのIPADicで形態素解析 -> 「ござる よ」のような機能語を連結してtokenizeする
    '''
    def __init__(self):
        super().__init__()
        # self.mecab = MeCab.Tagger("-Ochasen")

    def __mecab(self, sentence: str) -> str:
        p = Popen(["mecab", "-Ochasen"], stdin=PIPE, stdout=PIPE)
        o, e = p.communicate(sentence.encode("utf-8"))
        ret = o.decode("utf-8")

        if "[!tmp.empty()]" in ret:
            raise RuntimeError("MeCab Error: " + ret)

        return ret

    def tokenize(self, sentence: str) -> List[str]:
        '''
        実際のトークナイズする．

        :param sentence: 文，str

        :return: トークナイズされた文，List[str]
        '''
        return self._remove(self.wakachi(sentence)).split(" ")

    def wakachi(self, text: str) -> str:
        # parse = [item.split("\t") for item in self.mecab.parse(text).split("\n")]
        parse = [item.split("\t") for item in self.__mecab(text).split("\n")]
        parse = [p for p in parse if len(p) > 3]

        if len(parse) < 2:
            return text

        words = []

        app_count = 0
        for i, p in enumerate(parse):
            if p[3] == TYPE and i > 0:
                words[app_count - 1] += p[0]
            else:
                words.append(p[0])
                app_count += 1

        return " ".join(words)

    def __call__(self, sentence: str) -> List[str]:
        '''
        tokenizeをコールします．

        :param sentence: 文，str

        :return: トークナイズされた文，List[str]
        '''
        return self.tokenize(sentence)

    def _remove(self, text: str) -> str:
        return text.replace("\r", "").replace("\n", "").strip()
