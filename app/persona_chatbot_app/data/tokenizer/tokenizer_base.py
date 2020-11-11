from typing import List


class TokenizerBase():
    '''
    Tokenizerのベースクラス
    '''
    def __init__(self):
        super().__init__()

    def tokenize(self, sentence: str) -> List[str]:
        '''
        実際のトークナイズする．

        :param sentence: 文，str

        :return: トークナイズされた文，List[str]
        '''
        raise NotImplementedError("tokenize() を実装してください")

    def __call__(self, sentence: str) -> List[str]:
        '''
        tokenizeをコールします．

        :param sentence: 文，str

        :return: トークナイズされた文，List[str]
        '''
        return self.tokenize()
