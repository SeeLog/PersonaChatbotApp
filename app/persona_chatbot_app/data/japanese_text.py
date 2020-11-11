from torchtext import data
from typing import Callable, List

from .special_tokens import SOS_token, EOS_token, PAD_token, UNK_token
from .source_target_base import SourceTargetBase


class JapaneseText(SourceTargetBase):
    '''
    日本語でsourceとtargetのテキストを管理するクラス
    '''
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        max_length: int = 64,
    ) -> None:
        '''
        初期化します．

        :param tokenizer: CallableなTokenizerを指定します．
        Callableならなんでもよいです．
        '''
        super().__init__()

        self.src = data.Field(
            sequential=True,
            tokenize=tokenizer,
            tokenizer_language="ja",
            pad_token=PAD_token,
            init_token=SOS_token,
            eos_token=EOS_token,
            unk_token=UNK_token,
            lower=True,
            batch_first=True,
            fix_length=max_length,
        )
        self.tgt = data.Field(
            sequential=True,
            tokenize=tokenizer,
            tokenizer_language="ja",
            pad_token=PAD_token,
            init_token=SOS_token,
            eos_token=EOS_token,
            unk_token=UNK_token,
            lower=True,
            batch_first=True,
            fix_length=max_length,
        )

class JapaneseTextWithID(SourceTargetBase):
    '''
    日本語でsourceとtargetのテキストを管理するクラス
    '''
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        max_length: int = 64,
    ) -> None:
        '''
        初期化します．

        :param tokenizer: CallableなTokenizerを指定します．
        Callableならなんでもよいです．
        '''
        super().__init__()

        self.id = data.Field(
            sequential=False,
            use_vocab=False,
        )

        self.src = data.Field(
            sequential=True,
            tokenize=tokenizer,
            tokenizer_language="ja",
            pad_token=PAD_token,
            init_token=SOS_token,
            eos_token=EOS_token,
            unk_token=UNK_token,
            lower=True,
            batch_first=True,
            fix_length=max_length,
        )
        self.tgt = data.Field(
            sequential=True,
            tokenize=tokenizer,
            tokenizer_language="ja",
            pad_token=PAD_token,
            init_token=SOS_token,
            eos_token=EOS_token,
            unk_token=UNK_token,
            lower=True,
            batch_first=True,
            fix_length=max_length,
        )
