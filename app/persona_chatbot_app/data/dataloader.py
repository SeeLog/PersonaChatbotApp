from persona_chatbot_app.data.japanese_text import JapaneseTextWithID
import torch
import torchtext
from torchtext import data
import numpy as np

from .source_target_base import SourceTargetBase

from typing import Optional

ID = "id"
SOURCE = "src"
TARGET = "tgt"


class DataLoader():
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        fields: SourceTargetBase,
        batch_size: int = 256,
        max_vocab_size: int = 20000,
        device: torch.device = "cpu",
        vocab: Optional[torchtext.vocab.Vocab] = None,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.max_vocab_size = max_vocab_size

        # region build train, also build vocab
        self.train_dataset = data.TabularDataset(
            path=train_path,
            format="tsv",
            fields=[
                (SOURCE, fields.src),
                (TARGET, fields.tgt),
            ],
            csv_reader_params={"quotechar": None},
        )

        if vocab is None:
            self.train_dataset.fields[SOURCE].build_vocab(
                self.train_dataset.src,
                self.train_dataset.tgt,
                max_size=max_vocab_size,
            )
        else:
            self.train_dataset.fields[SOURCE].vocab = vocab

        self.train_dataset.fields[TARGET].vocab = self.train_dataset.fields[SOURCE].vocab

        # endregion

        # region build valid/test

        self.valid_dataset = data.TabularDataset(
            path=valid_path,
            format="tsv",
            fields=[
                (SOURCE, fields.src),
                (TARGET, fields.tgt),
            ],
            csv_reader_params={"quotechar": None},
        )

        self.test_dataset = data.TabularDataset(
            path=test_path,
            format="tsv",
            fields=[
                (SOURCE, fields.src),
                (TARGET, fields.tgt),
            ],
            csv_reader_params={"quotechar": None},
        )
        # endregion

        self.vocab_size = self.get_vocab_size()
        self.fields = fields

    def get_vocab_size(self) -> int:
        '''
        Vocabのサイズを取得します

        :return: vocab_size
        '''

        return len(self.train_dataset.fields[TARGET].vocab)

    def _get_batch(self, dataset: data.TabularDataset, train: bool) -> data.BucketIterator:
        return data.BucketIterator(dataset=dataset, batch_size=self.batch_size, device=self.device, repeat=False, train=train, sort=False)

    def get_train_batch(self) -> data.BucketIterator:
        '''
        Trainのミニバッチを作るイテレータを生成します

        :return: ミニバッチイテレータ
        '''
        return self._get_batch(self.train_dataset, train=True)

    def get_valid_batch(self) -> data.BucketIterator:
        '''
        Validationのミニバッチを作るイテレータを生成します

        :return: ミニバッチイテレータ
        '''
        return self._get_batch(self.valid_dataset, train=False)

    def get_test_batch(self) -> data.BucketIterator:
        '''
        Testのミニバッチを作るイテレータを生成します

        :return: ミニバッチイテレータ
        '''
        return self._get_batch(self.test_dataset, train=False)


class MiniBatch():
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=1):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    @staticmethod
    def from_src_tgt(src, tgt, pad: int = 1) -> "MiniBatch":
        MiniBatch(src, tgt, pad=pad)


class PersonaDataLoader():
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        fields: JapaneseTextWithID,
        batch_size: int = 256,
        max_vocab_size: int = 20000,
        device: torch.device = "cpu",
        vocab: Optional[torchtext.vocab.Vocab] = None,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.max_vocab_size = max_vocab_size

        # region build train, also build vocab
        self.train_dataset = data.TabularDataset(
            path=train_path,
            format="tsv",
            fields=[
                (ID, fields.id),
                (SOURCE, fields.src),
                (TARGET, fields.tgt),
            ],
            csv_reader_params={"quotechar": None},
        )

        if vocab is None:
            self.train_dataset.fields[SOURCE].build_vocab(
                self.train_dataset.src,
                self.train_dataset.tgt,
                max_size=max_vocab_size,
            )
        else:
            self.train_dataset.fields[SOURCE].vocab = vocab

        self.train_dataset.fields[TARGET].vocab = self.train_dataset.fields[SOURCE].vocab

        # endregion

        # region build valid/test

        self.valid_dataset = data.TabularDataset(
            path=valid_path,
            format="tsv",
            fields=[
                (ID, fields.id),
                (SOURCE, fields.src),
                (TARGET, fields.tgt),
            ],
            csv_reader_params={"quotechar": None},
        )

        self.test_dataset = data.TabularDataset(
            path=test_path,
            format="tsv",
            fields=[
                (ID, fields.id),
                (SOURCE, fields.src),
                (TARGET, fields.tgt),
            ],
            csv_reader_params={"quotechar": None},
        )
        # endregion

        self.vocab_size = self.get_vocab_size()
        self.fields = fields

    def get_vocab_size(self) -> int:
        '''
        Vocabのサイズを取得します

        :return: vocab_size
        '''

        return len(self.train_dataset.fields[TARGET].vocab)

    def _get_batch(self, dataset: data.TabularDataset, train: bool) -> data.BucketIterator:
        return data.BucketIterator(dataset=dataset, batch_size=self.batch_size, device=self.device, repeat=False, train=train, sort=False)

    def get_train_batch(self) -> data.BucketIterator:
        '''
        Trainのミニバッチを作るイテレータを生成します

        :return: ミニバッチイテレータ
        '''
        return self._get_batch(self.train_dataset, train=True)

    def get_valid_batch(self) -> data.BucketIterator:
        '''
        Validationのミニバッチを作るイテレータを生成します

        :return: ミニバッチイテレータ
        '''
        return self._get_batch(self.valid_dataset, train=False)

    def get_test_batch(self) -> data.BucketIterator:
        '''
        Testのミニバッチを作るイテレータを生成します

        :return: ミニバッチイテレータ
        '''
        return self._get_batch(self.test_dataset, train=False)


class PersonaMiniBatch():
    def __init__(self, src, trg=None, src_persona=None, tgt_persona=None, src_persona_dict=None, tgt_persona_dict=None, pad=1):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if src_persona is not None and src_persona_dict is not None:
            batch_size, seq_len = self.src.shape
            device = self.src.device
            personas = torch.tensor([src_persona_dict[int(persona_id)] for persona_id in src_persona]).repeat(1, 1, seq_len).view(batch_size, seq_len, -1)
            personas.to(device)
            self.src_persona = personas

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

            if tgt_persona is not None and tgt_persona_dict is not None:
                # personas: [batch_size, 300?]
                batch_size, seq_len = self.trg.shape
                device = self.trg.device
                personas = torch.tensor([tgt_persona_dict[int(persona_id)] for persona_id in tgt_persona]).to(device).float().repeat(1, 1, seq_len).view(batch_size, seq_len, -1)
                # maskはforward時に適応されるため無理にここでしなくていい
                self.trg_persona = personas.to(device)




    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    @staticmethod
    def from_src_tgt(src, tgt, pad: int = 1) -> "MiniBatch":
        MiniBatch(src, tgt, pad=pad)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
