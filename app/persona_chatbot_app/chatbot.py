import torch
import torch.nn.functional as F
import numpy as np

from typing import List

from .data.japanese_text import JapaneseTextWithID
from .model.transformer.models import TargetPersonaEncoderDecoder


class ChatBotBase():
    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, text: str) -> str:
        raise NotImplementedError()


class TargetPersonaChatBot(ChatBotBase):
    def __init__(self, model: TargetPersonaEncoderDecoder, fields: JapaneseTextWithID, device: torch.device, pad_idx=1, sos_idx=2, eos_idx=3, unk_idx=0, max_len=64) -> None:
        super().__init__(model)

        self.model = model
        self.fields = fields
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        self.max_len = max_len

    def __call__(self, text: str, persona: torch.tensor, decode_method="top", top_k=50, top_p=0.95) -> str:
        '''
        テキストとペルソナを受け取って応答を返す

        :param text: 入力文，str
        :param persona: ペルソナテンソル
        :param decode_method: デコーディング手法, default: "top"
        :param top_k: top-kの値, default: 50
        :param top_p: top-pの値, default: 0.95

        :return: モデルの応答をテキストで返す
        '''
        self.model.eval()

        with torch.no_grad():
            src = self.fields.src.process([self.fields.src.tokenize(text)]).to(self.device)
            src_mask = (src != self.pad_idx).unsqueeze(-2)
            if decode_method.lower() == "top":
                out = self.top_k_top_p_decoding(src, persona, top_k, top_p)
            else:
                out = self.greedy_decode(src, src_mask, persona, max_len=self.max_len, start_symbol=self.sos_idx)

            ret = []

            for i in range(1, out.size(1)):
                sym = self.fields.tgt.vocab.itos[out[0, i]]
                if sym == self.fields.tgt.vocab.itos[self.eos_idx]:
                    break
                ret.append(sym)

            return " ".join(ret)

    def generate_with_batch(self, src: torch.tensor, tgt_persona: torch.tensor) -> List[str]:
        self.model.eval()
        with torch.no_grad():
            src_mask = (src != self.pad_idx).unsqueeze(-2)

            out = self.greedy_decode(src, src_mask, tgt_persona, max_len=self.max_len, start_symbol=self.sos_idx)

            ret = []

            for batch_i in range(out.size(0)):
                tgt_list = []
                for i in range(1, out.size(1)):
                    sym = self.fields.tgt.vocab.itos[out[batch_i, i]]
                    if sym == self.fields.tgt.vocab.itos[self.eos_idx]:
                        break
                    tgt_list.append(sym)
                ret.append(" ".join(tgt_list))

        return ret

    def greedy_decode(self, src: torch.tensor, src_mask: torch.tensor, tgt_persona: torch.tensor, max_len: torch.tensor, start_symbol: int):
        """
        Greedy Decoding

        :param model: encode()とdecode()が実装されたモデル
        :param src: Source Tensor
        :param src_mask: Source Mask
        :param tgt_persona: ターゲットのペルソナ
        :param max_len: Max Length
        :param start_symbol: SOSシンボル

        :return: Greedy DecodingされたTensor
        """
        memory = self.model.encode(src, src_mask)
        ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = self.model.decode(memory, src_mask, ys, self.subsequent_mask(ys.size(1)).type_as(src.data), tgt_persona[:, :ys.size(1), :])
            prob = self.model.generator(out[:, -1])

            # unk prob set to zero.
            prob[:, self.unk_idx].fill_(float("-inf"))
            if i == 0:
                # <eos> prob set to zero.
                prob[:, self.eos_idx].fill_(float("-inf"))

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(-1)

            ys = torch.cat([ys, next_word], dim=1)

            if torch.sum(torch.max(ys == self.eos_idx, dim=-1)[0]) == ys.size(0):
                # <eos> が全ミニバッチで出現した
                break
        return ys


    def top_k_top_p_decoding(self, src: torch.tensor, tgt_persona: torch.tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = float("-inf")) -> torch.tensor:
        """
        top-k top-p サンプリングを用いてデコーディングを行います．

        :param src: 入力テンソル，[batch_size, src_len]
        :param top_k: 上位いくつを生存させるか？, default: 0
        :param top_p: 上位何割を生存させるか？, default: 0.0
        :param filter_value: 失格したやつをどの値に設定するか？, default: float("-inf")

        :return: デコードされたテンソル
        """
        self.model.eval()
        with torch.no_grad():
            src_mask = (src != self.pad_idx).unsqueeze(-2)
            memory = self.model.encode(src, src_mask)
            ys = torch.zeros(src.size(0), 1).fill_(self.sos_idx).type_as(src.data)

            for i in range(self.max_len - 1):
                out = self.model.decode(memory, src_mask, ys, self.subsequent_mask(ys.size(1)).type_as(src.data), tgt_persona[:, :ys.size(1), :])
                logits = self.model.generator(out[:, -1])

                # unk prob set to zero.
                logits[:, self.unk_idx].fill_(float("-inf"))
                if i == 0:
                    # <eos> prob set to zero.
                    logits[:, self.eos_idx].fill_(float("-inf"))
                # filter top-k and top-p -> calc probabilities
                logits = self.top_k_top_p_filtering(logits, top_k, top_p, filter_value)
                prob = F.softmax(logits, dim=-1)
                next_word = torch.multinomial(prob, 1)

                ys = torch.cat([ys, next_word], dim=1)

                if torch.sum(torch.max(ys == self.eos_idx, dim=-1)[0]) == ys.size(0):
                    # <eos> が全ミニバッチで出現した
                    break

        return ys

    def top_k_top_p_filtering(self, logits: torch.tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = float("-inf")) -> torch.tensor:
        """
        top-k top-p を用いて不要な確率値を0にセットする(log probの場合を想定しているのでデフォルトは -inf にセットする)
        一応 return しているが，内部で logits の中身を操作しているため内容が変わる可能性が高い．
        気になる場合は logits を渡す前に clone() しておくと良いでしょう．

        :param logits: モデルが吐いた logits, [batch_size, vocab_size]
        :param top_k: 上位いくつを生存させるか？, default: 0, 無効
        :param top_p: いくつ以上の確率値のものを生存させるか？, default: 0.0, 無効
        :param filter_value: 消す単語の値としてなにをセットするか？, default: float("-inf")

        :return: フィルター（というよりマスク？）をしたlogitsを返します，[batch_size, vocab_size]
        """
        # 溢れたらまずいのでminを通しておく
        top_k = min(top_k, logits.size(-1))

        if top_k > 0:
            # top-k を適応
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # top-p を適応
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # しきい値以上の prob を削除する(上位からtop-p分だけ残し，それ以降は削除)
            sorted_indices_to_remove = cumulative_probs > top_p
            # 右シフトして一番確率の高い単語は残す(そうじゃないとどの単語も生成確率0になる)
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            # Trueとなっているインデックスの削除
            for i in range(logits.size(0)):
                # バッチ対応
                indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i, :]]
                logits[i, indices_to_remove] = filter_value

        return logits

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
