from typing import Callable, List
import gensim
import argparse
from tqdm import tqdm
import numpy as np


class PersonaExtractor():
    """
    ペルソナを抽出するクラス
    """
    def __init__(self, model_path: str, tokenizer: Callable, dim: int = 256, last_word_length: int = 5, verbose=False) -> None:
        """
        イニシャライザ

        :param model_path: Style-sensitive word vectorsの学習済みモデル
        :param tokenizer: トークナイザ，Callableなもの
        :param dim: ペルソナベクトルの次元
        :param last_word_length: ペルソナベクトルを構成するのに用いる単語数
        """

        self.model_path = model_path
        self.tokenizer = tokenizer

        self.dim = dim
        self.last_word_length = last_word_length
        self.verbose = verbose

        self._load()



    def _load(self) -> None:
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)

    def extract(self, text: str) -> np.ndarray:
        """
        ペルソナベクトルの抽出

        :param text: 普通のテキスト，トークナイズ不要

        :return: ペルソナベクトル
        """
        words = self.tokenizer(text)
        n_words = []
        for word in words:
            if word in self.model.wv.vocab:
                n_words.append(word)

        while len(n_words) >= self.last_word_length:
            vec = np.zeros(self.dim)
            count = 0
            word_vec_list = []

            for word in n_words:
                if word in self.model.wv.vocab:
                    vec += self.model[word][:self.dim]
                    word_vec_list.append(self.model[word][:self.dim])
                    count += 1

            vec = vec / count

            if len(n_words) <= self.last_word_length:
                if self.verbose:
                    print(n_words)
                    return vec, n_words
                return vec

            idx = self._get_nearest_k(word_vec_list, vec, 1)[0]
            del n_words[idx]

        # 短い単語はこうなるよ
        vec = np.zeros(self.dim)
        count = 0
        for word in n_words:
            if word in self.model.wv.vocab:
                vec += self.model[word][:self.dim]
                count += 1

        if count == 0:
            # print("WARN: ALL UNK Persona")
            # print(text)
            if self.verbose:
                return None, None
            return None

        vec = vec / count

        if self.verbose:
            return vec, n_words
        else:
            return vec

    def _cos_similarity(self, v1: np.ndarray, v2: np.ndarray):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _top_k(self, lst, k):
        unsorted_max_indices = np.argpartition(-lst, k)[:k]
        y = lst[unsorted_max_indices]
        indices = np.argsort(-y)

        max_k_indices = unsorted_max_indices[indices]

        return max_k_indices

    def _get_nearest_k(self, vec_list, vec, k):
        # 最近傍のベクトルを探す
        cos_sims = [self._cos_similarity(vec, v) for v in vec_list]
        return self._top_k(np.array(cos_sims), k)
