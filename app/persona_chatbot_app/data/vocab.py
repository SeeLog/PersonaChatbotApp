from collections import defaultdict


class Vocab():
    def __init__(self):
        self.stoi = defaultdict()
        self.itos = []

    def __len__(self):
        return len(self.stoi)
