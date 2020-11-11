class Option():
    def __init__(self):
        self.embedding_size = 256       # default: 512
        self.hidden_size = 1024          # default: 2048
        self.num_head = 8
        self.num_layer = 6
        self.num_epoch = 100
        self.max_length = 64
        self.batch_size = 24

        self.pad_idx = 1
        self.dropout = 0.3

        # except special tokens
        self.max_vocab_size = 20000

    def __str__(self):
        ret = []
        for key, value in self.__dict__.items():
            ret.append(key + ": " + str(value))

        return "\n".join(ret)


class SentencePieceSimpleOption(Option):
    def __init__(self):
        super().__init__()
        self.embedding_size = 256
        self.hidden_size = 1024
        self.batch_size = 40
