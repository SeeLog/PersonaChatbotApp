from torchtext import data


class SourceTargetBase():
    def __init__(self) -> None:
        super().__init__()
        self.src: data.Field = None
        self.tgt: data.Field = None
