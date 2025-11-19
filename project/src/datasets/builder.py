from .config import DatasetConfig


class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def build(self):
        print(self.config)
