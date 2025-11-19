from .config import DatasetConfig
from .downloaders.git_downloader import GitDownloader
from .extractors.shapenet_extractor import ShapeNetExtractor


class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def build(self):
        print(self.config)

    def download(self, force: bool = False):
        GitDownloader(
            repo_url=self.config.download.repo_url,
            local_dir=self.config.download.local_dir,
        ).download(force)

        return self

    def extract(self, force: bool = False):
        ShapeNetExtractor(
            input_dir=self.config.download.local_dir,
            output_dir=self.config.extracted_dir,
        ).extract(force)

        return self

    def prepare(self):
        return self

    def syntehsize(self):

        return self
