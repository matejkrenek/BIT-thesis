from torch_geometric.data import Data, InMemoryDataset


class ShapeNetV2Dataset(InMemoryDataset):

    category_ids = {
        "airplane": "02691156",
        "ashcan": "02747177",
        "bag": "02773838",
        "basket": "02801938",
        "bathtub": "02808440",
        "bed": "02818832",
        "bench": "02828884",
        "birdhouse": "02843684",
        "bookshelf": "02871439",
        "bottle": "02876657",
        "bowl": "02880940",
        "bus": "02924116",
        "cabinet": "02933112",
        "camera": "02942699",
        "can": "02946921",
        "cap": "02954340",
        "car": "02958343",
        "chair": "03001627",
        "clock": "03046257",
        "keyboard": "03085013",
        "dishwasher": "03207941",
        "display": "03211117",
        "earphone": "03261776",
        "faucet": "03325088",
        "file": "03337140",
        "guitar": "03467517",
        "helmet": "03513137",
        "jar": "03593526",
        "knife": "03624134",
        "lamp": "03636649",
        "laptop": "03642806",
        "loudspeaker": "03691459",
        "mailbox": "03710193",
        "microphone": "03759954",
        "microwave": "03761084",
        "motorcycle": "03790512",
        "mug": "03797390",
        "piano": "03928116",
        "pillow": "03938244",
        "pistol": "03948459",
        "pot": "03991062",
        "printer": "04004475",
        "remotecontrol": "04074963",
        "rifle": "04090263",
        "rocket": "04099429",
        "skateboard": "04225987",
        "sofa": "04256520",
        "stove": "04330267",
        "table": "04379243",
        "telephone": "04401088",
        "tower": "04460130",
        "train": "04468005",
        "vessel": "04530566",
        "washer": "04554684",
    }

    def __init__(
        self,
        root: str,
        categories: Optional[Union[str, List[str]]] = None,
        include_normals: bool = True,
        split: str = "trainval",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        pass

    def download(self) -> None:
        pass

    def process_filenames(self, filenames: List[str]) -> List[Data]:
        pass

    def process(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self)}, " f"categories={self.categories})"
        )
