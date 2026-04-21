from core import bootstrap
from dataset import CO3DDataset, PhotogrammetricDataset


def main():
    cfg = bootstrap(data_subdir="")

    base_dataset = CO3DDataset(
        root=cfg.data_dir / "CO3D",
        samples_per_category=5,
        categories=["tv"],
    )

    dataset = PhotogrammetricDataset(
        dataset=base_dataset, frames_per_sample=20, frames_strategy="uniform"
    )

    for sample in dataset:
        if sample != "None":
            print(sample)
            exit(0)


if __name__ == "__main__":
    main()
