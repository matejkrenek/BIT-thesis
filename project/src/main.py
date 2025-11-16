from visualization.viewers.sample import SampleViewer
from utils.io import load_ply

if __name__ == "__main__":
    pairs = []

    clean = load_ply("data/examples/bunny.ply")
    corrupted = load_ply("data/examples/bunny.ply")
    pairs.append((clean, corrupted))

    clean = load_ply("data/examples/bunny.ply")
    corrupted = load_ply("data/examples/bunny.ply")
    pairs.append((clean, corrupted))

    viewer = SampleViewer(pairs)
    viewer.show()
