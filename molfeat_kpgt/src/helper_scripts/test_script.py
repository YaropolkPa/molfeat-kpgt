import sys
from multiprocessing import Pool
import pandas as pd

sys.path.append("..")

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized


if __name__ == "__main__":
    # df = pd.read_csv("./datasets/freesolv/freesolv.csv")
    df = pd.read_csv("../datasets/freesolv/freesolv.csv")
    smiless = df.smiles.values.tolist()
    generator = RDKit2DNormalized()
    # data = generator.process(smiles)
    features_map = Pool(32).imap(generator.process, smiless)
    arr = list(features_map)
    print(arr)