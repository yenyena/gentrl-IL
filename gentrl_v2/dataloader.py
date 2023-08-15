import torch

from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class MolecularDataset(Dataset):
    def __init__(self, descr, props):
        df = pd.read_csv(descr['path'])
        self.smiles = list(df[descr['smiles']].values)
        self.props = torch.zeros(len(self.smiles), len(props)).float()

        for i, prop in enumerate(props):
            self.props[:, i] = torch.from_numpy(df[descr[prop]].values)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.props[idx, :]