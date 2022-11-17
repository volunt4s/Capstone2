import numpy as np
import pickle
import torch

if __name__ == "__main__":
    with open('data/X_MFE.pickle', 'rb') as f:
        train_x = pickle.load(f)
        train_x = torch.tensor(train_x, dtype=torch.float64)

    with open('data/y.pickle', 'rb') as f:
        train_y = pickle.load(f)
        train_y = torch.tensor(train_y, dtype=torch.long)

    print(f"Input dataset shape : {train_x.shape}")