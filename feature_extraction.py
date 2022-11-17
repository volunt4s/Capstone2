import numpy as np
import pandas as pd
import pickle
import os
from skimage.transform import resize
from extract_manual_features import extract_features

if __name__ == "__main__":
    print("READ DATA")
    df = pd.read_pickle('data/LSWMD.pkl')
    df = df.drop(['waferIndex', 'trianTestLabel', 'lotName'], axis=1)
    df['failureNum'] = df.failureType

    mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
                    'Near-full': 7, 'none': 8}
    df = df.replace({'failureNum': mapping_type})

    df_withlabel = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 7)]
    df_withlabel = df_withlabel.drop(df_withlabel[df_withlabel['dieSize'] < 100].index.tolist()).reset_index()

    print("EXTRACT FEATURE")
    X_mfe = extract_features(df_withlabel)
    y = np.array(df_withlabel['failureNum']).astype(np.int64)

    print("SAVE DATA")
    # Save preprocessed data as pickle files
    with open('data/X_MFE.pickle', 'wb') as f:
        pickle.dump(X_mfe, f, protocol=4)
    with open('data/y.pickle', 'wb') as f:
        pickle.dump(y, f, protocol=4)
