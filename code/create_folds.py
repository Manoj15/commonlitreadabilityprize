import pandas as pd
import numpy as np
from sklearn import model_selection

def create_kfolds(df,target_col, seed):

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    return df

def create_Stratkfolds(df,target_col, seed):

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    ### This was taken from https://www.kaggle.com/abhishek/step-1-create-folds
    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(df))))
    
    # bin targets
    df.loc[:, "bins"] = pd.cut(
        df[target_col], bins=num_bins, labels=False
    )

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y = df.bins.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    return df