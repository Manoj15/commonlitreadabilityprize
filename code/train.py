# Importing Packages
import pandas as pd
import numpy as np
# local imports
import sys
sys.path.append(r"c:/Users/manho/Documents/Hackathons/Kaggle/commonlitreadabilityprize")

# Custom Modules
from create_folds import create_kfolds, create_Stratkfolds
from utils.seed_config import set_seed
from feature_extract import extract_features
from config import config_param
from preprocesser import convert_scoring_error
# from models.bert_regressor import BERTRegresssor
from models.automl_pycaret import AutoMl_Pycaret

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    # Reading Data
    df_train = pd.read_csv('./input/train.csv')
    df_test = pd.read_csv('./input/test.csv')
    test_idx = df_test["id"].values
    target_col = 'target'   
    predictions = None

    set_seed(config_param.SEED)

    # df_train_feat = extract_features(df_train)
    # df_train_feat.to_csv('input/feat_extract.csv')
    df_train_feat = pd.read_csv('input/feat_extract.csv')
    print('Features Extarcted')

    # Converting scoring error variability to accuracy to weight sample while training
    df_train_feat['standard_error'] = convert_scoring_error(df_train_feat['standard_error'].values)

    df = create_Stratkfolds(df_train_feat, target_col, config_param.SEED)

    for FOLD in FOLD_MAPPPING.keys():
        

        print(" Fold Number : {0}".format(str(FOLD)))
        train_df = df[(df.kfold.isin(FOLD_MAPPPING.get(FOLD)))].reset_index(drop=True)
        valid_df = df[(df.kfold==FOLD)].reset_index(drop=True)

        caret_model = AutoMl_Pycaret(train = train_df, val= valid_df, fold = FOLD, sample_weights=train_df['standard_error'].values)
        caret_model.run()


        # #  Run BERT Model
        # bert = BERTRegresssor(train_df, valid_df)
        # bert.train()