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

    df_train_feat = extract_features(df_train)
    print('Features Extarcted')

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


    #     ytrain = train_df[target_col].values
    #     yvalid = valid_df[target_col].values

    #     train_df = train_df.drop(["sig_id","kfold"] + target_cols, axis=1)
    #     valid_df = valid_df.drop(["sig_id", "kfold"] + target_cols, axis=1)

    #     # Sort columns based on train df
    #     valid_df = valid_df[train_df.columns]            
        
    #     # Oversampling
    #     train_df, ytrain = oversample_minority_svm(train_df, ytrain)

    #     predictions = None

    #     for r in seed:
    #         # data is ready to train
    #         print(MODEL)
    #         clf = dispatcher.MODELS[MODEL]
    #         setattr(clf, 'random_state', r) 
    #         print(target_col)
    #         clf.fit(train_df, ytrain)
    #         preds_valid = clf.predict_proba(valid_df)[:, 1]
    #         pred_test = clf.predict_proba(test_df)[:, 1]

            
    #     print("Fold : ", FOLD)
    #     print("train_shape : ", str(train_df.shape))
    #     print("valid_shape : ", str(valid_df.shape))
    #     print('Train Class Ratio : ', str((np.count_nonzero(ytrain == 1)/ytrain.shape[0])*100))
    #     print('Valid Class Ratio : ', str((np.count_nonzero(yvalid == 1)/ytrain.shape[0])*100))
    #     # print('Class Ratio : ', str(np.count_nonzero(ytrain == 1)))
    #     print('AUC of {0} is '.format(target_col),metrics.roc_auc_score(yvalid, preds))
    #     print('Log Loss of {0} is '.format(target_col),metrics.log_loss(yvalid, preds))
    #     # auc = []
    #     # auc.append(metrics.roc_auc_score(yvalid, preds))
    #     # print(auc)
    #     # print(preds[:5])

    # break

    # if SAVE == True:
    #     joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    #     joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")