import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prep_text(text_df):
    text_df = text_df.str.replace("\n","",regex=False) 
    return text_df.str.replace("\'s",r"s",regex=True).values

def convert_scoring_error(std_error):

    std_error = 1 - std_error
    scaler = MinMaxScaler()
    std_error  =  scaler.fit_transform(std_error.reshape(-1, 1))
    return std_error