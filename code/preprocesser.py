import pandas as pd
import numpy as np

def prep_text(text_df):
    text_df = text_df.str.replace("\n","",regex=False) 
    return text_df.str.replace("\'s",r"s",regex=True).values

