import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from keras.preprocessing.sequence import pad_sequences

class Preprocessor:
    
    def __init__(self, dataframe_list:List[pd.DataFrame]):
        self.dataframe_list = dataframe_list
        self.random_seed = 10
        self.padding_max_length = 6400
        self.scaler_name2scaler_class = {
            "standard": StandardScaler,
            "normalizer": Normalizer,
            "minmax": MinMaxScaler,   
        }
        
    def set_random_seed(self) -> None:
        # set random seed
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
    def add_padding(self) -> List[pd.DataFrame]:
        padded_data_list = pad_sequences(
            self.dataframe_list,
            padding='post',
            maxlen=self.padding_max_length,
            dtype='float64'
            )
        return padded_data_list
    
    def apply_scaling(self, dataframe_list: List[pd.DataFrame], scaler_name:str) -> List[pd.DataFrame]:
        scaler = self.scaler_name2scaler_class.get(scaler_name)()
        scaled_df_list = [scaler.fit_transform(df) for df in dataframe_list]
        return scaled_df_list
        