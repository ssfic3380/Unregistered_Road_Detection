import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from keras.preprocessing.sequence import pad_sequences

class Preprocessor:
    
    def __init__(self):
        self.random_seed = 10
        self.padding_max_length = 1000
        self.scaler_name2scaler_class = {
            "standard": StandardScaler,
            "normalizer": Normalizer,
            "minmax": MinMaxScaler,
            "robust": RobustScaler  
        }
        
    def set_random_seed(self) -> None:
        # set random seed
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
    def add_padding(self, data_frame_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        padded_data_list = pad_sequences(
            data_frame_list,
            padding='post',
            maxlen=self.padding_max_length,
            dtype='float64'
            )
        return padded_data_list
    
    def apply_scaling(self, dataframe_list: List[pd.DataFrame], scaler_name:str) -> List[pd.DataFrame]:
        assert scaler_name in self.scaler_name2scaler_class.keys(), f"❗ Wrong input : {self.scaler_name2scaler_class.keys()} could be"
        selected_scaler_class = self.scaler_name2scaler_class.get(scaler_name)
        scaler = selected_scaler_class()
        
        df_length_list = []
        for df in dataframe_list:
             df_length_list.append(len(df))
             
        concated_df = pd.concat(dataframe_list, axis=0) # row-wise concatenate dataframe list
        print(f"◽ Max X : {concated_df['mercX'].max()}")
        print(f"◽ Min X : {concated_df['mercX'].min()}")
        print(f"◽ Max Y : {concated_df['mercY'].max()}")
        print(f"◽ Min Y : {concated_df['mercY'].min()}")

        scaled_concated_ndarray = scaler.fit_transform(concated_df)
        scaled_concated_df = pd.DataFrame(scaled_concated_ndarray, columns = ['mercX', 'mercY'])
        
        scaled_df_list = []
        for df_len in df_length_list:
            turncated_df = scaled_concated_df.truncate(before=0,
                                                       after=df_len-1,
                                                       axis=0
                                                       )
            scaled_df_list.append(turncated_df)
        
        return scaled_df_list
        