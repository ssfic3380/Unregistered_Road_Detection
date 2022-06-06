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
            "robust": RobustScaler,
            "none": None
        }
        

    def set_random_seed(self) -> None:
        # set random seed
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)


    def set_padding_max_length(self, length: int) -> None:
        self.padding_max_length = length


    def add_padding(self, data_frame_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        padded_data_list = pad_sequences(
            data_frame_list,
            padding='post',
            maxlen=self.padding_max_length,
            dtype='float64'
            )
        
        print(f"◽ Total dataset shape : {np.shape(padded_data_list)}")
        return padded_data_list


    def apply_scaling(self, dataframe_list: List[pd.DataFrame], scaler_name: str) -> List[pd.DataFrame]:
        assert scaler_name in self.scaler_name2scaler_class.keys(), f"❗ Wrong input : {self.scaler_name2scaler_class.keys()} could be"
        if (scaler_name == "none"):
            return dataframe_list

        selected_scaler_class = self.scaler_name2scaler_class.get(scaler_name)
        scaler = selected_scaler_class()
        
        df_length_list = []
        for df in dataframe_list:
             df_length_list.append(len(df))
             
        concated_df = pd.concat(dataframe_list, axis=0) # row-wise concatenate dataframe list

        concated_df_column_list = concated_df.columns.values.tolist()
        for column in concated_df_column_list:
            print(f"◽ Max {column} : {concated_df[column].max()}")
            print(f"◽ Min {column} : {concated_df[column].min()}")

        scaled_concated_ndarray = scaler.fit_transform(concated_df)
        scaled_concated_df = pd.DataFrame(scaled_concated_ndarray, columns = concated_df_column_list)
        
        scaled_df_list = []
        for df_len in df_length_list:
            turncated_df = scaled_concated_df.truncate(before=0,
                                                       after=df_len-1,
                                                       axis=0
                                                       )
            scaled_df_list.append(turncated_df)
        
        return scaled_df_list
        