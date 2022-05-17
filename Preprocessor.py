import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List
from keras.preprocessing.sequence import pad_sequences

class Preprocessor:
    
    def __init__(self, dataframe_list:List[pd.DataFrame]):
        self.dataframe_list = dataframe_list
        self.random_seed = 10
        self.padding_max_length = 6400
        
    def set_random_seed(self) -> None:
        # set random seed
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
    def add_padding(self):
        padded_data_list = pad_sequences(
            self.dataframe_list,
            padding='post',
            maxlen=self.padding_max_length,
            dtype='float64'
            )
        return padded_data_list