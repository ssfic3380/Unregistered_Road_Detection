import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List

class Preprocessor:
    def __init__(self, df_list:List[pd.DataFrame]):
        self.df_list = df_list
        self.random_seed = 10
        
    def set_random_seed(self) -> None:
        # set random seed
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)