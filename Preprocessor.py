import pandas as pd
import numpy as np
import tensorflow as tf

class Preprocessor:
    def __init__(self, dataset:pd.DataFrame):
        self.dataset = dataset,
        self.random_seed = 10
        
    def set_random_seed(self) -> None:
        # set random seed
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
    
    