from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)#device_count={'GPU':1}
)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector, Masking, Embedding
from keras.models import Model
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences


class Modeller:
    # This class creates an LSTM model
    # return_sequences=True: decoder의 input에 encoder의 모든 output이 각각 사용됨
    # RepeadVector: decoder의 input에 encoder의 제일 마지막 output만 사용됨
    
    def __init__(self, dataframe_list: List[pd.DataFrame]):
        self.dataframe_list = dataframe_list
        self.activation_func = 'relu' # could be 'relu', 'tanh', 'sigmoid'
        self.n_epochs = 1
        self.batch_size = 20
        
        
    def get_model(self) -> tf.keras.Model:
        df_shape = self.dataframe_list.shape
        input_df_list = Input(shape=(df_shape[1], df_shape[2])) # determine input shape
        masked_df_list = Masking(mask_value=0.)(input_df_list) # mask input
        
        encoder = LSTM(64,
                       activation=self.activation_func,
                       return_sequences=False
                       )(masked_df_list)
        
        encoded_feature = RepeatVector(df_shape[1])(encoder)
        
        decoder = LSTM(64,
                       activation=self.activation_func,
                       return_sequences=True
                       )(encoded_feature)
        
        output_df_list = TimeDistributed(Dense(df_shape[2]))(decoder)
        
        lstm_autoencoder = Model(inputs=input_df_list, outputs=output_df_list)
        lstm_autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                 loss='mae',
                                 metrics=['acc']
                                 )
        lstm_autoencoder.summary()
        
        return lstm_autoencoder
    
    
    def train_model(self) -> tf.keras.Model:
        lstm_autoencoder = self.get_model()
        history = lstm_autoencoder.fit(self.dataframe_list,
                                       self.dataframe_list,
                                       epochs=self.n_epochs,
                                       batch_size=self.batch_size,
                                       validation_split=0.05
                                       ).history
        return lstm_autoencoder
    
    
    def get_prediction_of_model(self, model: tf.keras.Model) -> None:
        ## 모델의 예측값 확인
        xhat = model.predict(self.dataframe_list, verbose=0)
        print('---Predicted---')
        print(xhat)
        print('---Actual---')
        print(self.dataframe_list)
    
    
    def evaluate_model(self, model: tf.keras.Model) -> None:
        # 모델 평가하기
        loss_and_metrics = model.evaluate(self.dataframe_list, 
                                          self.dataframe_list,
                                          batch_size=self.batch_size
                                          )
        print('## evaluation loss and_metrics ##')
        print(loss_and_metrics)