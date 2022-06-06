from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Masking
from keras.models import Model
from keras import regularizers


class Modeller:
    # This class creates an LSTM model
    # return_sequences=True: decoder의 input에 encoder의 모든 output이 각각 사용됨
    # RepeatVector: decoder의 input에 encoder의 제일 마지막 output만 사용됨
    
    def __init__(self, data_list: List[pd.DataFrame]):
        self.data_list = data_list
        self.activation_func = 'relu' # could be 'relu', 'tanh', 'sigmoid'
        self.n_epochs = 10
        self.batch_size = 20
        
        
    def get_1layer_model(self) -> tf.keras.Model:
        df_shape = self.data_list.shape
        input_data_list = Input(shape=(df_shape[1], df_shape[2])) # determine input shape
        masked_data_list = Masking(mask_value=0.)(input_data_list) # mask input
        
        encoder = LSTM(128,
                       activation=self.activation_func,
                       return_sequences=False
                       )(masked_data_list)
        
        encoded_feature = RepeatVector(df_shape[1])(encoder)
        
        decoder = LSTM(128,
                       activation=self.activation_func,
                       return_sequences=True
                       )(encoded_feature)
        
        output_data_list = TimeDistributed(Dense(df_shape[2]))(decoder)
        
        lstm_autoencoder = Model(inputs=input_data_list, outputs=output_data_list)
        lstm_autoencoder.compile(#optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                 optimizer='adam',
                                 loss='mse',
                                 metrics=['acc']
                                 )
        lstm_autoencoder.summary()
        
        return lstm_autoencoder


    def get_2layer_model(self) -> tf.keras.Model:
        df_shape = self.data_list.shape
        input_data_list = Input(shape=(df_shape[1], df_shape[2])) # determine input shape
        masked_data_list = Masking(mask_value=0.)(input_data_list) # mask input
        
        encoder_1 = LSTM(24,
                        activation=self.activation_func,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l2(0.00)
                        )(masked_data_list)

        encoder_2 = LSTM(10,
                        activation=self.activation_func,
                        return_sequences=False
                        )(encoder_1)
        
        encoded_feature = RepeatVector(df_shape[1])(encoder_2)
        
        decoder_1 = LSTM(10,
                        activation=self.activation_func,
                        return_sequences=True
                        )(encoded_feature)

        decoder_2 = LSTM(24,
                        activation=self.activation_func,
                        return_sequences=True
                        )(decoder_1)
        
        output_data_list = TimeDistributed(Dense(df_shape[2]))(decoder_2)
        
        lstm_autoencoder = Model(inputs=input_data_list, outputs=output_data_list)
        lstm_autoencoder.compile(#optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                                 optimizer='adam',
                                 loss='mae',
                                 metrics=['acc']
                                 )
        lstm_autoencoder.summary()
        
        return lstm_autoencoder
    
    
    def train_model(self, layer_num: int) -> tf.keras.Model:
        if layer_num == 1:
            lstm_autoencoder = self.get_1layer_model()
        elif layer_num == 2:
            lstm_autoencoder = self.get_2layer_model()

        history = lstm_autoencoder.fit(self.data_list,
                                       self.data_list,
                                       epochs=self.n_epochs,
                                       batch_size=self.batch_size,
                                       validation_split=0.05
                                       ).history
        return lstm_autoencoder
    
    
    def get_prediction_of_model(self, model: tf.keras.Model) -> List[pd.DataFrame]:
        # get prediction of model
        predicted_data_list = model.predict(self.data_list, verbose=0)
        print('---Predicted---')
        print(predicted_data_list)
        print('---Actual---')
        print(self.data_list)
        return predicted_data_list
    
    
    def evaluate_model(self, model: tf.keras.Model) -> None:
        # evaluate model
        loss_and_metrics = model.evaluate(self.data_list, 
                                          self.data_list,
                                          batch_size=self.batch_size
                                          )
        print('## evaluation loss and_metrics ##')
        print(loss_and_metrics)