import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.data_loader import Data_loader
from src.preprocessor import Preprocessor
from src.plotter import Plotter
from src.modeling import Modeller

# 테스트용
import pandas as pd
import numpy as np

def main():

    # for multiple datasets
    # Data load
    loader = Data_loader()
    dataframe_list = loader.load_dataframe_list()
    #train_list, test_list = loader.split_datasets(dataframe_list, train_ratio = 0.8)
    print(f"✅ Data load done")
    
    # Plot dataframe_list to see distribution
    plotter = Plotter(dataframe_list)
    #plotter.plot_distribution()
    print(f"✅ Data visualize done")
    
    # Data preprocess
    preprocessor = Preprocessor()
    preprocessor.set_random_seed()
    # scaling
    scaled_dataframe_list = preprocessor.apply_scaling(dataframe_list, 'none') # could be 'standard', 'normalizer', 'minmax', 'robust'
                                                                               # if you don't need scaling, 'none'
    # padding
    #max_length = plotter.get_length()
    max_length = 100 # 임시
    preprocessor.set_padding_max_length(max_length)
    padded_dataframe_list = preprocessor.add_padding(scaled_dataframe_list)
    print(f"✅ Data preprocess done")

    # Modeling
    modeller = Modeller(padded_dataframe_list)
    model = modeller.train_model(layer_num=2)
    predicted_dataframe_list = modeller.get_prediction_of_model(model)
    modeller.evaluate_model(model)
    print(f"✅ Modeling done")

    # Plot route
    plotter.plot_multi_route(padded_dataframe_list, "multi_actual")
    plotter.plot_multi_route(predicted_dataframe_list, "multi_predicted")    
    print(f"✅ Route visualize done")
    exit()


    # for single dataset
    # Data load
    loader = Data_loader()
    data_list = loader.load_data_list()
    temporalized_data_list = loader.temporalize(data_list)
    print(f"✅ Data load done")

    # Plotter
    plotter = Plotter(data_list)

    # Data preprocess
    preprocessor = Preprocessor()
    preprocessor.set_random_seed()
    # scaling
    scaled_data_list = preprocessor.apply_scaling(temporalized_data_list, 'none') # could be 'standard', 'normalizer', 'minmax', 'robust'
                                                                                  # if you don't need scaling, 'none'

    # Modeling
    modeller = Modeller(scaled_data_list)
    model = modeller.train_model(layer_num=2)
    predicted_data_list = modeller.get_prediction_of_model(model)
    modeller.evaluate_model(model)
    print(f"✅ Modeling done")

    # Plot route
    original_data_list = data_list
    predicted_data_list = loader.untemporalize(predicted_data_list)
    plotter.plot_single_route(original_data_list, "single_actual")
    plotter.plot_single_route(predicted_data_list, "single_predicted")
    print(f"✅ Route visualize done")
    
    
if __name__ == "__main__":
    main()