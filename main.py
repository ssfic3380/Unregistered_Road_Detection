import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.data_loader import Data_loader
from src.preprocessor import Preprocessor
from src.plotter import Plotter
from src.modeling import Modeller

def main():
    # Data load
    loader = Data_loader()
    dataframe_list = loader.load_datasets()
    print(f"✅ Data load done")
    
    # Data preprocesse
    preprocessor = Preprocessor()
    preprocessor.set_random_seed()
    scaled_dataframe_list = preprocessor.apply_scaling(dataframe_list, 'minmax') # could be 'standard', 'normalizer', 'minmax'
    
    # Plot dataframe_list to see distribution
    #plotter = Plotter(dataframe_list)
    #plotter.plot_distribution()
    #print(f"✅ Data visualize done")
    
    # padding
    padded_data_list = preprocessor.add_padding(scaled_dataframe_list)
    print(f"✅ Data preprocess done")
    
    # test print
    #print(padded_data_list[0].head())
    #print(f"❗ Sample : <data_list[0].index> == {dataframe_list[0].index}")
    
    # modeling
    modeller = Modeller(padded_data_list)
    model = modeller.train_model()
    modeller.get_prediction_of_model(model)
    modeller.evaluate_model(model)
    
if __name__ == "__main__":
    main()