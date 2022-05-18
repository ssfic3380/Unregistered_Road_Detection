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
    #train_list, test_list = loader.split_datasets(dataframe_list, train_ratio = 0.8)
    #print(f"◽ Total dataset length : {len(dataframe_list)}")
    #print(f"◽ Training dataset length : {len(train_list)}")
    #print(f"◽ Test dataset length : {len(test_list)}")
    #print(f"◽ Sample Dataframe shape : {train_list[0].shape} (❗ Each dataframe can have a different length)")
    print(f"✅ Data load done")
    
    # Plot dataframe_list to see distribution
    plotter = Plotter(dataframe_list)
    plotter.plot_distribution()
    print(f"✅ Data visualize done")
    
    # Data preprocess
    preprocessor = Preprocessor()
    preprocessor.set_random_seed()
    max_length = plotter.get_length()
    preprocessor.set_padding_max_length(max_length)
    scaled_dataframe_list = preprocessor.apply_scaling(dataframe_list, 'minmax') # could be 'standard', 'normalizer', 'minmax', 'robust'

    # padding
    padded_data_list = preprocessor.add_padding(scaled_dataframe_list)
    print(f"✅ Data preprocess done")
    
    # modeling
    modeller = Modeller(padded_data_list)
    model = modeller.train_model()
    modeller.get_prediction_of_model(model)
    modeller.evaluate_model(model)
    print(f"✅ Modeling done")
    
if __name__ == "__main__":
    main()