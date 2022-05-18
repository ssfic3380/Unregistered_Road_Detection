from src.data_loader import Data_loader
from src.preprocessor import Preprocessor
from src.plotter import Plotter

def main():
    # Data load
    loader = Data_loader()
    dataframe_list = loader.load_datasets()
    print(f"✅ Data load done")
    
    # Data preprocesse
    preprocessor = Preprocessor(dataframe_list)
    preprocessor.set_random_seed()
    preprocessor.apply_scaling(dataframe_list, 'minmax') # could be 'standard', 'normalizer', 'minmax'
    print(f"✅ Data preprocess done")
    
    # Plot dataframe_list to see distribution
    plotter = Plotter(dataframe_list)
    plotter.plot_distribution()
    print(f"✅ Data visualize done")
    
    # padding
    padded_data_list = preprocessor.add_padding()
    
    # test print
    print(padded_data_list[0].head())
    print(f"❗ Sample : <data_list[0].index> == {dataframe_list[0].index}")
    
if __name__ == "__main__":
    main()