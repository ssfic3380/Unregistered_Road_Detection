from data_loader import Data_loader
from preprocessor import Preprocessor
from plotter import Plotter

if __name__ == "__main__":
    
    # 데이터 불러오기
    loader = Data_loader()
    dataframe_list = loader.load_datasets()
    print(f"✅ Data load done")
    
    # 전처리 하기
    preprocessor = Preprocessor(dataframe_list)
    preprocessor.set_random_seed()
    preprocessor.apply_scaling(dataframe_list, '추가행')
    print(f"✅ Data preprocess done")
    
    # data 길이의 분포 알아보기
    plotter = Plotter(dataframe_list)
    plotter.plot_distribution()
    print(f"✅ Data visualize done")
    
    # 선택된 길이에 맞춰서 padding 수행
    padded_data_list = preprocessor.add_padding()
    
    print(f"❗ Sample : <data_list[0].index> == {dataframe_list[0].index}")