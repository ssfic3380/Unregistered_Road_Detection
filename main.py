from data_loader import Data_loader
from preprocessor import Preprocessor
from plotter import Plotter

if __name__ == "__main__":
    
    # 데이터 불러오기
    loader = Data_loader()
    df_list = loader.load_datasets()
    print(f"✅ Data load done")
    
    # 전처리 하기
    preprocessor = Preprocessor(df_list)
    preprocessor.set_random_seed()
    
    # data 길이의 분포 알아보기
    plotter = Plotter(df_list)
    plotter.plot_distribution()


    # 선택된 길이에 맞춰서 padding 수행
    # max_length = 6400
    # padded_data_list = add_padding(data_list, max_length)

    # data_list[0].index