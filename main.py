from data_loader import Data_loader
from preprocessor import Preprocessor

if __name__ == "__main__":
    
    # 데이터 불러오기
    loader = Data_loader()
    df_list = loader.load_datasets()
    
    # 전처리 하기
    preprocessor = Preprocessor()
    preprocessor.set_random_seed()