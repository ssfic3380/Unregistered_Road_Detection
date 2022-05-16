from os import listdir
from typing import List
import pandas as pd
import os

class Data_loader:
    
    
    def __init__(self):
        self.log_segement_info_path = '../prev_src/log_segment_info/'
        self.use_cols = ['mercX', 'mercY']
        self.users = ['0mnEB226qqgHE79KLEfxRj6fiEK2',
                      '9z6F6ewGNzV4a6z0vpenjZH21Ar1',
                      ]
        
    def load_datasets(self) -> List[pd.DataFrame]:
        data_list = []
        
        for user in self.users:
            files_dir = self.log_segement_info_path + user
            for file_name in listdir(files_dir):
       
                # log 파일이 아닐 경우 continue
                if 'segment' not in file_name: 
                    continue
                
                # log 파일 load
                file_dir = "/".join([files_dir, file_name])
                try:
                    file_df = pd.read_csv(file_dir, usecols=self.use_cols)
                    # log의 길이가 0이면 스킵
                    if len(file_df) == 0:
                        print(f"❗ Warning: [{file_dir}]'s log length is zero")
                        continue
                    data_list.append(file_df)
                except Exception as error_message:
                    print(error_message)
                    
        return data_list