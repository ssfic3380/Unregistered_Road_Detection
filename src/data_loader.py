from os import listdir
from typing import List
import pandas as pd

class Data_loader:  
    
    def __init__(self):
        self.log_segment_info_path = '../prev_src/log_segment_info/'
        self.use_cols = ['mercX', 'mercY']
        self.users_hash = ['01WPXP7OfDQtMeFqczOs0yoKms32',
                           'OMmQkOhTgUfNdgJ6Hx9EPe7zReg1',
                           '2Ea7BdHDdPbCo4XngUzyIX2yBou1',
                           '77Q8hcwlfPSslea9rjAEiXDPB3R2',
                           'watlo38e0TU89EyhbKwxSXX5IPN2',
                           'ueyx35DAWfhXcO1BQo6aEOePWKe2',
                           'eEg0e2E1CoP3V6LSJxUQCDacNas1',
                           'JTXKl8ZBWFgdNWGHlxwXGvxJEbO2',
                           'UW0IPlU2hGOWDLgLZwA451BQyml1',
                           'ahpAq1Fwj8eO594l3CJ1r6kFD1K2',
                           'jIgXm6uwLuc9GwTxW4OoQausc162',
                           'uh1sF6AqiAQN0wYLQnsTFk5txsx1',
                           'dN4AlgMOWhRAwauOvHGYEyvLRFX2',
                           'MsMsBYIva2SYvmInH8xwVjqeG9c2',
                           'BcGek89jZPWdAIXQA1x0i4BYVqc2',
                           'qCnCGVTvptPul1yEO7VBDErX9Y43',
                           '16viBpfO0tc2DcTfeTa9TjN6Nam2',
                           'UeWTlLoudahes7rCWdatYFdDBS33',
                           'wWORa3eelHPOIiNEMtqxqvvTxKq2',
                           'YDTSNtW1C4cAcGlAY2jwRrwMSi82',
                           'a48Ae0rgwGYoPP4eVsjHLcfhB673',
                           '74mbmeYO4SMCBZ0zER2fPRwZxnA3',
                           'NlSC6XZw89SVqpbYnFBOl1dSQxw2',
                           'rVzIqYuDhvW0iTVfoFFspj3LwBi1',
                           'b6D5uc9tOvaYJ1bcDBFoVgzpaqJ3',
                           'yrNIQAh3qiOfYaOnfCBUloEftbm1',
                           'yrAH0r8mtkQNNejCKdBi5towyCv1',
                           'cQfrq9eo6kYvaiBb09zmde8XjrD3',
                           'yjCBTfEVL0dwmUFzU3GvqroV0XC3',
                           'wQmN5JirFEdKQb00Cnaz7CvmYOp1']
        
    def load_datasets(self) -> List[pd.DataFrame]:
        data_list = []
        
        for user_hash in self.users_hash:
            files_dir = self.log_segment_info_path + user_hash
            for file_name in listdir(files_dir):
       
                # log 파일이 아닐 경우 continue
                if 'segment' not in file_name: 
                    continue
                
                # log 파일 load
                file_path = "/".join([files_dir, file_name])
                file_path = f"{files_dir}/{file_name}"
                try:
                    file_df = pd.read_csv(file_path, usecols=self.use_cols)
                    # log의 길이가 0이면 스킵
                    if len(file_df) == 0:
                        print(f"❗ Warning: [{file_path}]'s log length is zero")
                        continue
                    data_list.append(file_df)
                except Exception as error_message:
                    print(error_message)
                    
        return data_list