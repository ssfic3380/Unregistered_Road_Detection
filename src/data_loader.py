from os import listdir, path
from typing import List, Tuple
import pandas as pd
import numpy as np

class Data_loader:  
    
    def __init__(self):
        self.log_segment_info_path = '../prev_src/log_segment_info/'
        #self.use_cols = ['mercX', 'mercY', 'latitude', 'longitude', 'direction', 'seg_lat1', 'seg_long1', 'seg_lat2', 'seg_long2']
        #self.use_cols = ['mercX', 'mercY', 'latitude', 'longitude', 'direction']
        self.use_cols = ['latitude', 'longitude']

        # for multiple datasets
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
        #self.users_hash = ['0mnEB226qqgHE79KLEfxRj6fiEK2']

        # for single dataset
        self.window_size = 1
        self.user_hash = '0mnEB226qqgHE79KLEfxRj6fiEK2'
        #self.log_segment_info_name = 'log641608610_segment_info.csv'
        self.log_segment_info_name = 'log636556604_segment_info.csv'


    # load many datasets
    def load_dataframe_list(self) -> List[pd.DataFrame]:
        dataframe_list = []
        
        for user_hash in self.users_hash:
            files_dir = self.log_segment_info_path + user_hash
            for file_name in listdir(files_dir):
       
                # skip if it is not log file
                if 'segment' not in file_name: 
                    continue
                
                # load log files
                #file_path = "/".join([files_dir, file_name])
                #file_path = f"{files_dir}/{file_name}"
                file_path = path.join(files_dir, file_name)
                try:
                    file = pd.read_csv(file_path, usecols=self.use_cols)
                    # skip if log length is zero
                    if len(file) == 0:
                        print(f"❗ Warning: [{file_path}]'s log length is zero")
                        continue
                    # delete NaN
                    file = file.fillna(0)
                    # add in dataframe_list
                    dataframe_list.append(file)
                except Exception as error_message:
                    print(error_message)
        
        print(f"◽ Total dataset length : {len(dataframe_list)}")
        return dataframe_list


    # load one dataset
    def load_data_list(self) -> np.array:
        file_path = path.join(self.log_segment_info_path, self.user_hash, self.log_segment_info_name)
        try:
            file = pd.read_csv(file_path, usecols=self.use_cols)
            file = file.fillna(0)

            data_list = file.to_numpy()
        except Exception as error_message:
            print(error_message)

        print(f"◽ Dataset length : {len(data_list)}")
        print(f"◽ Dataset shape : {data_list.shape}")
        return data_list


    def split_datasets(self, data_frame_list: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_end_idx = int(len(data_frame_list) * train_ratio)
        train_datasets = data_frame_list[:train_end_idx]
        test_datasets =  data_frame_list[train_end_idx:]
        
        print(f"◽ Training dataset length : {len(train_datasets)}")
        print(f"◽ Test dataset length : {len(test_datasets)}")
        return train_datasets, test_datasets

    
    def temporalize(self, original_data_list: np.array) -> np.array:
        temporalized_data_list = []

        for start in range(self.window_size - 1, len(original_data_list)):
            target_data = []
            for target_index in range(start - self.window_size+1, start+1):
                target_data.append(original_data_list[[target_index], :])
            temporalized_data_list.append(target_data)

        temporalized_data_list = np.array(temporalized_data_list)
        temporalized_data_list = temporalized_data_list.reshape(temporalized_data_list.shape[0], self.window_size, len(self.use_cols))

        print(f"◽ Temporalized dataset shape : {temporalized_data_list.shape}")
        return temporalized_data_list


    def untemporalize(self, temporalized_data_list: np.array) -> np.array:
        untemporalized_data_list = []

        for target in range(0, len(temporalized_data_list)):
            untemporalized_data_list.append(temporalized_data_list[target, 0])

            if (target == len(temporalized_data_list) - 1):
                for index in range(1, self.window_size):
                    untemporalized_data_list.append(temporalized_data_list[target, index])

        untemporalized_data_list = np.array(untemporalized_data_list)

        print(f"◽ Untemporalized dataset shape : {untemporalized_data_list.shape}")
        return untemporalized_data_list