
from os import listdir


class Loader:
    
    def __init__(self, path):
        self.data_path = './log_segment_info/'
    
    def load_dataset(self, users, usecols):
        data_list = []
    
        for user in users:
            for file in listdir('./log_segment_info/'+user):
                log_name = file.split("_")[0]

                # log{logname}_info.csv 파일을 load
                log_csv_name= './log_segment_info/'+user+"/"+file
                
                # log 파일이 아니면 스킵
                if('segment' not in file):
                    continue

                if os.path.isfile(log_csv_name):
                    csv = pd.read_csv(log_csv_name, usecols=usecols)
                    # log의 길이가 0이면 스킵
                    if len(csv) == 0:
                        print("log length is 0")
                        continue

                    # segment matching이 실패한 행 제외
                    #csv = csv[csv['seg_id'].notnull()]

                    data_list.append(csv)
                else:
                    print("no log info")

        return data_list