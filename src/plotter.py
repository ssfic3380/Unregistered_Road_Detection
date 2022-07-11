from typing import List

import folium
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import numpy as np
import pandas as pd
import statistics
from scipy.stats import kde

class Plotter:
    
    def __init__(self, dataframe_list: pd.DataFrame):
        self.dataframe_list = dataframe_list
        self.data_len_list = [len(data) for data in self.dataframe_list]
        self.mean = statistics.mean(self.data_len_list)
        self.std = statistics.stdev(self.data_len_list)

        self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
        self.map_path = '../prev_src/maps_hys/'


    def plot_distribution(self) -> None:        
        # plot matplotlib histogram
        bins = find_bins(self.data_len_list, 200.0)
        plt.hist(self.data_len_list, bins=bins)
        plt.axvline(self.mean, color='k', linestyle='dashed')
        plt.axvline(self.mean + 0.5*self.std, color='g', linestyle='dashed')
        plt.axvline(self.mean - 0.5*self.std, color='g', linestyle='dashed')
        plt.axvline(self.mean + self.std, color='y', linestyle='dashed')
        plt.axvline(self.mean - self.std, color='y', linestyle='dashed')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.title('Histogram')
        plt.show()


    # 평균 + 0.5표준편차 위치의 값을 padding 길이로 설정하기 위한 메소드
    def get_length(self) -> int:
        return int(self.mean + 0.5*self.std)


    def plot_multi_route(self, data_list: List[pd.DataFrame], save_file_name: str) -> None:
        map_osm = folium.Map(location=[37.6257746354002, 126.817219583318], zoom_start=12)
        cnt = 0

        routes = []
        for data in data_list:
            for lat_long in data:
                if lat_long[0] == 0:
                    continue
                else:
                    routes.append(lat_long)
        
            for index in range(len(routes) - 1):
                loc = [routes[index], routes[index+1]]
                folium.PolyLine(loc,
                                color=self.colors[cnt%8],
                                weight=5,
                                opacity=0.8).add_to(map_osm)
            cnt = cnt + 1
        
        map_osm.save(self.map_path + save_file_name + '.html')


    def plot_single_route(self, data_list: List[pd.DataFrame], save_file_name: str) -> None:
        map_osm = folium.Map(location=[37.6257746354002, 126.817219583318], zoom_start=12)
        cnt = 0

        routes = []
        for lat_long in data_list:
            if lat_long[0] == 0:
                continue
            else:
                routes.append(lat_long)
        
        for index in range(len(routes) - 1):
            loc = [routes[index], routes[index+1]]
            folium.PolyLine(loc,
                            color=self.colors[cnt%8],
                            weight=5,
                            opacity=0.8).add_to(map_osm)

        cnt = cnt + 1
        
        map_osm.save(self.map_path + save_file_name + '.html')


# plot_distribution에 사용되는 메소드 (적절한 bins를 찾는 메소드)
def find_bins(observations: List, width: float) -> np.ndarray:
    minimmum = np.min(observations)
    maximmum = np.max(observations)
    bound_min = -1.0 * (minimmum % width - minimmum)
    bound_max = maximmum - maximmum % width + width
    n = int((bound_max - bound_min) / width) + 1
    bins = np.linspace(bound_min, bound_max, n)
    return bins