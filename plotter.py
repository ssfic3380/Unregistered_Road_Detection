from typing import List
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kde

class Plotter:
    
    def __init__(self, dataframe_list: pd.DataFrame):
        self.dataframe_list = dataframe_list

    def plot_distribution(self) -> None:
        # segment 길이 추출
        data_len_list = [len(data) for data in self.dataframe_list]
        
        # plot matplotlib histogram
        bins = find_bins(data_len_list, 200.0)
        plt.hist(data_len_list, bins=bins)
        plt.show()
        
        # plot seaborn histogram
        sns.histplot(
            data_len_list,
            kde=True,
            stat="count",
            linewidth=0,
            bins=bins
            )
        plt.show()
        
        # 가우시안 커널 밀도 추정
        density = kde.gaussian_kde(data_len_list)
        x = np.linspace(0, 9000, 300)
        y = density(x)

        plt.plot(x, y)
        plt.title("Density Plot of the data")
        plt.show()
        
def find_bins(observations: List, width: float):
    minimmum = np.min(observations)
    maximmum = np.max(observations)
    bound_min = -1.0 * (minimmum % width - minimmum)
    bound_max = maximmum - maximmum % width + width
    n = int((bound_max - bound_min) / width) + 1
    bins = np.linspace(bound_min, bound_max, n)
    return bins