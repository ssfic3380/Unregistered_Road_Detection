from typing import List
import seaborn as sns
sns.set(color_codes=True)
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kde

class Plotter:
    
    def __init__(self, dataframe_list: pd.DataFrame):
        self.dataframe_list = dataframe_list
        self.data_len_list = [len(data) for data in self.dataframe_list]
        self.mean = statistics.mean(self.data_len_list)
        self.std = statistics.stdev(self.data_len_list)


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

    def get_length(self) -> int:
        return int(self.mean + 0.5*self.std)


def find_bins(observations: List, width: float) -> np.ndarray:
    minimmum = np.min(observations)
    maximmum = np.max(observations)
    bound_min = -1.0 * (minimmum % width - minimmum)
    bound_max = maximmum - maximmum % width + width
    n = int((bound_max - bound_min) / width) + 1
    bins = np.linspace(bound_min, bound_max, n)
    return bins