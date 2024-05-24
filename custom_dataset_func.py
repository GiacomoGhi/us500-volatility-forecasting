import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset


class CustomDatasetFunc(Dataset):

    def __init__(self, min_x, max_x, num_data, window_size, transform = None) -> None:

        # Memorizzo le trasformazioni che potro' fare sui dati.
        self.transform = transform
        
        # Imposto la grandezza della sequenza dati.
        self.window_size = window_size
        
        self.x_data = np.linspace(min_x, max_x, num_data).astype(np.float32)
        self.y_data = np.sin(self.x_data / 10).astype(np.float32)
        
        self.df = pd.DataFrame({'X': self.x_data, 'Y': self.y_data})
        self.df.index.name = 'Id'

    def __len__(self):
        return len(self.x_data) - self.window_size

    def __getitem__(self, index):
        windowed_x = self.y_data[index : index + self.window_size]
        windowed_y = self.y_data[index + self.window_size : index + self.window_size + 1]
        return windowed_x, windowed_y
    
    def dump_csv(self, filename : str):
        
        out_path = Path('./data/')
        
        if not out_path.exists():
            out_path.mkdir()
            
        filename = out_path / filename
        self.df.to_csv(filename, index=True)

    def show_data(self, data_start = None, data_end = None):            
        
        data_start = 0 if data_start is None else data_start
        data_end = len(self.df) if data_end is None else data_end
        
        s = min(data_start, data_end)
        e = max(data_start, data_end)
        
        s = s if (s >= 0 and s <= len(self.df)) else 0
        e = e if (e >= 0 and e <= len(self.df)) else 0
        
        self.df['Y'][s:e].plot()
        plt.show()

if __name__ == '__main__':

    num_data    = 1000
    va_perc     = 0.10
    te_perc     = 0.10
    window_size = 40

    te_data = int(num_data * te_perc)
    va_data = int(num_data * va_perc)
    tr_data = int(num_data - te_data - va_data)

    tr_cdf = CustomDatasetFunc(0, tr_data - 1, tr_data, window_size)
    
    for i, data in enumerate(tr_cdf):
        if i == 5:
            break
        print(f'Sample {i}\n|_Sequence:\t{data[0]}\n|_Next element:\t{data[1]}')

    tr_cdf.show_data()
    tr_cdf.dump_csv('train.csv')
    
    va_cdf = CustomDatasetFunc(0, va_data - 1, va_data, window_size)
    
    for i, data in enumerate(va_cdf):
        if i == 5:
            break
        print(f'Sample {i}\n|_Sequence:\t{data[0]}\n|_Next element:\t{data[1]}')

    va_cdf.show_data()
    va_cdf.dump_csv('val.csv')
    
    te_cdf = CustomDatasetFunc(0, te_data - 1, te_data, window_size)
    
    for i, data in enumerate(te_cdf):
        if i == 5:
            break
        print(f'Sample {i}\n|_Sequence:\t{data[0]}\n|_Next element:\t{data[1]}')

    te_cdf.show_data()
    te_cdf.dump_csv('test.csv')