import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class CustomDatasetCsv(Dataset):
    def __init__(self, root: str, window_size: int, transform=None, debug: bool = False) -> None:
        self.transform = transform
        self.debug = debug
        self.window_size = window_size
        self.data_path = Path(root)
        self.scaler = MinMaxScaler(feature_range=(0, 1)) 

        if not self.__analyze_file():
            sys.exit(-1)

        if not self.__try_open_as_dataframe():
            sys.exit(-1)

        if not self.__check_structure():
            sys.exit(-1)

        self.__load_data_and_labels()

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, index):
        windowed_x = self.data[index : index + self.window_size]
        windowed_y = self.data[index + self.window_size]
        return windowed_x, windowed_y

    def __analyze_file(self) -> bool:
        if self.debug:
            print(f'Analisi del file dati: {self.data_path.as_posix()}')

        if self.data_path.exists():
            if self.data_path.is_dir():
                if self.debug:
                    print(f'{self.data_path.as_posix()} deve essere un file, non una cartella.')
                return False
        else:
            if self.debug:
                print(f'File {self.data_path.as_posix()} inesistente.')
            return False

        if self.data_path.suffix != '.csv':
            if self.debug:
                print('Il file deve avere estensione csv.')

        if self.debug:
            print(f'Il file di dati e\' valido.')
        return True

    def __try_open_as_dataframe(self) -> bool:
        try:
            self.df = pd.read_csv(self.data_path)
            if self.debug:
                print(f'File aperto correttamente con Pandas.')
            return True
        except:
            if self.debug:
                print(f'Non e\' stato possibile aprire il file con Pandas.')
            return False        

    def __check_structure(self) -> bool:
        required_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volatility']
        condition_1 = len(self.df.shape) == 2
        condition_2 = all(col in self.df.columns for col in required_columns)
        condition_3 = len(self.df) > 0

        if condition_1 and condition_2 and condition_3:
            if self.debug:
                print(f'La struttura del file {self.data_path} e\' valida.')
            return True
        else:
            if self.debug:
                print(f'La struttura del file {self.data_path} non e\' valida.')
            return False

    def __load_data_and_labels(self) -> None:
        self.data = self.df.to_numpy().astype(np.float32)
        self.data = self.scaler.fit_transform(self.data)

    def show_data(self, data_start=None, data_end=None):
        data_start = 0 if data_start is None else data_start
        data_end = len(self.df) if data_end is None else data_end

        s = min(data_start, data_end)
        e = max(data_start, data_end)

        s = s if (s >= 0 and s <= len(self.df)) else 0
        e = e if (e >= 0 and e <= len(self.df)) else 0

        self.df[['Close']][s:e].plot()
        plt.show()


if __name__ == '__main__':
    window_size = 4
    
    tr_cdc = CustomDatasetCsv('./processed-data/D1/train.csv', window_size, debug=True)
    
    for i, data in enumerate(tr_cdc):
        if i == 5:
            break
        print(f'Sample {i}\n|_Sequence:\n{data[0]}\n|_Next element:\t{data[1]}')

    tr_cdc.show_data()
    
    va_cdc = CustomDatasetCsv('./processed-data/D1/val.csv', window_size, debug=True)
    
    for i, data in enumerate(va_cdc):
        if i == 5:
            break
        print(f'Sample {i}\n|_Sequence:\n{data[0]}\n|_Next element:\t{data[1]}')

    va_cdc.show_data()
    
    te_cdc = CustomDatasetCsv('./processed-data/D1/test.csv', window_size, debug=True)
    
    for i, data in enumerate(te_cdc):
        if i == 5:
            break
        print(f'Sample {i}\n|_Sequence:\n{data[0]}\n|_Next element:\t{data[1]}')

    te_cdc.show_data()
