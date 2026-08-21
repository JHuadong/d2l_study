import os

import pandas as pd

from d2l import torch

if __name__ == '__main__':
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'question2_2_1th.csv')
    with open(data_file, 'w') as f:
        f.write('id,MSSubClass,MSZoning,LotFrontage,NumRooms,Alley,Price\n')
        f.write('1,60,RL,65,NA,Pave,127500\n')
        f.write('2,70,NA,70,2,NA,106000\n')
        f.write('3,NA,NA,80,4,NA,178100\n')
        f.write('4,NA,NA,90,NA,NA,140000\n')

    original_data = pd.read_csv(data_file)
    print(original_data.head())

    # remove the columns with the highest number of NaN
    null_counts = original_data.isnull().sum()
    max_null_count = null_counts.max()
    cols_to_drop = null_counts[null_counts == max_null_count].index.tolist()
    original_data.drop(columns=cols_to_drop, inplace=True)
    print('-----------------After dropping columns with the highest number of null values-----------------')
    print(original_data)



    # convert the pre-processed dataset into tensor format
    original_data = pd.get_dummies(original_data, dummy_na=True)
    original_data = torch.tensor(original_data.values)
    print('-----------------After converting dataset into tensor format-----------------')
    print(original_data)