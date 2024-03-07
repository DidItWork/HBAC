import torch
from torch.utils.data import Dataset
import pandas as pd
from os import scandir

class HMS_Dataset(Dataset):

    def __init__(self):

        super(HMS_Dataset, self).__init__()

        train_list = pd.read_csv("/home/benluo/HBAC/data/hbac/train.csv")

        self.get_dataset(train_list)

    def get_data(self, data_list):

    def __getitem__(self, index) -> dict:
        
        
    
    def __len__(self):

        return len([])
    

if __name__ == "__main__":

    train_set = HMS_Dataset()