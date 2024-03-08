import torch
from torch.utils.data import Dataset
import pandas as pd
from os import scandir, join
import numpy as np
from sklearn.model_selection import train_test_split

class HMS_Dataset(Dataset):

    def __init__(self, 
                 root_path : str = "/home/benluo/HBAC/data/hbac/",
                 eval : bool = True,
                 test_size : float = 0.2) -> None:

        super(HMS_Dataset, self).__init__()

        self.data_root = root_path

        self.train_list = pd.read_csv(join(root_path, "train.csv"))
        
        if eval:
            self.train_list, self.val_list = train_test_split(self.train_list, test_size=test_size)
        else:
            self.val_list = None

        self.test_list = pd.read_csv(join(root_path, "test.csv"))

        #0 for training, 1 for validation, 2 for testing
        self.mode = 0

        self.eeg_sample_freq = 200 # 200 Hz
        self.spec_sample_freq = 0.5 # 0.5 Hz


    def __getitem__(self, index) -> dict:

        if self.mode < 2:
        
            if self.mode == 0:
                row = self.train_list.iloc[index]
            else:
                row = self.val_list.iloc[index]

            eeg_id = row["eeg_id"]
            eeg_sub_id = row["eeg_sub_id"]
            eeg_label_offset_seconds = row["eeg_label_offset_seconds"]
            spec_id = row["spectrogram_id"]
            spec_sub_id = row["spectrogram_sub_id"]
            spec_label_offset_seconds = row["spectogram_label_offset_seconds"]
            label_id = row["label_id"]
            patient_id = row["patient_id"]
            expert_consensus = row["expert_consensus"]
            seizure_vote = row["seizure_vote"]
            lpd_vote = row["lpd_vote"]
            gpd_vote = row["gpd_vote"]
            lrda_vote = row["lrda_vote"]
            grda_vote = row["grda_vote"]
            other_vote = row["other_vote"]

            eeg = pd.read_parquet(join(self.data_root, "train_eeg/", eeg_id, ".parquet"))
            spec = pd.read_parquet(join(self.data_root, "train_spectrograms/", spec_id, ".parquet"))

            #EEG Sub-sampling
            start = eeg_label_offset_seconds*self.eeg_sample_freq
            end = (eeg_label_offset_seconds+50)*self.eeg_sample_freq
            eeg = eeg.iloc[start:end+1]
            eeg = np.array(eeg)

            #Spectrogram Sub-sampling
            start = int(spec_label_offset_seconds*self.spec_sample_freq)
            end = int((spec_label_offset_seconds+600)*self.spec_sample_freq)
            spec = spec.iloc[start:end+1]
            spec = np.array(spec)

            label = np.array([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])/np.sum(label)

            return eeg, spec, label


        row = self.test_list.iloc[index]

        eeg_id = row["eeg_id"]
        spec_id = row["spectrogram_id"]
        patient_id = row["patient_id"]
        
        eeg = pd.read_parquet(join(self.data_root, "test_eeg/", eeg_id, ".parquet"))
        spec = pd.read_parquet(join(self.data_root, "test_spectrograms/", spec_id, ".parquet"))
    
        return eeg, spec

    def __len__(self):

        if self.mode == 0:
            return self.train_list.shape[0]
        elif self.mode == 1:
            return self.val_list.shape[0]

        return self.test_list.shape[0]
    

if __name__ == "__main__":

    train_set = HMS_Dataset()