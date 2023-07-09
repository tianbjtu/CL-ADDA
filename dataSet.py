from torch.utils.data import Dataset
from tool import *


class MyDataset(Dataset):

  def __init__(self, movie_data_fc,subjInfo_label):

    self.subjInfof_label=subjInfo_label
    self.movie_data_fc=movie_data_fc

  def __getitem__(self, index):

    result_label=self.subjInfof_label[index]
    result=self.movie_data_fc[index]
    return (result,result_label)

  def __len__(self):
    return len(self.subjInfof_label)
