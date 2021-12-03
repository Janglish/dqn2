import os
import time
import torch
import pandas as pd
from collections import namedtuple

class Logger:
    
    def __init__(self, dir_path="logs", file_name="logs", seed=0):
        self.dir_path = dir_path
        self.file_name = file_name
        self.start_time = time.time()
        os.makedirs(self.dir_path, exist_ok=True)
        self.df = pd.DataFrame()
        
        torch.manual_seed(seed)
    
    def write(self, reward=0):
        total_time = time.time() - self.start_time
        df = pd.DataFrame({'reward' : [reward], 'total_time' : [total_time]})
        self.df = self.df.append(df)
        self.df.to_csv(self.dir_path + '/' + self.file_name + '.csv', index=False)
        
    def save(self, model, file_path="", file_name=""):
        torch.save(model, file_path + file_name + ".pth")
        
    def load(self, file_path="", file_name=""):
        model = torch.load(file_path + file_name + ".pth")
        model.eval()
        return model