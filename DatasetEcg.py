import torch
import torch.nn as nn
import os
import numpy as np

class DatasetECG(torch.utils.data.Dataset):
    def __init__(self, path_dir:str):
        super().__init__()
        
        self.path_dir = path_dir
        
        self.dir_list = sorted(os.listdir(path_dir))        

        print(len(self.dir_list))

        
    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, idx):

        class_id = self.dir_list[idx][0] 

        if class_id == 'N':
            class_id = 0
            img_path = os.path.join(self.path_dir, self.dir_list[idx])

        elif class_id == 'A':
            class_id = 1
            img_path = os.path.join(self.path_dir, self.dir_list[idx])

        elif class_id == 'L':
            class_id = 2
            img_path = os.path.join(self.path_dir, self.dir_list[idx])
        
        elif class_id == 'R':
            class_id = 3
            img_path = os.path.join(self.path_dir, self.dir_list[idx])

        elif class_id == 'V':
            class_id = 4
            img_path = os.path.join(self.path_dir, self.dir_list[idx])
        
        img = np.load(img_path)

        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)
        
        return t_img, t_class_id