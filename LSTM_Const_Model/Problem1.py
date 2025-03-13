import torch
import torch.nn as nn
import numpy as np
import h5py


# This reads the matlab data from the .mat file provided
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_strain(self):
        strain = np.array(self.data['strain']).transpose(2,0,1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2,0,1)
        return torch.tensor(stress, dtype=torch.float32)

######################### Data processing #############################
# Read data from .mat file
path = 'Data/Material_A.mat' #Define your data path here
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()
