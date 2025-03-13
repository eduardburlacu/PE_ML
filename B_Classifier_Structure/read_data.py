import torch
import numpy as np
import h5py


# This reads the matlab data from the .mat file provided
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_load_apply(self):
        strain = np.array(self.data['load_apply']).transpose(1,0)
        return torch.tensor(strain, dtype=torch.float32)

    def get_result(self):
        stress = np.array(self.data['result']).transpose(1,0)
        return torch.tensor(stress, dtype=torch.float32)

######################### Data processing #############################
# Read data from .mat file
path = 'Data/Eiffel_data.mat' #Define your data path here
data_reader = MatRead(path)
load = data_reader.get_load_apply()
result = data_reader.get_result()
pass