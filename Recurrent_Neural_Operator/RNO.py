import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import logging

import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#Define your data path
TRAIN_PATH = r'./viscodata_3mat.mat'

# Device configuration
to_cuda = False
to_mps = False
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    to_cuda = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    to_mps = True

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_mps=False, to_float=True, device=None):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_mps = to_mps
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()
        self.device = device

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)
            if self.device:
                x = x.to(device)

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_mps(self, to_mps):
        self.to_mps = to_mps

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class RNO(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_input, layer_hidden):
        super(RNO, self).__init__()

        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j + 1]))
            if j != len(layer_input) - 1:
                self.layers.append(nn.SELU())

        self.hidden_layers = nn.ModuleList()
        self.hidden_size   = hidden_size

        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j + 1]))
            if j != len(layer_hidden) - 1:
                self.hidden_layers.append(nn.SELU())

    def forward(self, input, output, hidden, dt):
        h0 = hidden
        h = torch.cat((output, hidden), 1)
        for _, m in enumerate(self.hidden_layers):
            h = m(h)

        h = h*dt + h0
        combined = torch.cat((output, (output-input)/dt, hidden), 1)
        x = combined
        for _, l in enumerate(self.layers):
            x = l(x)

        output = x.squeeze(1)
        hidden = h
        return output, hidden

    def initHidden(self,b_size):

        return torch.zeros(b_size, self.hidden_size, device=device)

# Normalize your data using the min-max normalizer
class DataNormalizer(object):
    """
    Input shape: (dataset_size, time_steps, 1).
    Normalize the strain/stress data to have range 0,1.
    """

    def __init__(self, data, epsilon=1e-4):
        self.epsilon = epsilon
        # Compute min and max across dataset_size and time_steps
        self.min = torch.amin(data, dim=(0, 1), keepdim=True)
        self.max = torch.amax(data, dim=(0, 1), keepdim=True)

        self.range = self.max - self.min
        # Prevent zero-range issues
        self.range = torch.where(self.range > self.epsilon, self.range, torch.tensor(1.0))

    @torch.no_grad()
    def normalize(self, data):
        return (data - self.min) / self.range

    @torch.no_grad()
    def denormalize(self, data):
        return data * self.range + self.min

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Set the seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.use_deterministic_algorithms(True)

    # define train and test data
    Ntotal = 400
    train_size = 320
    test_start = 320

    N_test = Ntotal - test_start

    # Read data from the .mat file
    F_FIELD = 'epsi_tol'

    SIG_FIELD = 'sigma_tol'

    # Define loss function
    loss_func = nn.MSELoss()
    ######### Preprocessing data ####################
    temp = torch.zeros(Ntotal, 1, device=device)

    data_loader = MatReader(TRAIN_PATH, to_cuda=to_cuda, to_mps=to_mps, device=device)
    data_input = data_loader.read_field(F_FIELD).contiguous().view(Ntotal, -1)
    data_output = data_loader.read_field(SIG_FIELD).contiguous().view(Ntotal, -1)

    # We down sample the data to a coarser grid in time. This is to help saving the training time
    s = cfg.s # initially was 4

    data_input = data_input[:, 0::s]
    data_output = data_output[:, 0::s]

    inputsize = data_input.size()[1]

    input_normalizer = DataNormalizer(data_input)
    output_normalizer = DataNormalizer(data_output)

    data_input = input_normalizer.normalize(data_input)
    data_output = output_normalizer.normalize(data_output)

    # define train and test data
    frac_validation = 0.10
    val_start = int((1 - frac_validation) * train_size)
    x_train = data_input[0:val_start, :]
    y_train = data_output[0:val_start, :]

    x_val = data_input[val_start:train_size, :]
    y_val = data_output[val_start:train_size, :]
    valsize = x_val.shape[0]
    # define the time increment dt in the RNO
    dt = 1.0 / (y_train.shape[1] - 1)

    x_test = data_input[test_start:Ntotal, :]
    y_test = data_output[test_start:Ntotal, :]
    testsize = x_test.shape[0]

    # Define number of hidden variables to use
    input_dim = 1
    output_dim = 1

    # Define RNO
    layer_input = [
        input_dim + output_dim + 2,
        100, 100, 100,
        output_dim
    ]

    layer_hidden = [output_dim + 2, 20, 20, 2]


    # Use command-line arguments
    n_hidden = cfg.hidden_size

    net = RNO(input_dim, n_hidden, output_dim,layer_input,layer_hidden)

    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")
    if device is not None:
        net.to(device)

    # Optimizer and learning drate scheduler
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # Number of training epochs
    epochs = cfg.epochs
    # Batch size
    b_size = cfg.batch_size

    # Wrap training data in loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=b_size,
        shuffle=True,
    )

    # Train neural net
    T = inputsize
    train_err = np.zeros((epochs,))
    val_err = np.zeros((epochs,))
    test_err = np.zeros((epochs,))
    y_val_approx = torch.zeros(valsize, inputsize, device=device)
    y_test_approx = torch.zeros(testsize, inputsize, device=device)

    for ep in tqdm(range(epochs), leave=True, position=0):
        scheduler.step()
        train_loss = 0.0
        test_loss  = 0.0
        for x, y in train_loader:
            hidden = net.initHidden(b_size)
            optimizer.zero_grad()
            y_approx = torch.zeros(b_size,T, device=device)
            y_true  = y
            y_approx[:,0] = y_true[:,0]
            for i in range(1,T):
                y_approx[:,i], hidden = net(
                    x[:,i].unsqueeze(1),
                    x[:,i-1].unsqueeze(1),
                    hidden,
                    dt
                )

            loss = loss_func(y_approx,y_true)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            hidden_val = net.initHidden(valsize)
            y_val_approx[:,0] = y_val[:,0]
            for j in range(1,T):
               y_val_approx[:, j], hidden_val = net(
                   x_val[:, j].unsqueeze(1),
                   x_val[:, j-1].unsqueeze(1),
                   hidden_val,
                   dt
               )
            val_loss = loss_func(y_val_approx,y_val).item()

            hidden_test = net.initHidden(testsize)
            y_test_approx[:,0] = y_test[:,0]
            for j in range(1,T):
               y_test_approx[:, j], hidden_test = net(
                   x_test[:, j].unsqueeze(1),
                   x_test[:, j-1].unsqueeze(1),
                   hidden_test,
                   dt
               )
            test_loss = loss_func(y_test_approx,y_test).item()

        train_err[ep] = train_loss/len(train_loader)
        val_err[ep] = val_loss
        test_err[ep]  = test_loss

        logging.info(f"Epoch:{ep+1}/{epochs} | Train:{train_err[ep]:.6f} | Val:{val_err[ep]:.6f} | Test:{test_err[ep]:.6f}")
    # Plot the training, validation and test losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_err, label='Train Loss')
    plt.plot(val_err, label='Validation Loss')
    plt.plot(test_err, label='Test Loss')
    plt.semilogy()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Losses')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    # Plot some examples from the test set

    # First, denormalize the predictions and the true values
    y_test_approx = output_normalizer.denormalize(y_test_approx)
    y_test = output_normalizer.denormalize(y_test)
    # Plot the first 5 examples in the same figure with more subfigures
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))
    for i in range(5):
        axs[i].plot(y_test[i].cpu().numpy(), label='True')
        axs[i].plot(y_test_approx[i].cpu().numpy(), label='Predicted')
        axs[i].set_title(f"Example {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()
    # Save the model
    torch.save(net.state_dict(), cfg.save_path)

if __name__ == "__main__":
    main()
