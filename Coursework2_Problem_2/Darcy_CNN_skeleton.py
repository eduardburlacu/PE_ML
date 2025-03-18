import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py
from models import UNet2D
from tqdm import tqdm

# Define Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # print('x.shape',x.shape)
        # print('y.shape',y.shape)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)
    
# Define normalizer, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.device = self.mean.device

    def encode(self, x):
        x = (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std.to(x.device) + self.eps)) + self.mean.to(x.device)
        return x

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

# Define network  

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, 1, kernel_size=3, padding=1)
        )


    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = out.squeeze(1)
        return out




if __name__ == '__main__':
    # Define the device to be used by student or grader
    # I have MPC accelerator, but most people use cuda
    use_unet = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    #torch.set_default_device(device)
    ############################# Data processing #############################
    # Read data from mat
    train_path = 'Darcy_2D_data_train.mat'
    test_path = 'Darcy_2D_data_test.mat'

    data_reader = MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Normalize data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = UnitGaussianNormalizer(u_train)

    print(a_train.shape)
    print(a_test.shape)
    print(u_train.shape)
    print(u_test.shape)

    # Create data loader
    batch_size = 128
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    ############################# Define and train network #############################
    # Create RNN instance, define loss function and optimizer
    in_channels = 1
    out_channels = 1
    num_layers = 3
    pool_size = 2
    channel_width = 32
    step_size = 20
    gamma = 0.94

    lr = 1e-2

    net = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=channel_width,
        num_layers=num_layers,
        pool_size=pool_size
    ) if use_unet else CNN()
    net.to(device)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train network
    epochs = 400 #200 # Number of epochs
    print("Start training CNN for {} epochs...".format(epochs))
    start_time = time()
    
    loss_train_list = []
    loss_test_list = []
    x = []
    u_normalizer.to(device)
    a_test = a_test.to(device)
    for epoch in tqdm(range(epochs)):
        net.train(True)
        trainloss = 0
        for i, data in enumerate(train_loader):
            input, target = data
            input, target = input.to(device), target.to(device)
            output = net(input) # Forward
            output = u_normalizer.decode(output)
            l = loss_func(output, target) # Calculate loss

            optimizer.zero_grad() # Clear gradients
            l.backward() # Backward
            optimizer.step() # Update parameters
            scheduler.step() # Update learning rate

            trainloss += l.item()    

        # Test
        net.eval()
        with torch.no_grad():

            test_output = net(a_test)
            test_output = u_normalizer.decode(test_output)
            testloss = loss_func(test_output, u_test.to(device)).item()

        # Print train loss every 10 epochs
        if epoch % 10 == 0:
            print("epoch:{}, train loss:{}, test loss:{}".format(epoch, trainloss/len(train_loader), testloss))

        loss_train_list.append(trainloss/len(train_loader))
        loss_test_list.append(testloss)
        x.append(epoch)

    total_time = time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Traing time: {}'.format(total_time_str))
    print("Train loss:{}".format(trainloss/len(train_loader)))
    print("Test loss:{}".format(testloss))
    
    ############################# Plot #############################
    plt.figure(1)
    plt.plot(x, loss_train_list, label='Train loss')
    plt.plot(x, loss_test_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.05)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Plotting the contour plots
    with torch.no_grad():
        a_test_device = a_test.to(device)
        test_output = net(a_test_device)
        test_output = u_normalizer.decode(test_output)

    # Select a sample from the test set
    sample_id = 0  # You can change this to plot different samples
    u_true_sample = u_test[sample_id].cpu().numpy()
    u_pred_sample = test_output[sample_id].cpu().numpy()

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmin = min(u_true_sample.min(), u_pred_sample.min())
    vmax = max(u_true_sample.max(), u_pred_sample.max())
    # Plot the true solution
    im1 = axes[0].contourf(u_true_sample, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title('True Solution')
    fig.colorbar(im1, ax=axes[0])

    # Plot the predicted solution
    im2 = axes[1].contourf(u_pred_sample.squeeze(0), cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted Solution')
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()
