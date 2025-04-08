import torch
import matplotlib.pyplot as plt
import numpy as np

from RNO import RNO, device, to_cuda, to_mps, TRAIN_PATH, MatReader, DataNormalizer

def plot_rno_network_actual_hidden(model_path, sample_input, t=50):
    # Load model
    input_dim = 1
    output_dim = 1
    hidden_size = 2
    layer_input = [input_dim + output_dim + hidden_size, 100, 100, 100, output_dim]
    layer_hidden = [output_dim + hidden_size, 20, 20, hidden_size]

    net = RNO(input_dim, hidden_size, output_dim, layer_input, layer_hidden)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    if device:
        net.to(device)

    # Initialize hidden and pass data until time t
    x0 = sample_input[:, 0].unsqueeze(1)
    hidden = net.initHidden(sample_input.size(0))
    dt = 1.0 / (sample_input.size(1) - 1)  # or match how dt is set in your script
    for i in range(1, t + 1):
        # previous output is sample_input[:, i-1], current input is sample_input[:, i]
        x_current = sample_input[:, i].unsqueeze(1)
        x_previous = sample_input[:, i-1].unsqueeze(1)
        _, hidden = net(x_current, x_previous, hidden, dt)

    # Now replicate hidden-layers logic at time t
    # Combine output=x_t with hidden
    h_in = torch.cat((x0, hidden), dim=1)
    hidden_weights, hidden_activations = [], []
    for layer in net.hidden_layers:
        if isinstance(layer, torch.nn.Linear):
            hidden_weights.append(layer.weight.detach().cpu().numpy())
        h_in = layer(h_in)
        if isinstance(layer, torch.nn.Linear):
            hidden_activations.append(h_in.detach().cpu().numpy())

    # Now replicate forward-layers logic at time t
    forward_input = torch.cat((x0, (x0 - x0).detach(), hidden), dim=1)
    forward_weights, forward_activations = [], []
    for layer in net.layers:
        if isinstance(layer, torch.nn.Linear):
            forward_weights.append(layer.weight.detach().cpu().numpy())
        forward_input = layer(forward_input)
        if isinstance(layer, torch.nn.Linear):
            forward_activations.append(forward_input.detach().cpu().numpy())

    # Plot all hidden-layer weights/activations separately to avoid shape mismatch
    fig_hidden, axs_hidden = plt.subplots(len(hidden_weights), 2, figsize=(10, 10))
    for i, (w, act) in enumerate(zip(hidden_weights, hidden_activations)):
        im_w = axs_hidden[i, 0].imshow(w, aspect='auto', cmap='viridis')
        axs_hidden[i, 0].set_title(f'Hidden Layer {i} Weights')
        axs_hidden[i, 0].set_yticks(np.arange(w.shape[0]))
        axs_hidden[i, 0].set_xticks(np.arange(w.shape[1]))
        fig_hidden.colorbar(im_w, ax=axs_hidden[i, 0], fraction=0.046, pad=0.04)

        im_a = axs_hidden[i, 1].imshow(act, aspect='auto', cmap='plasma')
        axs_hidden[i, 1].set_title(f'Hidden Layer {i} Activations')
        axs_hidden[i, 1].set_yticks(np.arange(act.shape[0]))
        axs_hidden[i, 1].set_xticks(np.arange(act.shape[1]))
        fig_hidden.colorbar(im_a, ax=axs_hidden[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Plot all forward-layer weights/activations separately
    fig_forward, axs_forward = plt.subplots(len(forward_weights), 2, figsize=(10, 10))
    for i, (w, act) in enumerate(zip(forward_weights, forward_activations)):
        im_w = axs_forward[i, 0].imshow(w, aspect='auto', cmap='viridis')
        axs_forward[i, 0].set_title(f'Forward Layer {i} Weights')
        axs_forward[i, 0].set_yticks(np.arange(0, w.shape[0], 10))
        axs_forward[i, 0].set_xticks(np.arange(0, w.shape[1], 10 if i> 0 else 1))
        fig_forward.colorbar(im_w, ax=axs_forward[i, 0], fraction=0.046, pad=0.04)

        im_a = axs_forward[i, 1].imshow(act, aspect='auto', cmap='plasma')
        axs_forward[i, 1].set_title(f'Forward Layer {i} Activations')
        axs_forward[i, 1].set_yticks(np.arange(0, act.shape[0], 1))
        axs_forward[i, 1].set_xticks(np.arange(0, act.shape[1], 10))
        fig_forward.colorbar(im_a, ax=axs_forward[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    data_loader = MatReader(TRAIN_PATH, to_cuda=to_cuda, to_mps=to_mps, device=device)
    Ntotal = 400
    test_start = 320
    s = 4
    data_input = data_loader.read_field('epsi_tol').contiguous().view(Ntotal, -1)
    data_input = data_input[:, 0::s]
    input_normalizer = DataNormalizer(data_input)
    data_input = input_normalizer.normalize(data_input)
    x_test = data_input[test_start:Ntotal, :]
    sample = x_test[0].unsqueeze(0)
    plot_rno_network_actual_hidden('outputs_s=4/model_s=4.pt', sample, t=249)