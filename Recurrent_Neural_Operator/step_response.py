import torch
import numpy as np
import matplotlib.pyplot as plt

from RNO import RNO, device, to_cuda, to_mps, TRAIN_PATH, MatReader, DataNormalizer

def logistic_curve(T=300, amplitude=0.10, midpoint=50.0, k=0.2):
    """
    Creates a smooth step [0 .. amplitude] using a logistic (sigmoid) curve.
    T: total points
    amplitude: final value
    midpoint: center for the transition
    k: steepness factor
    """
    t = np.arange(T)
    y = amplitude / (1.0 + np.exp(-k * (t - midpoint)))
    return y

def plot_step_response(model_path='outputs_s=4/model_s=4.pt'):
    # 1) Build normalizer from dataset
    data_loader = MatReader(TRAIN_PATH, to_cuda=to_cuda, to_mps=to_mps, device=device)
    raw_data = data_loader.read_field('epsi_tol').contiguous()  # shape [Ntotal, time_steps]
    s = 4
    raw_data = raw_data[:, 0::s]  # down-sample
    input_normalizer = DataNormalizer(raw_data)

    # 2) Create step input: T=300, step starts at t=20 with amplitude=0.10
    T = 300
    #start_idx = 30
    #ramp_steps = 30
    #step = np.zeros((1, T), dtype=np.float32)
    # Ramp from 0.0 to 0.10 over the specified steps
    #for i in range(ramp_steps):
    #    step[0, start_idx + i] = 0.10 * (i + 1) / ramp_steps

    # Keep strain at 0.10 for the rest of the time
    #step[0, start_idx + ramp_steps:] = 0.10

    step = logistic_curve(T=T, amplitude=0.10, midpoint=100, k=0.25)

    step_tensor = torch.from_numpy(step).float()
    if device:
        step_tensor = step_tensor.to(device)

    # 3) Normalize step input
    step_norm = input_normalizer.normalize(step_tensor)

    # 4) Load model and run the step input
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

    dt = 1.0 / (T - 1)
    hidden = net.initHidden(step_norm.size(0))
    sigma_pred = torch.zeros_like(step_norm)
    sigma_pred[:, 0] = step_norm[:, 0]

    for i in range(1, T):
        sigma_pred[:, i], hidden = net(
            step_norm[:, i].unsqueeze(1),
            step_norm[:, i-1].unsqueeze(1),
            hidden,
            dt
        )

    # 5) Denormalize predicted sigma
    # Use a separate normalizer if you had one for sigma, or reuse input_normalizer for illustration
    sigma_denorm = input_normalizer.denormalize(sigma_pred)

    # 6) Plot the step input epsilon(t) and predicted sigma(t)
    time_axis = np.linspace(0,1, T)
    epsilon_denorm = step_tensor.detach().cpu().numpy().squeeze()
    sigma_denorm_np = sigma_denorm.detach().cpu().numpy().squeeze()

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, epsilon_denorm, label='epsilon\\(t\\)')
    plt.ylabel('Strain')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, sigma_denorm_np, label='sigma\\(t\\)', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Stress')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_step_response()