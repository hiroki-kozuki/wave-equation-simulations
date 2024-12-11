import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Create the 'pinnplot' directory if it doesn't exist
if not os.path.exists('pinnplot'):
    os.makedirs('pinnplot')

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load data
activation_times_data = np.load('activation_times.npy')
positions_to_record = np.array([2.0, 4.0, 6.0, 8.0])
speeds_list = np.load('speeds_list.npy', allow_pickle=True)
change_positions_list = np.load('change_positions_list.npy', allow_pickle=True)
start_positions = np.load('start_positions.npy')

inputs = torch.tensor(activation_times_data[:, :4], dtype=torch.float32)

interp_positions = np.linspace(2, 8, 200)
interp_positions_tensor = torch.tensor(interp_positions, dtype=torch.float32).unsqueeze(0).repeat(inputs.shape[0], 1)

num_samples = inputs.shape[0]
sample_change_positions = change_positions_list[0]
num_changes = len(sample_change_positions)  # number of speed change points
# means num_changes+1 speed segments

# Pre-convert speeds and change_positions to tensors to avoid repeated conversions
speeds_tensors = []
change_positions_tensors = []
for s, cp in zip(speeds_list, change_positions_list):
    s_arr = np.array(s, dtype=float)
    cp_arr = np.array(cp, dtype=float) if len(cp) > 0 else np.array([], dtype=float)
    speeds_tensors.append(torch.tensor(s_arr, dtype=torch.float32))
    change_positions_tensors.append(torch.tensor(cp_arr, dtype=torch.float32))


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Network remains the same
        self.fc1 = nn.Linear(4 + 1, 32)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.ReLU()
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.ReLU()
        self.fc7 = nn.Linear(32, 32)
        self.fc8 = nn.ReLU()
        self.fc9 = nn.Linear(32, 1)

    def forward(self, x, activation_times):
        activation_times_expanded = activation_times.unsqueeze(1).expand(-1, x.shape[1], -1)
        x_expanded = x.unsqueeze(-1)
        input = torch.cat([activation_times_expanded, x_expanded], dim=2)
        out = self.fc1(input)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)
        out = self.fc9(out)
        return out.squeeze(-1)


def compute_activation_time(start_pos, xi, speeds, change_positions):
    # Combine start position, change positions, and xi into segment boundaries
    x_points = [start_pos, *change_positions, xi]
    t = 0.0

    # Process each segment
    for i in range(len(speeds)):
        x0 = x_points[i]
        x1 = min(x_points[i + 1], xi)  # Ensure we don't go beyond xi
        if x1 <= x0:
            continue  # Skip invalid or zero-length segments

        distance = x1 - x0
        time_contrib = distance / speeds[i]
        #print(f"Segment {i}: x0={x0}, x1={x1}, speed={speeds[i]}, time_contrib={time_contrib}")
        t += time_contrib

        # Break early if xi falls within the current segment
        if x1 == xi:
            break

    return t



def loss_function(model, x_interp, activation_times, positions_known,
                  speeds_batch_tensors, change_positions_batch_tensors, weight_eikonal=0.01):

    activation_times_pred = model(x_interp, activation_times)
    positions_known_tensor = torch.tensor(positions_known, dtype=torch.float32).unsqueeze(0)
    positions_known_tensor = positions_known_tensor.expand(activation_times.shape[0], -1)
    activation_times_pred_known = model(positions_known_tensor, activation_times)
    data_loss = nn.MSELoss()(activation_times_pred_known, activation_times)

    x_interp.requires_grad_(True)
    activation_times_pred = model(x_interp, activation_times)
    grad_activation_times = torch.autograd.grad(
        outputs=activation_times_pred,
        inputs=x_interp,
        grad_outputs=torch.ones_like(activation_times_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    batch_size, num_points = x_interp.shape[0], x_interp.shape[1]
    c_values_list = []

    # Vectorized approach to determine segments:
    # We'll use torch.searchsorted on each sample's change_positions to find which segment each x belongs to.
    # segments: 0 to num_changes
    # If x < cp[0], segment = 0
    # If cp[k-1] <= x < cp[k], segment = k
    # If x >= cp[num_changes-1], segment = num_changes
    # searchsorted with side='right' can handle this logic.

    for i in range(batch_size):
        speeds_sample = speeds_batch_tensors[i]
        cp_sample = change_positions_batch_tensors[i]

        # If no changes, there's only one segment
        if num_changes == 0:
            # All x have same c
            c_values = speeds_sample[0].expand(num_points)
        else:
            # Use searchsorted
            # searchsorted finds the insertion indices, for a given x, 
            # seg_idx = how many cp are < x
            seg_idxs = torch.searchsorted(cp_sample, x_interp[i], right=False)
            # seg_idxs now gives a segment index in [0, num_changes]
            # speeds_sample[seg_idx] gives speed for that segment
            c_values = speeds_sample[seg_idxs]
        c_values_list.append(c_values)

    c = torch.stack(c_values_list, dim=0)  # [batch_size, num_points]

    # Eikonal residual
    eikonal_residual = (((grad_activation_times * c) - 1.0)**2).mean()
    total_loss = data_loss + weight_eikonal * eikonal_residual
    return total_loss, data_loss.item(), eikonal_residual.item()


model = PINN()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

num_epochs = 2000
batch_size = 32
indices = np.arange(num_samples)

# No need for list_to_tensor_list now since we have pre-converted
def get_batch_data(idx_list):
    activation_times_batch = inputs[idx_list]
    x_batch = interp_positions_tensor[idx_list]
    # Just index into pre-converted tensors
    speeds_batch_tensors_batch = [speeds_tensors[i] for i in idx_list]
    change_positions_batch_tensors_batch = [change_positions_tensors[i] for i in idx_list]
    return activation_times_batch, x_batch, speeds_batch_tensors_batch, change_positions_batch_tensors_batch

for epoch in range(num_epochs):
    np.random.shuffle(indices)
    epoch_total_loss = 0.0
    epoch_data_loss = 0.0
    epoch_eikonal_loss = 0.0
    for i in range(0, num_samples, batch_size):
        idx_list = indices[i:i+batch_size]
        activation_times_batch, x_batch, speeds_batch_tensors_batch, change_positions_batch_tensors_batch = get_batch_data(idx_list)

        optimizer.zero_grad()
        total_loss, dloss, eloss = loss_function(model, x_batch, activation_times_batch,
                                                 positions_to_record,
                                                 speeds_batch_tensors_batch,
                                                 change_positions_batch_tensors_batch,
                                                 weight_eikonal=0.01)
        total_loss.backward()
        optimizer.step()

        epoch_total_loss += total_loss.item()
        epoch_data_loss += dloss
        epoch_eikonal_loss += eloss

    num_batches = (num_samples + batch_size - 1) // batch_size
    avg_total_loss = epoch_total_loss / num_batches
    avg_data_loss = epoch_data_loss / num_batches
    avg_eikonal_loss = epoch_eikonal_loss / num_batches

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Total Loss: {avg_total_loss:.6f}, Data Loss: {avg_data_loss:.6f}, Eikonal Loss: {avg_eikonal_loss:.6f}")

def plot_training_samples(sample_indices):
    for sample_idx in sample_indices:
        with torch.no_grad():
            activation_times_sample = inputs[sample_idx:sample_idx+1]
            x_sample = interp_positions_tensor[sample_idx:sample_idx+1]
            activation_times_pred = model(x_sample, activation_times_sample).squeeze(0).numpy()
            x_sample_np = x_sample.squeeze(0).numpy()

            speeds_sample = speeds_list[sample_idx]
            change_positions_sample = change_positions_list[sample_idx]

            start_pos_sample = start_positions[sample_idx]

            true_activation_times = []
            for xi in x_sample_np:
                time = compute_activation_time(start_pos_sample, xi, speeds_sample, change_positions_sample)
                true_activation_times.append(time)
            true_activation_times = np.array(true_activation_times)

            print(f"Sample {sample_idx} - Speeds: {speeds_sample}, Change Positions: {change_positions_sample}")

        plt.figure(figsize=(10, 6))
        plt.plot(x_sample_np, activation_times_pred, label='Predicted Activation Times', linestyle='--')
        plt.plot(x_sample_np, true_activation_times, label='True Activation Times', linestyle='-')
        plt.scatter(positions_to_record, activation_times_sample.squeeze(0).numpy(), color='red', label='Input Activation Times')
        for idx_cp, pos in enumerate(change_positions_sample):
            plt.axvline(x=pos, color='grey', linestyle=':', label='Speed Change Point' if idx_cp == 0 else "")
            plt.plot(pos, compute_activation_time(start_pos_sample, pos, speeds_sample, change_positions_sample), 'x', color='black')
        plt.xlabel('Position')
        plt.ylabel('Activation Time')
        plt.title(f'Training Sample {sample_idx} - Activation Times with Variable Speeds')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'pinnplot/training_sample_{sample_idx}.png')
        plt.close()

sample_indices_to_plot = [0, 10, 20, 30, 40]
plot_training_samples(sample_indices_to_plot)

def visualize_distributions():
    first_segment_speeds = [s[0] for s in speeds_list]
    last_segment_speeds = [s[-1] for s in speeds_list]

    if num_changes > 0:
        first_changes = [cp[0] for cp in change_positions_list]
    else:
        first_changes = []

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(last_segment_speeds, bins=20, edgecolor='black')
    plt.title('Distribution of Last Segment Speeds')
    plt.xlabel('Speed')
    plt.ylabel('Frequency')

    if num_changes > 0:
        plt.subplot(1, 2, 2)
        plt.hist(first_changes, bins=20, range=(2, 8), edgecolor='black')
        plt.title('Distribution of First Change Position')
        plt.xlabel('Position')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('pinnplot/variable_distribution.png')
    plt.close()

visualize_distributions()

# Testing with specific data
new_start_pos = 0.0
new_num_changes = num_changes
new_change_positions = np.linspace(3, 7, new_num_changes)
print(f"New change positions: {new_change_positions}")

test_speeds = [1.0]
for _ in range(new_num_changes):
    factor = np.random.uniform(0.5, 2.0)
    test_speeds.append(test_speeds[-1]*factor)
test_speeds = np.array(test_speeds, dtype=float)
print(f"New speeds: {test_speeds}")

positions_known = positions_to_record
new_activation_times = []
for xi in positions_known:
    time = compute_activation_time(new_start_pos, xi, test_speeds, new_change_positions)
    new_activation_times.append(time)
new_activation_times_tensor = torch.tensor(new_activation_times, dtype=torch.float32).unsqueeze(0)

x_sample_np = interp_positions
x_sample_test = torch.tensor(x_sample_np, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    activation_times_pred = model(x_sample_test, new_activation_times_tensor).squeeze(0).numpy()

true_activation_times_test = []
for xi in x_sample_np:
    time = compute_activation_time(new_start_pos, xi, test_speeds, new_change_positions)
    true_activation_times_test.append(time)
true_activation_times_test = np.array(true_activation_times_test)

plt.figure(figsize=(10, 6))
plt.plot(x_sample_np, activation_times_pred, label='Predicted Activation Times', linestyle='--')
plt.plot(x_sample_np, true_activation_times_test, label='True Activation Times', linestyle='-')
plt.scatter(positions_known, new_activation_times, color='green', label='New Input Activation Times')
for idx, pos in enumerate(new_change_positions):
    plt.axvline(x=pos, color='grey', linestyle=':', label='Speed Change Point' if idx == 0 else "")
    plt.plot(pos, compute_activation_time(new_start_pos, pos, test_speeds, new_change_positions), 'x', color='black')
plt.xlabel('Position')
plt.ylabel('Activation Time')
plt.title('PINN Prediction on New Data with Variable Speed Changes')
plt.legend()
plt.grid(True)
plt.savefig('pinnplot/test_sample.png')
plt.close()
