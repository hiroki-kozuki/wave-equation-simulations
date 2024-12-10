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

# Load training data
activation_times_data = np.load('activation_times.npy')
positions_to_record = np.array([2.0, 4.0, 6.0, 8.0])

# Load speeds, change positions, and start positions
speeds_list = np.load('speeds_list.npy', allow_pickle=True)
change_positions_list = np.load('change_positions_list.npy', allow_pickle=True)
start_positions = np.load('start_positions.npy')

# Prepare training data
inputs = activation_times_data[:, :4]  # Four activation times at positions 2, 4, 6, 8
inputs = torch.tensor(inputs, dtype=torch.float32)  # Shape: [num_samples, 4]

# We will interpolate activation times at positions between 2 and 8
interp_positions = np.linspace(2, 8, 200)
interp_positions_tensor = torch.tensor(interp_positions, dtype=torch.float32).unsqueeze(0).repeat(inputs.shape[0], 1)  # Shape: [num_samples, num_points]

# Define the PINN with adjusted neurons per layer
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Reduced neurons per layer to 32
        self.fc1 = nn.Linear(4 + 1, 64)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.ReLU()
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.ReLU()
        self.fc9 = nn.Linear(64, 1)

    def forward(self, x, activation_times):
        # x: [batch_size, num_points]
        # activation_times: [batch_size, 4]
        activation_times_expanded = activation_times.unsqueeze(1).expand(-1, x.shape[1], -1)  # [batch_size, num_points, 4]
        x_expanded = x.unsqueeze(-1)  # [batch_size, num_points, 1]
        input = torch.cat([activation_times_expanded, x_expanded], dim=2)  # [batch_size, num_points, 5]
        out = self.fc1(input)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)
        out = self.fc9(out)
        return out.squeeze(-1)  # [batch_size, num_points]

# Loss function with data loss and eikonal residual
def loss_function(model, x_interp, activation_times, positions_known, speeds_batch, change_positions_batch, weight_eikonal=0.01):
    # Predictions at interpolated positions
    activation_times_pred = model(x_interp, activation_times)  # Shape: [batch_size, num_points]

    # Data loss at known positions (positions_known)
    positions_known_tensor = torch.tensor(positions_known, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 4]
    positions_known_tensor = positions_known_tensor.expand(activation_times.shape[0], -1)  # Shape: [batch_size, 4]

    activation_times_pred_known = model(positions_known_tensor, activation_times)  # Shape: [batch_size, 4]
    data_loss = nn.MSELoss()(activation_times_pred_known, activation_times)  # Scalar

    # Eikonal residual over interpolated positions
    x_interp.requires_grad_(True)
    activation_times_pred = model(x_interp, activation_times)
    grad_activation_times = torch.autograd.grad(
        outputs=activation_times_pred,
        inputs=x_interp,
        grad_outputs=torch.ones_like(activation_times_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # Shape: [batch_size, num_points]

    # Compute speed c(x) for each sample in the batch
    c_list = []
    for i in range(x_interp.shape[0]):  # batch_size
        x = x_interp[i]  # Shape: [num_points]
        speeds = speeds_batch[i]  # Shape: [3]
        change_positions = change_positions_batch[i]  # Shape: [2]

        # Define speed function for this sample
        c = torch.zeros_like(x)
        c = torch.where(x < change_positions[0], speeds[0], c)
        c = torch.where((x >= change_positions[0]) & (x < change_positions[1]), speeds[1], c)
        c = torch.where(x >= change_positions[1], speeds[2], c)
        c_list.append(c)

    c = torch.stack(c_list, dim=0)  # Shape: [batch_size, num_points]

    # Eikonal residual
    eikonal_residual = (((grad_activation_times * c) - 1.0) ** 2).mean()  # Scalar

    # Total loss: weighted sum of data loss and eikonal residual
    total_loss = data_loss*1.3 + weight_eikonal * eikonal_residual
    return total_loss, data_loss.item(), eikonal_residual.item()

# Initialize model and optimizer
model = PINN()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Using AdamW optimizer with weight decay

# Prepare speeds and change positions as tensors
speeds_tensor = torch.tensor(speeds_list.tolist(), dtype=torch.float32)  # Shape: [num_samples, 3]
change_positions_tensor = torch.tensor(change_positions_list.tolist(), dtype=torch.float32)  # Shape: [num_samples, 2]

# Training loop
num_epochs = 4000  # Reduced from 7000 to 2000
batch_size = 32  # Adjusted batch size
num_samples = inputs.shape[0]

for epoch in range(num_epochs):
    permutation = torch.randperm(num_samples)
    epoch_total_loss = 0.0
    epoch_data_loss = 0.0
    epoch_eikonal_loss = 0.0
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i+batch_size]
        activation_times_batch = inputs[indices]  # Shape: [batch_size, 4]
        x_batch = interp_positions_tensor[indices]  # Shape: [batch_size, num_points]
        speeds_batch = speeds_tensor[indices]  # Shape: [batch_size, 3]
        change_positions_batch = change_positions_tensor[indices]  # Shape: [batch_size, 2]

        optimizer.zero_grad()
        total_loss, data_loss_value, eikonal_loss_value = loss_function(
            model, x_batch, activation_times_batch, positions_to_record, speeds_batch, change_positions_batch, weight_eikonal=0.01
        )
        total_loss.backward()
        optimizer.step()

        # Sum losses over batches
        epoch_total_loss += total_loss.item()
        epoch_data_loss += data_loss_value
        epoch_eikonal_loss += eikonal_loss_value

    # Average losses over the number of batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    avg_total_loss = epoch_total_loss / num_batches
    avg_data_loss = epoch_data_loss / num_batches
    avg_eikonal_loss = epoch_eikonal_loss / num_batches

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Total Loss: {avg_total_loss:.6f}, Data Loss: {avg_data_loss:.6f}, Eikonal Loss: {avg_eikonal_loss:.6f}")

# Function to compute activation time
def compute_activation_time(start_pos, xi, speeds, change_positions):
    x_points = [start_pos, *change_positions, xi]
    t = 0.0
    for i in range(3):
        x0 = x_points[i]
        x1 = x_points[i+1]
        if x1 <= x0:
            continue  # Skip if x1 is not ahead of x0
        distance = x1 - x0
        t += distance / speeds[i]
    return t

# Plot the activation times from positions 2 to 8 for multiple samples
def plot_training_samples(sample_indices):
    for sample_idx in sample_indices:
        with torch.no_grad():
            activation_times_sample = inputs[sample_idx:sample_idx+1]  # Shape: [1, 4]
            x_sample = interp_positions_tensor[sample_idx:sample_idx+1]  # Shape: [1, num_points]
            activation_times_pred = model(x_sample, activation_times_sample).squeeze(0).numpy()
            x_sample_np = x_sample.squeeze(0).numpy()

            # Get the actual speeds and change positions for the sample
            speeds_sample = speeds_tensor[sample_idx].numpy()
            change_positions_sample = change_positions_tensor[sample_idx].numpy()

            # Get the starting position for the sample
            start_pos_sample = start_positions[sample_idx]

            # Compute true activation times across interp_positions
            true_activation_times = []
            for xi in x_sample_np:
                time = compute_activation_time(start_pos_sample, xi, speeds_sample, change_positions_sample)
                true_activation_times.append(time)
            true_activation_times = np.array(true_activation_times)

            # Print speeds and change positions
            print(f"Sample {sample_idx} - Speeds: {speeds_sample}, Change Positions: {change_positions_sample}")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x_sample_np, activation_times_pred, label='Predicted Activation Times', linestyle='--')
        plt.plot(x_sample_np, true_activation_times, label='True Activation Times', linestyle='-')
        plt.scatter(positions_to_record, activation_times_sample.squeeze(0).numpy(), color='red', label='Input Activation Times')
        # Plot vertical lines at speed change positions
        for idx_cp, pos in enumerate(change_positions_sample):
            plt.axvline(x=pos, color='grey', linestyle=':', label='Speed Change Point' if idx_cp == 0 else "")
            plt.plot(pos, compute_activation_time(start_pos_sample, pos, speeds_sample, change_positions_sample), 'x', color='black')
        plt.xlabel('Position')
        plt.ylabel('Activation Time')
        plt.title(f'Training Sample {sample_idx} - Activation Times with Variable Speeds')
        plt.legend()
        plt.grid(True)
        # Save the plot
        plt.savefig(f'pinnplot/training_sample_{sample_idx}64datfast.png')
        plt.close()

# Plot activation times for multiple samples
sample_indices_to_plot = [0, 10, 20, 30, 40]  # Adjust indices based on printed change positions
plot_training_samples(sample_indices_to_plot)

# Visualize distributions of speeds and change positions
def visualize_distributions():
    # Convert lists to arrays
    speeds_array = np.array(speeds_list)
    change_positions_array = np.array(change_positions_list)

    # Plot histograms of the speeds
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(speeds_array[:, i], bins=20, range=(0.5, 2.0), edgecolor='black')
        plt.title(f'Distribution of Speed {i+1}')
        plt.xlabel('Speed')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('pinnplot/speeds_distribution64datfast.png')
    plt.close()

    # Plot histograms of the change positions
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(change_positions_array[:, 0], bins=20, range=(2, 8), edgecolor='black')
    plt.title('Distribution of First Change Position')
    plt.xlabel('Position')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(change_positions_array[:, 1], bins=20, range=(2, 8), edgecolor='black')
    plt.title('Distribution of Second Change Position')
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('pinnplot/change_positions_distribution64datfast.png')
    plt.close()

# Call the function to visualize distributions
visualize_distributions()

# Testing with specific data
# Set new start position for the impulse
new_start_pos = 0.0  # Adjust as needed

# Set specific change positions at 3.3 and 6.5
new_change_positions = np.array([3.3, 6.5])
print(f"New change positions: {new_change_positions}")

# Assign significantly different speeds to make the changes noticeable
new_speeds = np.array([0.5, 2.0, 0.3])  # Adjusted to make speed changes more pronounced
print(f"New speeds: {new_speeds}")

# Compute activation times at known positions
positions_known = positions_to_record
new_activation_times = []
for xi in positions_known:
    time = compute_activation_time(new_start_pos, xi, new_speeds, new_change_positions)
    new_activation_times.append(time)
new_activation_times_tensor = torch.tensor(new_activation_times, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 4]

# Prepare x_sample for testing
x_sample_np = interp_positions  # Use the same interpolated positions
x_sample_test = torch.tensor(x_sample_np, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_points]

with torch.no_grad():
    activation_times_pred = model(x_sample_test, new_activation_times_tensor).squeeze(0).numpy()

    # Compute true activation times across interp_positions for testing
    true_activation_times_test = []
    for xi in x_sample_np:
        time = compute_activation_time(new_start_pos, xi, new_speeds, new_change_positions)
        true_activation_times_test.append(time)
    true_activation_times_test = np.array(true_activation_times_test)

# Plotting for testing data
plt.figure(figsize=(10, 6))
plt.plot(x_sample_np, activation_times_pred, label='Predicted Activation Times', linestyle='--')
plt.plot(x_sample_np, true_activation_times_test, label='True Activation Times', linestyle='-')
plt.scatter(positions_known, new_activation_times, color='green', label='New Input Activation Times')
# Plot vertical lines at speed change positions and add markers
for idx, pos in enumerate(new_change_positions):
    plt.axvline(x=pos, color='grey', linestyle=':', label='Speed Change Point' if idx == 0 else "")
    plt.plot(pos, compute_activation_time(new_start_pos, pos, new_speeds, new_change_positions), 'x', color='black')
plt.xlabel('Position')
plt.ylabel('Activation Time')
plt.title('PINN Prediction on New Data with Specific Variable Speeds')
plt.legend()
plt.grid(True)
# Save the plot
plt.savefig('pinnplot/test_sample64datfast.png')
plt.close()
