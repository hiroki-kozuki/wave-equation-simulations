'''
PINN Training and Validation:

    Seeding: torch - initial network weights and biases (other internal randomness)
             np - random shuffling of training data indicies
    loads data sets electrogram postions

    tensor(1) setup and interpolation:
        - inputs: the inputs of the PINN - the electrogram activation times
        - interp_positions: array of 200 evenly spaced positions where model will predict activation times
              - converted from numpy array to PyTorch tensor of shape (200,)
              - unsqueeze(0): adds a batch dimension - results in shape (1,200)
              - .repeat(inputs.shape[0],1): repeats the interpolation positions tensor across all training samples - shape (num_samples,200)

    The PINN is defined as a class that inherits from the torch.nn.Module which encapsulates the architecture and the forward pass
          - PINN decisions: 4+1 input dimension - electrogram activation times + one of the 200 positions for interpolation **
                            Hidden layers -  4 fully-connected w/ ReLU activation functions
                            Output layer has single neuron as nn predicts single scalar value for the position
                            Forward method:
                                  - activation_times_expanded = activation_times.unsqueeze(1).expand(-1, x.shape[1], -1):
                                          - designed to align the activation_times tensor with the shape of x so that both can be concatenated later
                                          - unsqueeze(1) adds a new dimension at position 1 - activation times becomes shape (batch_size,1,4)
                                          - .expand(-1, x.shape[1], -1): expand the second dimension to match the number of interpolation points in x
                                          - essentially, each interpolation position needs to be associated with the sames set of activation times values for that batch
                                  - x_expanded = x.unsqueeze(-1):
                                          - prepares x for concatenation by adding new dimension
                                          - shape becomes (batch_size, num_points, 1)
                                  - input = torch.cat([activation_times_expanded, x_expanded], dim=2):
                                          - Combines activation_times_expanded and x_expanded into a single tensor that will be fed into the neural network.
                                  - PINN application:
                                          - The first layer transforms the input features into a higher-dimensional space, which allows the network to learn complex relationships
                                          - Then, applies the ReLU activation function to the output of the first layer to introduce non-linearity.
                                          - Subsequent layers progressively develop abstraction
                                          - Final layer reduces dimensionality to 1
                                          - The output is reshaped to (batch_size, num_points) for compatibility with downstream operations.
      Duplicated activation time calc
      Wraps inputs and interp_positions_tensor into a PyTorch dataset and splits it into batches for efficient training.
      Enables parallel data loading if using a GPU.

      Instantiation of model, optimiser and scaler:
            - model = PINN().to(device): creates instance of PINN class and transfers model to specified device which can be a CPU or GPU
            - optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5):
                  - This initializes the optimizer, which updates the model’s weights during training to minimize the loss.
                  - AdamW is a variant of the Adam optimizer that includes decoupled weight decay.
                  - Weight decay helps regularize the model and prevent overfitting by adding a penalty proportional to the size of the weights.
            - scaler = GradScaler(): part of PyTorch's automatic mixed precision toolkit
                  - It scales the gradients dynamically to prevent underflow or overflow during backpropagation.
                  - GradScaler keeps track of when to unscale the gradients and adjusts the scaling factor during training.

      Loss Function:
            - first predicts all activation times based on current model
            - records activation times for electrogram positions in new tensor
            - uses this to find data loss using nn.MSEloss()
            - Eikonal loss calculation:
                    - the activation times are recalculated with dynamic gradient calc; this isn't done from start as longer gradient tracking more expensive than just recalculating
                    - then computes the gradient of activation_times_pred with respect to x_interp for each sample in the batch.
                    - eikonal_residual = (((grad_activation_times * c) - 1.0)**2).mean():
                            - grad_activation_times * c: Scales the gradients by the speed.
                            - - 1.0: Penalizes deviations from the Eikonal equation.
                            - **2: Square of the residual ensures non-negative loss.

            - speed (c) values for each position need to assigned:
                    - Initialize containers to store segment-wise speed (c) values for each batch.
                    - Loop over the batch to process individual speed and change point tensors.
                    - Assign segment-specific speeds for each position in x_interp[i]:
                          - torch.searchsorted: Determines which segment (speeds_sample) each position belongs to based on cp_sample.
                    - append c_values to c_values_list
                    - then stack all batch speed values into a tensor c of shape (batch_size, num_points)

            - Transition loss calculation:
                    - Initializes the total transition loss for the current batch to 0.0.
                    - Iterates over each sample in the batch. The batch size determines how many samples are processed together during training.
                    - Initializes a list to store the "true" activation times at speed change points for the current sample.
                    - Extracts the speed change positions (cp_sample) for the current sample. This is a tensor containing the positions where the speed changes occur.
                    - Extracts the speed values (speeds_sample) for the current sample. These speeds correspond to the segments defined by the speed change points.
                    - Iterates through each speed change position in the current sample.
                    - Converts the speed change position (change_point) from a tensor to a Python float (xi) for easier processing.
                    - Computes the "true" activation time (true_time) at the speed change position.
                    - Appends the calculated "true" activation time at the speed change position to the true_values_near_points list.
                    - Passes the speed change positions (cp_sample) through the model to compute the predicted activation times at those positions.
                    - Computes the Mean Squared Error (MSE) between the predicted activation times (predicted_values) and the "true" activation times (true_values_tensor).

        Training Loop:

            - Iterates over a specified number of epochs (num_epochs), where one epoch represents a full pass over the entire dataset.
            - Randomly shuffles the data indices to ensure that each batch in the dataset is different across epochs. This introduces randomness into training, reducing overfitting.
            - Initializes accumulators for tracking the total, data, Eikonal, and transition losses over the epoch. These are reset at the start of each epoch.
            - Selects a subset of shuffled indices for the current batch.
            - Retrieves the inputs and associated ground truth values (e.g., activation times, speeds, and change positions) for the current batch by calling the get_batch_data function.
            - Clears gradients from the previous iteration to prevent accumulation, as PyTorch accumulates gradients by default.
            - Computes the total loss and its components using the loss_function
            - Computes gradients of the loss function with respect to the model parameters using backpropagation.
            - Updates the model parameters based on the computed gradients and the optimizer's learning rate.
            - Accumulates the losses for the current epoch. .item() extracts the scalar value from the PyTorch tensor.
            - Calculates the average loss over all batches in the epoch by dividing the accumulated loss by the total number of batches.

perhaps enforce gradient change around speed change point somehow


(1): tensors are the core data structure of PyTorch - somewhat analagous to NumPy arrays but come with additional features for deep learning and scientific computation:
              - can have arbitrary dimensions
              - automatically handles broadcasting - matches tensor dimensions in operations
              - easier to manipulate
              - automatic differentiation - automatically computes gradients during back propagation
              - can seemlessly transition between CPU and GPU
              - can represent batches of data for parallelised operations


'''
#!pip install sympy==1.10.1 --no-deps
#!pip install --upgrade sympy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json

# Create the 'pinnplot' and 'debug_logs' directories if they don't exist
if not os.path.exists('pinnplot'):
    os.makedirs('pinnplot')
if not os.path.exists('debug_logs'):
    os.makedirs('debug_logs')

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
    x_points = [start_pos, *change_positions, xi]
    t = 0.0

    for i in range(len(speeds)):
        x0 = x_points[i]
        x1 = min(x_points[i + 1], xi)
        if x1 <= x0:
            continue

        distance = x1 - x0
        time_contrib = distance / speeds[i]
        t += time_contrib

        if x1 == xi:
            break

    return t

def compute_activation_time_transition(start_pos, xi, speeds, change_positions,timestart):
    x_points = [start_pos, *change_positions]
    # print('x_points',x_points)
    # print('speeds',speeds)

    t = timestart

    for i in range(len(speeds)):
      if x_points[i] == xi:
        t += (x_points[i]-x_points[i-1])/speeds[i-1]


    return t

def loss_function(model, x_interp, activation_times, positions_known,
                  speeds_batch_tensors, change_positions_batch_tensors,
                  weight_eikonal, weight_transition):

    activation_times_pred = model(x_interp, activation_times)
    positions_known_tensor = torch.tensor(positions_known, dtype=torch.float32).unsqueeze(0)
    positions_known_tensor = positions_known_tensor.expand(activation_times.shape[0], -1)
    activation_times_pred_known = model(positions_known_tensor, activation_times)
    data_loss = nn.MSELoss()(activation_times_pred_known, activation_times)
    #print('data',activation_times,activation_times_pred_known,data_loss)
  # data tensor([[1.9024, 3.4446, 3.7411, 3.9756]]) tensor([[-0.1456, -0.1255, -0.1164, -0.1343]], grad_fn=<SqueezeBackward1>) tensor(12.1776, grad_fn=<MseLossBackward0>)

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

    for i in range(batch_size):
        speeds_sample = speeds_batch_tensors[i]
        cp_sample = change_positions_batch_tensors[i]

        if len(cp_sample) == 0:
            c_values = speeds_sample[0].expand(num_points)
        else:
            seg_idxs = torch.searchsorted(cp_sample, x_interp[i], right=False)
            c_values = speeds_sample[seg_idxs]
        c_values_list.append(c_values)

    c = torch.stack(c_values_list, dim=0)
    eikonal_residual = (((grad_activation_times * c) - 1.0)**2).mean()

    transition_loss = 0.0
    for i in range(batch_size):
        true_values_near_points = []
        cp_sample = change_positions_batch_tensors[i]
        speeds_sample = speeds_batch_tensors[i]
        timestart = activation_times[i].tolist()[0]
        for change_point in cp_sample:
            xi = change_point.item()

            true_time = compute_activation_time_transition(positions_known[0], xi, speeds_sample.tolist(), cp_sample.tolist(),timestart)
            true_values_near_points.append(true_time)
            # print(positions_known[0], xi, speeds_sample.tolist(), cp_sample.tolist(),timestart)
            # print(true_time)
            timestart = true_time
            # 2.0 3.462552309036255 [1.0, 6.746002674102783, 18.148475646972656] [3.462552309036255, 7.335063934326172] 1.90237295627594
            # 3.364925265312195
            # 2.0 7.335063934326172 [1.0, 6.746002674102783, 18.148475646972656] [3.462552309036255, 7.335063934326172] 3.444594383239746
            # 4.018639756829071


        true_values_tensor = torch.tensor(true_values_near_points, dtype=torch.float32)
        predicted_values = model(cp_sample.clone().detach().unsqueeze(0), activation_times[i:i+1])

        transition_loss += nn.MSELoss()(predicted_values, true_values_tensor.unsqueeze(0))

    transition_loss /= batch_size

    # Save debug information
    debug_info = {
        "data_loss": data_loss.item(),
        "eikonal_loss": eikonal_residual.item(),
        "transition_loss": transition_loss.item(),
    }
    with open(f'debug_logs/loss_debug_epoch_{epoch}.json', 'w') as f:
        json.dump(debug_info, f, indent=4)

    total_loss = data_loss + weight_eikonal * eikonal_residual + weight_transition * transition_loss
    return total_loss, data_loss.item(), eikonal_residual.item(), transition_loss.item()

model = PINN()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 2000
batch_size = 32
indices = np.arange(num_samples)

def get_batch_data(idx_list):
    activation_times_batch = inputs[idx_list]
    x_batch = interp_positions_tensor[idx_list]
    speeds_batch_tensors_batch = [speeds_tensors[i] for i in idx_list]
    change_positions_batch_tensors_batch = [change_positions_tensors[i] for i in idx_list]
    return activation_times_batch, x_batch, speeds_batch_tensors_batch, change_positions_batch_tensors_batch

for epoch in range(num_epochs):
    np.random.shuffle(indices)
    epoch_total_loss = 0.0
    epoch_data_loss = 0.0
    epoch_eikonal_loss = 0.0
    epoch_transition_loss = 0.0

    for i in range(0, num_samples, batch_size):
        idx_list = indices[i:i+batch_size]
        activation_times_batch, x_batch, speeds_batch_tensors_batch, change_positions_batch_tensors_batch = get_batch_data(idx_list)

        optimizer.zero_grad()
        total_loss, dloss, eloss, tloss = loss_function(model, x_batch, activation_times_batch,
                                                        positions_to_record,
                                                        speeds_batch_tensors_batch,
                                                        change_positions_batch_tensors_batch,
                                                        weight_eikonal=0.05, weight_transition=1.5)
        total_loss.backward()
        optimizer.step()

        epoch_total_loss += total_loss.item()
        epoch_data_loss += dloss
        epoch_eikonal_loss += eloss
        epoch_transition_loss += tloss

    num_batches = (num_samples + batch_size - 1) // batch_size
    avg_total_loss = epoch_total_loss / num_batches
    avg_data_loss = epoch_data_loss / num_batches
    avg_eikonal_loss = epoch_eikonal_loss / num_batches
    avg_transition_loss = epoch_transition_loss / num_batches

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Total Loss: {avg_total_loss:.6f}, Data Loss: {avg_data_loss:.6f}, "
              f"Eikonal Loss: {avg_eikonal_loss:.6f}, Transition Loss: {avg_transition_loss:.6f}")



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

        plt.figure(figsize=(10, 6))
        plt.plot(x_sample_np, activation_times_pred, label='Predicted Activation Times', linestyle='--')
        plt.plot(x_sample_np, true_activation_times, label='True Activation Times', linestyle='-')
        plt.scatter(positions_to_record, activation_times_sample.squeeze(0).numpy(), color='red', label='Input Activation Times')
        for idx_cp, pos in enumerate(change_positions_sample):
            plt.axvline(x=pos, color='grey', linestyle=':', label='Speed Change Point' if idx_cp == 0 else "")
            speed_change_activ = compute_activation_time(start_pos_sample, pos, speeds_sample, change_positions_sample)
            #print(speed_change_activ,'spchac')
            plt.plot(pos, compute_activation_time(start_pos_sample, pos, speeds_sample, change_positions_sample), 'x', color='black')
        plt.xlabel('Position')
        plt.ylabel('Activation Time')
        plt.title(f'Training Sample {sample_idx} - Activation Times with Variable Speeds')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'pinnplot/training_sample_{sample_idx}eik.png')
        plt.close()

sample_indices_to_plot = [0, 10, 20, 30, 40,50,60,70,80,90]
plot_training_samples(sample_indices_to_plot)

def visualize_distributions():
    first_segment_speeds = [s[0] for s in speeds_list]
    last_segment_speeds = [s[-1] for s in speeds_list]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(last_segment_speeds, bins=20, edgecolor='black')
    plt.title('Distribution of Last Segment Speeds')
    plt.xlabel('Speed')
    plt.ylabel('Frequency')

    if len(change_positions_list[0]) > 0:
        first_changes = [cp[0] for cp in change_positions_list]
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
new_num_changes = len(change_positions_list[0])
new_change_positions = np.linspace(3, 7, new_num_changes)

test_speeds = [1.0]
for _ in range(new_num_changes):
    factor = np.random.uniform(0.5, 2.0)
    test_speeds.append(test_speeds[-1] * factor)
test_speeds = np.array(test_speeds, dtype=float)

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
    speed_change_activ = compute_activation_time(new_start_pos, pos, test_speeds, new_change_positions)
    plt.plot(pos,speed_change_activ, 'x', color='black')
    #print(speed_change_activ,'spchac')
plt.xlabel('Position')
plt.ylabel('Activation Time')
plt.title('PINN Prediction on New Data with Variable Speed Changes')
plt.legend()
plt.grid(True)
plt.savefig('pinnplot/test_sample.png')
plt.close()