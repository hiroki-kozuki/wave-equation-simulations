import numpy as np
import matplotlib.pyplot as plt
import os

# For reproducibility
np.random.seed(0)

# Parameters
num_impulses = 500  # Number of training samples
start_positions = np.random.uniform(-1, 1, num_impulses)
positions_to_record = np.array([2, 4, 6, 8])

# Lists to store speeds and change positions for each impulse
speeds_list = []
change_positions_list = []

# Function to compute activation time considering speed changes
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

# Generate activation times data
activation_times_data = []

# Define the number of bins for stratification
num_bins = 10
bin_edges = np.linspace(2, 8, num_bins + 1)

for idx in range(num_impulses):
    while True:
        # Generate two change positions using stratified sampling
        bins = np.random.choice(range(num_bins), size=2, replace=False)
        change_positions = []
        for bin_index in bins:
            # Randomly select a position within the bin
            bin_start = bin_edges[bin_index]
            bin_end = bin_edges[bin_index + 1]
            pos = np.random.uniform(bin_start, bin_end)
            change_positions.append(pos)
        change_positions = np.sort(change_positions)
        # Ensure the two change positions are sufficiently apart
        min_separation = 1.5  # Increased from 0.5 to 1.5
        if abs(change_positions[1] - change_positions[0]) >= min_separation:
            break
    change_positions_list.append(change_positions)

    # Assign random speeds for each segment
    speeds = np.random.uniform(0.2, 3.0, 3)
    speeds_list.append(speeds)

    start_pos = start_positions[idx]

    # Compute activation times at positions_to_record
    activation_times = []
    for xi in positions_to_record:
        time = compute_activation_time(start_pos, xi, speeds, change_positions)
        activation_times.append(time)
    activation_times_data.append(activation_times)

# Save activation times data
activation_times_data = np.array(activation_times_data)
np.save('activation_times.npy', activation_times_data)

# Save speeds and change_positions for reference
np.save('speeds_list.npy', speeds_list)
np.save('change_positions_list.npy', change_positions_list)

# Save start_positions
np.save('start_positions.npy', start_positions)

# Optional: Visualize the distributions
def visualize_distributions():
    # Load change_positions_list
    change_positions_array = np.array(change_positions_list)

    # Plot histograms of the change positions
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(change_positions_array[:, 0], bins=bin_edges, edgecolor='black')
    plt.title('Distribution of First Change Position')
    plt.xlabel('Position')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(change_positions_array[:, 1], bins=bin_edges, edgecolor='black')
    plt.title('Distribution of Second Change Position')
    plt.xlabel('Position')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Load speeds_list
    speeds_array = np.array(speeds_list)

    # Plot histograms of the speeds
    plt.figure(figsize=(10, 4))

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(speeds_array[:, i], bins=20, range=(0.5, 2.0), edgecolor='black')
        plt.title(f'Distribution of Speed {i+1}')
        plt.xlabel('Speed')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Uncomment the line below to visualize the distributions
    # visualize_distributions()
    pass
