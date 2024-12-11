import numpy as np
import matplotlib.pyplot as plt
import os

# For reproducibility
np.random.seed(0)

# Parameters
num_impulses = 500  # Number of training samples
start_positions = np.random.uniform(-1, 1, num_impulses)
positions_to_record = np.array([2, 4, 6, 8])

# New parameter for variable number of speed changes
num_changes = 2  # You can change this to any number of speed changes
# This means there will be num_changes+1 speed segments

# Lists to store speeds and change positions for each impulse
speeds_list = []
change_positions_list = []

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
        print(f"Segment {i}: x0={x0}, x1={x1}, speed={speeds[i]}, time_contrib={time_contrib}")
        t += time_contrib

        # Break early if xi falls within the current segment
        if x1 == xi:
            break

    return t

activation_times_data = []

num_bins = 10
bin_edges = np.linspace(2, 8, num_bins + 1)
min_separation = 1.5

for idx in range(num_impulses):
    while True:
        bins = np.random.choice(range(num_bins), size=num_changes, replace=False)
        change_positions = []
        for bin_index in bins:
            bin_start = bin_edges[bin_index]
            bin_end = bin_edges[bin_index + 1]
            pos = np.random.uniform(bin_start, bin_end)
            change_positions.append(pos)
        change_positions = np.sort(change_positions)

        if num_changes > 1:
            if all((change_positions[i+1] - change_positions[i]) >= min_separation for i in range(num_changes-1)):
                break
        else:
            # If num_changes == 1 or 0, just break
            break

    # Speeds: first is 1.0, subsequent are multiples
    speeds = [1.0]
    for _ in range(num_changes):
        factor = np.random.uniform(0.5, 2.0)
        speeds.append(speeds[-1] * factor)
    speeds = np.array(speeds, dtype=float)

    speeds_list.append(speeds)
    change_positions = np.array(change_positions, dtype=float)
    change_positions_list.append(change_positions)

    start_pos = start_positions[idx]

    activation_times = []
    for xi in positions_to_record:
        time = compute_activation_time(start_pos, xi, speeds, change_positions)
        activation_times.append(time)
    activation_times_data.append(activation_times)

activation_times_data = np.array(activation_times_data, dtype=float)
speeds_list = np.array(speeds_list, dtype=object)  # variable lengths
change_positions_list = np.array(change_positions_list, dtype=object)

np.save('activation_times.npy', activation_times_data)
np.save('speeds_list.npy', speeds_list, allow_pickle=True)
np.save('change_positions_list.npy', change_positions_list, allow_pickle=True)
np.save('start_positions.npy', start_positions)
