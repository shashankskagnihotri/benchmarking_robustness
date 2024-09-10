import matplotlib.pyplot as plt
import numpy as np
import os


def parse_log_file(filename):
    # To store utilization for all 4 GPUs
    gpu_data = {i: [] for i in range(4)}

    with open(filename, "r") as f:
        current_gpu = 0
        for line in f:
            line = line.strip()

            # Skip header lines and empty lines
            if line.startswith("utilization.gpu") or not line:
                current_gpu = 0  # Reset current GPU index after the header
                continue

            # Split the line into GPU utilization and memory utilization
            values = line.split(",")
            if len(values) == 2:
                try:
                    gpu_util = int(values[0].strip().replace("%", ""))
                    mem_util = int(values[1].strip().replace("%", ""))

                    # Store the data in the corresponding GPU's list
                    gpu_data[current_gpu].append((gpu_util, mem_util))

                    # Move to the next GPU (0 to 3)
                    current_gpu += 1
                except ValueError as e:
                    print(f"Error parsing line: {line} -> {e}")
                    continue
            else:
                print(f"Skipping malformed line: {line}")

    return gpu_data


def save_utilization_plots(gpu_data, log_filename):
    # Generate time points assuming 1 minute intervals
    num_measurements = len(
        gpu_data[0]
    )  # Assuming all GPUs have the same number of measurements
    time = np.arange(0, num_measurements)

    # Get the directory and base filename (without .log extension)
    base_filename = os.path.splitext(log_filename)[0]

    # ---- GPU Utilization Plot ----
    plt.figure(figsize=(10, 6))
    for i in range(4):
        gpu_util = [x[0] for x in gpu_data[i]]  # Extract GPU utilization for GPU i
        plt.plot(time, gpu_util, label=f"GPU {i+1} Utilization")

    plt.xlabel("Time (minutes)")
    plt.ylabel("GPU Utilization (%)")
    plt.title("GPU Utilization Over Time for 4 GPUs")
    plt.legend()
    plt.grid(True)

    # Save the GPU utilization plot
    gpu_plot_filename = f"{base_filename}_gpu.png"
    plt.savefig(gpu_plot_filename)
    print(f"GPU Utilization Plot saved as {gpu_plot_filename}")
    plt.close()

    # ---- Memory Utilization Plot ----
    plt.figure(figsize=(10, 6))
    for i in range(4):
        mem_util = [x[1] for x in gpu_data[i]]  # Extract Memory utilization for GPU i
        plt.plot(time, mem_util, label=f"GPU {i+1} Memory Utilization")

    plt.xlabel("Time (minutes)")
    plt.ylabel("Memory Utilization (%)")
    plt.title("Memory Utilization Over Time for 4 GPUs")
    plt.legend()
    plt.grid(True)

    # Save the Memory utilization plot
    mem_plot_filename = f"{base_filename}_memory.png"
    plt.savefig(mem_plot_filename)
    print(f"Memory Utilization Plot saved as {mem_plot_filename}")
    plt.close()


def calculate_averages(gpu_data):
    # Per-GPU averages
    gpu_averages = []
    for i in range(4):
        gpu_util = [x[0] for x in gpu_data[i]]  # GPU utilization only for GPU i
        mem_util = [x[1] for x in gpu_data[i]]  # Memory utilization only for GPU i
        avg_gpu = np.mean(gpu_util)
        avg_mem = np.mean(mem_util)
        gpu_averages.append((avg_gpu, avg_mem))

    # Overall node average (across all 4 GPUs at each time step)
    node_gpu_util = np.mean([[x[0] for x in gpu_data[i]] for i in range(4)], axis=0)
    node_mem_util = np.mean([[x[1] for x in gpu_data[i]] for i in range(4)], axis=0)

    avg_node_gpu = np.mean(node_gpu_util)
    avg_node_mem = np.mean(node_mem_util)

    return gpu_averages, avg_node_gpu, avg_node_mem


# Example usage:
filename = "slurm/train_results/gpu_usage_atss_cascade_rcnn_convnext_swin-b.log"

# Step 1: Parse the log file
gpu_data = parse_log_file(filename)

# Step 2: Save the GPU and Memory utilization plots
save_utilization_plots(gpu_data, filename)

# Step 3: Calculate and print averages
gpu_averages, avg_node_gpu, avg_node_mem = calculate_averages(gpu_data)

print("Per-GPU Averages (GPU %, Memory %):")
for i, (avg_gpu, avg_mem) in enumerate(gpu_averages):
    print(f"GPU {i+1}: {avg_gpu:.2f}%, {avg_mem:.2f}%")

print(f"\nOverall Node Average GPU Utilization: {avg_node_gpu:.2f}%")
print(f"Overall Node Average Memory Utilization: {avg_node_mem:.2f}%")
