def parse_step_times(file_path):
    # Dictionary to store the results
    # Structure: {job_name: {epoch: {num_replicas: (estimate, true)}}}
    results = {}
    
    # Read the file and process line by line
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('estimated step time and true step time'):
                # Split the line into components
                # Format: "estimated step time and true step time cifar10-0 80 8 0.1450263771681528 0.23039008378982537"
                parts = line.strip().split()
                
                # Extract values (adjusted indices)
                job_name = parts[7]  # cifar10-0
                epoch = int(parts[8])  # 80
                num_replicas = int(parts[9])  # 8
                estimate = float(parts[10])  # 0.1450263771681528
                true_value = float(parts[11])  # 0.23039008378982537
                
                # Create nested dictionary structure if it doesn't exist
                if job_name not in results:
                    results[job_name] = {}
                if epoch not in results[job_name]:
                    results[job_name][epoch] = {}
                
                # Store the values
                results[job_name][epoch][num_replicas] = (estimate, true_value)
    
    return results

def plot_step_times(results):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure with subplots for each job and epoch combination
    job_epochs = [(job, epoch) for job in results for epoch in results[job]]
    n_plots = len(job_epochs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5*n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, (job_name, epoch) in zip(axes, job_epochs):
        # Prepare data for plotting
        epoch_data = results[job_name][epoch]
        replicas = []
        estimates = []
        true_values = []

        for num_replica, (estimate, true_value) in epoch_data.items():
            replicas.append(num_replica)
            estimates.append(estimate)
            true_values.append(true_value)

        # Create scatter plot
        ax.scatter(replicas, estimates, label='Estimated', alpha=0.6, marker='o')
        ax.scatter(replicas, true_values, label='True', alpha=0.6, marker='x')

        # Set x-axis to log scale and customize ticks
        ax.set_xscale('log')
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.set_xticklabels(['4', '8', '16', '32', '64'])

        # Customize plot
        ax.set_title(f'{job_name} - Epoch {epoch}')
        ax.set_xlabel('Number of Replicas')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.savefig('step_times_comparison.png')
    plt.show()
    plt.close()

# Only run this code when the script is executed directly
if __name__ == "__main__":
    # Example usage:
    file_path = 'results/test_cifar2.txt'
    step_times = parse_step_times(file_path)
    plot_step_times(step_times)
