import matplotlib.pyplot as plt
import re
import pandas as pd
import os

def read_csv_values(filename):
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Dictionary to store results
    results = {}
    
    # Process each row
    for _, row in df.iterrows():
        app, job_id = row['name'].split('-')
        job_id = int(job_id)
        epoch = int(row['epoch'])
        replicas = int(row['replicas'])
        goodput = float(row['goodput'])
        throughput = float(row['throughput'])
        efficiency = float(row['efficiency'])
        
        # Initialize nested dictionaries if they don't exist
        if app not in results:
            results[app] = {}
        if job_id not in results[app]:
            results[app][job_id] = {}
        if epoch not in results[app][job_id]:
            results[app][job_id][epoch] = {}
            
        results[app][job_id][epoch][replicas] = (goodput, throughput, efficiency)
    
    return results

def plot_goodput_curves(app_name, epoch, path1, path2, entry=0):
    # epoch = "all": average over all epochs
    #path1: true values
    #entry=0: goodput
    #entry=1: throughput
    #entry=2: efficiency
    #entry=3: speedup
    plt.figure(figsize=(8, 8))  # Make figure square
    
    # Define the replicas we want to plot
    plotting_replicas = [1, 2, 4, 8, 12, 16, 20, 32, 64]

    # values[app][job_id][epoch][replicas] = (goodput, throughput, efficiency)
    values1 = read_csv_values(path1)
    replicas_1 = []
    plot_values_1 = []

    if app_name in values1:
        if epoch == "all":
            mean1 = {}
            for epoch_t in values1[app_name][0].keys():
                data = values1[app_name][0][epoch_t]
                # Filter replicas based on plotting_replicas
                replicas_1 = sorted([r for r in data.keys() if r in plotting_replicas])
                if entry < 3:
                    for r in replicas_1:
                        if r not in mean1:
                            mean1[r] = data[r][entry]
                        else:
                            mean1[r] += data[r][entry]
                elif entry == 3:
                    for r in replicas_1:
                        if r not in mean1:
                            mean1[r] = data[r][0] / data[1][0]
                        else: 
                            mean1[r] += data[r][0] / data[1][0]
                else:
                    raise ValueError(f'Invalid entry: {entry}')
            for r in mean1.keys():
                mean1[r] = mean1[r] / len(values1[app_name][0])
            replicas_1 = sorted(mean1.keys())
            plot_values_1 = [mean1[r] for r in replicas_1]
            plt.plot(replicas_1, plot_values_1, '-', linewidth=1,
                    label=f'True average values')
        else:
            if epoch in values1[app_name][0]:
                data = values1[app_name][0][epoch]
                # Filter replicas based on plotting_replicas
                replicas_1 = sorted([r for r in data.keys() if r in plotting_replicas])
                if entry < 3:
                    plot_values_1 = [1 / data[r][entry] if data[r][entry] != 0 else 0 for r in replicas_1] # remember to take away the 1/x
                elif entry == 3:
                    plot_values_1 = [data[r][0] / data[1][0] for r in replicas_1]
                else:
                    raise ValueError(f'Invalid entry: {entry}')

                plt.plot(replicas_1, plot_values_1, '-', linewidth=1,
                        label=f'True values')
    
    values2 = read_csv_values(path2)
    replicas_2 = []
    plot_values_2 = []

    if app_name in values2:
        if epoch == "all":
            mean2 = {}
            for epoch_t in values2[app_name][0].keys():
                data = values2[app_name][0][epoch_t]
                # Filter replicas based on plotting_replicas
                replicas_2 = sorted([r for r in data.keys() if r in plotting_replicas])
                if entry < 3:
                    for r in replicas_2:
                        if r not in mean2:
                            mean2[r] = data[r][entry]
                        else:
                            mean2[r] += data[r][entry]
                elif entry == 3:
                    for r in replicas_2:
                        if r not in mean2:
                            mean2[r] = data[r][0] / data[1][0]
                        else: 
                            mean2[r] += data[r][0] / data[1][0]
                else:
                    raise ValueError(f'Invalid entry: {entry}')
            for r in mean2.keys():
                mean2[r] = mean2[r] / len(values2[app_name][0])
            replicas_2 = sorted(mean2.keys())
            plot_values_2 = [mean2[r] for r in replicas_2]
            plt.plot(replicas_2, plot_values_2, '-', linewidth=1,
                    label=f'Estimated average values')
        else:
            if epoch in values2[app_name][0]:
                data = values2[app_name][0][epoch]
                # Filter replicas based on plotting_replicas
                replicas_2 = sorted([r for r in data.keys() if r in plotting_replicas])
                if entry < 3:
                    plot_values_2 = [1 / data[r][entry] for r in replicas_2] # remember to take away the 1/x
                elif entry == 3:
                    plot_values_2 = [data[r][0] / data[1][0] for r in replicas_2]
                else:
                    raise ValueError(f'Invalid entry: {entry}')

            plt.plot(replicas_2, plot_values_2, '-', linewidth=1,
                    label=f'Estimated values')

    # Set x-axis to log scale
    plt.xscale('log', base=2)
    
    # Customize x-axis ticks to show powers of 2
    plt.xticks(plotting_replicas, [str(x) for x in plotting_replicas])

    # Find the maximum value for x-axis (replicas)
    max_replicas = max(
        max(replicas_1) if replicas_1 else 0,
        max(replicas_2) if replicas_2 else 0
    )
    
    # Find the maximum value for y-axis 
    max_value = max(
        max(plot_values_1) if plot_values_1 else 0,
        max(plot_values_2) if plot_values_2 else 0
    )
    
    # Set the axes limits
    # plt.xlim(0, max_replicas)
    # plt.ylim(0, max_value * 1.1)  # Add 10% margin on top
    
    # Plot the linear scaling line
    if entry == 3:
        plt.plot([0, max_replicas], [0, max_replicas], '--', 
                    label='Linear scaling (ideal)')
    
    plt.xlabel('Number of Replicas')
    if entry == 0:
        label = 'Goodput'
    elif entry == 1:
        label = 'Throughput'
    elif entry == 2:
        label = 'Efficiency'
    elif entry == 3:
        label = 'Speedup'
    plt.ylabel(label)
    plt.title(f'{app_name} (Epoch {epoch})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('./results/figs3', exist_ok=True)
    
    # Modified filename to use and indicate goodput
    plt.savefig(f'./results/figs3/logx_{label}_{app_name}_epoch{epoch}.png', 
                bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


true_values_path = './profile_fix_localbsz/cleaned_speedup_raw.csv'
estimated_values_path = './profile_fix_localbsz/converted_values.csv'

# for i in range(0, 4):
#     plot_goodput_curves('cifar', 50 , true_values_path, estimated_values_path, i)


plot_goodput_curves('cifar10', 20 , true_values_path, estimated_values_path, 1)
