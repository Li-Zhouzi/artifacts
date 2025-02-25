import matplotlib.pyplot as plt
import re
import pandas as pd
import os

def read_speedups(filename):
    # Dictionary to store results and base_goodput
    # Structure: {app_name: {job_id: {epoch: {'speedups': {replicas: speedup}, 'base_goodput': value}}}}
    results = {}
    
    with open(filename, 'r') as f:
        for line in f:
            # Parse lines like "speedup cifar10-15 41 14 5.043129464059592 10.0406335873742"
            # Format: speedup {app_name}-{job_id} {epoch} {replicas} {speedup} {base_goodput}
            match = re.search(r'speedup (\w+)-(\d+) (\d+) (\d+) ([\d.]+) ([\d.]+)', line)
            if match:
                app, job_id, epoch, replicas, speedup, base_goodput = match.groups()
                job_id = int(job_id)
                epoch = int(epoch)
                replicas = int(replicas)
                speedup = float(speedup)
                base_goodput = float(base_goodput)
                
                # Initialize nested dictionaries if they don't exist
                if app not in results:
                    results[app] = {}
                if job_id not in results[app]:
                    results[app][job_id] = {}
                if epoch not in results[app][job_id]:
                    results[app][job_id][epoch] = {'speedups': {}, 'base_goodput': base_goodput}
                
                results[app][job_id][epoch]['speedups'][replicas] = speedup
    
    return results

def read_values(filename):
    # Dictionary to store values
    # Structure: {app_name: {job_id: {epoch: {replicas: goodput, throughput, efficiency}}}}
    results = {}
    
    with open(filename, 'r') as f:
        for line in f:
            # Parse lines like "speedup deepspeech2-0 20 6 6.39404366208027 1.7547310390695834 6.697217671670815 0.9547313489789993"
            # Format: speedup {app_name}-{job_id} {epoch} {replicas} {goodput} {base_goodput} {throughput} {efficiency}
            match = re.search(r'speedup (\w+)-(\d+) (\d+) (\d+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)', line)
            if match:
                app, job_id, epoch, replicas, goodput, base_goodput, throughput, efficiency = match.groups()
                job_id = int(job_id)
                epoch = int(epoch)
                replicas = int(replicas)
                goodput = float(goodput)
                throughput = float(throughput)
                efficiency = float(efficiency)
                
                # Initialize nested dictionaries if they don't exist
                if app not in results:
                    results[app] = {}
                if job_id not in results[app]:
                    results[app][job_id] = {}
                if epoch not in results[app][job_id]:
                    results[app][job_id][epoch] = {}
                
                results[app][job_id][epoch][replicas] = (goodput, throughput, efficiency)
    
    return results

def read_csv_speedups(filename):
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
        speedup = float(row['speedup'])
        base_goodput = float(row['base_goodput'])
        
        # Initialize nested dictionaries if they don't exist
        if app not in results:
            results[app] = {}
        if job_id not in results[app]:
            results[app][job_id] = {}
        if epoch not in results[app][job_id]:
            results[app][job_id][epoch] = {'speedups': {}, 'base_goodput': base_goodput}
            
        results[app][job_id][epoch]['speedups'][replicas] = speedup
    
    return results

def plot_goodput_curves(app_name, epoch, results_test, results_csv, job_id=None):
    plt.figure(figsize=(8, 8))  # Make figure square
    
    # Plot test.txt results if available
    replicas_test = []
    goodput_values_test = []
    base_goodput_test = None
    if app_name in results_test:
        # Use provided job_id or get the maximum one
        job_id_test = job_id if job_id is not None else max(results_test[app_name].keys())
        if epoch in results_test[app_name][job_id_test]:
            data = results_test[app_name][job_id_test][epoch]
            speedups = data['speedups']
            base_goodput_test = data['base_goodput']
            replicas_test = sorted(speedups.keys())
            # Calculate goodput by multiplying speedup by base_goodput
            goodput_values_test = [speedups[r] * base_goodput_test for r in replicas_test]
            plt.plot(replicas_test, goodput_values_test, '-', linewidth=1,
                    label=f'Estimation\nBase goodput: {base_goodput_test:.2f}')
    
    # Plot CSV results if available (no job_id needed)
    replicas_csv = []
    goodput_values_csv = []
    base_goodput_csv = None
    if app_name in results_csv:
        if epoch in results_csv[app_name][0]:  # CSV data is always under job_id 1
            data = results_csv[app_name][0][epoch]
            speedups = data['speedups']
            base_goodput_csv = data['base_goodput']
            replicas_csv = sorted(speedups.keys())
            # Calculate goodput by multiplying speedup by base_goodput
            goodput_values_csv = [speedups[r] * base_goodput_csv for r in replicas_csv]
            plt.plot(replicas_csv, goodput_values_csv, '-', linewidth=1,
                    label=f'True goodput\nBase goodput: {base_goodput_csv:.2f}')
    
    # Find the maximum value for x-axis (replicas)
    max_replicas = max(
        max(replicas_test) if replicas_test else 0,
        max(replicas_csv) if replicas_csv else 0
    )
    
    # Find the maximum value for y-axis (goodput)
    max_goodput = max(
        max(goodput_values_test) if goodput_values_test else 0,
        max(goodput_values_csv) if goodput_values_csv else 0
    )
    
    # Set the axes limits
    plt.xlim(0, max_replicas)
    plt.ylim(0, max_goodput * 1.1)  # Add 10% margin on top
    
    # Plot the linear scaling line
    if base_goodput_test:
        plt.plot([0, max_replicas], [0, base_goodput_test * max_replicas], '--', 
                label='Linear scaling (ideal)')
    
    plt.xlabel('Number of Replicas')
    plt.ylabel('Goodput')
    plt.title(f'Goodput Curves for {app_name} (Job {job_id}, Epoch {epoch})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('./results/figs', exist_ok=True)
    
    # Get the actual job_id (use max if None) - only from test results
    actual_job_id = job_id if job_id is not None else max(results_test[app_name].keys()) if app_name in results_test else 1
    
    # Modified filename to use actual_job_id and indicate goodput
    plt.savefig(f'./results/figs/Goodput_{app_name}_job{actual_job_id}_epoch{epoch}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def convert_txt_to_csv_format(txt_filename, output_filename):
    # First read all data using existing read_speedups function
    results = read_speedups(txt_filename)
    
    # Open output file
    with open(output_filename, 'w') as f:
        # Write header
        f.write('name,epoch,replicas,speedup,base_goodput\n')
        
        # For each application
        for app_name in results:
            # Get the largest job_id for this application
            max_job_id = max(results[app_name].keys())
            
            # Get data for this job
            job_data = results[app_name][max_job_id]
            
            # For each epoch in this job
            for epoch in job_data:
                # Get speedups and base_goodput
                speedups = job_data[epoch]['speedups']
                base_goodput = job_data[epoch]['base_goodput']
                
                # Write each replica's data
                for replicas, speedup in speedups.items():
                    f.write(f'{app_name}-{max_job_id},{epoch},{replicas},{speedup},{base_goodput}\n')


def convert_values_to_csv_format(txt_filename, output_filename):
    # First read all data using existing read_speedups function
    results = read_values(txt_filename)
    
    # Open output file
    with open(output_filename, 'w') as f:
        # Write header
        f.write('name,epoch,replicas,goodput,throughput,efficiency\n')
        
        # For each application
        for app_name in results:
            # Get the largest job_id for this application
            max_job_id = max(results[app_name].keys())
            
            # Get data for this job
            job_data = results[app_name][max_job_id]
            
            # For each epoch in this job
            for epoch in job_data:
                # Get speedups and base_goodput
                values = job_data[epoch]
                
                # Write each replica's data
                for replicas, triple_t in values.items():
                    goodput = triple_t[0]
                    throughput = triple_t[1]
                    efficiency = triple_t[2]
                    f.write(f'{app_name}-{max_job_id},{epoch},{replicas},{goodput},{throughput},{efficiency}\n')


# Example usage
# results_test = read_speedups('./results/sia_test.txt')
# results_csv = read_csv_speedups('./profile_all2/cleaned_speedup_raw.csv')

# # # Plot for each application at a specific epoch
# applications = ['cifar10', 'deepspeech2', 'bert']
# # applications = ['cifar10']
# epoch_to_plot = [75, 75, 0]  # Change this to the epoch you want to plot
# # epoch_to_plot = [76]  # Change this to the epoch you want to plot
# job_id_to_plot = [None, None, None]  # Change this to the job_id you want to plot
# # job_id_to_plot = [10]  # Change this to the job_id you want to plot
# for i, app in enumerate(applications):
#     plot_goodput_curves(app, epoch_to_plot[i], results_test, results_csv, job_id_to_plot[i])

# Example usage:
# convert_txt_to_csv_format('./results/sia_test.txt', './profile_all/converted_test.csv')
convert_values_to_csv_format('./results/test_fix_localbsz.txt', './profile_fix_localbsz/converted_values.csv')
