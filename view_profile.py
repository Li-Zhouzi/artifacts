import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file, skip the header row
df = pd.read_csv('traces/cifar10/placements-aws.csv', 
                 names=['placement', 'local_bsz', 'step_time', 'sync_time'],
                 skiprows=1)  # Skip the first row

# Convert local_bsz to integer
df['local_bsz'] = df['local_bsz'].astype(int)
df['step_time'] = df['step_time'].astype(float)
df['sync_time'] = df['sync_time'].astype(float)

# Parameters from epoch 2
params_epoch2 = {
    'alpha_c': 0.020427763111669755,
    'beta_c': 0.0006846836959649913,
    'alpha_n': 0.01456101437196375,
    'beta_n': 0.007176994242966395,
    'alpha_r': 0.0682931949541188,
    'beta_r': 0.0006950884967738235,
    'gamma': 1.2792475977722224
}

# Parameters from epoch 80
params_epoch80 = {
    'alpha_c': 0.020440073013824046,
    'beta_c': 0.0006841696458893314,
    'alpha_n': 0.01472544898256349,
    'beta_n': 0.007335994650384245,
    'alpha_r': 0.06896511904558286,
    'beta_r': 0.0007137767983382892,
    'gamma': 1.2805934713730076
}

def pred_compute_time(params, bsz):
    return params['alpha_c'] + params['beta_c'] * bsz

def pred_sync_time(params, num_nodes, num_replicas):
    if num_replicas == 1:
        return 0
    if num_nodes == 1:
        return params['alpha_r'] + params['beta_r'] * num_replicas
    else:
        return params['alpha_n'] + params['beta_n'] * num_replicas
    
def pred_step_time(params, num_nodes, num_replicas, bsz):
    compute_time = pred_compute_time(params, bsz)
    sync_time = pred_sync_time(params, num_nodes, num_replicas)
    return (compute_time ** params['gamma'] + sync_time ** params['gamma']) ** (1 / params['gamma'])


def plot_compute_vs_bsz():
    # Calculate compute time (step_time - sync_time)
    df['compute_time'] = df['step_time'] - df['sync_time']
    
    plt.figure(figsize=(10, 6))
    # Plot actual data
    plt.scatter(df['local_bsz'], df['compute_time'], label='Actual')
    
    # Plot predicted data
    bsz_range = np.arange(min(df['local_bsz']), max(df['local_bsz']) + 1)
    # Predictions for epoch 2
    pred_times_2 = [pred_compute_time(params_epoch2, bsz) for bsz in bsz_range]
    plt.plot(bsz_range, pred_times_2, 'r-', label='Predicted (Epoch 2)')
    # Predictions for epoch 80
    # pred_times_80 = [pred_compute_time(params_epoch80, bsz) for bsz in bsz_range]
    # plt.plot(bsz_range, pred_times_80, 'g-', label='Predicted (Epoch 80)')
    
    plt.xlabel('Local Batch Size')
    plt.ylabel('Compute Time (Step Time - Sync Time)')
    plt.title('Compute Time vs Local Batch Size')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sync_vs_replicas(bsz):
    # Filter rows where local_bsz matches the input bsz
    filtered_df = df[df['local_bsz'] == bsz].copy()
    
    # Calculate placement information
    filtered_df['num_nodes'] = filtered_df['placement'].astype(str).str.len()
    filtered_df['num_replicas'] = filtered_df['placement'].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()))
    
    # Create separate dataframes for single and multi node
    single_node_df = filtered_df[filtered_df['num_nodes'] == 1]
    multi_node_df = filtered_df[filtered_df['num_nodes'] > 1]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.scatter(single_node_df['num_replicas'], single_node_df['sync_time'], 
                color='red', label='Single Node (Actual)')
    plt.scatter(multi_node_df['num_replicas'], multi_node_df['sync_time'], 
                color='blue', label='Multi Node (Actual)')
    
    # Create ranges based on actual data points
    single_replicas_range = np.arange(min(single_node_df['num_replicas']), max(single_node_df['num_replicas']) + 1)
    multi_replicas_range = np.arange(min(multi_node_df['num_replicas']), max(multi_node_df['num_replicas']) + 1)
    
    # Plot predictions as lines
    # Epoch 2
    single_pred_2 = [pred_sync_time(params_epoch2, 1, r) for r in single_replicas_range]
    plt.plot(single_replicas_range, single_pred_2, 'r-', label='Single Node (Epoch 2)')
    
    multi_pred_2 = [pred_sync_time(params_epoch2, 2, r) for r in multi_replicas_range]
    plt.plot(multi_replicas_range, multi_pred_2, 'b-', label='Multi Node (Epoch 2)')
    
    # Epoch 80
    # single_pred_80 = [pred_sync_time(params_epoch80, 1, r) for r in single_replicas_range]
    # plt.plot(single_replicas_range, single_pred_80, 'r--', label='Single Node (Epoch 80)')
    
    # multi_pred_80 = [pred_sync_time(params_epoch80, 2, r) for r in multi_replicas_range]
    # plt.plot(multi_replicas_range, multi_pred_80, 'b--', label='Multi Node (Epoch 80)')
    
    plt.xlabel('Number of Replicas')
    plt.ylabel('Sync Time')
    plt.title(f'Sync Time vs Number of Replicas (local_bsz = {bsz})')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sync_vs_bsz_single_node():
    # Filter for single node data
    single_node_df = df[df['placement'].astype(str).str.len() == 1].copy()
    
    # Calculate num_replicas for each placement
    single_node_df['num_replicas'] = single_node_df['placement'].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()))
    
    # Get 5 random unique placements
    unique_placements = single_node_df['placement'].unique()
    selected_placements = np.random.choice(unique_placements, size=min(5, len(unique_placements)), replace=False)
    selected_placements = np.sort(selected_placements)  # Sort for consistent legend
    
    plt.figure(figsize=(10, 6))
    
    # Plot each placement with different color
    for placement in selected_placements:
        placement_df = single_node_df[single_node_df['placement'] == placement]
        num_replicas = placement_df['num_replicas'].iloc[0]
        plt.scatter(placement_df['local_bsz'], placement_df['sync_time'], 
                   label=f'{placement}')
        
        # Add prediction line
        bsz_range = np.arange(min(placement_df['local_bsz']), max(placement_df['local_bsz']) + 1)
        pred_times = [pred_sync_time(params_epoch2, 1, num_replicas) for _ in bsz_range]
        plt.plot(bsz_range, pred_times, '--')
    
    plt.xlabel('Local Batch Size')
    plt.ylabel('Sync Time')
    plt.title('Sync Time vs Batch Size (Single Node)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sync_vs_bsz_multi_node():
    # Filter for multi node data
    multi_node_df = df[df['placement'].astype(str).str.len() > 1].copy()
    
    # Calculate num_replicas for each placement
    multi_node_df['num_replicas'] = multi_node_df['placement'].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()))
    
    # Get 5 random unique placements
    unique_placements = multi_node_df['placement'].unique()
    selected_placements = np.random.choice(unique_placements, size=min(5, len(unique_placements)), replace=False)
    selected_placements = np.sort(selected_placements)  # Sort for consistent legend
    
    plt.figure(figsize=(10, 6))
    
    # Plot each placement with different color
    for placement in selected_placements:
        placement_df = multi_node_df[multi_node_df['placement'] == placement]
        num_replicas = placement_df['num_replicas'].iloc[0]
        num_nodes = len(str(placement))
        plt.scatter(placement_df['local_bsz'], placement_df['sync_time'], 
                   label=f'{placement}')
        
        # Add prediction line
        bsz_range = np.arange(min(placement_df['local_bsz']), max(placement_df['local_bsz']) + 1)
        pred_times = [pred_sync_time(params_epoch2, num_nodes, num_replicas) for _ in bsz_range]
        plt.plot(bsz_range, pred_times, '--')
    
    plt.xlabel('Local Batch Size')
    plt.ylabel('Sync Time')
    plt.title('Sync Time vs Batch Size (Multi Node)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_step_vs_bsz():
    # Calculate num_nodes and num_replicas for each placement
    df['num_nodes'] = df['placement'].astype(str).str.len()
    df['num_replicas'] = df['placement'].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()))
    
    # Get 5 random unique placements
    unique_placements = df['placement'].unique()
    selected_placements = np.random.choice(unique_placements, size=min(5, len(unique_placements)), replace=False)
    selected_placements = np.sort(selected_placements)  # Sort for consistent legend
    
    plt.figure(figsize=(10, 6))
    
    # Plot each placement with different color
    for placement in selected_placements:
        placement_df = df[df['placement'] == placement]
        num_replicas = placement_df['num_replicas'].iloc[0]
        num_nodes = placement_df['num_nodes'].iloc[0]
        
        # Plot actual data
        plt.scatter(placement_df['local_bsz'], placement_df['step_time'], 
                   label=f'{placement}')
        
        # Add prediction line
        bsz_range = np.arange(min(placement_df['local_bsz']), max(placement_df['local_bsz']) + 1)
        pred_times = [pred_step_time(params_epoch2, num_nodes, num_replicas, bsz) 
                     for bsz in bsz_range]
        plt.plot(bsz_range, pred_times, '--')
    
    plt.xlabel('Local Batch Size')
    plt.ylabel('Step Time')
    plt.title('Step Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_step_vs_replicas(bsz):
    # Filter rows where local_bsz matches the input bsz
    filtered_df = df[df['local_bsz'] == bsz].copy()
    
    # Calculate placement information
    filtered_df['num_nodes'] = filtered_df['placement'].astype(str).str.len()
    filtered_df['num_replicas'] = filtered_df['placement'].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()))
    
    # Create separate dataframes for single and multi node
    single_node_df = filtered_df[filtered_df['num_nodes'] == 1]
    multi_node_df = filtered_df[filtered_df['num_nodes'] > 1]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.scatter(single_node_df['num_replicas'], single_node_df['step_time'], 
                color='red', label='Single Node (Actual)')
    plt.scatter(multi_node_df['num_replicas'], multi_node_df['step_time'], 
                color='blue', label='Multi Node (Actual)')
    
    # Create ranges based on actual data points
    single_replicas_range = np.arange(min(single_node_df['num_replicas']), max(single_node_df['num_replicas']) + 1)
    multi_replicas_range = np.arange(min(multi_node_df['num_replicas']), max(multi_node_df['num_replicas']) + 1)
    
    # Plot predictions as lines
    # Single node predictions
    single_pred = [pred_step_time(params_epoch2, 1, r, bsz) for r in single_replicas_range]
    plt.plot(single_replicas_range, single_pred, 'r-', label='Single Node (Predicted)')
    
    # Multi node predictions
    multi_pred = [pred_step_time(params_epoch2, 2, r, bsz) for r in multi_replicas_range]
    plt.plot(multi_replicas_range, multi_pred, 'b-', label='Multi Node (Predicted)')
    
    plt.xlabel('Number of Replicas')
    plt.ylabel('Step Time')
    plt.title(f'Step Time vs Number of Replicas (local_bsz = {bsz})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call both plotting functions
# plot_compute_vs_bsz()
plot_sync_vs_replicas(64)
# plot_sync_vs_bsz_single_node()
# plot_sync_vs_bsz_multi_node()
# plot_step_vs_bsz()
plot_step_vs_replicas(64)