import argparse
import csv
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import numpy as np
import os

MAX_REPLICAS = 64


def single_data(input_file, output_file):
    # Read the input CSV file
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Store header
        for row in reader:
            name, epoch, replicas, goodput, throughput, efficiency = row
            # name = name.split("-")[0] # only take the job type
            data[name][int(epoch)][int(replicas)].append((float(goodput), float(throughput), float(efficiency)))
    # Process the data and write to output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write header
        
        for name in sorted(data.keys()):
            for epoch in sorted(data[name].keys()):
                for replicas in range(1, MAX_REPLICAS+1):  # Assuming max 64 replicas
                    if replicas in data[name][epoch]:
                        values = data[name][epoch][replicas]
                        if len(values) == 1:
                            goodput, throughput, efficiency = values[0]
                        elif len(values) == 2:
                            goodput, throughput, efficiency = max(values, key=lambda x: x[0])[0]
                        else:
                            goodputs = [x[0] for x in values]
                            median_goodput = statistics.median(goodputs)
                            idx = goodputs.index(median_goodput)
                            goodput, throughput, efficiency = values[idx]
                        writer.writerow([name, epoch, replicas, goodput, throughput, efficiency])
                    else:
                        writer.writerow([name, epoch, replicas, 0.0, 0.0, 0.0])

def visualize_speedups(jobs, input_file):
    # Read the cleaned CSV file
    data = {}
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            name, epoch, replicas, speedup, _ = row
            key = (name, int(epoch))
            if key not in data:
                data[key] = []
            data[key].append(float(speedup))

    # Set up the plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(jobs)))

    # Plot speedup for each job
    for (name, epoch), color in zip(jobs, colors):
        key = (name, epoch)
        if key in data:
            speedups = data[key]
            replicas = range(1, len(speedups) + 1)
            plt.plot(replicas, speedups, label=f'{name} (epoch {epoch})', color=color)
        else:
            print(f"Warning: No data found for {name} (epoch {epoch})")

    # Add reference line for linear speedup
    max_replicas = max(len(data[key]) for key in data if key in jobs)
    # plt.plot([1, max_replicas], [1, max_replicas], 'k--', label='Linear Speedup')

    # Customize the plot
    plt.xlabel('Number of Replicas')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Replicas')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xscale('log', base=2)
    # plt.yscale('log', base=2)
    # plt.xlim(1, max_replicas)
    # plt.ylim(1, max_replicas)

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def list2file(d, output_file):
    # Write the increasing data to the output CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "epoch", "replicas", "goodput", "throughput", "efficiency"])
        writer.writerows(d)
        
def file2dict(input_file):
    # Read the input CSV file
    data = defaultdict(lambda: defaultdict(dict))
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name, epoch, replicas, speedup, base_goodput = row.values()
            data[name][int(epoch)][int(replicas)] = (float(speedup), float(base_goodput))
    return data

def reshape(data):
    # Process the data to make speedup increasing
    increasing_data = []
    for name in data:
        for epoch in data[name]:
            max_speedup = 0
            for replicas in range(1, MAX_REPLICAS+1):  # Assuming max 64 replicas
                if replicas in data[name][epoch]:
                    speedup, base_goodput = data[name][epoch][replicas]
                    # max_speedup = max(max_speedup, speedup)
                    increasing_data.append([name, epoch, replicas, speedup, base_goodput])
                else:
                    break  # Stop if we reach a missing replica count
    return increasing_data

def clean_data(input_file, dir="profiling"):
    directory, filename = os.path.split(input_file)
    new_filename = f"cleaned_{filename}"
    output_file = os.path.join(directory, new_filename)
    single_data(input_file, output_file)
    
    # d = file2dict(mid_file)
    # reshaped = reshape(d)
    # list2file(reshaped, output_file)
    # os.remove(mid_file)
    os.remove(input_file)

def load_goodput(goodput_path):
    with open(goodput_path, "r") as fd:
        df = csv.DictReader(fd)
        goodput_dict = dict()
        for row in df:
            if row["name"] not in goodput_dict:
                goodput_dict[row["name"]] = {}
            goodput_dict[row["name"]][int(row["epoch"])] = (float(row["goodput"]), float(row["throughput"]), float(row["efficiency"]))
    return goodput_dict

def load_speedup(speedup_dir):
    goodput_dicts = dict()
    for i in range(1, MAX_REPLICAS+1):
        path = os.path.join(speedup_dir, f"{i}gpu.csv")
        goodput_dicts[i] = load_goodput(path)

    speedup_dict = {}
    rows = []
    for num_gpus in range(1, MAX_REPLICAS+1):
        goodput_dict = goodput_dicts[num_gpus]
        for name in goodput_dict:
            if name not in speedup_dict:
                speedup_dict[name] = {}
            for epoch in goodput_dict[name]:
                if epoch not in speedup_dict[name]:
                    speedup_dict[name][epoch] = {}
                goodput, throughput, efficiency = goodput_dict[name][epoch]
                rows.append([name, epoch, num_gpus, goodput, throughput, efficiency])
    rows.sort() 
    return rows
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approximate", type=bool, default=False)
    parser.add_argument("--monotone", type=bool, default=True)
    parser.add_argument("speedup_path")
    args = parser.parse_args()
    input_file = args.speedup_path
    if not args.approximate:
        speedup = load_speedup(args.speedup_path)
        list2file(speedup, os.path.join(args.speedup_path, "speedup_raw.csv"))
        input_file = os.path.join(args.speedup_path, "speedup_raw.csv")
    clean_data(input_file)