from typing import List
import numpy as np
import matplotlib.pyplot as plt

def plot_results(results: List[dict], title: str, xlabel: str, ylabel: str, filename: str):
    plt.figure(figsize=(10, 6))
    
    for result in results:
        plt.plot(result['input_sizes'], result['values'], marker='o', label=result['label'])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_performance(performance_data: dict):
    results = []
    for name, data in performance_data.items():
        results.append({
            'label': name,
            'input_sizes': list(data.keys()),
            'values': [data[size]['throughput'] for size in data.keys()]
        })
    
    plot_results(results, 'Hash Function Performance', 'Input Size (bytes)', 'Throughput (MB/s)', 'performance_comparison.png')

def visualize_entropy(entropy_data: dict):
    results = []
    for name, data in entropy_data.items():
        results.append({
            'label': name,
            'input_sizes': list(data.keys()),
            'values': [data[size]['entropy_score'] for size in data.keys()]
        })
    
    plot_results(results, 'Entropy Analysis', 'Input Size (bytes)', 'Entropy Score', 'entropy_analysis.png')

def visualize_collision(collision_data: dict):
    results = []
    for name, data in collision_data.items():
        results.append({
            'label': name,
            'input_sizes': list(data.keys()),
            'values': [data[size]['collisions'] for size in data.keys()]
        })
    
    plot_results(results, 'Collision Resistance', 'Input Size (bytes)', 'Number of Collisions', 'collision_analysis.png')

def visualize_avalanche(avalanche_data: dict):
    results = []
    for name, data in avalanche_data.items():
        results.append({
            'label': name,
            'input_sizes': list(data.keys()),
            'values': [data[size]['average_bit_change_percentage'] for size in data.keys()]
        })
    
    plot_results(results, 'Avalanche Effect Analysis', 'Input Size (bytes)', 'Average Bit Change (%)', 'avalanche_analysis.png')