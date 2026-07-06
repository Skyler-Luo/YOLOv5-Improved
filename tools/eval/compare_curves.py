import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(csv_paths, labels, metric_column, output_path):
    plt.figure(figsize=(10, 8))
    
    for csv_path, label in zip(csv_paths, labels):
        if not os.path.exists(csv_path):
            print(f"Warning: file {csv_path} does not exist. Skipping.")
            continue
        try:
            df = pd.read_csv(csv_path)
            # Strip column names because YOLOv5 csvs often have leading spaces
            df.columns = [c.strip() for c in df.columns]
            
            if metric_column in df.columns:
                epochs = np.arange(len(df))
                plt.plot(epochs, df[metric_column], label=label, linewidth=2)
            else:
                available_cols = list(df.columns)
                # Try finding matching column without strict prefix
                matched = False
                for col in df.columns:
                    if metric_column in col or col in metric_column:
                        plt.plot(epochs, df[col], label=label, linewidth=2)
                        matched = True
                        break
                if not matched:
                    print(f"Warning: Column {metric_column} not found in {csv_path}. Available: {available_cols}")
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric_column, fontsize=14)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'YOLO Comparison: {metric_column}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Curve comparison plot saved to: {output_path}")

def plot_scatter(scatter_string, output_path):
    # Format of scatter_string: "yolov5n:0.892:1.8,yolov5n-light:0.885:1.4"
    try:
        models = []
        accuracies = []
        latencies = []
        markers = ['o', 's', '^', 'D', 'p', '*', 'X', 'P']
        
        for item in scatter_string.split(','):
            parts = item.split(':')
            if len(parts) == 3:
                name, acc, lat = parts
                models.append(name)
                accuracies.append(float(acc))
                latencies.append(float(lat))
                
        if not models:
            print("No scatter data parsed. Please use format: 'Name:Accuracy:Latency'")
            return
            
        plt.figure(figsize=(10, 8))
        for i, (name, acc, lat) in enumerate(zip(models, accuracies, latencies)):
            marker = markers[i % len(markers)]
            plt.scatter(lat, acc, label=name, marker=marker, s=300)
            # Add label annotation near points
            plt.annotate(name, (lat, acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11)
            
        plt.xlabel('Inference Latency (ms)', fontsize=14)
        plt.ylabel('mAP / Accuracy', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Latency vs. Accuracy Comparison', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Scatter plot saved to: {output_path}")
    except Exception as e:
        print(f"Failed to plot scatter: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvs', type=str, default='', help='comma-separated results.csv paths')
    parser.add_argument('--labels', type=str, default='', help='comma-separated labels for each curve')
    parser.add_argument('--metric', type=str, default='metrics/mAP_0.5', help='metric column to plot')
    parser.add_argument('--output', type=str, default='runs/mAP50_curve.png', help='output path for curves')
    parser.add_argument('--scatter-data', type=str, default='', help='format: name:acc:lat,name:acc:lat')
    parser.add_argument('--scatter-output', type=str, default='runs/latency_vs_accuracy.png', help='output path for scatter plot')
    
    args = parser.parse_args()
    
    if args.csvs:
        csv_paths = [x.strip() for x in args.csvs.split(',')]
        if args.labels:
            labels = [x.strip() for x in args.labels.split(',')]
        else:
            labels = [os.path.basename(os.path.dirname(x)) for x in csv_paths]
            
        # Pad labels if counts don't match
        while len(labels) < len(csv_paths):
            labels.append(f"Model_{len(labels)+1}")
            
        plot_curves(csv_paths, labels, args.metric, args.output)
        
    if args.scatter_data:
        plot_scatter(args.scatter_data, args.scatter_output)
