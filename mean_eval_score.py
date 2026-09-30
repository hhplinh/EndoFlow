import argparse
import glob
import json
from pathlib import Path
import numpy as np
import os
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate mean PSNR, SSIM, and LPIPS scores.')
    parser.add_argument('--batch_to_csv', action='store_true',
                        help='Whether to convert the results to a CSV file (default: False).')
    parser.add_argument('--batch_file', type=str, default='batch_input.json',
                        help='Path to a text file containing a list of result directories to process (one per line). If not provided, the script will process all subdirectories in --results_dir.')
    parser.add_argument('--csv_path', type=str, default='mean_results.csv',
                        help='Path to save the CSV file if --batch_to_csv is set (default: "mean_results.csv").')
    parser.add_argument('--results_dir', type=str,
                        help='Directory containing the result JSON files.')
    parser.add_argument('--iter_key', type=str, default='ours_6000',
                        help='Key in the JSON file to extract the metrics from (default: "ours_6000").')
    parser.add_argument('--metrics', nargs='+', default=['PSNR', 'SSIM', 'LPIPS'],)
    return parser.parse_args()

def compute_mean_results(results_dir, iter_key, metrics):
    result_dirs = glob.glob(results_dir + '/*/')  # Assuming each subdirectory contains a results.json file

    metrics = metrics + ["Gaussians", "TrainingTimeSeconds", "FPS", "PeakGPUMemoryMB", "PeakGPUMemoryReservedMB"]

    total_results = {
        iter_key: {
            metric: [] for metric in metrics
        }
    }

    cnt_results = 0

    for res_dir in result_dirs:
        result_path = Path(res_dir) / 'results.json'
        if result_path.exists():
            cnt_results += 1
            with open(result_path, 'r') as f:
                result = json.load(f)
                for metric in metrics:
                    total_results[iter_key][metric].append(result[iter_key][metric])

    print(f"Aggregated results from {cnt_results} samples:")
    for metric in metrics:
        print(f"Mean {metric}: {np.mean(total_results[iter_key][metric]):.4f} ± {np.std(total_results[iter_key][metric]):.4f}")

    for metric in metrics:
        total_results[iter_key][metric] = {
            'mean': np.mean(total_results[iter_key][metric]),
            'std': np.std(total_results[iter_key][metric])
        }

    total_results["num_samples"] = cnt_results
    return total_results

def compute_mean_result_per_view(results_dir, iter_key, metrics):
    result_dirs = glob.glob(results_dir + '/*/')  # Assuming each subdirectory contains a results.json file

    view_results = {
        iter_key: {
            metric: {} for metric in metrics
        }
    }

    for res_dir in result_dirs:
        result_path = Path(res_dir) / 'per_view.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                result = json.load(f)
                for metric in metrics:
                    for view, value in result[iter_key][metric].items():
                        if view not in view_results[iter_key][metric]:
                            view_results[iter_key][metric][view] = []
                        view_results[iter_key][metric][view].append(value)

    # Compute means for each view
    for metric in metrics:
        for view in view_results[iter_key][metric]:
            view_results[iter_key][metric][view] = {
                'mean': np.mean(view_results[iter_key][metric][view]),
                'std': np.std(view_results[iter_key][metric][view])
            }
    
    view_results["num_samples"] = len(result_dirs)
    return view_results

def from_batch_to_csv_w_config(batch_file, csv_path):
    with open(batch_file, 'r') as f:
        data = json.load(f)

    methods = data['method']
    work_dir = data['work_dir']
    configs = data['config']
    fields = data['field']

    csv_fields = ["Method", "Video", "Config"] + fields

    csv_rows = []

    for method in methods:
        for entry in data['data']:
            name = entry['name']
            path = entry['path']
            iter_key = entry['iter_key']

            for config in configs:
                results_dir = os.path.join(work_dir, method, f"output/{path}_{config}")
                print(f"Processing results from: {results_dir}")
                assert Path(results_dir).exists(), f"Directory {results_dir} does not exist. Please check the path and naming convention."
                mean_results = compute_mean_results(results_dir, iter_key, args.metrics)
                csv_rows.append([method, name, config] + [mean_results[iter_key][field]['mean'] for field in fields])
        
    # Write to CSV    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_fields)
        writer.writerows(csv_rows)
    
def from_batch_to_csv_raw(batch_file, csv_path):
    with open(batch_file, 'r') as f:
        data = json.load(f)

    methods = data['method']
    work_dir = data['work_dir']
    fields = data['field']

    csv_fields = ["Method", "Video"] + fields

    csv_rows = []

    for method in methods:
        for entry in data['data']:
            name = entry['name']
            path = entry['path']
            iter_key = entry['iter_key']

            results_dir = os.path.join(work_dir, method, f"output/{path}")
            print(f"Processing results from: {results_dir}")
            assert Path(results_dir).exists(), f"Directory {results_dir} does not exist. Please check the path and naming convention."
            mean_results = compute_mean_results(results_dir, iter_key, args.metrics)
            csv_rows.append([method, name] + [mean_results[iter_key][field]['mean'] for field in fields])
        
    # Write to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_fields)
        writer.writerows(csv_rows)

def from_batch_to_csv(batch_file, csv_path):
    with open(batch_file, 'r') as f:
        data = json.load(f)

    if 'config' in data and len(data['config']) > 0:
        from_batch_to_csv_w_config(batch_file, csv_path)
    else:
        from_batch_to_csv_raw(batch_file, csv_path)

if __name__ == '__main__':
    args = parse_args()

    if args.batch_to_csv:
        from_batch_to_csv(args.batch_file, args.csv_path)
    else:
        print("Evaluating results from:", args.results_dir)

        mean_results = compute_mean_results(args.results_dir, args.iter_key, args.metrics)
        view_results = compute_mean_result_per_view(args.results_dir, args.iter_key, args.metrics)

        with open(Path(args.results_dir) / 'mean_results.json', 'w') as f:
            json.dump(mean_results, f, indent=4)

        with open(Path(args.results_dir) / 'mean_per_view.json', 'w') as f:
            json.dump(view_results, f, indent=4)