import os
import csv
import re
from collections import defaultdict

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 4:
            accuracy_line = lines[-4]  # Fourth last line
            parts = accuracy_line.split()
            if len(parts) >= 2:
                try:
                    accuracy = float(parts[1])
                    return accuracy
                except ValueError:
                    print(f"Warning: Could not parse accuracy from {file_path}")
    return None

def process_logs(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.log'):
            file_path = os.path.join(folder_path, filename)
            
            # Extract aug_type and aug_value using regex
            match = re.match(r'yolo-tiny-(.+?)[-_](.+?)\.log', filename)
            if match:
                aug_type, aug_value = match.groups()
                
                # Convert aug_value to float if possible, otherwise keep as string
                try:
                    aug_value = float(aug_value)
                except ValueError:
                    pass  # Keep as string if it's not a number
                
                accuracy = parse_log_file(file_path)
                
                if accuracy is not None:
                    results.append([aug_type, aug_value, accuracy])
            else:
                print(f"Warning: Couldn't parse filename: {filename}")
    return results

def sort_results_by_type(results):
    # Group results by augmentation type
    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result[0]].append(result)
    
    # Sort each group by accuracy in descending order
    for aug_type in grouped_results:
        grouped_results[aug_type].sort(key=lambda x: x[2], reverse=True)
    
    # Flatten the sorted groups back into a list
    sorted_results = []
    for aug_type in sorted(grouped_results.keys()):
        sorted_results.extend(grouped_results[aug_type])
    
    return sorted_results

def main():
    aug_folder = 'yolo_aug_logs'
    hsv_folder = 'yolo_hsv_logs'
    
    all_results = process_logs(aug_folder) + process_logs(hsv_folder)
    
    # Sort results by augmentation type, then by accuracy
    sorted_results = sort_results_by_type(all_results)
    
    with open('augmentation_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Augmentation Type', 'Value', 'Top1 Accuracy'])
        writer.writerows(sorted_results)

if __name__ == "__main__":
    main()