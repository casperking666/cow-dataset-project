import os
import csv
import re

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract augmentation type and value from line 8
    aug_info = lines[7].strip().split(':')
    aug_type = aug_info[0].strip()
    aug_value = aug_info[1].strip()
    
    # Extract top1 accuracy from the last line using regex
    last_line = lines[-1].strip()
    match = re.search(r"'eval_accuracy':\s*([\d.]+)", last_line)
    if match:
        top1_accuracy = float(match.group(1))
    else:
        print(f"Warning: Could not find eval_accuracy in file {file_path}")
        top1_accuracy = None
    
    return aug_type, aug_value, top1_accuracy

def main():
    log_folder = 'vit_aug_logs'  # Update this to your folder path if different
    results = []

    for filename in os.listdir(log_folder):
        if filename.endswith('.log'):
            file_path = os.path.join(log_folder, filename)
            try:
                aug_type, aug_value, top1_accuracy = parse_log_file(file_path)
                if top1_accuracy is not None:
                    results.append((aug_type, aug_value, top1_accuracy))
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    # Sort results by augmentation type and value
    results.sort(key=lambda x: (x[0], x[1]))

    # Save results to CSV file
    output_file = 'augmentation_results_vit.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Augmentation Type', 'Value', 'Top1 Accuracy'])
        csvwriter.writerows(results)

    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    main()