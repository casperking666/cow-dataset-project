import os

# Define directories
logs_dir = './logs_round2'
summary_file = './summary_results_round2_150e.txt'

# Initialize list to store results
results = []

# Function to extract relevant information from a log file
def extract_info(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
        if lines[2].split()[0] == "Traceback":
            # print("shabi")
            return
        
        # Extract parameters (line 4)
        params_line = lines[3].strip()
        hsv_h = params_line.split('hsv_h=')[1].split()[0]
        hsv_s = params_line.split('hsv_s=')[1].split()[0]
        hsv_v = params_line.split('hsv_v=')[1].split()[0]
        
        # Extract GPU info
        gpu_line = lines[2].strip()
        if "CUDA" in gpu_line:
            gpu = gpu_line.split("CUDA:0 (")[1].split(',')[0]
        else:
            gpu = "N/A"
        
        # Extract best epoch (look for "Best results observed at epoch")
        best_epoch = 'N/A'
        # for line in lines:
        #     if "Best results observed at epoch" in line:
        #         best_epoch = line.split()[-6]
        #         break
        
        # Extract top-1 accuracy (line -4)
        top1_accuracy_line = lines[-4].strip()
        top1_accuracy = top1_accuracy_line.split('all ')[-1].split()[0]
        
        return (hsv_h, hsv_s, hsv_v, gpu, best_epoch, top1_accuracy)

# Loop through log files
for log_file in os.listdir(logs_dir):
    if log_file.endswith('.log') and log_file != "job_164.log" and log_file != "job_230.log":
        log_file_path = os.path.join(logs_dir, log_file)
        if extract_info(log_file_path) != None:
            hsv_h, hsv_s, hsv_v, gpu, best_epoch, top1_accuracy = extract_info(log_file_path)
        
            # Extract job id from filename
            job_id = log_file.split('_')[1].split('.')[0]
            
            # Append results
            results.append((job_id, hsv_h, hsv_s, hsv_v, gpu, best_epoch, top1_accuracy))

# Sort results by top-1 accuracy
results.sort(key=lambda x: float(x[-1]), reverse=True)

# Write summary to file
with open(summary_file, 'w') as f:
    f.write("Job ID\tHSV_H\tHSV_S\tHSV_V\tGPU\tBest Epoch\tTop-1 Accuracy\n")
    for result in results:
        f.write("\t".join(result) + "\n")

print("Summary of results saved to", summary_file)
