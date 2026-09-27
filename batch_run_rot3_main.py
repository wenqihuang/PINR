import os
import subprocess
import multiprocessing


subject_list = ['subject01']


slice_list = [[123, 124, 129]]


rot = 3             # number of rotated views
total_R = 2
R = int(total_R*rot)      # per-view acceleration: rot views at 1/R lines each = 1/total_R of a full k-space

print(f'rot = {rot}, R = {R}')


# Define available GPUs
gpu_list = ['0']
failed_log_file = f"failed_jobs_rot{rot}.log"

# Function to run the command on a specific GPU
def run_job(subject_id, slc, gpu_id):
    cmd = [
        "python", "main.py",
        "--config", f"configs/config_simulated_rot{rot}.yml",
        "--subject_id", subject_id,
        "--slice_idx", str(slc),
        "--R", str(R),
        "--device", gpu_id
    ]
    
    command_str = " ".join(cmd)
    print(f"Running on GPU {gpu_id}: {command_str}")

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running: {command_str}")
        with open(failed_log_file, "a") as f:
            f.write(f"Failed command: {command_str}\nError: {e}\n\n")

# Create a list of all jobs
jobs = []
for i in range(len(subject_list)):
    subject_id = subject_list[i]
    for slc in slice_list[i]:
        jobs.append((subject_id, slc))

# Use multiprocessing to distribute jobs across GPUs
with multiprocessing.Pool(len(gpu_list)) as pool:
    pool.starmap(run_job, [(subject_id, slc, gpu_list[i % len(gpu_list)]) for i, (subject_id, slc) in enumerate(jobs)], chunksize=1)

print("All jobs completed.")

