import os
import sys
from simple_slurm import Slurm
from shutil import copyfile
import datetime

folder_path = "/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/multi_ophys/multi-feature"

for indiv_file in os.listdir(folder_path):
    python_file = os.path.join(folder_path, indiv_file)
    if os.path.isdir(python_file):
        continue
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    jobdir = "/home/jeromel/Documents/Projects/Deep2P/ClusterJobs/"
    output_terminal = os.path.join(
        jobdir, run_uid + os.path.basename(python_file) + "_running_terminal.txt"
    )

    arg_to_pass = ""

    # call the `sbatch` command to run the jobs
    slurm.sbatch(python_executable+' '+python_file+' '+ arg_to_pass +' '+ Slurm.SLURM_ARRAY_TASK_I + " > " + output_terminal)
