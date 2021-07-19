import os
import sys
from simple_slurm import Slurm
from shutil import copyfile
import datetime

folder_path = "/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/paper_generation_code/multi_ophys/multi-feature"

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

    python_executable="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env/bin/python"
    
    # instantiate a Slurm object
    slurm = Slurm(
        cpus_per_task=1,
        nodes=1,
        gpus=1,
        tasks-per-node=16,
        mem='250G',
        partition= 'braintv',
        time='48:00:00', 
        job_name="tf_deepInterp",
        output=jobdir+f'{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    # call the `sbatch` command to run the jobs
    slurm.sbatch(python_executable+' '+python_file)
