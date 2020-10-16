import os
import sys
from pbstools import PythonJob
from shutil import copyfile
import datetime

folder_path = "/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/multi_ophys/multi-eaggerly"

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

    job_settings = {
        "queue": "braintv",
        "mem": "250g",
        "walltime": "200:00:00",
        "ppn": 16,
        "gpus": 1,
    }

    job_settings.update(
        {
            "outfile": os.path.join(jobdir, "$PBS_JOBID.out"),
            "errfile": os.path.join(jobdir, "$PBS_JOBID.err"),
            "email": "jeromel@alleninstitute.org",
            "email_options": "a",
        }
    )

    arg_to_pass = ""

    PythonJob(
        python_file,
        python_executable="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env/bin/python",  # "/home/jeromel/.conda/envs/deep_work_gpu/bin/python",
        conda_env="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env",  # "deep_work_gpu",
        jobname="tf_" + os.path.basename(python_file),
        python_args=arg_to_pass + " > " + output_terminal,
        **job_settings
    ).run(dryrun=False)
