import os
import sys
from pbstools import PythonJob
from shutil import copyfile
import datetime
import numpy as np

python_file = r"/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/cluster_lib/generic_ephys_transfer_inference_sync.py"

output_folder = "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/simulated_ground_truth/deep_interp"
model_file = "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/778998620_419112_20181114_probeD/trained_models/unet_single_ephys_1024_mean_squared_error_2020_02_29_15_28/2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"
dat_file = "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/simulated_ground_truth/continuous_sim.dat

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")
jobdir = output_folder

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

output_terminal = os.path.join(jobdir, run_uid + "_running_terminal.txt")
script_basename = os.path.basename(__file__)
copyfile(
    os.path.realpath(__file__), os.path.join(jobdir, run_uid + "_" + script_basename)
)

job_settings = {
    "queue": "braintv",
    "mem": "50g",
    "walltime": "48:00:00",
    "ppn": 8,
}

job_settings.update(
    {
        "outfile": os.path.join(jobdir, "$PBS_JOBID.out"),
        "errfile": os.path.join(jobdir, "$PBS_JOBID.err"),
        "email": "jeromel@alleninstitute.org",
        "email_options": "a",
    }
)

arg_to_pass = (
    " --dat_file "
    + dat_file
    + " --output_folder "
    + output_folder
    + " --model_file "
    + model_file
)

PythonJob(
    python_file,
    python_executable="/home/jeromel/.conda/envs/deep_work2/bin/python",
    conda_env="deep_work2",
    jobname="ephys_syncer",
    python_args=arg_to_pass + " > " + output_terminal,
    **job_settings
).run(dryrun=False)
