import os
from pbstools import PythonJob
from shutil import copyfile
import datetime

python_file = (r"/home/jeromel/Documents/Projects/Deep2P/repos/" +
               r"deepinterpolation/examples/paper_generation_code/" +
               r"cluster_lib/generic_ophys_transfer_inference_sync.py")

output_folder = (r"/allen/programs/braintv/workgroups/neuralcoding/2p_data/" +
                 r"single_plane/ground_truth/denoised")

model_file = (r"/allen/programs/braintv/workgroups/ophysdev/OPhysCore/" +
              r"Deep2p/" +
              r"unet_single_1024_mean_absolute_error_Ai93_2019_09_11_23_32/" +
              r"2019_09_11_23_32_unet_single_1024_mean_absolute_error" +
              r"_Ai93-0450.h5")

dat_file = (r"/allen/programs/braintv/workgroups/neuralcoding/2p_data/" +
            r"single_plane/ground_truth/20191215_raw_noisy.h5")

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")
jobdir = output_folder

try:
    os.mkdir(jobdir)
except Exception:
    print("folder already exists")

output_terminal = os.path.join(jobdir, run_uid + "_running_terminal.txt")
script_basename = os.path.basename(__file__)
copyfile(
    os.path.realpath(__file__), os.path.join(jobdir,
                                             run_uid + "_" + script_basename)
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
    jobname="ophys_syncer",
    python_args=arg_to_pass + " > " + output_terminal,
    **job_settings
).run(dryrun=False)
