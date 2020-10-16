import os
import sys
from pbstools import PythonJob
from shutil import copyfile
import datetime
import numpy as np

python_file = r"/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/cluster_lib/generic_ephys_process_sync.py"

sub_folder = "processed_2020_03_02"
data_file = "continuous.dat2"

data_folders = [
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/767871931_415149_20181024_probeE",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/768515987_412809_20181025_probeF",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/778998620_419112_20181114_probeE",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/787025148_416356_20181128_probeC",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/771160300_415148_20181031_probeE",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/779839471_419118_20181115_probeE",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/781842082_419119_20181119_probeC",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/793224716_424445_20181211_probeF",
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/794812542_418196_20181213_probeF",
]

for local_data in data_folders:
    output_folder = local_data + "/" + sub_folder
    dat_file = local_data + "/" + data_file

    model_file = "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/778998620_419112_20181114_probeD/trained_models/unet_single_ephys_1024_mean_squared_error_2020_02_29_15_28/2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"

    nb_probes = 384
    raw_data = np.memmap(dat_file, dtype="int16")
    img_per_movie = int(raw_data.size / nb_probes)
    pre_post_frame = 30
    pre_post_omission = 1

    end_frame = img_per_movie - pre_post_frame - pre_post_omission - 1

    nb_jobs = 200
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")
    jobdir = output_folder

    start_frame = pre_post_omission + pre_post_frame

    try:
        os.mkdir(jobdir)
    except:
        print("folder already exists")

    output_terminal = os.path.join(jobdir, run_uid + "_running_terminal.txt")
    script_basename = os.path.basename(__file__)
    copyfile(
        os.path.realpath(__file__),
        os.path.join(jobdir, run_uid + "_" + script_basename),
    )

    job_settings = {
        "queue": "braintv",
        "mem": "250g",
        "walltime": "24:00:00",
        "ppn": 16,
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
    arg_to_pass += (
        " --start_frame "
        + str(start_frame)
        + " --end_frame "
        + str(end_frame)
        + " --pre_post_frame "
        + str(pre_post_frame)
    )
    arg_to_pass += (
        " --nb_jobs " + str(nb_jobs) + " --pre_post_omission " + str(pre_post_omission)
    )

    PythonJob(
        python_file,
        python_executable="/home/jeromel/.conda/envs/deep_work2/bin/python",
        conda_env="deep_work2",
        jobname="movie_2p",
        python_args=arg_to_pass + " > " + output_terminal,
        **job_settings
    ).run(dryrun=False)
