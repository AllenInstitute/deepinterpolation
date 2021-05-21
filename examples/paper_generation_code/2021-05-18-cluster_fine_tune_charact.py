import os
from pbstools import PythonJob
import datetime

folder_path = r"/allen/programs/braintv/workgroups/neuralcoding/2p_data/"\
    "single_plane/2021-05-14-simulation-naomi/"

raw_path_folder = r"/allen/programs/braintv/workgroups/neuralcoding/2p_data/"\
    "single_plane/2021-05-14-simulation-naomi/"

list_exp_folder = [
    #   'typicalVolume80mW',
    #   'typicalVolume160mW',
    'sparseLabelingVolume'
]

raw_model = r"/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Deep2p/"\
    "unet_single_1024_mean_absolute_error_Ai93_2019_09_11_23_32/"\
    "2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"

python_file = os.path.join(
    "/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/"
    "examples/paper_generation_code",
    "2021-05-18-fine_tune_tif_charact.py",
)

for indiv_id in list_exp_folder:
    path_tif = os.path.join(
        raw_path_folder, indiv_id, indiv_id + "noisy.tif")

    local_output_path = os.path.join(
        folder_path, indiv_id, 'deepinterp')

    try:
        os.mkdir(local_output_path)
    except Exception:
        print("folder already exists")

    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

    jobdir = local_output_path

    output_terminal = os.path.join(
        jobdir, run_uid +
        os.path.basename(python_file) + "_running_terminal.txt"
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

    arg_to_pass = (
        " --file_h5_path "
        + path_tif
        + " --raw_model_path "
        + raw_model
        + " --output_path "
        + local_output_path
    )

    PythonJob(
        python_file,
        python_executable="/allen/programs/braintv/workgroups/nc-ophys/"
        "Jeromel/conda/tf20-env/bin/python",
        conda_env="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/"
        "conda/tf20-env",
        jobname="tf_" + os.path.basename(python_file),
        python_args=arg_to_pass + " > " + output_terminal,
        **job_settings
    ).run(dryrun=False)
