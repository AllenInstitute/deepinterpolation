import os
from pbstools import PythonJob
from shutil import copyfile
import datetime
import glob

raw_path_folder = r"/allen/programs/braintv/workgroups/mct-t300/"\
    "CalibrationTF/rawdata"
output_path = r"/allen/programs/braintv/workgroups/neuralcoding/"\
    "2p_data/single_plane/2021-05-13-transfer_traning_oephys"
path_models = r"/allen/programs/braintv/workgroups/neuralcoding/"\
    "2p_data/single_plane/2021-05-13-transfer_traning_oephys"
list_exp_id = [
    102932,
    102939,
    102941,
    102945,
]

all_models = os.listdir(path_models)

for indiv_id in list_exp_id:
    try:
        path_tif = os.path.join(raw_path_folder, str(indiv_id) + "_2.tif")

        jobdir = os.path.join(output_path, str(indiv_id))

        raw_model = os.path.join(output_path, str(indiv_id), "*_model.h5")
        raw_model = glob.glob(raw_model)[0]

        python_file = r"/home/jeromel/Documents/Projects/Deep2P/repos/"\
            "deepinterpolation/examples/paper_generation_code/cluster_lib/"\
            "single_tif_section_inferrence.py"

        model_file = raw_model

        pre_post_frame = 30
        pre_post_omission = 0

        now = datetime.datetime.now()
        run_uid = now.strftime("%Y_%m_%d_%H_%M")

        start_frame = 0
        end_frame = -1
        batch_size = 5
        save_raw = True
        output_type = "float32"

        output_terminal = os.path.join(
            jobdir, run_uid + "_running_terminal.txt")
        script_basename = os.path.basename(__file__)
        copyfile(
            os.path.realpath(__file__), os.path.join(
                jobdir, run_uid + "_" + script_basename)
        )

        local_path = os.path.join(
            jobdir, "movie_" + os.path.basename(model_file))

        job_settings = {
            "queue": "braintv",
            "mem": "180g",
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

        arg_to_pass = [
            "--movie_path "
            + path_tif
            + " --frame_start "
            + str(start_frame)
            + " --frame_end "
            + str(end_frame)
            + " --output_file "
            + local_path
            + " --model_file "
            + model_file
            + " --batch_size "
            + str(batch_size)
            + " --pre_post_frame "
            + str(pre_post_frame)
            + " --pre_post_omission "
            + str(pre_post_omission)
            + " --save_raw "
            + str(save_raw)
        ]

        PythonJob(
            python_file,
            python_executable="/allen/programs/braintv/workgroups/nc-ophys/"
            "Jeromel/conda/tf20-env/bin/python",
            conda_env="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/"
            "conda/tf20-env",
            jobname="movie_2p",
            python_args=arg_to_pass[0] + " > " + output_terminal,
            **job_settings
        ).run(dryrun=False)
    except Exception:
        print("experiment associated could not be found")
