import os
from pbstools import PythonJob
import glob

input_dir = "/home/jeromel/Documents/Projects/Deep2P/input_data/"
output_folder = "/allen/programs/braintv/workgroups/neuralcoding/2p_data/2020_09_02_validation"
access_to_models = "/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Deep2p/unet_single_1024_mean_absolute_error_Ai93_2019_09_11_23_32"
python_file = "/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/cluster_lib/single_ophys_json_inferrence.py"

models_path = os.listdir(access_to_models)
pre_frame = 30
post_frame = 30
batch_size = 5

json_path = os.path.join(
    input_dir, "2020-06-16-val-semi-large-single-plane-Ai93-norm.json"
)

for model_file in models_path:
    if ".h5" in model_file:
        try:
            os.mkdir(output_folder)
        except:
            print("Folder already created")

        jobdir = os.path.join(
            output_folder, "validation_" +
            os.path.splitext(os.path.basename(model_file))[0]
        )

        try:
            os.mkdir(jobdir)
        except:
            print("Folder already created")
            files = glob.glob(os.path.join(jobdir, "*"))
            for f in files:
                os.remove(f)

        local_path = os.path.join(
            jobdir, "validation_out_" + os.path.splitext(os.path.basename(model_file))[0] + ".hdf5")

        job_settings = {
            "queue": "braintv",
            "mem": "180g",
            "walltime": "12:00:00",
            "ppn": 16,
        }

        out_file = os.path.join(jobdir, "$PBS_JOBID.out")

        job_settings.update(
            {
                "outfile": out_file,
                "errfile": os.path.join(jobdir, "$PBS_JOBID.err"),
                "email": "jeromel@alleninstitute.org",
                "email_options": "a",
            }
        )

        arg_to_pass = [
            "--json_path "
            + json_path
            + " --output_file "
            + local_path
            + " --model_file "
            + os.path.join(access_to_models, model_file)
            + " --batch_size "
            + str(batch_size)
            + " --pre_frame "
            + str(pre_frame)
            + " --post_frame "
            + str(post_frame)
        ]

        PythonJob(
            python_file,
            python_executable="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env/bin/python",
            conda_env="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env",
            jobname="movie_2p",
            python_args=arg_to_pass[0],
            **job_settings
        ).run(dryrun=False)
