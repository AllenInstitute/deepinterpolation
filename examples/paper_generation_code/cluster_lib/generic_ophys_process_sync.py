import os
import sys
from pbstools import PythonJob
import sys, getopt
import numpy as np
import glob
import h5py
import time
import glob
import shutil


def main(argv):
    opts, args = getopt.getopt(
        argv,
        [],
        [
            "output_folder=",
            "model_file=",
            "start_frame=",
            "end_frame=",
            "pre_frame=",
            "post_frame=",
            "nb_jobs=",
            "h5_file=",
            "save_raw=",
            "output_type=",
        ],
    )

    # Default
    save_raw = False
    output_type = 'float16'

    for opt, arg in opts:
        if opt == "--output_folder":
            output_folder = arg
        if opt == "--model_file":
            model_file = arg
        if opt == "--start_frame":
            start_frame = int(arg)
        if opt == "--end_frame":
            end_frame = int(arg)
        if opt == "--pre_frame":
            pre_frame = int(arg)
        if opt == "--post_frame":
            post_frame = int(arg)
        if opt == "--nb_jobs":
            nb_jobs = int(arg)
        if opt == "--h5_file":
            h5_file = arg
        if opt == "--save_raw":
            save_raw == bool(arg)
        if opt == "--output_type":
            output_type == arg
    try:
        os.mkdir(output_folder)
    except:
        print("Folder already created")

    batch_size = 5

    # We infer the movie in chunks
    block_size = np.int(np.ceil((end_frame - start_frame) / nb_jobs))

    # We force block_size to be a multiple of batch size
    block_size = int(np.floor(block_size / batch_size) * batch_size)

    jobdir = os.path.join(
        output_folder, "tmp_" + os.path.splitext(os.path.basename(model_file))[0]
    )
    try:
        os.mkdir(jobdir)
    except:
        print("Folder already created")
        files = glob.glob(os.path.join(jobdir, "*"))
        for f in files:
            os.remove(f)

    python_file = "/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/cluster_lib/single_ophys_section_inferrence.py"

    list_files_check = []
    for index, local_start_frame in enumerate(
        np.arange(start_frame, end_frame, block_size)
    ):
        local_path = os.path.join(jobdir, "movie_" + str(index) + ".hdf5")
        local_end_frame = np.min([end_frame, local_start_frame + block_size - 1])
        job_settings = {
            "queue": "braintv",
            "mem": "180g",
            "walltime": "48:00:00",
            "ppn": 16,
        }
        out_file = os.path.join(jobdir, "$PBS_JOBID.out")
        list_files_check.append(local_path + ".done")

        job_settings.update(
            {
                "outfile": out_file,
                "errfile": os.path.join(jobdir, "$PBS_JOBID.err"),
                "email": "jeromel@alleninstitute.org",
                "email_options": "a",
            }
        )

        arg_to_pass = [
            "--movie_path "
            + h5_file
            + " --frame_start "
            + str(local_start_frame)
            + " --frame_end "
            + str(local_end_frame)
            + " --output_file "
            + local_path
            + " --model_file "
            + model_file
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

    # We wait for the jobs to complete
    stay_in_loop = True
    while stay_in_loop:
        time.sleep(60)
        nb_file = 0
        for indiv_file in list_files_check:
            if os.path.isfile(indiv_file):
                nb_file += 1

        if nb_file == len(list_files_check):
            stay_in_loop = False

    # We merge the files
    output_merged = os.path.join(output_folder, "movie_" + os.path.basename(model_file))

    list_files = glob.glob(os.path.join(jobdir, "*.hdf5"))
    list_files = sorted(
        list_files, key=lambda x: int(x.split("movie_")[1].split(".hdf5")[0])
    )

    nb_frames = 0
    for each_file in list_files:
        with h5py.File(each_file, "r") as file_handle:
            local_shape = file_handle["data"].shape
            nb_frames = nb_frames + local_shape[0]

    final_shape = list(local_shape)
    final_shape[0] = nb_frames

    global_index_frame = 0
    with h5py.File(output_merged, "w") as file_handle:
        dset_out = file_handle.create_dataset(
            "data",
            shape=final_shape,
            chunks=(1, final_shape[1], final_shape[2]),
            dtype=output_type,
        )

        for each_file in list_files:
            with h5py.File(each_file, "r") as file_handle:
                local_shape = file_handle["data"].shape
                dset_out[
                    global_index_frame : global_index_frame + local_shape[0], :, :, :
                ] = file_handle["data"][:, :, :, :].astype(output_type)
                global_index_frame += local_shape[0]

        if save_raw:
            raw_out = file_handle.create_dataset(
                "raw",
                shape=final_shape,
                chunks=(1, final_shape[1], final_shape[2]),
                dtype=output_type,
            )

            with h5py.File(h5_file, "r") as file_handle_raw:
                for index in np.arange(start_frame, end_frame):
                    raw_out[index, :, :, :] = file_handle_raw["data"][index, :, :, :].astype(output_type)

    shutil.rmtree(jobdir)


if __name__ == "__main__":
    main(sys.argv[1:])
