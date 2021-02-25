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
import datetime


def main(argv):
    opts, args = getopt.getopt(
        argv,
        [],
        [
            "output_folder=",
            "model_file=",
            "dat_file=",
        ],
    )

    for opt, arg in opts:
        if opt == "--output_folder":
            output_folder = arg
        if opt == "--model_file":
            model_file = arg
        if opt == "--dat_file":
            dat_file = arg
    try:
        os.mkdir(output_folder)
    except:
        print("Folder already created")

    # We fist fine-tune the model

    python_file = "/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/cluster_lib/single_ephys_transfer_trainer.py"

    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")
    output_terminal = os.path.join(output_folder, 
                                   run_uid+'_running_terminal.txt')

    
    nb_probes = 384
    raw_data = np.memmap(dat_file, dtype="int16")
    img_per_movie = int(raw_data.size / nb_probes)
    pre_post_frame = 30
    batch_size = 100
    nb_jobs = 100
    training_samples = 100000 # 10000000
    pre_post_omission = 1
    start_frame = pre_post_omission + pre_post_frame
    end_frame = img_per_movie - pre_post_frame - pre_post_omission - 1

    job_settings = {'queue': 'braintv',
                    'mem': '250g',
                    'walltime': '48:00:00',
                    'ppn': 24,
                    'gpus': 1
                    } 

    job_settings.update({
                    'outfile': os.path.join(output_folder, '$PBS_JOBID.out'),
                    'errfile': os.path.join(output_folder, '$PBS_JOBID.err'),
                    'email': 'jeromel@alleninstitute.org',
                    'email_options': 'a'
                    })
            
    arg_to_pass = [
        "--movie_path "
        + dat_file
        + " --train_frame_start "
        + str(20000)
        + " --train_frame_end "
        + str(-1)
        + " --train_total_samples "
        + str(training_samples)
        + " --val_frame_start "
        + str(0)
        + " --val_frame_end "
        + str(19999)
        + " --val_total_samples "
        + str(-1)
        + " --output_path "
        + output_folder
        + " --model_file "
        + model_file
        + " --batch_size "
        + str(batch_size)
        + " --pre_post_frame "
        + str(pre_post_frame)
        + " --pre_post_omission "
        + str(pre_post_omission)
        + " --loss "
        + 'mean_squared_error'
        ]
]
    PythonJob(
        python_file,
        python_executable= (r"/allen/programs/braintv/workgroups/nc-ophys/" +
                            r"Jeromel/conda/tf20-env/bin/python"),  
        conda_env= (r"/allen/programs/braintv/workgroups/nc-ophys/Jeromel/"+
                    r"conda/tf20-env"),  
        jobname= 'fine_tuning_ephys',
        python_args= arg_to_pass[0]+' > '+output_terminal,
        **job_settings	
    ).run(dryrun=False)

    # We wait for the jobs to complete
    stay_in_loop = True
    while stay_in_loop:
        time.sleep(60)
        list_files = glob.glob(os.path.join(output_folder, "*_model.h5"))

        if len(list_files)>0:
            stay_in_loop = False
            
    new_model_file = list_files[0]

    # We infer the movie in chunks
    block_size = np.int(np.ceil((end_frame - start_frame) / nb_jobs))

    # We force block_size to be a multiple of batch size
    block_size = int(np.floor(block_size / batch_size) * batch_size)

    jobdir = os.path.join(
        output_folder, 
        "tmp_" + os.path.splitext(os.path.basename(model_file))[0]
    )
    try:
        os.mkdir(jobdir)
    except Exception:
        print("Folder already created")
        files = glob.glob(os.path.join(jobdir, "*"))
        for f in files:
            os.remove(f)

    python_file = (r"/home/jeromel/Documents/Projects/Deep2P/repos/"+
                   r"deepinterpolation/examples/cluster_lib/"+
                   r"single_ephys_section_inferrence.py")

    list_files_check = []
    for index, local_start_frame in enumerate(
        np.arange(start_frame, end_frame, block_size)
    ):
        local_path = os.path.join(jobdir, "movie_" + str(index) + ".hdf5")
        local_end_frame = np.min([end_frame, 
                                  local_start_frame + block_size - 1])
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
            + dat_file
            + " --frame_start "
            + str(local_start_frame)
            + " --frame_end "
            + str(local_end_frame)
            + " --output_file "
            + local_path
            + " --model_file "
            + new_model_file
            + " --batch_size "
            + str(batch_size)
            + " --pre_post_frame "
            + str(pre_post_frame)
            + " --pre_post_omission "
            + str(pre_post_omission)
        ]

        PythonJob(
            python_file,
            python_executable="/home/jeromel/.conda/envs/deep_work2/bin/python",
            conda_env=(r"/allen/programs/braintv/workgroups/nc-ophys/"+
                r"Jeromel/conda/tf20-env"),  
            jobname="ephys_inferrence",
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
    output_merged = os.path.join(output_folder, 
                                 "movie_" + os.path.basename(model_file))

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
            chunks=(1, final_shape[1], final_shape[2], final_shape[3]),
            dtype="float16",
        )

        for each_file in list_files:
            with h5py.File(each_file, "r") as file_handle:
                local_shape = file_handle["data"].shape
                dset_out[
                    global_index_frame : global_index_frame + local_shape[0], 
                    :, :, :
                ] = file_handle["data"][:, :, :, :]
                global_index_frame += local_shape[0]

    shutil.rmtree(jobdir)


if __name__ == "__main__":
    main(sys.argv[1:])

