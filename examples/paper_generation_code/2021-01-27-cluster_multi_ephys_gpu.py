import os
import sys
from pbstools import PythonJob
from shutil import copyfile
import datetime

python_file = r"/home/jeromel/Documents/Projects/Deep2P/repos/deepinterpolation/examples/paper_generation_code/2021-01-27-multi_ephys_training.py"

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")
jobdir = '/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/ephys/cluster/'
output_terminal = os.path.join(jobdir, run_uid+'_running_terminal.txt')

job_settings = {'queue': 'braintv',
                'mem': '250g',
                'walltime': '4000:00:00',
                'ppn': 24,
                'gpus': 1
                } 

job_settings.update({
                'outfile':os.path.join(jobdir, '$PBS_JOBID.out'),
                'errfile':os.path.join(jobdir, '$PBS_JOBID.err'),
                'email': 'jeromel@alleninstitute.org',
                'email_options': 'a'
                })

arg_to_pass =  ''

PythonJob(
    python_file,
    python_executable="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env/bin/python",  # "/home/jeromel/.conda/envs/deep_work_gpu/bin/python",
    conda_env="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env",  # "deep_work_gpu",
    jobname = 'movie_ephys',
    python_args = arg_to_pass+' > '+output_terminal,
    **job_settings	
).run(dryrun=False)


    

              
    
    
