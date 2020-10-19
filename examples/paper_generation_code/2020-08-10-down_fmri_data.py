from urllib.request import urlretrieve
import os
import requests
import numpy as np

base_preproc_path = 'https://openneuro.org/crn/datasets/ds001246/snapshots/1.2.1/files/derivatives:preproc-spm:output:sub-{:02d}:ses-perceptionTraining{:02d}:func:sub-{:02d}_ses-perceptionTraining{:02d}_task-perception_run-{:02d}_bold_preproc.nii.gz'

base_raw_path = r"https://openneuro.org/crn/datasets/ds001246/snapshots/1.2.1/files/sub-{:02d}:ses-perceptionTraining{:02d}:func:sub-{:02d}_ses-perceptionTraining{:02d}_task-perception_run-{:02d}_bold.nii.gz"
base_local_path = 'fmri_imagenet_sub_{}_percept_{}_run_{}.nii.gz'
list_perception_training = [1, 2, 3]
list_sub = [1, 2, 3, 4, 5]
runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for local_sub in list_sub:
    for local_run in runs:
        for local_perception_training in list_perception_training:
            prep_local_url = base_preproc_path.format(
                local_sub, local_perception_training, local_sub, local_perception_training, local_run)

            r = requests.head(prep_local_url, allow_redirects=True)
            if r.status_code == 200:
                # We only download if there is  a corresponding preprocessed file as those are "QCed"
                local_url = base_raw_path.format(local_sub, local_perception_training, local_sub,
                                                 local_perception_training, local_sub, local_perception_training, local_run)

                local_filename = base_local_path.format(
                    local_sub, local_perception_training, local_run)
                dst = '/home/ec2-user/fmri_data/training/'+local_filename
                try:
                    urlretrieve(local_url, dst)
                except:
                    print("issues with "+str(dst))


# remove files that had trouble downloading
list_files = os.listdir('/home/ec2-user/fmri_data/training/')
for file in list_files:
    local_path = os.path.join('/home/ec2-user/fmri_data/training/', file)
    if os.path.getsize(local_path) < 1024:
        os.remove(local_path)

"""
# We only keep 110 files to avoid memory issue
list_files = os.listdir('/home/ec2-user/fmri_data/training/')

total_files_nb = len(list_files)

list_del  = np.random.choice(list_files, total_files_nb-110, replace=False)

for file in list_del:
    local_path = os.path.join('/home/ec2-user/fmri_data/training/', file)
    os.remove(local_path)i
"""
