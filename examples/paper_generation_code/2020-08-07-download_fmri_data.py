from urllib.request import urlretrieve
import os

base_path = 'https://openneuro.org/crn/datasets/ds001246/snapshots/1.2.1/files/derivatives:preproc-spm:output:sub-{:02d}:ses-perceptionTraining{:02d}:func:sub-{:02d}_ses-perceptionTraining{:02d}_task-perception_run-{:02d}_bold_preproc.nii.gz'
base_local_path = 'fmri_imagenet_sub_{}_percept_{}_run_{}.nii.gz'
list_perception_training = [1,2,3]
list_sub = [1,2,3,4,5]
runs = [1,2,3,4,5,6,7,8,9,10]

for local_sub in list_sub:
    for local_run in runs:
        for local_perception_training in list_perception_training:
            local_url = base_path.format(local_sub, local_perception_training, local_sub, local_perception_training, local_run)

            local_filename = base_local_path.format(local_sub, local_perception_training, local_run)
            dst = '/home/ec2-user/fmri_data/training/'+local_filename
            try:
                urlretrieve(local_url, dst)
            except: 
                print("issues with "+str(dst))


list_files = os.listdir('/home/ec2-user/fmri_data/training/')
for file in list_files:
    local_path = os.path.join('/home/ec2-user/fmri_data/training/',file)
    if os.path.getsize(local_path) < 1024:
        os.remove(local_path)
