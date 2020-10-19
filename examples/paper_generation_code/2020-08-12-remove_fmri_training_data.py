import os
import numpy as np

# We only keep 80 files to avoid memory issue
list_files = os.listdir('/home/ec2-user/fmri_data/training/')

total_files_nb = len(list_files)

list_del  = np.random.choice(list_files, total_files_nb-80, replace=False)

for file in list_del:
    local_path = os.path.join('/home/ec2-user/fmri_data/training/', file)
    os.remove(local_path)
