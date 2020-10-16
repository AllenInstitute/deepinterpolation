import ophysextractor.utils.util as util 
from ophysextractor.datasets.motion_corr_physio import MotionCorrPhysio
from ophysextractor.datasets.lims_ophys_experiment import LimsOphysExperiment
import numpy as np
import os
import json
import random
import pickle

img_per_movie = 100000
pre_post_frame = 30
number_train_movies = 1   
output_dir = '/home/jeromel/Documents/Projects/Deep2P/input_data/'
ind_id = 727407138

all_passed = {}
cursor = util.mongo.db.ophys_session_log.find({'ophys_experiments.0.id':ind_id})
for indiv_c in cursor:
    local_exp = indiv_c['ophys_experiments'][0]['id']
    ophys_exp_path  = indiv_c['ophys_experiments'][0]['storage_directory']
    all_passed[local_exp]=ophys_exp_path

list_lims_id = list(all_passed.keys())
list_train = list_lims_id[:number_train_movies]

local_path_train = os.path.join(output_dir, '2019-09-17-single_movie_train.json')

def save_json_list(all_passed, list_selected, json_path):
    train_data = {}
    for indiv_exp in list_selected:
        try:
            local_dict = {}
            local_lims = LimsOphysExperiment(indiv_exp)
            movie_obj = MotionCorrPhysio(ophys_experiment = local_lims)
            length_array = movie_obj.data_pointer.shape[0]
            local_data = movie_obj.data_pointer[1:100,:,:].flatten()
            local_mean = np.mean(local_data)
            local_std = np.std(local_data)
            
            list_pull = np.random.choice(np.arange(pre_post_frame, length_array-pre_post_frame), img_per_movie, replace=False)
            local_dict['path'] = all_passed[indiv_exp]
            local_dict['frames'] = list_pull.tolist()
            local_dict['mean'] = local_mean         
            local_dict['std'] = local_std

            train_data[str(indiv_exp)] = local_dict
        except: 
            print("issues with "+str(indiv_exp))

    with open(json_path, 'w+') as json_file:
        json.dump(train_data, json_file)

save_json_list(all_passed, list_train, local_path_train)
