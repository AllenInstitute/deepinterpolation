import ophysextractor.utils.util as util 
from ophysextractor.datasets.motion_corr_physio import MotionCorrPhysio
from ophysextractor.datasets.lims_ophys_experiment import LimsOphysExperiment
import numpy as np
import os
import json
import random
import pickle

img_per_movie = 10000
pre_post_frame = 300
proportion_train = 90   
proportion_test = 10

output_dir = '/home/jeromel/Documents/Projects/Deep2P/input_data/'

all_passed = {}
cursor = util.mongo.db.ophys_session_log.find({ "$and": [{'ophys_experiments.0.workflow_state':'passed'},{'rig':{ "$in": ["MESO.1"]}}]})

for indiv_c in cursor:
    for plane_number in np.arange(8):
        try: 
            local_exp = indiv_c['ophys_experiments'][plane_number]['id']
            ophys_exp_path  = indiv_c['ophys_experiments'][plane_number]['storage_directory']
            all_passed[local_exp]=ophys_exp_path
        except: 
            print('Plane missing for '+str(local_exp))

list_lims_id = list(all_passed.keys())
random.shuffle(list_lims_id)

total_movies = len(list_lims_id)
number_train_movies = int(np.round(proportion_train*float(total_movies)/100))
number_test_movies = int(np.round(proportion_test*float(total_movies)/100))

list_train = list_lims_id[:number_train_movies]

list_after_train = np.setdiff1d(list_lims_id, list_train)
random.shuffle(list_after_train)
list_test = list_after_train[:number_test_movies]
local_path_train = os.path.join(output_dir, '2019-09-16-train-very-large-meso-norm.json')
local_path_test = os.path.join(output_dir, '2019-09-16-test-very-large-meso-norm.json')

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
            list_pull = np.random.randint(low = pre_post_frame, high = length_array-pre_post_frame, size=img_per_movie)
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
save_json_list(all_passed, list_test, local_path_test)